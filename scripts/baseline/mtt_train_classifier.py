import numpy as np
import torch

from clap_transfer_learning import CLAPTransferLearning
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import Dataset, DataLoader, random_split

seed = 42
seed_everything(seed, workers=True)


class AudioEmbeddingDataset(Dataset):
    """
    A custom PyTorch Dataset for loading audio embeddings and corresponding labels.
    
    Args:
        audio_embeddings_file (str): Path to the numpy file containing audio embeddings.
        labels_file (str): Path to the numpy file containing labels.
    """
    def __init__(self, audio_embeddings_file, labels_file):
        self.audio_embeddings = torch.from_numpy(np.load(audio_embeddings_file)).float()
        self.labels = torch.from_numpy(np.load(labels_file)).float()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return idx, self.labels[idx]


def prepare_data(audio_embeddings_file, labels_file, batch_size=64, train_split=0.8):
    """
    Prepares the training and validation data loaders.
    
    Args:
        audio_embeddings_file (str): Path to the audio embeddings file.
        labels_file (str): Path to the labels file.
        batch_size (int, optional): Batch size for data loaders. Default is 64.
        train_split (float, optional): Proportion of data to use for training. Default is 0.8.
    
    Returns:
        tuple: A tuple containing training and validation DataLoader instances.
    """
    full_dataset = AudioEmbeddingDataset(audio_embeddings_file, labels_file)
    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_dataloader, val_dataloader


def train_model(model, model_ckpt, train_dataloader, val_dataloader, model_dir="../../models/trained/", max_epochs=10):
    """
    Trains the CLAPTransferLearning model using PyTorch Lightning.
    
    Args:
        model (CLAPTransferLearning): The model to train.
        model_ckpt (str): Name for the checkpoint file.
        train_dataloader (DataLoader): DataLoader for training data.
        val_dataloader (DataLoader): DataLoader for validation data.
        model_dir (str, optional): Directory to save the best model checkpoint. Default is '../../models/trained/'.
        max_epochs (int, optional): Maximum number of training epochs. Default is 10.
    """
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=model_dir,
        filename=model_ckpt,
        save_top_k=1,
        mode="min",
    )

    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model, train_dataloader, val_dataloader)


def main():
    # Initialize variables
    train_audio_embeddings_file = "mtt_laion_embeddings/train_audio_embeddings.npy"
    train_labels_file = "mtt_laion_embeddings/train_labels.npy"
    model_ckpt = "mtt-laion-clap-multilabel-classifier-checkpoint"

    # Initialize the training and validation data
    train_dataloader, val_dataloader = prepare_data(train_audio_embeddings_file, train_labels_file)

    # Initialize the model
    model = CLAPTransferLearning(train_audio_embeddings_file)

    # Train the model using the given data
    train_model(model, model_ckpt, train_dataloader, val_dataloader)


if __name__ == "__main__":
    main()
