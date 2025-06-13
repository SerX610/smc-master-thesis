import numpy as np
import os
import torch

from clap_transfer_learning import CLAPTransferLearning
from dataclasses import dataclass
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import Dataset, DataLoader

seed = 42
seed_everything(seed, workers=True)


class AudioEmbeddingDataset(Dataset):
    """
    A custom PyTorch Dataset for loading audio embeddings and corresponding labels.

    Args:
        audio_embeddings_file (str): Path to the .npy file containing audio embeddings.
        labels_file (str): Path to the .npy file containing labels corresponding to the audio embeddings.
    """
    def __init__(self, audio_embeddings_file, labels_file):
        self.audio_embeddings = torch.from_numpy(np.load(audio_embeddings_file)).float()
        self.labels = torch.from_numpy(np.load(labels_file)).float()

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the index and the corresponding label.
        """
        return idx, self.labels[idx]


@dataclass
class CLAPTrainerConfig:
    """
    Configuration class for CLAPTrainer that holds various hyperparameters and file paths.
    """

    embeddings_folder: str = "mtt_embeddings"  # Path to the folder containing the embeddings
    model_dir: str = "../../models/trained/"  # Directory to save the model checkpoint
    model_ckpt: str = "mtt-clap-multilabel-classifier-checkpoint"  # Filename for saving the model checkpoint
    batch_size: int = 64  # Batch size for training
    max_epochs: int = 10  # Maximum number of training epochs


class CLAPTrainer:
    """
    Trainer class to handle data preparation, model initialization, and training.

    Args:
        config (CLAPTrainerConfig): A configuration object that contains training parameters.
    """
    def __init__(self, config: CLAPTrainerConfig):
        self.config = config

        # Ensure the model directory exists
        os.makedirs(self.config.model_dir, exist_ok=True)

        # Define dataset paths for training and validation
        self.train_audio_embeddings_file = f"{self.config.embeddings_folder}/train_audio_embeddings.npy"
        self.train_labels_file = f"{self.config.embeddings_folder}/train_labels.npy"
        self.valid_audio_embeddings_file = f"{self.config.embeddings_folder}/valid_audio_embeddings.npy"
        self.valid_labels_file = f"{self.config.embeddings_folder}/valid_labels.npy"

        # Initialize the DataLoader for both training and validation datasets
        self.train_dataloader = DataLoader(
            AudioEmbeddingDataset(self.train_audio_embeddings_file, self.train_labels_file),
            batch_size=self.config.batch_size, shuffle=True, num_workers=4
        )
        self.valid_dataloader = DataLoader(
            AudioEmbeddingDataset(self.valid_audio_embeddings_file, self.valid_labels_file),
            batch_size=self.config.batch_size, shuffle=False, num_workers=4
        )

        # Initialize the CLAP model for transfer learning
        self.model = CLAPTransferLearning(self.train_audio_embeddings_file)

    def train_model(self):
        """
        Trains the CLAP model using PyTorch Lightning with checkpointing.
        """

        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath=self.config.model_dir,
            filename=self.config.model_ckpt,
        )

        # Initialize the PyTorch Lightning trainer with GPU acceleration if available
        trainer = Trainer(
            max_epochs=self.config.max_epochs,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            callbacks=[checkpoint_callback],
        )

        # Train the model
        trainer.fit(self.model, self.train_dataloader, self.valid_dataloader)


def main():

    # Initialize configuration with specified parameters
    config = CLAPTrainerConfig(
        embeddings_folder="mtt_ms_embeddings",
        model_dir="../../models/trained/",
        model_ckpt="mtt-ms-clap-multilabel-classifier-checkpoint",
        batch_size=64,
        max_epochs=10,
    )

    # Initialize and train the model using the configuration.
    trainer = CLAPTrainer(config)
    trainer.train_model()


if __name__ == "__main__":
    main()
