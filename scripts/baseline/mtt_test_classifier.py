import json
import numpy as np
import torch

from clap_transfer_learning import CLAPTransferLearning
from pytorch_lightning import Trainer
from torch.utils.data import Dataset, DataLoader


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


def get_top_n_indices(test_labels_file, top_n=50):
    """
    Calculates the indices of the top N most frequent labels.
    
    Args:
        test_labels_file (str): Path to the labels file.
        top_n (int, optional): The number of top labels to select. Default is 50.

    Returns:
        np.ndarray: Array of indices of the top N most frequent labels.
    """
    test_labels = np.load(test_labels_file)
    label_counts = np.sum(test_labels, axis=0)
    sorted_indices = np.argsort(label_counts)[::-1]
    top_n_indices = sorted_indices[:top_n]
    return top_n_indices


def load_tag_index_mapping(tag_index_mapping_path):
    """
    Loads the tag to index mapping from a JSON file.
    
    Args:
        tag_index_mapping_path (str): Path to the JSON file with the tag index mapping.
    
    Returns:
        dict: The loaded tag index mapping.
    """
    with open(tag_index_mapping_path, 'r') as f:
        tag_index_mapping = json.load(f)
    return tag_index_mapping


def prepare_data(audio_embeddings_file, labels_file):
    """
    Prepares the testing data loader.
    
    Args:
        audio_embeddings_file (str): Path to the audio embeddings file.
        labels_file (str): Path to the labels file.
    
    Returns:
        DataLoader: DataLoader for testing data.
    """
    test_dataset = AudioEmbeddingDataset(audio_embeddings_file, labels_file)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
    return test_dataloader


def test_model(model, test_dataloader):
    """
    Tests the CLAPTransferLearning model using PyTorch Lightning.
    
    Args:
        model (CLAPTransferLearning): The model to test.
        test_dataloader (DataLoader): DataLoader for testing data.
    """
    trainer = Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu")
    trainer.test(model, test_dataloader)


def main():
    """
    Main function to run the model testing pipeline.
    """
    # Define paths to required files
    test_audio_embeddings_file = "mtt_laion_embeddings/test_audio_embeddings.npy"
    test_labels_file = "mtt_laion_embeddings/test_labels.npy"
    tag_index_mapping_path = "../../data/mtt/tag_index_mapping.json"
    model_ckpt = "../../models/trained/mtt-laion-clap-multilabel-classifier-checkpoint.ckpt"
    plot_dir = "../../results/mtt_laion_clap_multilabel_classification_confusion_matrix.png"

    # Get indices of the top N most frequent labels
    top_n_indices = get_top_n_indices(test_labels_file)

    # Load the tag-to-index mapping
    tag_index_mapping = load_tag_index_mapping(tag_index_mapping_path)

    # Prepare the test data loader
    test_dataloader = prepare_data(test_audio_embeddings_file, test_labels_file)

    # Load the pre-trained model from the checkpoint
    model = CLAPTransferLearning.load_from_checkpoint(
        model_ckpt,
        audio_embeddings_file=test_audio_embeddings_file,
        top_n_indices=top_n_indices,
        tag_index_mapping=tag_index_mapping,
        plot_dir=plot_dir
    )

    # Test the model with the test data
    test_model(model, test_dataloader)


if __name__=="__main__":
    main()
