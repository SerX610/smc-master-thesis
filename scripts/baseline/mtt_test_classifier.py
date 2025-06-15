import json
import numpy as np
import torch

from clap_transfer_learning import CLAPTransferLearning
from dataclasses import dataclass
from pytorch_lightning import Trainer
from torch.utils.data import Dataset, DataLoader


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
class CLAPTestConfig:
    """
    Configuration class for CLAPTester that holds various file paths and parameters for testing.
    """
    
    embeddings_folder: str = "mtt_embeddings"  # Path to the folder containing the embeddings
    model_ckpt: str = "../../models/trained/mtt-clap-multilabel-classifier-checkpoint.ckpt"  # Path to the model checkpoint
    batch_size: int = 64  # Batch size for testing
    tag_index_mapping_path: str = "../../data/mtt/MTAT_split/top50_tags.txt"  # Path to the tag list `.txt` file
    plot_dir: str = "../../results/mtt_clap_multilabel_classification_confusion_matrix.png"  # Directory to save the plot


class CLAPTester:
    """
    Tester class to handle data preparation, model loading, and testing.

    Args:
        config (CLAPTestConfig): A configuration object that contains testing parameters.
    """
    def __init__(self, config: CLAPTestConfig):
        self.config = config
        
        # Define dataset paths for testing
        self.test_audio_embeddings_file = f"{self.config.embeddings_folder}/test_audio_embeddings.npy"
        self.test_labels_file = f"{self.config.embeddings_folder}/test_labels.npy"

        # Initialize the DataLoader for testing dataset
        self.test_dataloader = DataLoader(
            AudioEmbeddingDataset(self.test_audio_embeddings_file, self.test_labels_file),
            batch_size=self.config.batch_size, shuffle=False, num_workers=4
        )

        # Load the tag-to-index mapping from the .txt file
        self.tag_index_mapping = self._load_tag_index_mapping_from_txt()

        # Load the model from the checkpoint
        self.model = CLAPTransferLearning.load_from_checkpoint(
            self.config.model_ckpt,
            audio_embeddings_file=self.test_audio_embeddings_file,
            tag_index_mapping=self.tag_index_mapping,
            plot_dir=self.config.plot_dir,
        )

    def test_model(self):
        """
        Tests the CLAP model using PyTorch Lightning.
        """
        trainer = Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu")
        trainer.test(self.model, self.test_dataloader)

    def _load_tag_index_mapping_from_txt(self):
        """
        Loads the tag-to-index mapping from a .txt file.
        
        Args:
            tag_index_mapping_path (str): Path to the .txt file with the tag list.
        
        Returns:
            dict: A dictionary mapping indices to tags.
        """
        # Read the file and create a list of tags
        with open(self.config.tag_index_mapping_path, 'r') as f:
            tags = [line.strip() for line in f.readlines()]
        
        # Create a dictionary where index is the key and tag is the value
        tag_index_mapping = {index: tag for index, tag in enumerate(tags)}
        
        return tag_index_mapping


def main():
    """
    Main function to run the model testing pipeline.
    """
    # Initialize configuration with specified parameters
    config = CLAPTestConfig(
        embeddings_folder="mtt_laion_embeddings", 
        model_ckpt="../../models/trained/mtt-laion-clap-multilabel-classifier-checkpoint.ckpt",
        batch_size=64,
        tag_index_mapping_path="../../data/mtt/MTAT_split/top50_tags.txt",
        plot_dir="../../results/mtt_laion_clap_multilabel_classification_confusion_matrix.png"
    )

    # Initialize the tester and run the test
    tester = CLAPTester(config)
    tester.test_model()


if __name__=="__main__":
    main()
