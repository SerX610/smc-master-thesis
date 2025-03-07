"""
This module provides a PyTorch Dataset class for loading and processing the
MagnaTagATune dataset. The MagnaTagATune dataset is a multi-label dataset for
music classification, featuring audio files with associated tags describing
genre, instrumentation, and other musical attributes. The dataset includes an
annotations file where each row corresponds to an audio clip, witha file path
and a set of binary tags. The provided class loads audio file paths and their
corresponding tag data in a PyTorch-friendly format.
"""

import os
import pandas as pd
import torch

from torch.utils.data import Dataset, random_split


class MagnaTagATuneDataset(Dataset):
    """
    A PyTorch Dataset class for loading the MagnaTagATune dataset directly from annotations_final.csv.
    It provides audio file paths and multi-label tags for classification tasks.
    """

    def __init__(self, root, annotations_file):
        """
        Args:
            root (str): Path to the root directory of the MagnaTagATune dataset.
            annotations_file (str): Name of the annotations CSV file.
        """
        self.root = root

        # Load the annotations file
        self.annotations_path = os.path.join(self.root, annotations_file)
        self.annotations = pd.read_csv(self.annotations_path, sep="\t")

        # Extract mp3 paths and tag columns (ignore clip_id)
        self.file_paths = self.annotations["mp3_path"].tolist()
        self.tag_columns = self.annotations.columns[1:-1]  # Exclude 'clip_id' and 'mp3_path'

        # Convert tag data to a tensor (one-hot encoding)
        self.tags = torch.tensor(self.annotations[self.tag_columns].values, dtype=torch.float32)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (file_path, target, one_hot_target)
        """
        # Get the audio path, tag values and one-hot tags
        file_path = os.path.join(self.root, self.file_paths[index])
        target = [tag for tag, value in zip(self.tag_columns, self.tags[index]) if value == 1]
        one_hot_target = self.tags[index]  # Tensor of shape (188,)

        return file_path, target, one_hot_target

    def __len__(self):
        return len(self.file_paths)


def split_dataset(dataset, train_ratio=0.75, seed=42):
    """
    Splits the dataset into training and test sets.
    
    Args:
        dataset (Dataset): The full dataset.
        train_ratio (float): The ratio of training data (default is 0.75).
        seed (int): Random seed for reproducibility.
    
    Returns:
        tuple: (train_dataset, test_dataset)
    """
    torch.manual_seed(seed)
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    return train_dataset, test_dataset
