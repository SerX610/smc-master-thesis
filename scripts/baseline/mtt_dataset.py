"""
This module provides a PyTorch Dataset class for loading and processing the
MagnaTagATune dataset. The MagnaTagATune dataset is a multi-label dataset for
music classification, featuring audio files with associated tags describing
genre, instrumentation, and other musical attributes. The provided class loads
audio file paths and their corresponding tag data in a PyTorch-friendly format.
"""

import numpy as np
import os
import pickle
import torch

from torch.utils.data import Dataset


class MagnaTagATuneDataset(Dataset):
    """
    A PyTorch Dataset class for loading the MagnaTagATune dataset using preprocessed files.
    This class loads MP3 file paths, corresponding tags, and one-hot encoded tag vectors for 
    the specified split (train, validation, or test).
    """
    
    def __init__(self, split, root="../../data/mtt"):
        """
        Args:
            split (str): One of 'train', 'valid', or 'test'.
            root (str, optional): Root directory for dataset files. Default is "../../data/mtt".
        """

        self.root = root
    
        # Define paths for file lists and tag data based on the split (train/valid/test)
        self.split_paths = {
            "train": (f"{self.root}/MTAT_split/train_list_pub.cP", f"{self.root}/MTAT_split/y_train_pub.npy"),
            "valid": (f"{self.root}/MTAT_split/valid_list_pub.cP", f"{self.root}/MTAT_split/y_valid_pub.npy"),
            "test": (f"{self.root}/MTAT_split/test_list_pub.cP", f"{self.root}/MTAT_split/y_test_pub.npy")
        }
        
        # Path to the top 50 tags file
        top_50_tags_path = f"{self.root}/MTAT_split/top50_tags.txt"

        # Ensure the provided split is valid
        if split not in self.split_paths:
            raise ValueError("Split must be one of 'train', 'valid', or 'test'")

        # Load file paths, tags, and one-hot encoded tags for the selected split
        file_paths, tag_file = self.split_paths[split]
        self.file_paths, self.tags, self.one_hot_tags = self._load_mp3_files_and_tags(file_paths, tag_file, top_50_tags_path)
        
    def __getitem__(self, index):
        """
        Retrieves an item from the dataset.

        Args:
            index (int): Index of the item to retrieve.

        Returns:
            tuple: A tuple containing:
                - file_path (str): Path to the MP3 file.
                - target_tags (list): List of tag names associated with the file.
                - one_hot_target (Tensor): One-hot encoded tags for the file.
        """
        file_path = os.path.join(self.root, self.file_paths[index])
        target = self.tags[index]
        one_hot_target = torch.tensor(self.one_hot_tags[index], dtype=torch.float32)

        return file_path, target, one_hot_target

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The total number of samples (files) in the dataset.
        """
        return len(self.file_paths)

    def _load_mp3_files_and_tags(self, file_paths, tag_file, top_50_tags_file):
        """
        Loads the MP3 file paths, corresponding tags, and one-hot encoded tag vectors.
        
        Args:
            file_paths (str): Path to the pickle file containing the list of file paths.
            tag_file (str): Path to the .npy file containing one-hot encoded tags.
            top_50_tags_file (str): Path to the text file containing the list of the top 50 tags.
            
        Returns:
            tuple: A tuple containing:
                - list of file paths for MP3 files.
                - list of tags corresponding to each file.
                - numpy array of one-hot encoded tag vectors.
        """
        # Load MP3 file paths (with '.mp3' extension)
        with open(file_paths, "rb") as f:
            files = pickle.load(f)
            mp3_files = [path.replace(".npy", ".mp3") for path in files]

        # Load one-hot encoded tags
        one_hot_tags = np.load(tag_file)

        # Load the list of top 50 tags
        with open(top_50_tags_file, "r") as f:
            top_50_tags = [line.strip() for line in f]

        # Map one-hot encoded tags to their respective tag names
        tags = [
            [top_50_tags[i] for i in range(len(top_50_tags)) if vector[i] == 1]
            for vector in one_hot_tags
        ]

        return mp3_files, tags, one_hot_tags
