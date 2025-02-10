"""
This module provides a PyTorch Dataset class for loading and processing the
GTZAN dataset. The GTZAN dataset consists of 1000 audio tracks, each 30
seconds long, categorized into 10 genres. The dataset is organized into
subdirectories, with each genre having its own folder containing 100 audio
files. However, the file "jazz/jazz.00054" is corrupted and has been removed
from the dataset.
"""

import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm


class GTZANDataset(Dataset):
    """
    A PyTorch Dataset class for loading the GTZAN dataset.
    It provides audio file paths, genre labels, and one-hot encoded targets for classification tasks.
    """
    def __init__(self, root: str, reading_transformations: nn.Module = None):
        """
        Args:
            root (str): Path to the root directory of the GTZAN dataset.
            reading_transformations (nn.Module): Optional transformations to apply to the audio files.
        """
        self.root = root
        self.audio_dir = os.path.join(self.root, "genres_original")
        self.targets, self.audio_paths = [], []
        self.pre_transformations = reading_transformations

        # Load metadata and prepare targets and audio paths
        print("Loading audio files")
        for genre in tqdm(os.listdir(self.audio_dir)):
            genre_dir = os.path.join(self.audio_dir, genre)
            if os.path.isdir(genre_dir):
                for file_name in os.listdir(genre_dir):
                    file_path = os.path.join(genre_dir, file_name)
                    self.targets.append(genre)
                    self.audio_paths.append(file_path)

        # Create class-to-index mapping
        self.classes = sorted(set(self.targets))
        self.class_to_idx = {genre: idx for idx, genre in enumerate(self.classes)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (audio_path, target, one_hot_target)
        """
        file_path, target = self.audio_paths[index], self.targets[index]
        idx = torch.tensor(self.class_to_idx[target])
        one_hot_target = torch.zeros(len(self.classes)).scatter_(0, idx, 1).reshape(1, -1)
        return file_path, target, one_hot_target

    def __len__(self):
        return len(self.audio_paths)
