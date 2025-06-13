"""
This module provides a PyTorch Dataset class for loading and processing the
Song Describer Dataset (SDD). The dataset consists of audio files with textual
descriptions (captions) that describe the musical content.
"""

import os
import pandas as pd

from torch.utils.data import Dataset


class SongDescriberDataset(Dataset):
    """
    A PyTorch Dataset class for loading the Song Describer Dataset (SDD) using preprocessed files.
    This class loads MP3 file paths and their corresponding captions in a PyTorch-friendly format.
    """
    
    def __init__(self, root="../../data/sdd", csv_file_name="song_describer.csv"):
        """
        Initializes the dataset by loading file paths and captions from a CSV file.

        Args:
            root (str, optional): Root directory for dataset files. Default is "../../data/sdd".
            csv_file_name (str, optional): CSV file name with the dataset metadata. Default is "song_describer.csv".
        """
        self.root = root
        csv_file = f"{self.root}/{csv_file_name}"
        self.file_paths, self.captions = self._load_data_from_csv(csv_file)
        
    def __getitem__(self, index):
        """
        Retrieves an item from the dataset.

        Args:
            index (int): Index of the item to retrieve.

        Returns:
            tuple: A tuple containing:
                - file_path (str): Path to the MP3 file.
                - caption (str): Text description of the audio file.
        """
        file_path = os.path.join(self.root, self.file_paths[index])
        caption = self.captions[index]
        
        return file_path, caption

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The total number of samples (files) in the dataset.
        """
        return len(self.file_paths)

    def _load_data_from_csv(self, csv_file):
        """
        Loads MP3 file paths and captions from a CSV file.
        
        Args:
            csv_file (str): Path to the CSV file containing metadata.
            
        Returns:
            tuple: A tuple containing:
                - list of file paths for MP3 files (formatted for 2-minute versions).
                - list of captions corresponding to each file.
        """
        # Read CSV file
        df = pd.read_csv(csv_file)

        # Filter to include only valid data entries
        df_valid = df[df['is_valid_subset'] == True]

        # Format file paths correctly
        file_paths = [f"audio/{path.replace('.mp3', '')}.2min.mp3" for path in df_valid['path']]
        captions = df_valid['caption'].tolist()

        return file_paths, captions
