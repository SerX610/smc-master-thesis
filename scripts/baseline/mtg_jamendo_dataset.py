import os
import subprocess
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    def __init__(self, root: str, download: bool = True):
        self.root = os.path.expanduser(root)
        if download:
            self.download()

    def __getitem__(self, index):
        raise NotImplementedError

    def download(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class MTGJamendo(AudioDataset):
    raise NotImplementedError