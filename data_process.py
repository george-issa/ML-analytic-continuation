# This script is for loading and processing the dataset
import numpy as np
import torch
from torch.utils.data import Dataset

class GreenFunctionDataset(Dataset):
    
    """Loading Green's functions as a custom dataset for PyTorch (as a tensor)."""
    
    def __init__(self, file_path, transform=None):
        
        # Load all Green's functions at once
        self.data = np.loadtxt(file_path, delimiter=",")
        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        
        x = self.data[idx]
        
        # Perform any transormation on the data
        # Could be normalization, augmentation, or noise addition
        if self.transform:
            x = self.transform(x)
        return x