import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pytorch_lightning as pl
from typing import Optional, Tuple

class PointwiseDataset(Dataset):
    def __init__(self, data_path: str, n_feature: int, n_target: int):
        self.data = np.load(data_path)
        self.n_feature = n_feature
        self.n_target = n_target
        
        # Split features and targets
        self.features = torch.FloatTensor(self.data[:, :n_feature])
        self.targets = torch.FloatTensor(self.data[:, n_feature:n_feature + n_target])
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class NIFDataModule(pl.LightningDataModule):
    def __init__(self, data_path: str, n_feature: int, n_target: int,
                 batch_size: int = 32, num_workers: int = 4):
        super().__init__()
        self.data_path = data_path
        self.n_feature = n_feature
        self.n_target = n_target
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            self.train_dataset = PointwiseDataset(
                self.data_path, self.n_feature, self.n_target
            )
            
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True
        )
