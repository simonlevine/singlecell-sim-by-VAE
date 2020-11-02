import pandas as pd
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from pipeline.paths import INTERMEDIATE_DATA_DIR


class SingleCellDataset:
    def __init__(self, fp):
        self.df = pd.read_json(fp)
        self.genes = [c for c in self.df.columns.tolist() if c ]
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        gene_expression, cell_type = self.X[idx,:]
        return gene_expression, cell_type
    
    @property
    def X(self):
        return self.df[self.genes].values()
    
    @property
    def y(self):
        return self.df["tissue"]


class SingleCellDataModule(pl.LightningDataModule):
    def __init__(self, train_fp, test_fp, val_fp, batch_size):
        super().__init__()
        self.train_fp = train_fp
        self.test_fp = test_fp
        self.val_fp = val_fp
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = SingleCellDataset(self.train_fp)
        self.test_dataset = SingleCellDataset(self.test_fp)
        self.val_dataset = SingleCellDataset(self.val_fp)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


def load_single_cell_data(batch_size=32):
    return SingleCellDataModule(
        INTERMEDIATE_DATA_DIR/"train_data.json",
        INTERMEDIATE_DATA_DIR/"test_data.json",
        INTERMEDIATE_DATA_DIR/"validate_data.json",
        batch_size
    )