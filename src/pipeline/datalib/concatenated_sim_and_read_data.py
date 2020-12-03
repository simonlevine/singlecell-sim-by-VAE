import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from . import real_data, sim_data

class ConcatDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, single_cell_datamodule, simulated_datamodule):
        super().__init__()
        self.single_cell = single_cell_datamodule
        self.simulated = simulated_datamodule
        self.batch_size = batch_size

    def setup(self, stage=None):        
        self.train_dataset = torch.utils.data.ChainDataset([
            self.simulated.train_dataset,
            self.single_cell.train_dataset,
        ])
        self.val_dataset = self.single_cell.val_dataset
        self.test_dataset = self.single_cell.test_dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


def load_combined_single_cell_data(batch_size=32):
    single_cell_datamodule = real_data.load_single_cell_data(batch_size)
    ventilator_status_encoder = single_cell_datamodule.train_dataset.ventilator_status_encoder
    simulated_datamodule = sim_data.load_simulated_single_cell_data(ventilator_status_encoder, batch_size)
    concat_datamodule = ConcatDataModule(batch_size, single_cell_datamodule, simulated_datamodule)
    concat_datamodule.setup()
    return concat_datamodule
