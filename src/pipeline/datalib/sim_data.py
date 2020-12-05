import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pipeline.helpers.paths import SIMULATED_GENE_EXPRESSION_FP


class SimulatedSingleCellDataset(torch.utils.data.IterableDataset):
    def __init__(self, fp, ventilator_status_encoder):
        self.df = pd.read_parquet(fp)
        self.genes = [n for n in self.df.columns if n not in {"cell_type", "ventilator_status"}]
        self.ventilator_status_encoder = ventilator_status_encoder

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row_idx = self.df.index[i]
        gene_expression = self.df.loc[row_idx, self.genes].values.astype(np.float32)
        cell_type = self.df.loc[row_idx, "cell_type"]
        ventilator_status = self.df.loc[row_idx, "ventilator_status"]
        if self.ventilator_status_encoder:
            ventilator_status, = self.ventilator_status_encoder.transform([ventilator_status])
        return gene_expression, cell_type, ventilator_status

    def __iter__(self):
        # I know this is a terrible hack and I am sorry
        for i in range(len(self)):
            yield self[i]


class SimulatedSingleCellDataModule(pl.LightningDataModule):
    def __init__(self, fp, batch_size, ventilator_status_encoder):
        super().__init__()
        self.fp = fp
        self.batch_size = batch_size
        self.ventilator_status_encoder = ventilator_status_encoder

    def setup(self, stage=None):
        self.train_dataset = SimulatedSingleCellDataset(self.fp, self.ventilator_status_encoder)


def load_simulated_single_cell_data(ventilator_status_encoder=None, batch_size=32):
    data_module = SimulatedSingleCellDataModule(SIMULATED_GENE_EXPRESSION_FP, batch_size, ventilator_status_encoder)
    data_module.setup()
    return data_module
