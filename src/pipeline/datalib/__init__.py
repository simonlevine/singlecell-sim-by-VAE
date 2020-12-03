import itertools as it
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scanpy as sc
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from pipeline.helpers.paths import DATA_SPLIT_FPS, SIMULATED_GENE_EXPRESSION_FP


###############
## REAL DATA ##
###############



class SingleCellDataset(torch.utils.data.IterableDataset):
    def __init__(self, anndata_fp, chunk_size=6000):
        self.annotations = sc.read_h5ad(anndata_fp, backed="r")
        self.genes = [str(gene) for gene in self.annotations.var_names.tolist()]
        self.cell_types = self.annotations.obs.cell_type_coarse.unique().tolist()
        self.ventilator_statuses = self.annotations.obs.Ventilated.unique().tolist()
        self.cell_type_encoder = None
        self.ventilator_status_encoder = None
        self.chunk_size = chunk_size

    def __iter__(self):
        return self.lazy_iter_annotations()

    def lazy_iter_annotations(self):
        for i in it.count():
            if len(self) < i * self.chunk_size:
                break
            s = slice(
                i * self.chunk_size,
                min([((i + 1) * self.chunk_size), len(self)])
            )
            gene_expressions = self.annotations[s, :].X
            n_rows, _ = gene_expressions.shape
            if 0 < n_rows:
                cell_type = self.annotations.obs.cell_type_coarse[s]
                cell_type = np.array(cell_type).reshape(1,-1)
                if self.cell_type_encoder:
                    cell_type = self.cell_type_encoder.transform(cell_type.T)
                ventilator_status = self.annotations.obs.Ventilated[s]
                ventilator_status = np.array(ventilator_status).reshape(1,-1)
                if self.ventilator_status_encoder:
                    ventilator_status = self.ventilator_status_encoder.transform(ventilator_status.T)
                for i in range(n_rows):
                    yield gene_expressions[i,:], cell_type[i], ventilator_status[i]

    def __len__(self):
        return len(self.annotations)


class SingleCellDataModule(pl.LightningDataModule):
    def __init__(self, train_fp, test_fp, val_fp, batch_size):
        super().__init__()
        self.train_fp = train_fp
        self.test_fp = test_fp
        self.val_fp = val_fp
        self.batch_size = batch_size
        self.genes = set()
        self.cell_types = set()
        self.ventilator_statuses = set()

    def setup(self, stage=None):
        self.train_dataset = SingleCellDataset(self.train_fp)
        self.test_dataset = SingleCellDataset(self.test_fp)
        self.val_dataset = SingleCellDataset(self.val_fp)
        for dataset in [self.train_dataset, self.test_dataset, self.val_dataset]:
            self.genes.update(dataset.genes)
            self.cell_types.update(dataset.cell_types)
            self.ventilator_statuses.update(dataset.ventilator_statuses)
        cell_type_encoder = LabelEncoder()
        cell_type_encoder.fit(np.array(list(self.cell_types)).reshape(-1, 1))
        ventilator_status_encoder = LabelEncoder()
        ventilator_status_encoder.fit(np.array(list(self.ventilator_statuses)).reshape(-1, 1))
        for dataset in [self.train_dataset, self.test_dataset, self.val_dataset]:
            dataset.cell_type_encoder = cell_type_encoder
            dataset.ventilator_status_encoder = ventilator_status_encoder

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

def load_single_cell_data(batch_size=32):
    data_module = SingleCellDataModule(*DATA_SPLIT_FPS, batch_size)
    data_module.setup()
    return data_module

###############
## FAKE DATA ##
###############

class SimulatedSingleCellDataset(torch.utils.data.Dataset):
    def __init__(self, fp):
        self.df = pd.read_parquet(fp)
        self.genes = [n for n in self.df.columns if n not in {"cell_type", "ventilator_status"}]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row_idx = self.dc.index[i]
        gene_expression = self.df.loc[row_idx, self.genes]
        cell_type = self.df.loc[row_idx, "cell_type"]
        ventilator_status = self.df.loc[row_idx, "ventilator_status"]
        return gene_expression, cell_type, ventilator_status


class SimulatedSingleCellDataModule(pl.LightningDataModule):
    def __init__(self, fp, batch_size):
        super().__init__()
        self.fp = fp
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = SimulatedSingleCellDataset(self.train_fp)


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


class ConcatDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, *datamodules):
        self.single_cell, self.simulated = datamodules
        self.batch_size = batch_size

    def setup(self, stage=None):

        self.single_cell.setup()
        self.simulated.setup()
        
        self.train_dataset = ConcatDataset(
                    self.single_cell.train_dataset,
                    self.simulated.train_dataset
                    )
        self.val_dataset = ConcatDataset(
                    self.single_cell.val_dataset,
                    self.simulated.val_dataset
                    )

        self.test_dataset = ConcatDataset(
                    self.single_cell.test_dataset,
                    self.simulated.test_dataset
                    )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


def load_combined_single_cell_data(batch_size=32):
    single_cell_datamodule = load_single_cell_data(*DATA_SPLIT_FPS, batch_size)
    simulated_datamodule = load_simulated_single_cell_data(SIMULATED_GENE_EXPRESSION_FP, batch_size)
    concat_datamodule = ConcatDataModule(batch_size,single_cell_datamodule,simulated_datamodule)
    concat_datamodule.setup()
    return concat_datamodule

    
