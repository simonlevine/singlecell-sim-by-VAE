import numpy as np
import pytorch_lightning as pl
import scanpy as sc
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader
from pipeline.helpers.paths import DATA_SPLIT_FPS


class SingleCellDataset:
    def __init__(self, anndata_fp):
        self.annotations = sc.read_h5ad(anndata_fp, backed="r")
        self.genes = [str(gene) for gene in self.annotations.var_names.tolist()]
        self.tissues = set(self.y)
        self.label_encoder = None
        
    def __len__(self):
        n, _ = self.annotations.shape
        return n
    
    def __getitem__(self, idx):
        gene_expression = self.annotations.X[idx,:]
        cell_type = self.annotations.obs.tissue[idx].decode()
        if self.label_encoder:
            M = self.label_encoder.transform([[cell_type]]).todense()
            cell_type = np.squeeze(np.asarray(M))
        return gene_expression, cell_type
    
    @property
    def X(self):
        return self.annotations.X
    
    @property
    def y(self):
        return [tissue.decode() for tissue in self.annotations.obs.tissue.tolist()]


class SingleCellDataModule(pl.LightningDataModule):
    def __init__(self, train_fp, test_fp, val_fp, batch_size):
        super().__init__()
        self.train_fp = train_fp
        self.test_fp = test_fp
        self.val_fp = val_fp
        self.batch_size = batch_size
        self.genes = set()
        self.tissues = set()

    def setup(self, stage=None):
        self.train_dataset = SingleCellDataset(self.train_fp)
        self.test_dataset = SingleCellDataset(self.test_fp)
        self.val_dataset = SingleCellDataset(self.val_fp)
        for dataset in [self.train_dataset, self.test_dataset, self.val_dataset]:
            self.genes.update(dataset.genes)
            self.tissues.update(dataset.tissues)
        label_encoder = OneHotEncoder()
        label_encoder.fit(np.array(list(self.tissues)).reshape(-1, 1))
        for dataset in [self.train_dataset, self.test_dataset, self.val_dataset]:
            dataset.label_encoder = label_encoder

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