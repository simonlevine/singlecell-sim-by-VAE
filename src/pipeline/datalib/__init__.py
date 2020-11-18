import itertools as it
import numpy as np
import pytorch_lightning as pl
import scanpy as sc
import torch
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader
from pipeline.helpers.paths import DATA_SPLIT_FPS


class SingleCellDataset(torch.utils.data.IterableDataset):
    def __init__(self, anndata_fp, chunk_size=6000):
        self.annotations = sc.read_h5ad(anndata_fp, backed="r")
        self.genes = [str(gene) for gene in self.annotations.var_names.tolist()]
        self.tissues = set(self.annotations.obs.tissue.tolist())
        self.label_encoder = None
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
                tissues = self.annotations.obs.tissue[s]
                tissues = np.array(tissues).reshape(1,-1)
                if self.label_encoder:
                    tissues = self.label_encoder.transform(tissues.T)
                for i in range(n_rows):
                    yield gene_expressions[i,:], tissues[i]

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
        self.tissues = set()

    def setup(self, stage=None):
        self.train_dataset = SingleCellDataset(self.train_fp)
        self.test_dataset = SingleCellDataset(self.test_fp)
        self.val_dataset = SingleCellDataset(self.val_fp)
        for dataset in [self.train_dataset, self.test_dataset, self.val_dataset]:
            self.genes.update(dataset.genes)
            self.tissues.update(dataset.tissues)
        label_encoder = OneHotEncoder(sparse=False)
        label_encoder.fit(np.array(list(self.tissues)).reshape(-1, 1))
        for dataset in [self.train_dataset, self.test_dataset, self.val_dataset]:
            dataset.label_encoder = label_encoder

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    # def val_dataloader(self):
    #     return DataLoader(self.val_dataset, batch_size=self.batch_size)

    # def test_dataloader(self):
    #     return DataLoader(self.test_dataset, batch_size=self.batch_size)


def load_single_cell_data(batch_size=32):
    data_module = SingleCellDataModule(*DATA_SPLIT_FPS, batch_size)
    data_module.setup()
    return data_module