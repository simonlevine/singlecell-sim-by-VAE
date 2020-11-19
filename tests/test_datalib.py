import pytest
from pipeline.datalib import load_single_cell_data


@pytest.fixture(scope="module")
def datamodule():
    return load_single_cell_data(batch_size=1)


@pytest.fixture(scope="module")
def train_dataset(datamodule):
    return datamodule.train_dataset


@pytest.fixture(scope="module")
def first_training_example(train_dataset):
    return next(iter(train_dataset))


def test_correct_dimensions(first_training_example, train_dataset, datamodule):
    gene_expression, cell_type = first_training_example
    n_genes, = gene_expression.shape
    _, n_cell_types = cell_type.shape
    assert n_genes == len(datamodule.genes)
    assert n_cell_types >= len(train_dataset.tissues)
    assert n_cell_types == len(datamodule.tissues)