import json
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from loguru import logger
from tqdm import tqdm
from pipeline.train_vae import LitVae1d
from pipeline.datalib import load_single_cell_data
from pipeline.helpers.paths import VAE_WEIGHTS_FP, VAE_METADATA_JSON_FP, SIMULATED_GENE_EXPRESSION_FP


def main():
    logger.info("loading VAE model")
    vae = rehydrate_vae()
    logger.info("loading data (takes about a minute)")
    data = load_single_cell_data(batch_size=256)
    logger.info("determing cell type class representation")
    class_mu = determine_mus_by_class(vae, data)
    simulated_gene_expressions = []
    n_samples_needed_per_cell_type = determine_n_samples_needed_per_class(data.train_dataset.annotations)
    logger.info("simulating gene expression")
    iter_ = n_samples_needed_per_cell_type.items()
    for (cell_type, ventilator_status), n_samples_needed  in tqdm(iter_):
        mu = class_mu[cell_type, ventilator_status]
        simulated_gene_expression = sample(n_samples_needed, vae, mu, epsilon=0.05)
        simulated_gene_expressions.append((
            [cell_type] * n_samples_needed,
            [ventilator_status] * n_samples_needed,
            simulated_gene_expression))
    cell_types, ventilator_statuses, gene_expressions = zip(*simulated_gene_expressions)
    cell_type_arr = np.array(sum(cell_types, []))
    ventilator_status_arr = np.array(sum(ventilator_statuses, []))
    gene_expressions = np.vstack(gene_expressions)
    logger.info("saving simulation results")
    pd.DataFrame(columns=data.train_dataset.genes, data=gene_expressions) \
        .assign(cell_type=cell_type_arr, ventilator_status=ventilator_status_arr) \
        .to_parquet(SIMULATED_GENE_EXPRESSION_FP, compression="GZIP")


def rehydrate_vae():
    with open(VAE_METADATA_JSON_FP) as f:
        vae_metadata = json.load(f)
    vae = LitVae1d(**vae_metadata)
    vae.load_state_dict(torch.load(VAE_WEIGHTS_FP))
    vae.eval()
    return vae.vae


def determine_n_samples_needed_per_class(covid: AnnData, minimum_cells_per_type=1000):
    """
    Args:
        covid (AnnData): observational data
        minimum_cells_per_type (int, optional): minimum number of cells to have after augmentation (i.e. simulated + observed). Defaults to 1000.

    Returns:
        Dict[byte, int]: how many cells per cell type/health status to get from simulation
    """
    class_counts = covid.obs.groupby(["cell_type_coarse", "Ventilated"]) \
        .apply(lambda _df: len(_df))
    cell_types2upsample = class_counts[class_counts < minimum_cells_per_type]
    return (minimum_cells_per_type - cell_types2upsample).to_dict()


def determine_mus_by_class(vae, data):
    train_dataloader = data.train_dataloader()
    cell_type_encoder = data.train_dataset.cell_type_encoder
    ventilator_status_encoder = data.train_dataset.ventilator_status_encoder
    class_mus = defaultdict(list)
    for batch in tqdm(train_dataloader):
        gene_expressions, cell_types, ventilator_statuses = batch
        cell_types = cell_type_encoder.inverse_transform(cell_types)
        ventilator_statuses = ventilator_status_encoder.inverse_transform(ventilator_statuses)
        with torch.no_grad():
            mus, _ = vae.encode(gene_expressions)
        batch_size, _ = gene_expressions.shape
        for i in range(batch_size):
            class_ = (cell_types[i].item(), ventilator_statuses[i].item())
            class_mus[class_].append(mus[i,:].numpy())
    class_mu = {class_: np.vstack(mus).mean(axis=0)
        for (class_, mus) in class_mus.items()}
    return class_mu


def sample(n, vae, mu: np.array, epsilon=0):
    latent_target = np.random.multivariate_normal(
        mu, cov=np.identity(mu.shape[0]) * epsilon, size=n)
    latent_target = torch.from_numpy(latent_target).to(torch.float32)
    with torch.no_grad():
        simulation_result = vae.decode(latent_target)
    return simulation_result.numpy()


if __name__ == "__main__":
    main()