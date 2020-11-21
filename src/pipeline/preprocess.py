import numpy as np
import pandas as pd
import scanpy as sc

import anndata
from tqdm import tqdm
from loguru import logger
from helpers.paths import RAW_DATA_FP, DATA_SPLIT_FPS,PREPROC_DATA_FP



"""Preprocessing recipes from the literature:

sc.pp.filter_genes(adata, min_counts=1)         # only consider genes with more than 1 count
sc.pp.normalize_per_cell(                       # normalize with total UMI count per cell
     adata, key_n_counts='n_counts_all'
)
filter_result = sc.pp.filter_genes_dispersion(  # select highly-variable genes
    adata.X, flavor='cell_ranger', n_top_genes=n_top_genes, log=False
)
adata = adata[:, filter_result.gene_subset]     # subset the genes
sc.pp.normalize_per_cell(adata)                 # renormalize after filtering
if log: sc.pp.log1p(adata)                      # log transform: adata.X = log(adata.X + 1)
sc.pp.scale(adata)                              # scale to unit variance and shift to zero mean

"""
def main():

    adata = sc.read(RAW_DATA_FP,
    backup_url='https://hosted-matrices-prod.s3-us-west-2.amazonaws.com/Single_cell_atlas_of_peripheral_immune_response_to_SARS_CoV_2_infection-25/Single_cell_atlas_of_peripheral_immune_response_to_SARS_CoV_2_infection.h5ad')

    logger.info('Computing QC metrics for ingested data...')
    sc.pp.calculate_qc_metrics(adata, inplace=True)
    sc.pp.recipe_zheng17(adata, n_top_genes=1000, log=True, plot=False, copy=False)
    adata.write(PREPROC_DATA_FP)

    # filtered = filter(adata)

    # normed = log_normalize(filtered)
    # normed.write(PREPROC_DATA_FP)
    # normed.write_h5ad(PREPROC_DATA_FP, compression="gzip")
 

# def filter(adata):
#     logger.info(f'Filtering out datapoints based on number of genes and minimal presence...')
#     logger.info('Setting lower bound on min number of genes to 200...')
#     sc.pp.filter_cells(adata, min_genes=200) #takes awhile

#     logger.info('Setting lower bound on min number of cells with genes to 5...')
#     sc.pp.filter_genes(adata, min_cells=5) #takes awhile

#     # logger.info(f'Filtering out genes with counts < 2000...')
#     # adata = adata[adata.obs.n_counts > 2000, :]
#     logger.info(f'Filtering out mitochondrial noise at 5% threshold...')
#     adata = adata[adata.obs.percent_counts_mt < 5, :]

#     return adata

# def log_normalize(adata):
#     sc.pp.normalize_total(adata, target_sum=1e4)
#     sc.pp.log1p(adata)
#     sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
#     logger.info('Identified highly variate genes. Filtering out...')
#     adata = adata[:, adata.var.highly_variable]
#     logger.info('Regressing out effects of percentage of mitochondrial genes expressed')
#     sc.pp.regress_out(adata, ['total_counts','percent_mt'])
#     logger.info('Scaling the data to unit variance, excluding values exceeding standard dev of 10.')
#     sc.pp.scale(adata,max_value=10)

#     return adata

if __name__=='__main__':
    main()