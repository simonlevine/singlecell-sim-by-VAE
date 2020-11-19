import numpy as np
import pandas as pd
import scanpy as sc

import anndata
from tqdm import tqdm
from loguru import logger
from pipeline.helpers.paths import RAW_DATA_FP, DATA_SPLIT_FPS,PREPROC_DATA_FP

def main():
    plot = False

    adata = anndata.read_h5ad(RAW_DATA_FP, backed="r")

    if plot == True:
        sc.pl.highest_expr_genes(adata, n_top=20, show=False).savefig('10_highest_expressed.png')

    filtered = filter(adata)
    normed = log_normalize(filtered)
    normed.write_h5ad(PREPROC_DATA_FP, compression="gzip")
 

def filter(adata):
    logger.info(f'Filtering out datapoints based on number of genes and minimal presence...')
    logger.info('Setting lower bound on min number of genes to 200...')
    adata=sc.pp.filter_cells(adata, min_genes=200)

    logger.info('Setting lower bound on min number of cells with genes to 5...')
    adata=sc.pp.filter_genes(adata, min_cells=5)

    logger.info(f'Filtering out genes with counts < 2500...')
    adata = adata[adata.obs.n_genes_by_counts < 2500, :]
    logger.info(f'Filtering out mitochondrial noise at 5% threshold...')
    adata = adata[adata.obs.pct_counts_mt < 5, :]

    return adata

def log_normalize(adata):
    adata=sc.pp.normalize_total(adata, target_sum=1e4)
    adata=sc.pp.log1p(adata)
    adata=sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    logger.info('Identified highly variate genes. Filtering out...')
    adata = adata[:, adata.var.highly_variable]
    logger.info('Regressing out effects of total counts per cell and the percentage of mitochondrial genes expressed')
    sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt'])
    logger.info('Scaling the data to unit variance, excluding values exceeding standard dev of 10.')
    sc.pp.scale(adata, max_value=10)

    return adata

if __name__=='__main__':
    main()