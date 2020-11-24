import scanpy as sc
from loguru import logger
from helpers.paths import RAW_DATA_FP,PREPROC_DATA_FP

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
    adata = sc.read(RAW_DATA_FP, backup_url='https://hosted-matrices-prod.s3-us-west-2.amazonaws.com/Single_cell_atlas_of_peripheral_immune_response_to_SARS_CoV_2_infection-25/Single_cell_atlas_of_peripheral_immune_response_to_SARS_CoV_2_infection.h5ad')
    logger.info('Computing QC metrics for ingested data...')
    sc.pp.calculate_qc_metrics(adata, inplace=True)
    sc.pp.recipe_zheng17(adata, n_top_genes=1000, log=True, plot=False, copy=False)
    adata.write(PREPROC_DATA_FP)

if __name__=='__main__':
    main()