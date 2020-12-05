import anndata
import random
import math
from pipeline.helpers.params import params
from pipeline.helpers.paths import RAW_DATA_FP, SUBSAMPLED_DATA_DIR

covid_dataset = anndata.read_h5ad(RAW_DATA_FP, backed="r")
n, _ = covid_dataset.shape
idxs = random.sample(range(n), k=math.floor(n*params.data.subsampling))
covid_dataset[idxs].write_h5ad(SUBSAMPLED_DATA_DIR, compression="gzip")
