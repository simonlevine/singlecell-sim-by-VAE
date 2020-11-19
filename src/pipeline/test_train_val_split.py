"""create an 8:1:1 split for train, test and validation"""

import anndata
from tqdm import tqdm
from loguru import logger
from pipeline.helpers.paths import RAW_DATA_FP, DATA_SPLIT_FPS

logger.info("Please note this process takes about 10 minutes")

hcl = anndata.read_h5ad(RAW_DATA_FP, backed="r")
n, _ = hcl.shape
split_idx_1 = int(n * 0.8)
split_idx_2 = int(n * 0.9)
train_fp, test_fp, val_fp = DATA_SPLIT_FPS
splits = [
    ("train", None, split_idx_1, train_fp),
    ("test", split_idx_1, split_idx_2, test_fp),
    ("val", split_idx_2, None, val_fp),
]
for split_type, i, j, outfp in tqdm(splits, unit="split"):
    logger.info("Extracting single cell {} data within [{},{})", split_type, i, j)
    split = hcl[slice(i,j), :]
    split.write_h5ad(outfp, compression="gzip")
    # bug workaround: after writing, we need to refresh the view
    hcl = anndata.read_h5ad(RAW_DATA_FP, backed="r")