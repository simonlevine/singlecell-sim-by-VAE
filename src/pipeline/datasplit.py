"""create an 8:1:1 split for train, test and validation"""

from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc

from paths import INTERMEDIATE_DATA_DIR

sc_data = sc.read_h5ad("data/raw/HCL_final_USE.h5ad")
genes = [m.decode() for m in sc_data.var.index.tolist()]
tissues = [m.decode() for m in sc_data.obs.tissue.tolist()]
df = pd.DataFrame(data=sc_data.X, columns=genes).assign(tissue=tissues)

train, validate, test = np.split(
    df.sample(frac=1, random_state=42),
    [int(.6*len(df)), int(.8*len(df))]
)
for split, split_name in [(train, "train"),
                          (validate, "validate"),
                          (test, "test")]:
    split.to_json(INTERMEDIATE_DATA_DIR/f"{split_name}_data.json.gz")
