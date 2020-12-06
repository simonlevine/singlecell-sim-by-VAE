#!./venv/bin/python3

"""run `make download_data` and `make sparsify_data` to regenerate the raw
data for the pipeline"""

import anndata
import numpy as np
from pipeline.helpers.paths import RAW_DATA_DENSE_FP, RAW_DATA_FP


def main():
    ann = anndata.read_h5ad(
        RAW_DATA_DENSE_FP,
        backed="r",
        as_sparse=["X", "raw/X"]
    )
    ann.uns = clean_up_dict(ann.uns.data)
    ann.write_h5ad(
        RAW_DATA_FP,
        as_dense=(),
        force_dense=False,
        compression="lzf"
    )


def clean_up_dict(dict_):
    """nestled convert bytes to unicode"""
    dict_ = dict_.copy()
    for k, v in dict_.items():
        if type(v) == bytes:
            dict_[k] = v.decode()
        elif type(v) == np.ndarray:
            dict_[k] = np.array([
                s.decode() if hasattr(s, "decode") else s
                for s in dict_[k]
            ])
        elif type(v) == dict:
            dict_[k] = clean_up_dict(dict_[k])
    return dict_


if __name__ == "__main__":
    main()