from pathlib import Path

INTERMEDIATE_DATA_DIR = Path("data/intermediate")
ARTIFACTS_DATA_DIR = Path("data/artifacts")
PARAMS_FP = Path("params.toml")
RAW_DATA_FP = "data/raw/HCL_final_USE.h5ad"
DATA_SPLIT_FPS = [
    INTERMEDIATE_DATA_DIR / f"hcl_{split_type}_data.h5ad"
    for split_type in ["train", "test", "val"]
]
MODEL_WEIGHTS_ONNX_FP = ARTIFACTS_DATA_DIR/"vae.onnx"