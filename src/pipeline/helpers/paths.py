from pathlib import Path

INTERMEDIATE_DATA_DIR = Path("data/intermediate")
ARTIFACTS_DATA_DIR = Path("data/artifacts")
PARAMS_FP = Path("params.toml")
RAW_DATA_FP = "data/raw/Single_cell_atlas_of_peripheral_immune_response_to_SARS_CoV_2_infection.h5ad"

DATA_SPLIT_FPS = [
    INTERMEDIATE_DATA_DIR / f"covid_{split_type}_data.h5ad"
    for split_type in ["train", "test", "val"]
]
VAE_WEIGHTS_FP = ARTIFACTS_DATA_DIR/"vae.pth"
VAE_METADATA_JSON_FP = ARTIFACTS_DATA_DIR/"vae.pth.json"