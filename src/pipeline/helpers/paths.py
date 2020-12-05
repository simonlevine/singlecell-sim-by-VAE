from pathlib import Path

INTERMEDIATE_DATA_DIR = Path("data/intermediate")
ARTIFACTS_DATA_DIR = Path("data/artifacts")
RAW_DATA_DIR = Path("data/raw")
PARAMS_FP = Path("params.toml")
RAW_DATA_FP         = RAW_DATA_DIR/"Single_cell_atlas_of_peripheral_immune_response_to_SARS_CoV_2_infection.h5ad"
SUBSAMPLED_DATA_DIR = RAW_DATA_DIR/"Single_cell_atlas_of_peripheral_immune_response_to_SARS_CoV_2_infection_subsampled.h5ad"
DATA_SPLIT_FPS = [
    INTERMEDIATE_DATA_DIR / f"covid_{split_type}_data.h5ad"
    for split_type in ["train", "test", "val"]
]
VAE_WEIGHTS_FP = ARTIFACTS_DATA_DIR/"vae.pth"
VAE_METADATA_JSON_FP = ARTIFACTS_DATA_DIR/"vae.pth.json"
SIMULATED_GENE_EXPRESSION_FP = ARTIFACTS_DATA_DIR/"simulation_gene_expression.pq"
CLASSIFER_W_SIMULATED_DATA_METRICS = ARTIFACTS_DATA_DIR/"classifer_w_simulated_data_metrics.json"
CLASSIFER_WOUT_SIMULATED_DATA_METRICS = ARTIFACTS_DATA_DIR/"classifer_wout_simulated_data_metrics.json"