import json
from collections import defaultdict
import numpy as np
from pipeline.trainlib.vae import Vanilla1dVAE
from pipeline.datalib import load_single_cell_data
from pipeline.helpers.paths import MODEL_WEIGHTS_ONNX_FP

def main():
    data = load_single_cell_data(batch_size=256)
    n_samples_needed_per_cell_type = ...

    class_mu = determine_mus_by_class(vae, data.train_dataloader())
    new_data = []
    for n_samples_needed, cell_type in n_samples_needed_per_cell_type():
        mu = class_mu[cell_type]
        row = sample(n_samples_needed, vae, mu, epsilon=0.05)
        new_data.append((cell_type, row))
        

def rehydrate_vae():
    with open(VAE_METADATA_JSON_FP) as f:
        vae_metadata = json.load(f)
    vae = Vanilla1dVAE(**vae_metadata)
    vae.load_state_dict(VAE_WEIGHTS_FP)
    vae.eval()
    return vae

def determine_mus_by_class(vae, train_dataloader):
    class_mus = defaultdict(list)
    for batch in train_dataloader:
        gene_expression, cell_type, ventilator_status = batch
        mu, _ = vae.encode(gene_expression)
        class_mus[cell_type].append(mu.item())
    class_mu = {class_: sum(mus) / len(mus)
        for (class_, mus) in class_mus.items()}
    return class_mu


def sample(n, vae, mu: np.array, epsilon=0):
    latent_target = np.random.multivariate_normal(
        mu, cov=np.identity() * epsilon, size=n)
    return vae.decode(latent_target)


if __name__ == "__main__":
    main()