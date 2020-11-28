import warnings
# ignore AnnDatas use of categorical
warnings.simplefilter(action="ignore", category=FutureWarning)
# ignore single threaded dataloader warning; AnnData  is single threaded
warnings.simplefilter(action="ignore", category=UserWarning)
import json
from typing import Optional, List
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from loguru import logger

from pipeline.trainlib.vae import Vanilla1dVAE
from pipeline.datalib import load_single_cell_data
from pipeline.helpers.params import params
from pipeline.helpers.paths import VAE_WEIGHTS_FP, VAE_METADATA_JSON_FP, INTERMEDIATE_DATA_DIR


import wandb
wandb.init(project='02718-vae')


def main():
    """use Newton-Raphson to determine the latent dimensions, retrain using the best one"""
    pl.seed_everything(42)
    logger.info("loading data (takes about a minute)")
    data = load_single_cell_data(batch_size=params.training.batch_size)
    latent_dims_best = tune_vae(32, data=data, batch_size=params.training.batch_size)
    train_vae(latent_dims_best, data, params.training.batch_size, VAE_WEIGHTS_FP, VAE_METADATA_JSON_FP)


def tune_vae(x_0, dx=1, n_iterations=10, **kwargs):
    """Tune number of latent dimensions using Newtons method and 
    Taylor approximation of derivative of VAE w.r.t number of
    latent dimensions

    Args:
        x_0 (int): number of latent dimensions, initial guess
        dx (int): delta (number of latent dimensions)
        n_iterations (int): number of newton iterations

    Returns: tuned number of latent dimensions
    """
    # for each latent dimension, store the log-likelihood
    # after running `train_vae` and return cached value
    # if requested again
    cache = {}
    def run_and_cache(x):
        if x in cache:
            return cache[x]
        _, negative_log_likelihood = train_vae(
            x, logging_enabled=False,
            # max_epochs=1,
            **kwargs)
        cache[x] = negative_log_likelihood
        return negative_log_likelihood
    x = x_0
    temperature = params.training.newton_temperature
    intermediary_results = []
    for i in range(n_iterations):
        logger.info("newton-rhapson {}/{}", i, n_iterations)
        while True:
            a = run_and_cache(x+dx)
            b = run_and_cache(x-dx)
            c = run_and_cache(x)
            f_prime = (a-b) / (2*dx)
            f_prime_prime = (a+b-2*c) / (dx**2)
            delta = (temperature * f_prime / f_prime_prime)
            if 1 < (x - delta):
                intermediary_results.append({
                    "i": i,
                    "f": c,
                    "f_prime": f_prime,
                    "f_prime_prime": f_prime_prime,
                    "temperature": temperature,
                    "x_n": x,
                    "x_n+1": int(x - delta),
                })
                x = int(x - delta)
                break
            else:
                logger.info(
                    "{} - {} results in a negative number of latent dimensions! "
                    "reducing temperature {} -> {}", x, delta, temperature, temperature*0.5)
                temperature = temperature * 0.5
    # report what happened during tuning
    negative_log_likelihoods = []
    for n_dimensions, ll in cache.items():
        negative_log_likelihoods.append({"n_dimensions": n_dimensions, "negative_log_likelihood": ll})
    negative_log_likelihoods_table = wandb.Table(dataframe=pd.DataFrame.from_records(negative_log_likelihoods))
    newton_table = wandb.Table(dataframe=pd.DataFrame.from_records(intermediary_results))
    wandb.log({"newton_raphson_progress": newton_table,
               "negative_log_likelihood": negative_log_likelihoods_table})
    return x

def train_vae(n_latent_dimensions,
              data,
              batch_size,
              model_path=None,
              model_metadata_path=None,
              max_epochs=None,
              logging_enabled=True):
    """train the VAE with a specific number of dimensions

    Args:
        n_latent_dimensions (int): number of latent dimensions
        data (pl.DataModule): HCL data
        model_path (Optional[Pathlike]): where to save model state dict, if specified
        model_metadata_path (Optional[Pathlike]): where to save the model metadata to properly deserialize state dict, if specified
        max_epochs (Optional[int]): number of epochs, overwriting the default in params.toml
        logging_enabled (bool): report to wand.ai?

    Returns (float): log-likelihood
    """
    n_latent_dimensions = int(n_latent_dimensions)
    M_N = batch_size / len(data.train_dataset)
    vae_kwargs = {"in_features": len(data.genes), "latent_dim": n_latent_dimensions, "M_N": M_N}
    vae = LitVae1d(**vae_kwargs)
    wandb_logger = WandbLogger(name=f"vae-{n_latent_dimensions}-latent-dims", project='02718-vae')
    train_opts = params.training.vae_trainer
    if max_epochs:
        train_opts["max_epochs"] = max_epochs
    trainer = pl.Trainer(
        callbacks=[pl.callbacks.ModelCheckpoint(
            dirpath=INTERMEDIATE_DATA_DIR,
            monitor="val_loss",
        )],
        logger=wandb_logger if logging_enabled else True,
        **train_opts,
    )
    trainer.fit(vae, data)
    logger.info("done.")
    if model_path:
        assert model_metadata_path is not None
        torch.save(vae.state_dict(), model_path)
        with open(model_metadata_path, "w") as f:
            json.dump(vae_kwargs, f)
    negative_log_likelihood = trainer.callback_metrics["negative_log_likelihood"].item()
    return vae, negative_log_likelihood


class LitVae1d(pl.LightningModule):

    """lightning wrapper for 1d VAE"""
    
    def __init__(self,
                 in_features: int,
                 latent_dim: int,
                 M_N: float,
                 hidden_dims: Optional[List] = None):

        """see: Vanilla1dVAE constructor"""

        super(LitVae1d, self).__init__()
        self.vae = Vanilla1dVAE(in_features, latent_dim, hidden_dims)
        self.M_N = M_N

    def forward(self, x):
        return self.vae.forward(x)
        
    def training_step(self, batch, batch_idx):
        x, *_ = batch
        x_reconstructed, _, mu, log_var  = self.forward(x)
        loss = self.vae.loss_function(x_reconstructed, x, mu, log_var, M_N=self.M_N)["loss"]
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, *_ = batch
        x_reconstructed, _, mu, log_var  = self.forward(x)
        loss = self.vae.loss_function(x_reconstructed, x, mu, log_var, M_N=self.M_N)["loss"]
        self.log("val_loss", loss)
        return {"val_loss": loss,
                "mu": mu,
                "n": x.shape[0]}

    def validation_epoch_end(self, validation_step_outputs):
        n = sum([x["n"] for x in validation_step_outputs])
        residual = torch.vstack([x["mu"] for x in validation_step_outputs])
        mu, sigma = 0, 1
        negative_log_likelihood = (n/2) * np.log(np.pi) \
                                + (n/2) * np.log(sigma**2) \
                                + (1/(2*sigma**2)) * torch.sum((residual - mu).pow(2)).item()
        return {"negative_log_likelihood": negative_log_likelihood}

    def configure_optimizers(self):
        return torch.optim.Adam(self.vae.parameters())


if __name__ == "__main__":
    main()