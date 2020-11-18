import warnings
# ignore AnnDatas use of categorical
warnings.simplefilter(action="ignore", category=FutureWarning)
# ignore single threaded dataloader warning; AnnData  is single threaded
warnings.simplefilter(action="ignore", category=UserWarning)
from collections import defaultdict
from typing import Optional, List
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from loguru import logger

from pipeline.trainlib.vae import Vanilla1dVAE
from pipeline.datalib import load_single_cell_data
from pipeline.helpers.params import params
from pipeline.helpers.paths import MODEL_WEIGHTS_ONNX_FP, INTERMEDIATE_DATA_DIR


def main():
    """use Newton-Raphson to determine the latent dimensions, retrain using the best one"""
    pl.seed_everything(42)
    data = load_single_cell_data(batch_size=params.training.batch_size)
    latent_dims_best = tune_vae(32, data=data, batch_size=params.training.batch_size)
    train_vae(latent_dims_best, data, params.training.batch_size, MODEL_WEIGHTS_ONNX_FP)


def tune_vae(x_0, dx=1, n_iterations=10, temperature=100, **kwargs):
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
    cache = defaultdict(lambda x: train_vae(x, **kwargs)[1])
    x = x_0
    for _ in range(n_iterations):
        a = cache[x+dx]
        b = cache[x-dx]
        c = cache[x]
        f_prime = (a-b) / (2*dx)
        f_prime_prime = (a+b-(2*c)) / (dx**2)
        x = x - (temperature * f_prime / f_prime_prime)
    return x
    """train the VAE with a specific number of dimensions

    Args:
        n_latent_dimensions (int): number of latent dimensions
        data (pl.DataModule): HCL data
        model_path (Optional[Pathlike]): where to save ONNX serialization, if specified

    Returns (float): log-likelihood
    """
    n_latent_dimensions = int(n_latent_dimensions)
    M_N = batch_size / len(data.train_dataset)
    vae = LitVae1d(in_features=len(data.genes), latent_dim=n_latent_dimensions, M_N=M_N)
    wandb_logger = WandbLogger(name=f"vae-{n_latent_dimensions}-latent-dims", project='02718-vae')
    trainer = pl.Trainer(
        callbacks=[pl.callbacks.ModelCheckpoint(
            dirpath=INTERMEDIATE_DATA_DIR,
            monitor="val_loss",
        )],
        logger=wandb_logger,
        **params.training.vae_trainer,
    )
    trainer.fit(vae, data)
    logger.info("done.")
    if model_path:
        vae.to_onnx(
            model_path,
            torch.randn((batch_size, len(data.genes))),
            export_params=True
        )
    log_likelihood = trainer.callback_metrics["log_likelihood"].item()
    return vae, log_likelihood


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
        x, _ = batch
        x_reconstructed, _, mu, log_var  = self.forward(x)
        loss = self.vae.loss_function(x_reconstructed, x, mu, log_var, M_N=self.M_N)["loss"]
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_reconstructed, _, mu, log_var  = self.forward(x)
        loss = self.vae.loss_function(x_reconstructed, x, mu, log_var, M_N=self.M_N)["loss"]
        self.log("val_loss", loss, prog_bar=True)
        return {"mu^2": torch.pow(mu.sum(axis=0), 2),
                "n": x.shape[0]}

    def validation_epoch_end(self, validation_step_outputs):
        n = sum([x["n"] for x in validation_step_outputs])
        log_likelihood = - 0.5*n*1.83 - 0.5*torch.stack([x["mu^2"] for x in validation_step_outputs]).sum()
        return {"log_likelihood": log_likelihood}

    def configure_optimizers(self):
        return torch.optim.Adam(self.vae.parameters())


if __name__ == "__main__":
    main()