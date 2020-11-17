import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)
from typing import Optional, List
import scipy
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from loguru import logger

from pipeline.trainlib.vae import Vanilla1dVAE
from pipeline.datalib import load_single_cell_data
from pipeline.helpers.params import params
from pipeline.helpers.paths import MODEL_WEIGHTS_ONNX_FP, INTERMEDIATE_DATA_DIR


def main():
    """use Newton-Raphson to determine the latent dimensions, retrain using the best one"""
    pl.seed_everything(42)
    data = load_single_cell_data(batch_size=params.training.batch_size)
    f = lambda x_0: train_vae_prime(x_0, data=data, batch_size=params.training.batch_size)
    latent_dims_best \
        = scipy.optimize.newton(f, params.training.latent_dimensions_initial_guess)
    logger.info("found {latent_dims_best} as best latent dimensionality")
    train_vae(latent_dims_best, data, params.training.batch_size, MODEL_WEIGHTS_ONNX_FP)


def train_vae_prime(x, dx=1, **kwargs):
    """Taylor approximation of derivative of VAE w.r.t number of latent dimensions

    Args:
        x (int): number of latent dimensions
        dx (int): delta (number of latent dimensions)
    """
    _, a = train_vae(x + dx, **kwargs)
    _, b = train_vae(x - dx, **kwargs)
    return (a-b) / (2*dx)


def train_vae(n_latent_dimensions, data, batch_size, model_path=None):
    """train the VAE with a specific number of dimensions

    Args:
        n_latent_dimensions (int): number of latent dimensions
        data (pl.DataModule): HCL data
        model_path (Optional[Pathlike]): where to save ONNX serialization, if specified

    Returns (float): log-likelihood
    """
    n_latent_dimensions = int(n_latent_dimensions)
    logger.info(f"training with {n_latent_dimensions} latent dimensions")
    M_N = batch_size / len(data.train_dataset)
    vae = LitVae1d(in_features=len(data.genes), latent_dim=n_latent_dimensions, M_N=M_N)
    trainer = pl.Trainer(
        callbacks=[pl.callbacks.ModelCheckpoint(
            dirpath=INTERMEDIATE_DATA_DIR,
            monitor='val_loss',
        )],
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

    def _step(self, batch, batch_idx):
        x, _ = batch
        x_reconstructed, _, mu, log_var  = self.forward(x)
        return self.vae.loss_function(x_reconstructed, x, mu, log_var, M_N=self.M_N)

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx)
    
    def validation_step(self, batch, batch_idx):
        x, _ = batch
        loss = self._step(batch, batch_idx)
        mu, log_var = self.vae.encode(x)
        self.log("val_loss", loss)
        return {"loss": loss,
                "mu^2": torch.pow(mu.sum(axis=0), 2),
                "n": x.shape[0]}

    def validation_epoch_end(self, validation_step_outputs):
        n = sum([x["n"] for x in validation_step_outputs])
        log_likelihood = - 0.5*n*1.83 - 0.5*torch.stack([x["mu^2"] for x in validation_step_outputs]).sum()
        self.log("log_likelihood", log_likelihood)

    def configure_optimizers(self):
        return torch.optim.Adam(self.vae.parameters())


if __name__ == "__main__":
    main()