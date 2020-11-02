from typing import List
import toml
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from pipeline.vae import Vanilla1dVAE
from pipeline.data import load_single_cell_data


class LitVae1d(pl.LightningModule):

    """lightning wrapper for 1d VAE"""
    
    def __init__(self,
                 in_features: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:

        """see: Vanilla1dVAE constructor"""

        super(LitVae1d, self).__init__()
        self.vae = Vanilla1dVAE(in_features, latent_dim, hidden_dims)

    def forward(self, batch):
        x, _ = batch
        return self.vae.forward(x)

    def _step(self, batch, batch_idx):
        x_reconstructed, x, _, _  = self.forward(batch)
        reconstruction_loss = F.mse_loss(x_reconstructed, x)
        return reconstruction_loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx)
    
    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.Adam(self.vae.parameters())


if __name__ == "__main__":
    data = load_single_cell_data(batch_size=32)
    vae = LitVae1d(in_features=data.genes, latent_dim=4)
    trainer_args = toml.loads("params.toml")["vae_trainer"]
    trainer = pl.Trainer(**trainer_args)
    trainer.fit(vae, data)