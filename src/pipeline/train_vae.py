"""train the VAE"""

from typing import List
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from pipeline.trainlib.vae import Vanilla1dVAE
from pipeline.datalib import load_single_cell_data
from pipeline.helpers.params import params
from pipeline.helpers.paths import MODEL_WEIGHTS_ONNX_FP


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

    def forward(self, x):
        return self.vae.forward(x)

    def _step(self, batch, batch_idx):
        x, _ = batch
        x_reconstructed, _, _, _  = self.forward(x)
        reconstruction_loss = F.mse_loss(x_reconstructed, x)
        return reconstruction_loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx)
    
    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.Adam(self.vae.parameters())


if __name__ == "__main__":
    batch_size = params.training.batch_size
    data = load_single_cell_data(batch_size)
    vae = LitVae1d(in_features=len(data.genes), latent_dim=params.model.latent_dimensions)
    trainer = pl.Trainer(**params.training.vae_trainer)
    trainer.fit(vae, data)
    vae.to_onnx(
        MODEL_WEIGHTS_ONNX_FP,
        torch.randn((batch_size, len(data.genes))),
        export_params=True
    )