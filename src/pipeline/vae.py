from typing import List
import models as vae_models
import torch
from torch import nn


class Vanilla1dVAE(vae_models.VanillaVAE):

    """substitute convolutional layers for linear layers"""

    def __init__(self,
                 in_features: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:

        """1D VAE for vector reconstruction

        modified lightly from VanillaVAE constructor
        
        Arguments:
            in_features: dimensions of an individual 1xM input vector
            latent_dim: dimensions of the latent layer
            hidden_dims: hidden layer dimensions
        """

        super(vae_models.VanillaVAE, self).__init__()

        self.latent_dim = latent_dim
        self.in_features = in_features

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_features, out_features=h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU())
            )
            in_features = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1])  # upsample step

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                    nn.BatchNorm1d(hidden_dims[i+1]),
                    nn.LeakyReLU()
                )
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Linear(hidden_dims[-1], self.in_features)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        # result = result.view(-1, 512, 1, 1)  # no need to shape when using linear layers
        result = self.decoder(result)
        result = self.final_layer(result)
        return result