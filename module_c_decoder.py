"""
Module C: DECODER (Δ₂)

Second constrained system. Reconstructs through ∅.
Produces Object Δ (generative excess) as side effect.
"""

import torch
import torch.nn as nn


class Decoder(nn.Module):
    """e_k (B, d, h, w) -> x_hat (B, C, H, W). Upsampling ×4."""

    def __init__(self, out_channels: int = 3, latent_dim: int = 64,
                 hidden_dims: tuple = (256, 128)):
        super().__init__()
        layers = [
            nn.Conv2d(latent_dim, hidden_dims[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dims[0]),
            nn.ReLU(inplace=True),
        ]
        ch_in = hidden_dims[0]
        for ch_out in hidden_dims[1:]:
            layers += [
                nn.ConvTranspose2d(ch_in, ch_out, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace=True),
            ]
            ch_in = ch_out
        # Final upsample to image_channels
        layers += [
            nn.ConvTranspose2d(ch_in, out_channels, kernel_size=4, stride=2, padding=1),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, e_k: torch.Tensor) -> torch.Tensor:
        return self.net(e_k)
