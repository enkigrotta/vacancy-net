"""
Module A: ENCODER (Prism ⤢Δ₁⤢)

Organ of distinction. Neural network x → z_e.
Double bind: detailed representations AND quantization-robust.
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """x (B, C, H, W) -> z_e (B, d, h, w). Downsampling ×4 for CIFAR-like."""

    def __init__(self, in_channels: int = 3, latent_dim: int = 64,
                 hidden_dims: tuple = (128, 256)):
        super().__init__()
        layers = []
        ch_in = in_channels
        for ch_out in hidden_dims:
            layers += [
                nn.Conv2d(ch_in, ch_out, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace=True),
            ]
            ch_in = ch_out
        layers += [
            nn.Conv2d(ch_in, latent_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(inplace=True),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
