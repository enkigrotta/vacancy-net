"""
BASELINE: Standard VQ-VAE (control for ∅-NET).

- Fixed K (never changes)
- EMA codebook updates
- Standard commitment loss
- No remainder tracking, no ⟳, no valve

Same Encoder/Decoder as VacancyNet for fair comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from module_a_encoder import Encoder
from module_c_decoder import Decoder
from config import Config


class BaselineVQVAE(nn.Module):
    """Standard VQ-VAE. Takes a Config (same interface as VacancyNet)."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.K = cfg.K_initial
        self.latent_dim = cfg.latent_dim
        self.beta = cfg.beta
        self.ema_decay = cfg.ema_decay_codebook

        self.encoder = Encoder(cfg.image_channels, cfg.latent_dim, cfg.hidden_dims)
        self.decoder = Decoder(cfg.image_channels, cfg.latent_dim,
                               tuple(reversed(cfg.hidden_dims)))

        self.register_buffer('codebook', torch.randn(self.K, self.latent_dim) * 0.1)
        self.register_buffer('ema_count', torch.zeros(self.K))
        self.register_buffer('ema_sum', torch.zeros(self.K, self.latent_dim))
        self._initialized = False

    def initialize_codebook(self, z_flat: torch.Tensor):
        from module_h_initializer import kmeans_init
        codes = kmeans_init(z_flat, self.K).to(self.codebook.device)
        self.codebook.copy_(codes)
        self.ema_sum.copy_(codes)
        self.ema_count.fill_(1.0)
        self._initialized = True

    def quantize(self, z_e: torch.Tensor):
        B, d, h, w = z_e.shape
        z_flat = z_e.permute(0, 2, 3, 1).reshape(-1, d)
        distances = (
            z_flat.pow(2).sum(1, keepdim=True)
            - 2 * z_flat @ self.codebook.t()
            + self.codebook.pow(2).sum(1, keepdim=True).t()
        )
        indices = distances.argmin(dim=1)
        e_k = self.codebook[indices].reshape(B, h, w, d).permute(0, 3, 1, 2)

        if self.training:
            with torch.no_grad():
                onehot = F.one_hot(indices, self.K).float()
                self.ema_count.mul_(self.ema_decay).add_(
                    onehot.sum(0), alpha=1 - self.ema_decay,
                )
                self.ema_sum.mul_(self.ema_decay).add_(
                    onehot.t() @ z_flat, alpha=1 - self.ema_decay,
                )
                n = self.ema_count.sum()
                cs = (self.ema_count + 1e-5) / (n + self.K * 1e-5) * n
                self.codebook.copy_(self.ema_sum / cs.unsqueeze(1))

        e_k_ste = z_e + (e_k - z_e).detach()
        indices = indices.reshape(B, h, w)
        return e_k_ste, indices

    def forward(self, x: torch.Tensor):
        z_e = self.encoder(x)
        e_k, indices = self.quantize(z_e)
        x_hat = self.decoder(e_k)

        L_recon = F.mse_loss(x_hat, x)
        L1 = (z_e - e_k.detach()).pow(2).mean()
        L2 = (z_e.detach() - e_k).pow(2).mean()
        L_commit = L1 + self.beta * L2
        L_total = L_recon + L_commit

        return {
            'x_hat': x_hat, 'z_e': z_e, 'indices': indices,
            'L_recon': L_recon, 'L_commit': L_commit, 'L_total': L_total,
        }
