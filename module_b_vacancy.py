"""
Module B: ∅_sg (Self-Governing Vacancy)

The vacancy. Irreversible quantization that destroys within-cell
information. Self-governed: parameters set by ⟳-protocol, not engineer.

STRUCTURAL INVARIANTS (enforced in code):
  - K_eff >= 2          (∅ requires at least two codes)
  - tau_k > 0           (hard argmin is the tau->0 limit, never reached)
  - delta NOT detached  (structural trace of what fell through ∅)
  - EMA codebook update (not gradient-based) — standard VQ-VAE-EMA
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfGoverningVacancy(nn.Module):
    """
    z_e (B, d, h, w) -> e_k (B, d, h, w), indices (B, h, w), delta (B, d, h, w)

    - Hard argmin in forward
    - Straight-through estimator in backward
    - Per-code temperature tau_k (currently used only as diagnostic / at valve)
    - Codebook updated via EMA on assigned vectors
    - Replaceable whole (Module H calls replace_codebook at ⟳)
    """

    def __init__(self, K: int, latent_dim: int, tau_init: float = 1.0,
                 ema_decay: float = 0.99):
        super().__init__()
        assert K >= 2, "∅ requires K_eff ≥ 2 at init (single code = no vacancy)"
        assert tau_init > 0, "tau_k must be strictly positive"
        self.K_eff = int(K)
        self.latent_dim = int(latent_dim)
        self.ema_decay = float(ema_decay)

        # Buffers (move with .to(device))
        self.register_buffer('codebook', torch.randn(K, latent_dim) * 0.1)
        self.register_buffer('tau_k', torch.full((K,), tau_init))
        self.register_buffer('ema_count', torch.zeros(K))
        self.register_buffer('ema_sum', torch.zeros(K, latent_dim))
        self._initialized = False

    # ---- Δ₀_initial ---------------------------------------------------------

    def initialize_from_data(self, z_e_flat: torch.Tensor):
        """K-means init on first batch(es). z_e_flat: (N, d)."""
        from module_h_initializer import kmeans_init
        codes = kmeans_init(z_e_flat, self.K_eff).to(self.codebook.device)
        self.codebook.copy_(codes)
        self.ema_sum.copy_(codes)
        self.ema_count.fill_(1.0)
        self._initialized = True

    # ---- Forward ------------------------------------------------------------

    def forward(self, z_e: torch.Tensor):
        B, d, h, w = z_e.shape
        assert d == self.latent_dim, f"latent_dim mismatch: {d} vs {self.latent_dim}"

        # Flatten: (B*h*w, d)
        z_flat = z_e.permute(0, 2, 3, 1).reshape(-1, d)

        # Squared distances to all codes
        distances = (
            z_flat.pow(2).sum(dim=1, keepdim=True)
            - 2 * z_flat @ self.codebook.t()
            + self.codebook.pow(2).sum(dim=1, keepdim=True).t()
        )  # (N, K_eff)

        # Hard assignment
        indices_flat = distances.argmin(dim=1)              # (N,)
        e_k_flat = self.codebook[indices_flat]              # (N, d)

        # EMA update (training only)
        if self.training:
            self._ema_update(z_flat.detach(), indices_flat)

        # Reshape back
        e_k = e_k_flat.reshape(B, h, w, d).permute(0, 3, 1, 2)
        indices = indices_flat.reshape(B, h, w)

        # Straight-through estimator: forward uses hard, backward uses z_e grad
        e_k_ste = z_e + (e_k - z_e).detach()

        # Residual (the remainder that falls into %). NOT detached in graph sense,
        # but we detach e_k so no grad flows into codebook through delta.
        delta = z_e - e_k.detach()

        return e_k_ste, indices, delta

    # ---- EMA codebook update (no gradient) ----------------------------------

    def _ema_update(self, z_flat: torch.Tensor, indices: torch.Tensor):
        with torch.no_grad():
            onehot = F.one_hot(indices, self.K_eff).float()     # (N, K)
            new_count = onehot.sum(dim=0)                        # (K,)
            new_sum = onehot.t() @ z_flat                        # (K, d)

            self.ema_count.mul_(self.ema_decay).add_(new_count, alpha=1 - self.ema_decay)
            self.ema_sum.mul_(self.ema_decay).add_(new_sum, alpha=1 - self.ema_decay)

            # Laplace smoothing
            n = self.ema_count.sum()
            count_smoothed = (self.ema_count + 1e-5) / (n + self.K_eff * 1e-5) * n
            self.codebook.copy_(self.ema_sum / count_smoothed.unsqueeze(1))

    # ---- Replacement at ⟳ ---------------------------------------------------

    def replace_codebook(self, new_codebook: torch.Tensor,
                         new_tau: torch.Tensor):
        """Replace whole codebook state. Called by Module H at ⟳ events."""
        K_new = int(new_codebook.shape[0])
        assert K_new >= 2, "∅ requires K_eff ≥ 2 after ⟳"
        assert (new_tau > 0).all(), "tau_k must be strictly positive after ⟳"
        assert new_codebook.shape[1] == self.latent_dim, \
            f"latent_dim mismatch in ⟳: {new_codebook.shape[1]} vs {self.latent_dim}"

        device = self.codebook.device
        self.K_eff = K_new
        # Replace buffers by assignment (sizes change)
        self.codebook = new_codebook.detach().clone().to(device)
        self.tau_k = new_tau.detach().clone().to(device)
        self.ema_count = torch.zeros(K_new, device=device)
        self.ema_sum = new_codebook.detach().clone().to(device)

    # ---- Diagnostics --------------------------------------------------------

    def get_usage(self, indices: torch.Tensor) -> torch.Tensor:
        """Vectorized per-code usage fraction from a batch of indices."""
        flat = indices.flatten()
        counts = torch.bincount(flat, minlength=self.K_eff).float()
        return counts / max(flat.numel(), 1)
