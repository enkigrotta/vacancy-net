"""
Module D: % ACCUMULATOR (Remainder / ⤢%⤢)

Organ of accumulation. Collects what falls through distinction.
Produces the signal that drives ⟳.

STATISTICS (per code k):
  delta_mean[k]  — δ̄_k: EMA of mean residual   (d,)  — drift signal
  delta_var[k]   — Var_k: EMA of scalar var     ( ,)  — ⤢ heterogeneity
  usage_count[k] — usage_k: EMA of assignments  ( ,)  — dead-code detection
  cross_corr     — C_{kk'}: code correlations   (K,K) — redundancy

At ⟳: reset() is called by the model (new epoch).
"""

import torch
import numpy as np


class RemainderAccumulator:
    """Vectorized statistical tracker. No trainable parameters."""

    def __init__(self, K_eff: int, latent_dim: int,
                 ema_decay: float = 0.99,
                 cross_corr_interval: int = 50,
                 device: str = 'cpu'):
        self.K_eff = int(K_eff)
        self.latent_dim = int(latent_dim)
        self.ema = float(ema_decay)
        self.cross_corr_interval = int(cross_corr_interval)
        self.device = device

        self.delta_mean = torch.zeros(K_eff, latent_dim, device=device)
        self.delta_var = torch.zeros(K_eff, device=device)
        self.usage_count = torch.zeros(K_eff, device=device)
        self.d_percent_dt = torch.zeros(K_eff, device=device)
        self.cross_corr = torch.zeros(K_eff, K_eff, device=device)

        self._batches_since_cross_corr = 0
        self._total_updates = 0

    # ---- Core update (vectorized) ------------------------------------------

    def update(self, delta: torch.Tensor, indices: torch.Tensor):
        """
        delta:   (B, d, h, w) — residuals from Module B (already detached)
        indices: (B, h, w)    — code assignments
        """
        with torch.no_grad():
            B, d, h, w = delta.shape
            N = B * h * w
            delta_flat = delta.permute(0, 2, 3, 1).reshape(N, d)      # (N, d)
            idx_flat = indices.reshape(N)                              # (N,)

            # Per-code SUM of deltas: scatter-add
            sums = torch.zeros(self.K_eff, d, device=delta.device)
            sums.index_add_(0, idx_flat, delta_flat)

            # Per-code COUNT
            counts = torch.bincount(idx_flat, minlength=self.K_eff).float()
            counts_safe = counts.clamp(min=1.0).unsqueeze(1)           # (K, 1)

            # Per-code MEAN
            means = sums / counts_safe                                 # (K, d)

            # Per-code VAR (scalar): mean over d of per-dim variance
            # Using E[x^2] - E[x]^2 per-dim then mean over d.
            sq_sums = torch.zeros(self.K_eff, d, device=delta.device)
            sq_sums.index_add_(0, idx_flat, delta_flat.pow(2))
            mean_sq = sq_sums / counts_safe
            var_per_dim = (mean_sq - means.pow(2)).clamp(min=0.0)      # (K, d)
            var_scalar = var_per_dim.mean(dim=1)                        # (K,)

            # Mask codes that got zero assignments this batch
            active = (counts > 0).float()                              # (K,)

            # EMA updates — only for active codes this batch
            a = self.ema
            self.delta_mean = (
                a * self.delta_mean
                + (1 - a) * means * active.unsqueeze(1)
                + (1 - active.unsqueeze(1)) * self.delta_mean  # inactive: keep old
            ) if False else (
                # Cleaner: update only where active, keep where not
                torch.where(active.unsqueeze(1).bool(),
                            a * self.delta_mean + (1 - a) * means,
                            self.delta_mean)
            )

            old_var = self.delta_var.clone()
            self.delta_var = torch.where(
                active.bool(),
                a * self.delta_var + (1 - a) * var_scalar,
                self.delta_var,
            )
            self.d_percent_dt = self.delta_var - old_var

            # Usage EMA (always updated — zero counts pull it down)
            self.usage_count = a * self.usage_count + (1 - a) * counts

            self._batches_since_cross_corr += 1
            self._total_updates += 1

            # Cross-correlation (expensive, periodic)
            if self._batches_since_cross_corr >= self.cross_corr_interval:
                self._update_cross_corr()
                self._batches_since_cross_corr = 0

    def _update_cross_corr(self):
        """C_{kk'} = cosine similarity between mean residual directions."""
        with torch.no_grad():
            m = self.delta_mean                                         # (K, d)
            norms = m.norm(dim=1, keepdim=True).clamp(min=1e-8)
            m_normed = m / norms
            self.cross_corr = m_normed @ m_normed.t()                  # (K, K)
            # Zero diagonal so we don't flag self as redundant
            self.cross_corr.fill_diagonal_(0.0)

    # ---- Reset at ⟳ ---------------------------------------------------------

    def reset(self, new_K_eff: int):
        """Called by model at ⟳: %-stats discarded, new epoch begins."""
        self.K_eff = int(new_K_eff)
        self.delta_mean = torch.zeros(new_K_eff, self.latent_dim, device=self.device)
        self.delta_var = torch.zeros(new_K_eff, device=self.device)
        self.usage_count = torch.zeros(new_K_eff, device=self.device)
        self.d_percent_dt = torch.zeros(new_K_eff, device=self.device)
        self.cross_corr = torch.zeros(new_K_eff, new_K_eff, device=self.device)
        self._batches_since_cross_corr = 0

    # ---- Snapshot for logging ----------------------------------------------

    def get_stats_snapshot(self) -> dict:
        with torch.no_grad():
            return {
                'K_eff': self.K_eff,
                'delta_var_mean': float(self.delta_var.mean().item()),
                'delta_var_max': float(self.delta_var.max().item()),
                'usage_min': float(self.usage_count.min().item()),
                'usage_max': float(self.usage_count.max().item()),
                'drift_norm_mean': float(self.delta_mean.norm(dim=1).mean().item()),
                'cross_corr_max_abs': float(self.cross_corr.abs().max().item()),
            }

    # ---- Queries used by Module F ------------------------------------------

    def overloaded_codes(self, percentile: float = 0.90) -> torch.Tensor:
        """Indices of codes whose Var_k is above the given percentile."""
        if self.K_eff < 2:
            return torch.tensor([], dtype=torch.long, device=self.device)
        q = torch.quantile(self.delta_var, percentile)
        return torch.nonzero(self.delta_var > q, as_tuple=False).flatten()

    def redundant_pairs(self, c_critical: float = 0.8) -> list:
        """List of (k1, k2) with k1 < k2 and |C_kk'| > c_critical."""
        K = self.K_eff
        if K < 2:
            return []
        iu = torch.triu_indices(K, K, offset=1, device=self.cross_corr.device)
        vals = self.cross_corr[iu[0], iu[1]]
        mask = vals.abs() > c_critical
        k1s = iu[0][mask].tolist()
        k2s = iu[1][mask].tolist()
        return list(zip(k1s, k2s))

    def dead_codes(self, K_eff: int, factor: float = 0.1) -> torch.Tensor:
        """Codes with usage below 1/(factor*K) fraction of mean usage."""
        if self.K_eff < 2:
            return torch.tensor([], dtype=torch.long, device=self.device)
        total = self.usage_count.sum().clamp(min=1.0)
        mean_share = total / self.K_eff
        threshold = mean_share * factor      # fraction of mean usage
        return torch.nonzero(self.usage_count < threshold, as_tuple=False).flatten()

    def drifted_codes(self, factor: float = 2.0) -> torch.Tensor:
        """Codes whose mean-residual norm exceeds factor × global mean norm."""
        if self.K_eff < 2:
            return torch.tensor([], dtype=torch.long, device=self.device)
        norms = self.delta_mean.norm(dim=1)
        thresh = norms.mean() * factor
        return torch.nonzero(norms > thresh, as_tuple=False).flatten()
