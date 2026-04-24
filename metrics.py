"""
Metrics for ∅-NET evaluation.

Keep it lightweight. FID needs external packages; provide a placeholder
that returns a reconstruction MSE as a proxy unless real FID is installed.
"""

import torch
import numpy as np


@torch.no_grad()
def codebook_utilization(indices: torch.Tensor, K: int) -> float:
    """Fraction of codebook entries that appear in this batch of indices."""
    flat = indices.flatten()
    used = torch.unique(flat).numel()
    return float(used / max(K, 1))


@torch.no_grad()
def reconstruction_mse(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    return float(((x - x_hat) ** 2).mean().item())


@torch.no_grad()
def compute_reconstruction_fid(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    """
    Placeholder FID — uses simple reconstruction MSE since real FID
    requires InceptionV3. Returns MSE for now.
    """
    return reconstruction_mse(x, x_hat)


@torch.no_grad()
def pca_spectrum(z_flat: torch.Tensor, k: int = 10) -> list:
    """Top-k singular values of z_flat (treated as (N, d))."""
    if z_flat.numel() == 0:
        return []
    z = z_flat - z_flat.mean(dim=0, keepdim=True)
    try:
        S = torch.linalg.svdvals(z)
        return S[:k].tolist()
    except Exception:
        return []


@torch.no_grad()
def average_usage(usage_count: torch.Tensor) -> dict:
    """Basic codebook-usage summary."""
    u = usage_count.float()
    return {
        'usage_mean': float(u.mean().item()),
        'usage_min': float(u.min().item()),
        'usage_max': float(u.max().item()),
        'active_fraction': float((u > 0).float().mean().item()),
    }
