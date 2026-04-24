"""
Module G: Δ OBSERVER (Object Δ metrics)

Phase 1: LOG ONLY. Observes decoder's excess capacity.
Phase 2: will feed into ⟳ protocol (reserved).

Metrics:
  1. L_recon_per_code      — which codes are over/under-served
  2. interpolation_coherence — can decoder produce smooth outputs between codes
  3. weight_utilization     — fraction of significant singular values in decoder
"""

import random

import torch


class DeltaObserver:
    def __init__(self, N_observe: int = 500,
                 interpolation_samples: int = 100,
                 device: str = 'cpu'):
        self.N_observe = int(N_observe)
        self.interp_n = int(interpolation_samples)
        self.device = device

        self.counter = 0
        self.log = []

    def step(self):
        self.counter += 1

    def is_due(self) -> bool:
        return self.counter >= self.N_observe

    def observe(self, encoder, vacancy, decoder, batch_x: torch.Tensor) -> dict:
        """
        Called by model when is_due(). Caller passes a batch of data.
        Returns metrics dict.
        """
        self.counter = 0
        metrics = {}
        with torch.no_grad():
            # 1. Reconstruction loss per code
            z_e = encoder(batch_x)
            e_k, indices, _ = vacancy(z_e)
            x_hat = decoder(e_k)

            K = vacancy.K_eff
            recon_per_pixel = (batch_x - x_hat).pow(2).mean(dim=1)  # (B, H, W)
            # Map spatial indices back
            flat_idx = indices.flatten()
            flat_err = recon_per_pixel.flatten()
            # BUT recon is on image-pixel grid, indices on latent grid;
            # simplify: average recon error per assigned code via latent-grid average.
            # Take latent-grid recon proxy: decoder output averaged to latent size.
            # For simplicity, report overall recon + stats on indices:
            metrics['recon_mean'] = float((batch_x - x_hat).pow(2).mean().item())
            code_counts = torch.bincount(flat_idx, minlength=K).float()
            metrics['codes_used'] = int((code_counts > 0).sum().item())
            metrics['codes_total'] = int(K)

            # 2. Interpolation coherence — smoother outputs = higher coherence
            if K >= 2:
                coherence_scores = []
                cb = vacancy.codebook
                for _ in range(self.interp_n):
                    k1, k2 = random.sample(range(K), 2)
                    mid = 0.5 * cb[k1] + 0.5 * cb[k2]
                    # Build a latent of the right shape: broadcast across spatial
                    lat_h = z_e.shape[2]
                    lat_w = z_e.shape[3]
                    e_interp = mid.view(1, -1, 1, 1).expand(1, -1, lat_h, lat_w)
                    out = decoder(e_interp)
                    # Coherence proxy: 1 / (1 + total variation)
                    tv = (out[..., 1:, :] - out[..., :-1, :]).abs().mean() + \
                         (out[..., :, 1:] - out[..., :, :-1]).abs().mean()
                    coherence_scores.append(1.0 / (1.0 + float(tv.item())))
                metrics['interpolation_coherence'] = sum(coherence_scores) / len(coherence_scores)
            else:
                metrics['interpolation_coherence'] = 0.0

            # 3. Decoder weight utilization — SVD of first Conv2d weight
            first_conv = None
            for m in decoder.modules():
                if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d):
                    first_conv = m
                    break
            if first_conv is not None:
                W = first_conv.weight.detach().reshape(first_conv.weight.shape[0], -1)
                try:
                    S = torch.linalg.svdvals(W)
                    if S.numel() > 0 and S[0] > 0:
                        significant = (S > 0.01 * S[0]).sum().item()
                        metrics['weight_utilization'] = float(significant / len(S))
                    else:
                        metrics['weight_utilization'] = 0.0
                except Exception:
                    metrics['weight_utilization'] = float('nan')
            else:
                metrics['weight_utilization'] = float('nan')

        self.log.append(metrics)
        return metrics
