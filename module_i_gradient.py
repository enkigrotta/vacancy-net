"""
Module I: GRADIENT ENGINE (standard backprop)

Nothing novel. Wraps Adam.

Updates:
  - Encoder (A): via gradient of L_total through STE
  - Decoder (C): via gradient of L_recon
  - Codebook (B): NOT via gradient — via EMA inside Module B

Reduces LR on COLLAPSE (emergency).
"""

import torch
import torch.optim as optim


class GradientEngine:
    def __init__(self, encoder_params, decoder_params,
                 lr: float = 3e-4, weight_decay: float = 0.0):
        params = list(encoder_params) + list(decoder_params)
        self.optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)

    def step(self, loss: torch.Tensor):
        self.optimizer.zero_grad()
        loss.backward()
        # Light gradient clipping for stability
        for g in self.optimizer.param_groups:
            for p in g['params']:
                if p.grad is not None:
                    torch.nn.utils.clip_grad_norm_([p], max_norm=5.0)
        self.optimizer.step()

    def reduce_lr(self, factor: float = 0.5):
        for g in self.optimizer.param_groups:
            g['lr'] = g['lr'] * factor

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']
