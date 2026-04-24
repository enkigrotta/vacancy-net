"""
Module E: ⫿ SYNTONE

Three roles (from ТЗ §2):
  1. commitment_loss   — the loss term (L_commit)
  2. tension_monitor   — diagnostic (RESONANCE / STRAINED / OVERLOADED / COLLAPSE)
  3. valve_shedding    — preemptive action (partial pressure release before ⟳)
"""

from enum import Enum
from collections import deque

import torch


class TensionState(Enum):
    RESONANCE = "resonance"
    STRAINED = "strained"
    OVERLOADED = "overloaded"
    COLLAPSE = "collapse"


class Syntone:
    def __init__(self, beta: float = 0.25,
                 utilization_resonance: float = 0.8,
                 utilization_strained: float = 0.5,
                 utilization_overloaded: float = 0.2,
                 commit_history_window: int = 100,
                 valve_nudge_rate: float = 0.1,
                 valve_tau_softening: float = 1.05,
                 device: str = 'cpu'):
        self.beta = float(beta)
        self.u_res = float(utilization_resonance)
        self.u_str = float(utilization_strained)
        self.u_ovl = float(utilization_overloaded)
        self.window = int(commit_history_window)
        self.valve_rate = float(valve_nudge_rate)
        self.valve_tau = float(valve_tau_softening)
        self.device = device

        self.commit_history = deque(maxlen=self.window)

    # ---- 1. Commitment loss (differentiable) --------------------------------

    def commitment_loss(self, z_e: torch.Tensor, e_k_ste: torch.Tensor) -> torch.Tensor:
        """
        L_commit = ||z_e - sg[e_k]||^2 + beta * ||sg[z_e] - e_k||^2

        First term moves encoder toward codes.
        Second term moves codes toward encoder (but codebook uses EMA, not grad —
        this term still helps shape encoder statistics).
        """
        L1 = (z_e - e_k_ste.detach()).pow(2).mean()
        L2 = (z_e.detach() - e_k_ste).pow(2).mean()
        loss = L1 + self.beta * L2
        self.commit_history.append(float(loss.detach().item()))
        return loss

    # ---- 2. Tension monitor -------------------------------------------------

    def tension_monitor(self, K_eff: int,
                        usage_count: torch.Tensor,
                        delta_mean: torch.Tensor) -> TensionState:
        """Classify ⫿ state from usage + L_commit trend + drift."""
        with torch.no_grad():
            # Active fraction
            active = (usage_count > 0).float().sum().item()
            utilization = active / max(K_eff, 1)

            # Commit trend (positive slope => L_commit rising => strain)
            commit_trend = self._commit_slope()

            # Drift (alignment between encoder and codebook)
            drift = float(delta_mean.norm(dim=1).mean().item())

        # Thresholds (copy of ТЗ table)
        if utilization >= self.u_res and commit_trend <= 0.0:
            return TensionState.RESONANCE
        if utilization >= self.u_str:
            return TensionState.STRAINED
        if utilization >= self.u_ovl:
            return TensionState.OVERLOADED
        return TensionState.COLLAPSE

    def _commit_slope(self) -> float:
        """Simple slope of commit history tail."""
        if len(self.commit_history) < 10:
            return 0.0
        tail = list(self.commit_history)[-min(50, len(self.commit_history)):]
        xs = torch.arange(len(tail), dtype=torch.float32)
        ys = torch.tensor(tail, dtype=torch.float32)
        xm, ym = xs.mean(), ys.mean()
        num = ((xs - xm) * (ys - ym)).sum()
        den = ((xs - xm) ** 2).sum().clamp(min=1e-8)
        return float((num / den).item())

    # ---- 3. Valve shedding --------------------------------------------------

    def valve_shedding(self,
                       codebook: torch.Tensor, tau_k: torch.Tensor,
                       usage_count: torch.Tensor, delta_var: torch.Tensor,
                       K_eff: int):
        """
        Preemptive action under STRAINED/OVERLOADED:
          - nudge underused codes toward overloaded regions
          - slightly soften τ on overloaded codes

        Returns (new_codebook, new_tau, did_shed).
        """
        with torch.no_grad():
            new_cb = codebook.clone()
            new_tau = tau_k.clone()

            total_usage = usage_count.sum().clamp(min=1.0)
            mean_share = total_usage / K_eff
            underused_mask = usage_count < (mean_share * 0.1)
            underused = torch.nonzero(underused_mask, as_tuple=False).flatten()

            if K_eff < 2 or underused.numel() == 0:
                return new_cb, new_tau, False

            # Overloaded = top quartile by variance
            q75 = torch.quantile(delta_var, 0.75)
            overloaded_mask = delta_var > q75
            overloaded = torch.nonzero(overloaded_mask, as_tuple=False).flatten()

            if overloaded.numel() == 0:
                return new_cb, new_tau, False

            # Nudge each underused toward its nearest overloaded
            u_vecs = new_cb[underused]                              # (U, d)
            o_vecs = new_cb[overloaded]                             # (O, d)
            # Pairwise distances
            dists = torch.cdist(u_vecs, o_vecs)                     # (U, O)
            nearest = dists.argmin(dim=1)                            # (U,)
            targets = o_vecs[nearest]                                # (U, d)
            new_cb[underused] = u_vecs + self.valve_rate * (targets - u_vecs)

            # Soften τ on overloaded (with hard cap to prevent runaway)
            new_tau[overloaded] = (new_tau[overloaded] * self.valve_tau).clamp(max=10.0)

            return new_cb, new_tau, True
