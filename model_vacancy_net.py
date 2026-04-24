"""
∅-NET Phase 1: Self-Governing Vacancy Network.

All nine modules assembled.

DATA FLOW (one batch):
  1. x       -> Module A (encoder)        -> z_e
  2. z_e     -> Module B (∅_sg)           -> e_k + delta
     delta  -> Module D (% accumulator)   -> stats
  3. e_k     -> Module C (decoder)        -> x_hat, L_recon
  4. z_e,e_k -> Module E (⫿)              -> L_commit + tension
     valve shedding if STRAINED/OVERLOADED
  5. L_total -> Module I                  -> weights updated
     Codebook in B updated via EMA (inside B)
  6. D + E   -> Module F (⟳)              -> plan (if triggered)
     plan   -> Module H                   -> new codebook -> B
     D.reset()
  7. G       -> periodic observation (log only in Phase 1)

STRUCTURAL INVARIANTS (asserted):
  - K_eff >= 2 at all times
  - After ⟳: accumulator.K_eff == vacancy.K_eff
  - COLLAPSE blocks ⟳ (protected in Module F)
  - delta is detached before hitting Module D (no grad leak)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from module_a_encoder import Encoder
from module_b_vacancy import SelfGoverningVacancy
from module_c_decoder import Decoder
from module_d_accumulator import RemainderAccumulator
from module_e_syntone import Syntone, TensionState
from module_f_replicant import ReplicantProtocol
from module_g_observer import DeltaObserver
from module_h_initializer import execute_restructuring_plan
from module_i_gradient import GradientEngine

from config import Config


class VacancyNet(nn.Module):
    """Self-governing VQ-VAE with remainder-driven restructuring."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        device = cfg.resolve_device()

        # ---- Neural (trainable) ----
        self.encoder = Encoder(cfg.image_channels, cfg.latent_dim, cfg.hidden_dims)
        self.decoder = Decoder(cfg.image_channels, cfg.latent_dim,
                               tuple(reversed(cfg.hidden_dims)))

        # ---- Vacancy (structural, buffer-only) ----
        self.vacancy = SelfGoverningVacancy(
            cfg.K_initial, cfg.latent_dim, cfg.tau_initial, cfg.ema_decay_codebook,
        )

        # ---- Stateless-from-nn POV; keep references on self so .to() works on their tensors ----
        self.accumulator = RemainderAccumulator(
            cfg.K_initial, cfg.latent_dim,
            cfg.ema_decay_stats, cfg.cross_corr_update_interval, device,
        )
        self.syntone = Syntone(
            cfg.beta,
            cfg.utilization_resonance, cfg.utilization_strained, cfg.utilization_overloaded,
            cfg.commit_history_window, cfg.valve_nudge_rate, cfg.valve_tau_softening,
            device,
        )
        self.replicant = ReplicantProtocol(
            cfg.T_cool, cfg.pressure_threshold,
            cfg.alpha_shift, cfg.var_critical_percentile,
            cfg.c_critical, cfg.drift_threshold_factor, device,
        )
        self.observer = DeltaObserver(cfg.N_observe, cfg.interpolation_samples, device)

        self.gradient_engine = None   # set by setup_gradient_engine after .to(device)

        # Counters
        self._total_batches = 0
        self._replicant_count = 0

    # ---- Setup --------------------------------------------------------------

    def setup_gradient_engine(self):
        self.gradient_engine = GradientEngine(
            self.encoder.parameters(),
            self.decoder.parameters(),
            self.cfg.learning_rate, self.cfg.weight_decay,
        )

    def _move_structural_buffers(self, device):
        """Move tensors inside non-nn.Module structural objects."""
        acc = self.accumulator
        acc.device = device
        acc.delta_mean = acc.delta_mean.to(device)
        acc.delta_var = acc.delta_var.to(device)
        acc.usage_count = acc.usage_count.to(device)
        acc.d_percent_dt = acc.d_percent_dt.to(device)
        acc.cross_corr = acc.cross_corr.to(device)
        self.syntone.device = device
        self.replicant.device = device
        self.observer.device = device

    def to(self, device, *args, **kwargs):
        out = super().to(device, *args, **kwargs)
        self._move_structural_buffers(str(device))
        return out

    # ---- Forward (no stats, no ⟳) ------------------------------------------

    def forward(self, x: torch.Tensor):
        z_e = self.encoder(x)
        e_k_ste, indices, delta = self.vacancy(z_e)
        x_hat = self.decoder(e_k_ste)
        L_recon = F.mse_loss(x_hat, x)
        L_commit = self.syntone.commitment_loss(z_e, e_k_ste)
        L_total = L_recon + L_commit
        return {
            'x_hat': x_hat, 'z_e': z_e, 'e_k': e_k_ste,
            'indices': indices, 'delta': delta,
            'L_recon': L_recon, 'L_commit': L_commit, 'L_total': L_total,
        }

    # ---- Full train step ----------------------------------------------------

    def train_step(self, x: torch.Tensor, epoch: int, batch: int) -> dict:
        self._total_batches += 1

        # 1–3: forward
        res = self.forward(x)

        # 2': update % accumulator (delta detached by construction)
        self.accumulator.update(res['delta'].detach(), res['indices'])

        # 4: tension monitor
        tension = self.syntone.tension_monitor(
            self.vacancy.K_eff,
            self.accumulator.usage_count,
            self.accumulator.delta_mean,
        )

        # 4': valve shedding
        did_shed = False
        if tension in (TensionState.STRAINED, TensionState.OVERLOADED):
            new_cb, new_tau, did_shed = self.syntone.valve_shedding(
                self.vacancy.codebook, self.vacancy.tau_k,
                self.accumulator.usage_count, self.accumulator.delta_var,
                self.vacancy.K_eff,
            )
            if did_shed:
                self.vacancy.codebook.copy_(new_cb)
                self.vacancy.tau_k.copy_(new_tau)

        if tension == TensionState.COLLAPSE and self.gradient_engine is not None:
            self.gradient_engine.reduce_lr(0.5)

        # 5: gradient
        if self.gradient_engine is not None:
            self.gradient_engine.step(res['L_total'])

        # 6: replicant
        self.replicant.step()
        replicant_event = None
        if self._replicant_count < self.cfg.max_replicant_events:
            should_trigger, plan = self.replicant.decide(
                self.accumulator, tension, self.vacancy.codebook,
            )
            if should_trigger and plan is not None:
                pre_stats = self.accumulator.get_stats_snapshot()
                K_old = self.vacancy.K_eff

                new_cb, new_tau, K_new = execute_restructuring_plan(
                    plan, self.vacancy.codebook, self.vacancy.tau_k,
                    device=str(x.device),
                )
                self.vacancy.replace_codebook(new_cb, new_tau)
                self.accumulator.reset(K_new)

                # INVARIANT check
                assert self.accumulator.K_eff == self.vacancy.K_eff, \
                    "⟳ invariant violated: accumulator.K_eff != vacancy.K_eff"
                assert self.vacancy.K_eff >= 2, "⟳ invariant: K_eff >= 2"

                self.replicant.record_event(plan, pre_stats, epoch, batch)
                self._replicant_count += 1

                replicant_event = {
                    'epoch': epoch, 'batch': batch,
                    'K_old': K_old, 'K_new': K_new,
                    'n_splits': len(plan['splits']),
                    'n_merges': len(plan['merges']),
                    'n_shifts': len(plan['shifts']),
                    'n_resurrections': len(plan['resurrections']),
                }

        # 7: observer tick
        self.observer.step()

        return {
            'L_recon': float(res['L_recon'].item()),
            'L_commit': float(res['L_commit'].item()),
            'L_total': float(res['L_total'].item()),
            'K_eff': int(self.vacancy.K_eff),
            'tension': tension.value,
            'valve_shed': bool(did_shed),
            'replicant_event': replicant_event,
        }

    # ---- Helpers ------------------------------------------------------------

    @torch.no_grad()
    def collect_init_data(self, dataloader, n_batches: int = 5,
                          device: str = 'cpu') -> torch.Tensor:
        self.eval()
        chunks = []
        for i, batch in enumerate(dataloader):
            if i >= n_batches:
                break
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device)
            z_e = self.encoder(x)
            z_flat = z_e.permute(0, 2, 3, 1).reshape(-1, self.cfg.latent_dim)
            chunks.append(z_flat)
        self.train()
        return torch.cat(chunks, dim=0) if chunks else torch.empty(0, self.cfg.latent_dim, device=device)
