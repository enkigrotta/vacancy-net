"""
Module F: ⟳ PROTOCOL (Replicant)

Decides whether to restructure based on %-pressure and ⫿-health.
Computes the restructuring plan (splits / merges / shifts / resurrections).

One protocol, unified. NO splits of responsibility with other modules.
"""

import torch

from module_e_syntone import TensionState


class ReplicantProtocol:
    def __init__(self, T_cool: int = 1000,
                 pressure_threshold: int = 3,
                 alpha_shift: float = 0.5,
                 var_critical_percentile: float = 0.9,
                 c_critical: float = 0.8,
                 drift_threshold_factor: float = 2.0,
                 device: str = 'cpu'):
        self.T_cool = int(T_cool)
        self.pressure_threshold = int(pressure_threshold)
        self.alpha_shift = float(alpha_shift)
        self.var_percentile = float(var_critical_percentile)
        self.c_critical = float(c_critical)
        self.drift_factor = float(drift_threshold_factor)
        self.device = device

        self.cooldown_counter = 0
        self.events = []       # list of {batch, epoch, plan_summary, pre_stats}

    # ---- Tick ---------------------------------------------------------------

    def step(self):
        if self.cooldown_counter < self.T_cool:
            self.cooldown_counter += 1

    # ---- Decision + plan ----------------------------------------------------

    def decide(self, accumulator, tension: TensionState,
               codebook: torch.Tensor):
        """Returns (should_trigger: bool, plan: dict|None)."""
        if self.cooldown_counter < self.T_cool:
            return False, None

        # Never restructure in COLLAPSE — system too fragile
        if tension == TensionState.COLLAPSE:
            return False, None

        overloaded = accumulator.overloaded_codes(self.var_percentile).tolist()
        redundant = accumulator.redundant_pairs(self.c_critical)
        dead = accumulator.dead_codes(accumulator.K_eff, factor=0.1).tolist()
        drifted = accumulator.drifted_codes(self.drift_factor).tolist()

        pressure = len(overloaded) + len(redundant) + len(dead)

        # In RESONANCE with no pressure, nothing to do
        if tension == TensionState.RESONANCE and pressure == 0:
            return False, None

        if pressure < self.pressure_threshold:
            return False, None

        plan = self._compute_plan(
            overloaded=overloaded,
            redundant=redundant,
            dead=dead,
            drifted=drifted,
            accumulator=accumulator,
            codebook=codebook,
        )

        # Reset cooldown after committing to a plan
        self.cooldown_counter = 0
        return True, plan

    # ---- Planner ------------------------------------------------------------

    def _compute_plan(self, overloaded, redundant, dead, drifted,
                      accumulator, codebook):
        K = codebook.shape[0]
        d = codebook.shape[1]
        device = codebook.device

        plan = {
            'splits': [],         # {code, direction, magnitude}
            'merges': [],         # {code1, code2, new_position}
            'shifts': [],         # {code, shift}
            'resurrections': [],  # {dead_code, new_position}
        }

        # Track codes reserved by earlier operations — avoid double-booking
        used = set()

        # 1. SPLITS — overloaded that aren't in redundant pairs
        red_flat = {k for pair in redundant for k in pair}
        for k in overloaded:
            if k in used or k in red_flat:
                continue
            # Split direction: principal direction of residual for code k
            direction = accumulator.delta_mean[k].clone()
            norm = direction.norm()
            if norm < 1e-8:
                # Use random direction if no meaningful drift
                direction = torch.randn(d, device=device)
                direction = direction / direction.norm().clamp(min=1e-8)
            else:
                direction = direction / norm
            magnitude = float(accumulator.delta_var[k].clamp(min=1e-6).sqrt().item())
            plan['splits'].append({
                'code': int(k),
                'direction': direction.detach(),
                'magnitude': magnitude,
            })
            used.add(k)

        # 2. MERGES — redundant pairs where neither code is being split
        for (k1, k2) in redundant:
            if k1 in used or k2 in used:
                continue
            new_pos = (codebook[k1] + codebook[k2]) / 2
            plan['merges'].append({
                'code1': int(k1),
                'code2': int(k2),
                'new_position': new_pos.detach(),
            })
            used.add(k1)
            used.add(k2)

        # 3. SHIFTS — drifted codes not otherwise touched
        for k in drifted:
            if k in used:
                continue
            plan['shifts'].append({
                'code': int(k),
                'shift': (self.alpha_shift * accumulator.delta_mean[k]).detach(),
            })
            used.add(k)

        # 4. RESURRECTIONS — dead codes placed near highest-Var codes
        if len(accumulator.delta_var) > 0:
            top = int(accumulator.delta_var.argmax().item())
        else:
            top = 0
        for k in dead:
            if k in used:
                continue
            perturb = torch.randn(d, device=device) * 0.05
            plan['resurrections'].append({
                'dead_code': int(k),
                'new_position': (codebook[top] + perturb).detach(),
            })
            used.add(k)

        return plan

    # ---- Event log ----------------------------------------------------------

    def record_event(self, plan: dict, pre_stats: dict,
                     epoch: int, batch: int):
        self.events.append({
            'epoch': epoch,
            'batch': batch,
            'n_splits': len(plan['splits']),
            'n_merges': len(plan['merges']),
            'n_shifts': len(plan['shifts']),
            'n_resurrections': len(plan['resurrections']),
            'pre_stats': pre_stats,
        })
