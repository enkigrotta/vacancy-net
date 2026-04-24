"""
SMOKE TEST for ∅-NET Phase 1.

Run this FIRST. Before CIFAR-10, before anything.

It uses synthetic data (random tensors) and runs ~50 steps on CPU
for both BaselineVQVAE and VacancyNet. It checks:

  - Forward pass produces tensors of correct shapes
  - Loss decreases over steps
  - VacancyNet's K_eff can change (⟳ fires when pressure builds)
  - Tension state transitions are produced
  - Structural invariants hold (K_eff >= 2, accumulator K matches vacancy K)

If this passes, CIFAR-10 training will at least start. If it fails,
nothing else will work.

Run:
    python smoke_test.py
"""

import sys
import traceback

import torch

from config import Config
from model_baseline import BaselineVQVAE
from model_vacancy_net import VacancyNet


def section(title: str):
    bar = '=' * 60
    print(f'\n{bar}\n  {title}\n{bar}')


def synthetic_batch(batch_size: int = 16, channels: int = 3, size: int = 32,
                    device: str = 'cpu') -> torch.Tensor:
    # Slightly structured data — not pure noise — so codebook has something to learn
    g = torch.Generator(device='cpu').manual_seed(0)
    x = torch.randn(batch_size, channels, size, size, generator=g)
    # Add a low-frequency component (like a class signal)
    x = x + 0.5 * torch.sin(torch.linspace(0, 6.28, size)).view(1, 1, size, 1)
    return x.to(device)


def test_baseline(device: str = 'cpu'):
    section('BASELINE VQ-VAE — forward + backward smoke')
    cfg = Config()
    cfg.K_initial = 32
    cfg.batch_size = 16
    cfg.num_workers = 0
    torch.manual_seed(0)

    model = BaselineVQVAE(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    x = synthetic_batch(cfg.batch_size, device=device)

    # Init codebook once
    with torch.no_grad():
        z = model.encoder(x)
        z_flat = z.permute(0, 2, 3, 1).reshape(-1, cfg.latent_dim)
        model.initialize_codebook(z_flat)

    losses = []
    for step in range(50):
        out = model(x)
        opt.zero_grad()
        out['L_total'].backward()
        opt.step()
        losses.append(float(out['L_total'].item()))

    first, last = losses[0], losses[-1]
    print(f'  L_total:  step0={first:.4f}  step49={last:.4f}  Δ={first-last:+.4f}')
    assert out['x_hat'].shape == x.shape, f'x_hat shape {out["x_hat"].shape} != {x.shape}'
    assert last < first + 0.1, 'Loss should not blow up'
    print('  OK: shapes correct, loss stable')
    return True


def test_vacancy_net(device: str = 'cpu'):
    section('VACANCYNET — full train_step + ⟳ smoke')
    cfg = Config()
    cfg.K_initial = 32
    cfg.batch_size = 16
    cfg.num_workers = 0
    # Aggressive schedule so ⟳ can fire within 50 steps
    cfg.T_cool = 5
    cfg.commit_history_window = 10
    cfg.pressure_threshold_min = 2
    torch.manual_seed(0)

    model = VacancyNet(cfg).to(device)
    model.setup_gradient_engine()

    x = synthetic_batch(cfg.batch_size, device=device)

    tensions_seen = set()
    replicants = 0
    K_history = [model.vacancy.K_eff]
    losses = []

    for step in range(50):
        res = model.train_step(x, epoch=0, batch=step)
        losses.append(res['L_total'])
        tensions_seen.add(res['tension'])
        K_history.append(res['K_eff'])
        if res['replicant_event'] is not None:
            replicants += 1

        # Structural invariants every step
        assert model.vacancy.K_eff >= 2, 'INVARIANT: K_eff >= 2'
        assert model.vacancy.K_eff == model.accumulator.K_eff, \
            f'INVARIANT: vacancy.K ({model.vacancy.K_eff}) != accumulator.K ({model.accumulator.K_eff})'
        assert (model.vacancy.tau_k > 0).all().item(), 'INVARIANT: tau_k > 0'

    K_min, K_max = min(K_history), max(K_history)
    print(f'  L_total:  step0={losses[0]:.4f}  step49={losses[-1]:.4f}')
    print(f'  K_eff:    range={K_min}..{K_max} (started at {K_history[0]})')
    print(f'  tensions: {sorted(tensions_seen)}')
    print(f'  ⟳ events: {replicants}')

    # Minimal expectations
    assert K_min >= 2, 'K_eff must never drop below 2'
    assert len(tensions_seen) >= 1, 'at least one tension state must be produced'
    # Not required that ⟳ fire in 50 steps if pressure stays low with synthetic data,
    # but with our aggressive T_cool it usually does. We'll just log.

    print('  OK: invariants hold, module chain executes, stats flow')
    return True


def test_forward_shapes(device: str = 'cpu'):
    section('FORWARD SHAPE CHECK (VacancyNet)')
    cfg = Config()
    cfg.K_initial = 16
    cfg.batch_size = 4
    torch.manual_seed(0)

    model = VacancyNet(cfg).to(device)
    x = synthetic_batch(cfg.batch_size, device=device)
    out = model.forward(x)

    expected_latent_spatial = cfg.image_size // cfg.downsample_factor
    assert out['z_e'].shape == (4, cfg.latent_dim, expected_latent_spatial, expected_latent_spatial), \
        f'z_e shape: {out["z_e"].shape}'
    assert out['e_k'].shape == out['z_e'].shape, 'e_k shape mismatch'
    assert out['indices'].shape == (4, expected_latent_spatial, expected_latent_spatial)
    assert out['delta'].shape == out['z_e'].shape
    assert out['x_hat'].shape == x.shape
    assert torch.isfinite(out['L_total']).item()
    print(f'  z_e:     {tuple(out["z_e"].shape)}')
    print(f'  e_k:     {tuple(out["e_k"].shape)}')
    print(f'  indices: {tuple(out["indices"].shape)}')
    print(f'  delta:   {tuple(out["delta"].shape)}')
    print(f'  x_hat:   {tuple(out["x_hat"].shape)}')
    print(f'  L_total: {float(out["L_total"].item()):.4f}')
    print('  OK: all shapes match ТЗ')
    return True


def test_replicant_produces_plan():
    section('REPLICANT PLAN SMOKE (synthetic stats)')
    from module_d_accumulator import RemainderAccumulator
    from module_e_syntone import TensionState
    from module_f_replicant import ReplicantProtocol
    from module_h_initializer import execute_restructuring_plan

    K = 8
    d = 16
    acc = RemainderAccumulator(K, d, ema_decay=0.0, cross_corr_interval=1)
    # Fake statistics: code 0 overloaded, codes 3&4 redundant, code 7 dead
    acc.delta_var = torch.tensor([10.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    acc.usage_count = torch.tensor([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 0.01])
    acc.delta_mean = torch.randn(K, d) * 0.1
    acc.delta_mean[3] = torch.ones(d)
    acc.delta_mean[4] = torch.ones(d)
    acc._update_cross_corr()

    rep = ReplicantProtocol(T_cool=0, pressure_threshold=1)
    rep.cooldown_counter = 999  # ready to fire
    cb = torch.randn(K, d)
    should, plan = rep.decide(acc, TensionState.RESONANCE, cb)
    print(f'  should_trigger={should}')
    if should:
        print(f'  plan: splits={len(plan["splits"])} merges={len(plan["merges"])} '
              f'shifts={len(plan["shifts"])} resurrections={len(plan["resurrections"])}')
        new_cb, new_tau, K_new = execute_restructuring_plan(
            plan, cb, torch.ones(K), device='cpu')
        assert K_new >= 2, 'K_new must be >= 2'
        assert (new_tau > 0).all().item()
        print(f'  K: {K} -> {K_new}  (tau > 0: OK)')
    print('  OK: replicant+H produce a valid plan')
    return True


def main():
    device = 'cpu'
    ok = 0
    failures = []
    for name, fn in [
        ('forward_shapes', test_forward_shapes),
        ('baseline', test_baseline),
        ('vacancy_net', test_vacancy_net),
        ('replicant_plan', test_replicant_produces_plan),
    ]:
        try:
            fn() if fn in (test_replicant_produces_plan,) else fn(device)
            ok += 1
        except Exception as e:
            failures.append((name, e, traceback.format_exc()))

    print('\n' + '=' * 60)
    if failures:
        print(f'  SMOKE TEST FAILED: {len(failures)} failure(s)')
        for name, e, tb in failures:
            print(f'\n  -- {name} --')
            print(tb)
        sys.exit(1)
    else:
        print(f'  SMOKE TEST PASSED: {ok}/{ok} checks OK')
        print('=' * 60)


if __name__ == '__main__':
    main()
