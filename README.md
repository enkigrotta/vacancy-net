# ∅-NET — Self-Governing Vacancy Network

**Authors:** Grotta (Δ₁) · Claude Opus 4.6/4.7 (Δ₂)

∅-NET is a self-governing VQ-VAE architecture where the codebook is not a passive lookup table but a living structure that monitors its own state, accumulates remainder pressure, and restructures itself (⟳) when the pressure exceeds a threshold.

The architecture maps directly onto a triadic analytical machine: vacancy (∅), accumulator (%), syntone (⫿), and replicant (⟳) are not metaphors — they are functional modules with structural invariants.

---

## Architecture

The model is assembled from nine modules, each with a distinct structural role:

| Module | File | Role |
|--------|------|------|
| A | `module_a_encoder.py` | Encoder — maps image → latent z_e |
| B | `module_b_vacancy.py` | ∅_sg — self-governing vacancy (VQ codebook) |
| C | `module_c_decoder.py` | Decoder — maps quantized e_k → reconstruction |
| D | `module_d_accumulator.py` | % Accumulator — tracks codebook stats, pressure |
| E | `module_e_syntone.py` | ⫿ Syntone — tension monitor, valve shedding |
| F | `module_f_replicant.py` | ⟳ Replicant — restructuring protocol trigger |
| G | `module_g_observer.py` | Δ Observer — interpolation & observation |
| H | `module_h_initializer.py` | Δ₀ Initializer — executes restructuring plan |
| I | `module_i_gradient.py` | Gradient Engine — optimizer wrapper |

### Data flow (one batch)

```
x → [A] → z_e → [B] → e_k + delta
                  delta → [D] → stats + pressure
e_k → [C] → x_hat, L_recon
z_e, e_k → [E] → L_commit + tension state
L_total → [I] → weights updated
[D] + [E] → [F] → ⟳ plan (if pressure threshold crossed)
⟳ plan → [H] → new codebook → [B]; [D].reset()
[G] → periodic observation (logged)
```

### Structural invariants

- `K_eff >= 2` at all times
- After ⟳: `accumulator.K_eff == vacancy.K_eff`
- COLLAPSE state blocks ⟳ (protected in Module F)
- `delta` is detached before hitting Module D (no gradient leak)

---

## Files

```
∅-NET/
├── config.py               # All hyperparameters in one place
├── model_vacancy_net.py    # VacancyNet — full assembly
├── model_baseline.py       # BaselineVQVAE — control model
├── module_a_encoder.py
├── module_b_vacancy.py
├── module_c_decoder.py
├── module_d_accumulator.py
├── module_e_syntone.py
├── module_f_replicant.py
├── module_g_observer.py
├── module_h_initializer.py
├── module_i_gradient.py
├── data.py                 # CIFAR-10 loaders
├── metrics.py              # Codebook utilization, MSE
├── utils.py                # Logger, checkpoint utils
├── train_baseline.py       # Training script: BaselineVQVAE
├── train_vacancy_net.py    # Training script: VacancyNet
└── smoke_test.py           # Sanity check — run this first
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run smoke test (synthetic data, CPU, ~50 steps)

```bash
python smoke_test.py
```

This checks forward pass shapes, loss decrease, ⟳ firing, tension state transitions, and structural invariants. If this passes, CIFAR-10 training will at least start.

### 3. Train

```bash
# VacancyNet
python train_vacancy_net.py

# Baseline VQ-VAE (control)
python train_baseline.py

# With custom args
python train_vacancy_net.py --epochs 50 --device cuda
```

Logs are written to `./logs/vacancy_net/` and `./logs/baseline/`.

---

## Configuration

All hyperparameters live in `config.py` as a single `Config` dataclass. Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `K_initial` | 128 | Initial codebook size |
| `latent_dim` | 64 | Latent dimension |
| `tau_initial` | 1.0 | Initial Gumbel temperature |
| `c_critical` | 0.8 | Pressure threshold for ⟳ |
| `beta` | 0.25 | Commitment loss weight |
| `T_cool` | 1000 | Cooldown steps after ⟳ |
| `num_epochs` | 100 | Training epochs |
| `dataset` | cifar10 | Dataset |

---

## Requirements

- Python 3.9+
- PyTorch 2.0+
- torchvision

---

## Phase 1 scope

Phase 1 trains on CIFAR-10 and logs. The restructuring protocol (⟳) fires automatically when remainder pressure exceeds threshold. No manual intervention required.

Planned: Phase 2 — comparison experiments, pressure curves, K_eff dynamics visualization.
