"""
Module H: Δ₀ INITIALIZER (Inheritance)

Two responsibilities:
  1. kmeans_init(z_e_flat, K)  — Δ₀_initial (used at setup)
  2. execute_restructuring_plan(plan, ...) — Δ₀' (used at every ⟳)

Both produce a codebook. The second one is the inheritance function:
    Δ₀' = f(plan, %-stats, old_codebook)
"""

import torch


# ---- K-means (mini, CPU-safe) -------------------------------------------

def kmeans_init(z_flat: torch.Tensor, K: int,
                n_iter: int = 20, seed: int = 0) -> torch.Tensor:
    """
    Simple k-means++ init + Lloyd iterations. Returns (K, d) centers.

    z_flat: (N, d)
    """
    N, d = z_flat.shape
    device = z_flat.device
    g = torch.Generator(device='cpu').manual_seed(seed)

    # k-means++ seeding
    idx0 = int(torch.randint(0, N, (1,), generator=g).item())
    centers = [z_flat[idx0].clone()]
    for _ in range(K - 1):
        cs = torch.stack(centers)                                 # (k, d)
        d2 = torch.cdist(z_flat, cs).pow(2).min(dim=1).values     # (N,)
        probs = d2 / d2.sum().clamp(min=1e-12)
        nxt = int(torch.multinomial(probs, 1, generator=g).item())
        centers.append(z_flat[nxt].clone())

    C = torch.stack(centers).to(device)                           # (K, d)

    # Lloyd iterations
    for _ in range(n_iter):
        dists = torch.cdist(z_flat, C)                            # (N, K)
        assign = dists.argmin(dim=1)                               # (N,)
        new_C = torch.zeros_like(C)
        counts = torch.bincount(assign, minlength=K).float()
        new_C.index_add_(0, assign, z_flat)
        counts_safe = counts.clamp(min=1.0).unsqueeze(1)
        new_C = new_C / counts_safe
        # For empty clusters, keep old center
        empty = counts == 0
        if empty.any():
            new_C[empty] = C[empty]
        if torch.allclose(new_C, C, atol=1e-6):
            C = new_C
            break
        C = new_C
    return C


# ---- Restructuring executor ---------------------------------------------

def execute_restructuring_plan(plan: dict,
                               old_codebook: torch.Tensor,
                               old_tau: torch.Tensor,
                               device: str = 'cpu'):
    """
    Apply the plan produced by Module F. Returns (new_codebook, new_tau, K_new).

    Order matters:
      1) Splits   — old K + n_splits
      2) Merges   — mark second-of-pair for removal
      3) Shifts
      4) Resurrections
      5) Remove merged codes — update K
    """
    cb = old_codebook.detach().clone()
    tau = old_tau.detach().clone()

    new_codes = [cb[i].clone() for i in range(cb.shape[0])]
    new_tau = [float(tau[i].item()) for i in range(tau.shape[0])]

    # 1. Splits: replace old code with +direction, append -direction
    for sp in plan['splits']:
        k = sp['code']
        direction = sp['direction'].to(cb.device)
        magnitude = float(sp['magnitude'])
        if k >= len(new_codes):
            continue
        base = new_codes[k].clone()
        new_codes[k] = base + magnitude * direction
        new_codes.append(base - magnitude * direction)
        new_tau.append(new_tau[k])       # inherit τ

    # 2. Merges: first of pair becomes centroid, second marked for removal
    to_remove = set()
    for mg in plan['merges']:
        k1 = mg['code1']
        k2 = mg['code2']
        if k1 >= len(new_codes) or k2 >= len(new_codes):
            continue
        if k1 in to_remove or k2 in to_remove:
            continue
        new_codes[k1] = mg['new_position'].to(cb.device)
        new_tau[k1] = 0.5 * (new_tau[k1] + new_tau[k2])
        to_remove.add(k2)

    # 3. Shifts
    for sh in plan['shifts']:
        k = sh['code']
        if k >= len(new_codes) or k in to_remove:
            continue
        new_codes[k] = new_codes[k] + sh['shift'].to(cb.device)

    # 4. Resurrections
    for res in plan['resurrections']:
        k = res['dead_code']
        if k >= len(new_codes):
            continue
        new_codes[k] = res['new_position'].to(cb.device)
        # If tau was somehow zeroed, reset to mean
        if new_tau[k] <= 0:
            mean_tau = sum(new_tau) / max(len(new_tau), 1)
            new_tau[k] = max(mean_tau, 1e-3)

    # 5. Remove merged codes (work on index lists to stay consistent)
    keep_idx = [i for i in range(len(new_codes)) if i not in to_remove]
    final_codes = [new_codes[i] for i in keep_idx]
    final_tau = [new_tau[i] for i in keep_idx]

    # Safety: keep K >= 2
    if len(final_codes) < 2:
        # Unmerge one: restore the last-removed
        for i in sorted(to_remove):
            final_codes.append(new_codes[i])
            final_tau.append(new_tau[i])
            if len(final_codes) >= 2:
                break

    new_codebook = torch.stack(final_codes, dim=0).to(device)
    new_tau_tensor = torch.tensor(final_tau, dtype=torch.float32, device=device)
    # Ensure tau > 0
    new_tau_tensor = new_tau_tensor.clamp(min=1e-3)
    return new_codebook, new_tau_tensor, new_codebook.shape[0]
