"""
∅-NET Phase 1 Configuration.

All hyperparameters in one place. No magic numbers.

Authors: Grotta (Δ₁) and Claude Opus 4.6/4.7 (Δ₂)
"""

from dataclasses import dataclass
import os


@dataclass
class Config:
    """Complete hyperparameter specification for ∅-NET Phase 1."""

    # === DATA ===
    dataset: str = "cifar10"
    image_channels: int = 3
    image_size: int = 32
    batch_size: int = 128
    num_workers: int = 2
    data_dir: str = "./data"

    # === ARCHITECTURE ===
    latent_dim: int = 64
    hidden_dims: tuple = (128, 256)
    downsample_factor: int = 4  # 32 -> 8

    # === MODULE B: ∅_sg (Vacancy) ===
    K_initial: int = 128
    tau_initial: float = 1.0
    ema_decay_codebook: float = 0.99

    # === MODULE D: % Accumulator ===
    ema_decay_stats: float = 0.99
    var_critical_percentile: float = 0.90
    c_critical: float = 0.8
    usage_critical_factor: float = 0.1   # dead if usage < 1/(factor*K)
    cross_corr_update_interval: int = 50

    # === MODULE E: ⫿ Syntone ===
    beta: float = 0.25
    utilization_resonance: float = 0.8
    utilization_strained: float = 0.5
    utilization_overloaded: float = 0.2
    commit_history_window: int = 100
    valve_nudge_rate: float = 0.1
    valve_tau_softening: float = 1.05

    # === MODULE F: ⟳ Protocol ===
    T_cool: int = 1000
    pressure_threshold_min: int = 3
    pressure_threshold_factor: float = 0.1
    alpha_shift: float = 0.5
    drift_threshold_factor: float = 2.0

    # === MODULE G: Δ Observer ===
    N_observe: int = 500
    interpolation_samples: int = 100

    # === MODULE H: Δ₀ Initializer ===
    init_method: str = "kmeans"
    kmeans_batches: int = 5

    # === MODULE I: Gradient Engine ===
    learning_rate: float = 3e-4
    weight_decay: float = 0.0

    # === TRAINING ===
    num_epochs: int = 100
    max_replicant_events: int = 20
    log_interval: int = 50
    save_interval: int = 10
    seed: int = 42

    # === LOGGING ===
    log_dir: str = "./logs"
    baseline_log_dir: str = "./logs/baseline"
    vacancy_net_log_dir: str = "./logs/vacancy_net"

    # === DEVICE ===
    device: str = "auto"  # "auto", "cuda", "cpu"

    def __post_init__(self):
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.baseline_log_dir, exist_ok=True)
        os.makedirs(self.vacancy_net_log_dir, exist_ok=True)

    @property
    def latent_spatial(self) -> int:
        return self.image_size // self.downsample_factor

    @property
    def pressure_threshold(self) -> int:
        return max(
            self.pressure_threshold_min,
            int(self.K_initial * self.pressure_threshold_factor),
        )

    def resolve_device(self) -> str:
        import torch
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device
