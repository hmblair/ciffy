"""Diffusion model components for generative modeling.

Provides utilities for training and sampling from diffusion models:
- Noise schedules (linear, cosine)
- Forward/reverse diffusion processes (DDPM, DDIM)
- Timestep embeddings
- EMA for model weights

Example:
    >>> from ciffy.nn.diffusion import CosineNoiseSchedule, DiffusionProcess, EMA
    >>>
    >>> schedule = CosineNoiseSchedule(num_timesteps=1000)
    >>> process = DiffusionProcess(schedule)
    >>> ema = EMA(model, decay=0.9999)
"""

from .process import (
    FixedSinusoidalEmbedding,
    NoiseSchedule,
    LinearNoiseSchedule,
    CosineNoiseSchedule,
    DiffusionProcess,
    TimestepEmbedding,
)
from .ema import (
    EMA,
    create_ema_model,
    update_ema_model,
)
from .trainer import (
    DiffusionModelConfig,
    DiffusionDataConfig,
    DiffusionConfig,
    DiffusionTrainer,
)
from .metrics import (
    TimestepLossProfile,
    SampleQualityMetrics,
    DiffusionMetrics,
    compute_denoising_loss,
    compute_timestep_loss_profile,
    compute_elbo,
    compute_sample_rmsd,
    evaluate_samples,
    compute_diffusion_metrics,
)

__all__ = [
    # Noise schedules
    "NoiseSchedule",
    "LinearNoiseSchedule",
    "CosineNoiseSchedule",
    # Diffusion process
    "DiffusionProcess",
    # Embeddings
    "FixedSinusoidalEmbedding",
    "TimestepEmbedding",
    # EMA utilities
    "EMA",
    "create_ema_model",
    "update_ema_model",
    # Trainer
    "DiffusionModelConfig",
    "DiffusionDataConfig",
    "DiffusionConfig",
    "DiffusionTrainer",
    # Metrics
    "TimestepLossProfile",
    "SampleQualityMetrics",
    "DiffusionMetrics",
    "compute_denoising_loss",
    "compute_timestep_loss_profile",
    "compute_elbo",
    "compute_sample_rmsd",
    "evaluate_samples",
    "compute_diffusion_metrics",
]
