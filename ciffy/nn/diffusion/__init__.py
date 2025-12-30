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
from .latent_denoiser import (
    LatentDenoiserConfig,
    LatentDenoiser,
)
from .latent_diffusion import (
    LatentDiffusionConfig,
    LatentDiffusionModel,
)
from .coordinate_denoiser import (
    CoordinateDenoiserConfig,
    CoordinateDenoiser,
)
from .coordinate_diffusion import (
    CoordinateDiffusionConfig,
    CoordinateDiffusionModel,
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
    # Config classes
    "DiffusionModelConfig",
    "DiffusionDataConfig",
    "DiffusionConfig",
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
    # Latent diffusion
    "LatentDenoiserConfig",
    "LatentDenoiser",
    "LatentDiffusionConfig",
    "LatentDiffusionModel",
    # Coordinate diffusion
    "CoordinateDenoiserConfig",
    "CoordinateDenoiser",
    "CoordinateDiffusionConfig",
    "CoordinateDiffusionModel",
]
