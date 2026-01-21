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
]
