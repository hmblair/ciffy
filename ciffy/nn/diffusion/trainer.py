"""Diffusion model configuration classes.

Provides configuration dataclasses for diffusion model training.

Example:
    >>> from ciffy.nn.diffusion.trainer import DiffusionConfig
    >>>
    >>> config = DiffusionConfig.from_yaml("config.yaml")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ..base_trainer import (
    BaseConfig,
    OutputConfig,
    TrainingConfig,
    WandbConfig,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Diffusion Configuration
# =============================================================================


@dataclass
class DiffusionModelConfig:
    """Diffusion model architecture configuration.

    Attributes:
        num_timesteps: Number of diffusion timesteps.
        noise_schedule: Type of noise schedule ('linear', 'cosine').
        ema_decay: EMA decay rate for model weights.
        ema_warmup_steps: Steps to warm up EMA decay.
    """

    num_timesteps: int = 1000
    noise_schedule: str = "cosine"
    ema_decay: float = 0.9999
    ema_warmup_steps: int = 2000


@dataclass
class DiffusionDataConfig:
    """Dataset configuration for diffusion training.

    Attributes:
        data_dir: Directory containing training data.
        batch_size: Training batch size.
        num_workers: DataLoader workers.
    """

    data_dir: str = ""
    batch_size: int = 32
    num_workers: int = 4


@dataclass
class DiffusionConfig(BaseConfig):
    """Full diffusion training configuration.

    Combines model, data, training, output, and logging configurations.

    Example:
        >>> config = DiffusionConfig(
        ...     model=DiffusionModelConfig(num_timesteps=1000),
        ...     data=DiffusionDataConfig(data_dir="./data"),
        ...     training=TrainingConfig(epochs=100),
        ... )
    """

    model: DiffusionModelConfig = field(default_factory=DiffusionModelConfig)
    data: DiffusionDataConfig = field(default_factory=DiffusionDataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)


__all__ = [
    "DiffusionModelConfig",
    "DiffusionDataConfig",
    "DiffusionConfig",
]
