"""Diffusion model trainer following the VAE trainer pattern.

Provides DiffusionConfig and DiffusionTrainer for training diffusion models.

Example:
    >>> from ciffy.nn.diffusion.trainer import DiffusionConfig, DiffusionTrainer
    >>>
    >>> config = DiffusionConfig.from_yaml("config.yaml")
    >>> trainer = DiffusionTrainer(config)
    >>> result = trainer.train()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    optim = None

if TYPE_CHECKING:
    import torch
    import torch.nn as nn
    import torch.optim as optim

from ..base_trainer import (
    BaseConfig,
    BaseTrainer,
    OutputConfig,
    TrainingConfig,
    WandbConfig,
)
from .process import CosineNoiseSchedule, DiffusionProcess, LinearNoiseSchedule
from .ema import EMA

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


# =============================================================================
# Diffusion Trainer
# =============================================================================


class DiffusionTrainer(BaseTrainer):
    """Trainer for diffusion models.

    Extends BaseTrainer with diffusion-specific functionality:
    - EMA weight tracking
    - Noise schedule management
    - Diffusion-specific loss function

    Example:
        >>> config = DiffusionConfig.from_yaml("config.yaml")
        >>> trainer = DiffusionTrainer(config)
        >>> result = trainer.train()
    """

    config: DiffusionConfig

    def __init__(
        self,
        config: DiffusionConfig,
        model: "nn.Module | None" = None,
        **kwargs: Any,
    ) -> None:
        """Initialize diffusion trainer.

        Args:
            config: Diffusion training configuration.
            model: Optional pre-initialized model.
            **kwargs: Additional arguments passed to BaseTrainer.
        """
        super().__init__(config=config, model=model, **kwargs)

        # Setup noise schedule
        if config.model.noise_schedule == "cosine":
            schedule = CosineNoiseSchedule(config.model.num_timesteps)
        else:
            schedule = LinearNoiseSchedule(config.model.num_timesteps)

        self.diffusion_process = DiffusionProcess(schedule)
        self.ema: EMA | None = None

    def _setup_ema(self, model: "nn.Module") -> None:
        """Setup EMA for the model.

        Args:
            model: Model to track with EMA.
        """
        self.ema = EMA(
            model,
            decay=self.config.model.ema_decay,
            warmup_steps=self.config.model.ema_warmup_steps,
        )

    def _compute_loss(
        self,
        model: "nn.Module",
        batch: Any,
    ) -> tuple["torch.Tensor", dict[str, float]]:
        """Compute diffusion loss for a batch.

        Args:
            model: Diffusion model.
            batch: Training batch.

        Returns:
            Tuple of (loss tensor, metrics dict).
        """
        raise NotImplementedError(
            "DiffusionTrainer._compute_loss must be implemented for your model."
        )


__all__ = [
    "DiffusionModelConfig",
    "DiffusionDataConfig",
    "DiffusionConfig",
    "DiffusionTrainer",
]
