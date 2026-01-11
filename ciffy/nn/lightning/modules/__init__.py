"""LightningModules for ciffy models."""

from .base import BaseCiffyModule
from .vae_base import BaseVAEModule, BaseVAEModelConfig
from .latent_diffusion import (
    LatentDiffusionDataConfig,
    LatentDiffusionFullConfig,
    LatentDiffusionModule,
)
from .coordinate_diffusion import (
    CoordinateDiffusionDataConfig,
    CoordinateDiffusionFullConfig,
    CoordinateDiffusionModule,
)
from .autoregressive import (
    ResidueLatentARModelTrainingConfig,
    ResidueLatentARModelFullConfig,
    ResidueLatentARModelModule,
    ResidueLatentARModelDataModule,
    PolymerLatentDataset,
)

__all__ = [
    # Base
    "BaseCiffyModule",
    "BaseVAEModule",
    "BaseVAEModelConfig",
    # Latent diffusion
    "LatentDiffusionDataConfig",
    "LatentDiffusionFullConfig",
    "LatentDiffusionModule",
    # Coordinate diffusion
    "CoordinateDiffusionDataConfig",
    "CoordinateDiffusionFullConfig",
    "CoordinateDiffusionModule",
    # Autoregressive
    "ResidueLatentARModelTrainingConfig",
    "ResidueLatentARModelFullConfig",
    "ResidueLatentARModelModule",
    "ResidueLatentARModelDataModule",
    "PolymerLatentDataset",
]
