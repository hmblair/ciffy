"""LightningModules for ciffy models."""

from .base import BaseCiffyModule
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
from .residue_flow import (
    ResidueFlowDataConfig,
    ResidueFlowFullConfig,
    ResidueFlowModelConfig,
    ResidueFlowModule,
)

__all__ = [
    # Base
    "BaseCiffyModule",
    # Latent diffusion
    "LatentDiffusionDataConfig",
    "LatentDiffusionFullConfig",
    "LatentDiffusionModule",
    # Coordinate diffusion
    "CoordinateDiffusionDataConfig",
    "CoordinateDiffusionFullConfig",
    "CoordinateDiffusionModule",
    # Residue flow
    "ResidueFlowDataConfig",
    "ResidueFlowFullConfig",
    "ResidueFlowModelConfig",
    "ResidueFlowModule",
]
