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
from .residue_flow import (
    ResidueFlowDataConfig,
    ResidueFlowFullConfig,
    ResidueFlowModelConfig,
    ResidueFlowModule,
)
from .consolidated_vae import (
    ConsolidatedVAEDataConfig,
    ConsolidatedVAEFullConfig,
    ConsolidatedVAEModelConfig,
    ConsolidatedVAEModule,
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
    # Residue flow
    "ResidueFlowDataConfig",
    "ResidueFlowFullConfig",
    "ResidueFlowModelConfig",
    "ResidueFlowModule",
    # Consolidated VAE
    "ConsolidatedVAEDataConfig",
    "ConsolidatedVAEFullConfig",
    "ConsolidatedVAEModelConfig",
    "ConsolidatedVAEModule",
]
