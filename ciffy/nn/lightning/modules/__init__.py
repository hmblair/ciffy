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
from .residue_vae import (
    ResidueVAEDataConfig,
    ResidueVAEFullConfig,
    ResidueVAEModelConfig,
    ResidueVAEModule,
)
from .attention_vae import (
    AttentionVAEDataConfig,
    AttentionVAEFullConfig,
    AttentionVAEModelConfig,
    AttentionResidueVAEModule,
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
    # Residue VAE (MLP)
    "ResidueVAEDataConfig",
    "ResidueVAEFullConfig",
    "ResidueVAEModelConfig",
    "ResidueVAEModule",
    # Attention VAE
    "AttentionVAEDataConfig",
    "AttentionVAEFullConfig",
    "AttentionVAEModelConfig",
    "AttentionResidueVAEModule",
    # Consolidated VAE
    "ConsolidatedVAEDataConfig",
    "ConsolidatedVAEFullConfig",
    "ConsolidatedVAEModelConfig",
    "ConsolidatedVAEModule",
]
