"""LightningDataModules for ciffy training."""

from .diffusion import LatentDiffusionDataModule, CoordinateDiffusionDataModule
from .flow import ResidueDataModule, FlowDataModule  # FlowDataModule is alias

__all__ = [
    "LatentDiffusionDataModule",
    "CoordinateDiffusionDataModule",
    "ResidueDataModule",
    "FlowDataModule",  # Backwards-compatible alias
]
