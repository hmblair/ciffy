"""LightningDataModules for ciffy training.

Note:
    ResidueDataModule (FlowDataModule) and ConsolidatedDataModule have been
    archived. For new residue-level modeling, use ciffy.nn.residue.ResidueVAE
    with ciffy.nn.PolymerDataset directly.
    Old code is in archive/nn/lightning/data/.
"""

from .diffusion import LatentDiffusionDataModule, CoordinateDiffusionDataModule

__all__ = [
    "LatentDiffusionDataModule",
    "CoordinateDiffusionDataModule",
]
