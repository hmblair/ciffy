"""Residue-level VAE models."""

from .model import ResidueVAE, ResidueVAEConfig
from .attention import AttentionResidueVAE, AttentionResidueVAEConfig

__all__ = [
    "ResidueVAE",
    "ResidueVAEConfig",
    "AttentionResidueVAE",
    "AttentionResidueVAEConfig",
]
