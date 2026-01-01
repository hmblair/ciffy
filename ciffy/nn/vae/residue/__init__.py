"""Residue-level VAE models."""

from .model import ResidueVAE, ResidueVAEConfig
from .attention import AttentionResidueVAE, AttentionResidueVAEConfig
from .invariant import InvariantResidueVAE, InvariantResidueVAEConfig
from .consolidated import ConsolidatedResidueVAE, ConsolidatedVAEConfig

__all__ = [
    "ResidueVAE",
    "ResidueVAEConfig",
    "AttentionResidueVAE",
    "AttentionResidueVAEConfig",
    "InvariantResidueVAE",
    "InvariantResidueVAEConfig",
    "ConsolidatedResidueVAE",
    "ConsolidatedVAEConfig",
]
