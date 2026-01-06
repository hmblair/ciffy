"""
Autoregressive models for polymer structure generation.

This module provides transformer-based autoregressive models for generating
polymer structures either in latent space or directly in coordinate space.

Models:
    - ResidueLatentARModel: Predicts latent vectors autoregressively
    - PolymerLatentARModel: End-to-end latent generation with decoders
    - CoordinateARModel: Direct coordinate prediction with global conditioning
    - AtomARModel: All-atom conditioned, residue-level output (recommended)
"""

from .latent import (
    ResidueLatentARModel,
    ResidueLatentARModelConfig,
    PolymerLatentARModel,
)

from .coordinate import (
    CoordinateARModel,
    CoordinateARModelConfig,
)

from .atom import (
    AtomARModel,
    AtomARModelConfig,
)

__all__ = [
    # Latent-space models
    "ResidueLatentARModel",
    "ResidueLatentARModelConfig",
    "PolymerLatentARModel",
    # Coordinate-space models (residue-level)
    "CoordinateARModel",
    "CoordinateARModelConfig",
    # All-atom conditioned model (recommended)
    "AtomARModel",
    "AtomARModelConfig",
]
