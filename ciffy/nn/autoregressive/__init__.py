"""
Autoregressive models for polymer structure generation.

This module provides transformer-based autoregressive models for generating
polymer structures either in latent space or directly in coordinate space.

Models:
    - ResidueLatentAR: Predicts latent vectors autoregressively
    - PolymerLatentAR: End-to-end latent generation with decoders
    - CoordinateAR: Direct coordinate prediction with global conditioning
    - AtomAR: All-atom conditioned, residue-level output (recommended)
"""

from .latent import (
    ResidueLatentAR,
    ResidueLatentARConfig,
    PolymerLatentAR,
)

from .coordinate import (
    CoordinateAR,
    CoordinateARConfig,
)

from .atom import (
    AtomAR,
    AtomARConfig,
)

__all__ = [
    # Latent-space models
    "ResidueLatentAR",
    "ResidueLatentARConfig",
    "PolymerLatentAR",
    # Coordinate-space models (residue-level)
    "CoordinateAR",
    "CoordinateARConfig",
    # All-atom conditioned model (recommended)
    "AtomAR",
    "AtomARConfig",
]
