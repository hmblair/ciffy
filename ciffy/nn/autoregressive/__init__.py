"""
Autoregressive models for polymer structure generation.

This module provides transformer-based autoregressive models for generating
polymer structures either in latent space or directly in coordinate space.

Models:
    - ResidueLatentAR: Predicts latent vectors autoregressively
    - PolymerLatentAR: End-to-end latent generation with decoders
    - CoordinateAR: Direct coordinate prediction with global conditioning
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

__all__ = [
    # Latent-space models
    "ResidueLatentAR",
    "ResidueLatentARConfig",
    "PolymerLatentAR",
    # Coordinate-space models
    "CoordinateAR",
    "CoordinateARConfig",
]
