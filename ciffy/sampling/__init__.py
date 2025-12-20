"""
Sampling utilities for generating realistic polymer conformations.

This module provides functions for sampling backbone dihedrals from
empirical distributions fitted to PDB data. Supports proteins and RNA.

Includes both rejection sampling and Langevin dynamics methods for
clash-aware autoregressive sampling.
"""

from .backbone import (
    ClashSamplingError,
    randomize_backbone,
    sample_autoregressive,
    sample_protein_autoregressive,
    sample_protein_autoregressive_langevin,
    sample_protein_dihedrals,
    sample_rna_autoregressive,
    sample_rna_autoregressive_langevin,
    sample_rna_dihedrals,
)
from .energy import ClashEnergy, CompositeEnergy, EnergyFunction, GMMEnergy
from .langevin import langevin_dynamics, langevin_dynamics_with_adaptation

__all__ = [
    # Backbone sampling functions
    "randomize_backbone",
    "sample_protein_dihedrals",
    "sample_rna_dihedrals",
    "sample_autoregressive",
    "sample_protein_autoregressive",
    "sample_protein_autoregressive_langevin",
    "sample_rna_autoregressive",
    "sample_rna_autoregressive_langevin",
    "ClashSamplingError",
    # Energy functions
    "EnergyFunction",
    "GMMEnergy",
    "ClashEnergy",
    "CompositeEnergy",
    # Langevin dynamics
    "langevin_dynamics",
    "langevin_dynamics_with_adaptation",
]
