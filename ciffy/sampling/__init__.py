"""
Sampling utilities for dihedral angle distributions.

This module provides:
- GMM-based dihedral sampling for proteins and RNA
- Energy functions for evaluating conformations
- Langevin dynamics for optimization

For generating complete polymer conformations, use :class:`ciffy.nn.flow.PolymerFlowModel`:

    >>> from ciffy.nn.flow import PolymerFlowModel
    >>> model = PolymerFlowModel.load("path/to/model")
    >>> samples = model.sample(sequence, n_samples=10)
"""

from .backbone import (
    ClashSamplingError,
    sample_protein_dihedrals,
    sample_rna_dihedrals,
)
from .energy import ClashEnergy, CompositeEnergy, EnergyFunction, GMMEnergy, StackingEnergy
from .langevin import langevin_dynamics, langevin_dynamics_with_adaptation

__all__ = [
    # Dihedral sampling (returns angle arrays)
    "sample_protein_dihedrals",
    "sample_rna_dihedrals",
    "ClashSamplingError",
    # Energy functions
    "EnergyFunction",
    "GMMEnergy",
    "ClashEnergy",
    "CompositeEnergy",
    "StackingEnergy",
    # Langevin dynamics
    "langevin_dynamics",
    "langevin_dynamics_with_adaptation",
]
