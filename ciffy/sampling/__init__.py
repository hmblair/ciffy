"""
Sampling utilities for generating realistic polymer conformations.

.. deprecated::
    This module is deprecated. The dihedral-based sampling functions no longer
    work after the internal coordinate system was removed.

    **Migration path**: Use :class:`ciffy.nn.flow.PolymerFlowModel` for generating
    realistic polymer conformations:

        >>> from ciffy.nn.flow import PolymerFlowModel
        >>> model = PolymerFlowModel.load("path/to/model")
        >>> samples = model.sample(sequence, n_samples=10)

    The flow-based approach provides:
    - Direct Cartesian coordinate generation
    - Latent space interpolation and sampling
    - No ring closure or DOF discovery issues
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
from .energy import ClashEnergy, CompositeEnergy, EnergyFunction, GMMEnergy, StackingEnergy
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
    "StackingEnergy",
    # Langevin dynamics
    "langevin_dynamics",
    "langevin_dynamics_with_adaptation",
]
