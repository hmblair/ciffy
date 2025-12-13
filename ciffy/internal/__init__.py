"""
Internal coordinate representation for molecular structures.

This module provides tools for converting between Cartesian (XYZ) and
internal (bond length, bond angle, dihedral) coordinate representations.

Main Classes:
    InternalPolymer: Stores molecular geometry in internal coordinates.

Main Functions:
    cartesian_to_internal: Convert Polymer to InternalPolymer.

Example:
    >>> import ciffy
    >>> polymer = ciffy.load("structure.cif", backend="torch")
    >>> internal = polymer.to_internal()
    >>>
    >>> # Access/modify internal coordinates
    >>> print(internal.dihedrals.shape)  # (N,) dihedral angles
    >>>
    >>> # Modify and convert back (differentiable)
    >>> internal.dihedrals.requires_grad_(True)
    >>> reconstructed = internal.to_cartesian()
"""

from .internal_polymer import InternalPolymer
from .graph import ZMatrixEntry, build_zmatrix, build_bond_graph
from .zmatrix import cartesian_to_internal
from .nerf import nerf_reconstruct
from .dihedrals import (
    PROTEIN_DIHEDRALS,
    NUCLEIC_ACID_DIHEDRALS,
    compute_dihedral_indices,
)

__all__ = [
    # Main class
    "InternalPolymer",
    # Z-matrix
    "ZMatrixEntry",
    "build_zmatrix",
    "build_bond_graph",
    # Conversion functions
    "cartesian_to_internal",
    "nerf_reconstruct",
    # Dihedral definitions
    "PROTEIN_DIHEDRALS",
    "NUCLEIC_ACID_DIHEDRALS",
    "compute_dihedral_indices",
]
