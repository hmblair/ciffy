"""
Internal coordinate representation for molecular structures.

This module provides tools for converting between Cartesian (XYZ) and
internal (bond length, bond angle, dihedral) coordinate representations.

Main Classes:
    CoordinateManager: Manages dual representation with lazy evaluation.
    ZMatrix: Z-matrix representation as (M, 4) array.

Example:
    >>> import ciffy
    >>> polymer = ciffy.load("structure.cif", backend="torch")
    >>>
    >>> # Access internal coordinates (computed lazily)
    >>> dihedrals = polymer.dihedrals  # (N,) dihedral angles
    >>> phi = polymer.dihedral(ciffy.DihedralType.PHI)  # Backbone phi angles
    >>>
    >>> # Modify dihedrals (triggers Cartesian reconstruction)
    >>> polymer.dihedrals = modified_dihedrals
"""

from .coordinates import CoordinateManager
from .graph import ZMatrix, build_bond_graph
from .nerf import nerf_reconstruct
from .dihedrals import (
    PROTEIN_DIHEDRALS,
    NUCLEIC_ACID_DIHEDRALS,
    compute_dihedral_indices,
    DIHEDRAL_TYPE_TO_INDEX,
    INDEX_TO_DIHEDRAL_TYPE,
)

__all__ = [
    # Main classes
    "CoordinateManager",
    "ZMatrix",
    # Graph functions
    "build_bond_graph",
    # NERF reconstruction
    "nerf_reconstruct",
    # Dihedral definitions
    "PROTEIN_DIHEDRALS",
    "NUCLEIC_ACID_DIHEDRALS",
    "compute_dihedral_indices",
    "DIHEDRAL_TYPE_TO_INDEX",
    "INDEX_TO_DIHEDRAL_TYPE",
]
