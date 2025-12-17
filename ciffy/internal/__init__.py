"""
Internal coordinate representation for molecular structures.

This module provides the public API for internal coordinate operations.

Main Class:
    CoordinateManager: Manages dual Cartesian/internal representation with lazy evaluation.

For users, the primary interaction is through the Polymer class, which uses
CoordinateManager internally. Direct use of CoordinateManager is rarely needed.

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

For backend operations (coordinate conversion, graph building, Z-matrix construction),
use ``ciffy.backend.dispatch``. This is an internal API and should not be needed
for typical use cases.
"""

from .coordinates import CoordinateManager
from .dihedrals import (
    PROTEIN_DIHEDRALS,
    NUCLEIC_ACID_DIHEDRALS,
    INDEX_TO_DIHEDRAL_TYPE,
)

__all__ = [
    # Main class
    "CoordinateManager",
    # Dihedral definitions
    "PROTEIN_DIHEDRALS",
    "NUCLEIC_ACID_DIHEDRALS",
    "INDEX_TO_DIHEDRAL_TYPE",
]
