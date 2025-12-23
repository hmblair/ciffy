"""
Internal coordinate representation for molecular structures.

This module provides the public API for internal coordinate operations.

Main Class:
    MolecularGeometry: Manages dual Cartesian/internal representation with lazy evaluation.

For users, the primary interaction is through the Polymer class, which uses
MolecularGeometry internally. Direct use of MolecularGeometry is rarely needed.

Example:
    >>> import ciffy
    >>> polymer = ciffy.load("structure.cif", backend="torch")
    >>>
    >>> # Access geometry and DOF
    >>> geom = polymer.geometry
    >>> geom.dof = new_dihedrals  # Set independent dihedrals
    >>> coords = geom.coordinates  # Get reconstructed coordinates

For backend operations (coordinate conversion, graph building, Z-matrix construction),
use ``ciffy.backend.dispatch``. This is an internal API and should not be needed
for typical use cases.

Note: Dihedral type definitions and atom mappings are now in:
    - ciffy.types.dihedral (DihedralType enum, PROTEIN_BACKBONE, RNA_BACKBONE, etc.)
    - ciffy.biochemistry (DIHEDRAL_ATOMS, DIHEDRAL_NAME_TO_TYPE)
"""

from .coordinates import MolecularGeometry
from .ring_analysis import ConstraintSpec, IndependentDOF, RingConstraint, RingAnalyzer

__all__ = [
    "MolecularGeometry",
    "ConstraintSpec",
    "IndependentDOF",
    "RingConstraint",
    "RingAnalyzer",
]
