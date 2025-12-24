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

# New unified constraint system
from .constraints import ClosureConstraints, ConstraintSystem, solve_closure
from .jacobian import discover_dof, compute_jacobian_analytical
from .differentiable import DOFToCartesian, dof_to_cartesian, TORCH_AVAILABLE

# Legacy exports (kept for backwards compatibility)
from .ring_analysis import ConstraintSpec, IndependentDOF, RingConstraint, RingAnalyzer

__all__ = [
    # Main class
    "MolecularGeometry",
    # New constraint system
    "ClosureConstraints",
    "ConstraintSystem",
    "solve_closure",
    "discover_dof",
    "compute_jacobian_analytical",
    "DOFToCartesian",
    "dof_to_cartesian",
    "TORCH_AVAILABLE",
    # Legacy (deprecated - will be removed in future version)
    "ConstraintSpec",
    "IndependentDOF",
    "RingConstraint",
    "RingAnalyzer",
]
