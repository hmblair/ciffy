"""
Backend-agnostic geometric primitives for molecular modeling.

This package provides pure geometric functions that work with both NumPy and PyTorch
backends. All functions are stateless and testable in isolation.

Submodules:
- primitives: Vector operations, Rodrigues rotation, CCD optimization, ring closure
- transforms: SE(3) transforms, frame computation, residue positioning
- alignment: Kabsch alignment algorithm
- rmsd: Coordinate RMSD computation
"""

# Vector operations
from .primitives import (
    cross,
    dot,
    norm,
    normalize,
)

# Trigonometric
from .primitives import (
    atan2,
    cos,
    sin,
)

# Array utilities
from .primitives import to_scalar

# Rotation
from .primitives import rodrigues_rotate

# CCD optimization
from .primitives import (
    optimal_rotation_to_target,
    project_to_rotation_circle,
)

# Ring closure
from .primitives import (
    circle_sphere_intersect,
    verify_closure_distance,
)

# SE(3) transforms
from .transforms import (
    LocalCoordinates,
    rodrigues,
    rotation_to_axis_angle,
    compute_relative_transform,
    apply_relative_transform,
    geodesic_so3,
    se3_loss,
)

# Frame computation
from .transforms import (
    extract_frame_positions,
    frame_from_positions,
    rigid_align,
)

# Kabsch alignment
from .alignment import (
    kabsch_rotation,
    kabsch_align,
)

# RMSD
from .rmsd import rmsd_coords

# Geometry projection (torch-only, differentiable)
from .projection import project_bond_lengths

# Geometry constraints (general system for bond/angle losses)
from .constraints import GeometryConstraints

__all__ = [
    # Vector operations
    "cross",
    "dot",
    "norm",
    "normalize",
    # Trigonometric
    "atan2",
    "cos",
    "sin",
    # Array utilities
    "to_scalar",
    # Rotation
    "rodrigues_rotate",
    # CCD optimization
    "optimal_rotation_to_target",
    "project_to_rotation_circle",
    # Ring closure
    "circle_sphere_intersect",
    "verify_closure_distance",
    # SE(3) transforms
    "LocalCoordinates",
    "rodrigues",
    "rotation_to_axis_angle",
    "compute_relative_transform",
    "apply_relative_transform",
    "geodesic_so3",
    "se3_loss",
    # Frame computation
    "extract_frame_positions",
    "frame_from_positions",
    "rigid_align",
    # Kabsch alignment
    "kabsch_rotation",
    "kabsch_align",
    # RMSD
    "rmsd_coords",
    # Geometry projection
    "project_bond_lengths",
    # Geometry constraints
    "GeometryConstraints",
]
