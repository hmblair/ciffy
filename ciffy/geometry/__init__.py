"""
Backend-agnostic geometric primitives for molecular modeling.

This package provides pure geometric functions that work with both NumPy and PyTorch
backends. All functions are stateless and testable in isolation.

Submodules:
- primitives: Vector operations, Rodrigues rotation, CCD optimization, ring closure
- transforms: SE(3) transforms, frame computation, residue positioning
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
from .primitives import (
    zeros_like,
    clone,
    to_scalar,
)

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
    rodrigues,
    rotation_to_axis_angle,
    axis_angle_to_rotation,
    compute_relative_transform,
    apply_relative_transform,
)

# Frame computation (new unified API)
from .transforms import (
    extract_frame_positions,
    frame_from_positions,
    is_purine,
)

# Frame computation functions
from .frames import (
    compute_glycosidic_frame,
    align_and_compute_transform,
)

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
    "zeros_like",
    "clone",
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
    "rodrigues",
    "rotation_to_axis_angle",
    "axis_angle_to_rotation",
    "compute_relative_transform",
    "apply_relative_transform",
    # Frame computation (new unified API)
    "extract_frame_positions",
    "frame_from_positions",
    "is_purine",
    # Frame computation functions
    "compute_glycosidic_frame",
    "align_and_compute_transform",
    # Geometry projection
    "project_bond_lengths",
    # Geometry constraints
    "GeometryConstraints",
]
