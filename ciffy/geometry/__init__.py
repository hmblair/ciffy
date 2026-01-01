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

# Frame computation (fast path with pre-resolved indices)
from .transforms import compute_frame_from_indices

# Frame computation (convenience wrappers)
from .transforms import (
    compute_o3p_frame,
    compute_p_frame,
    compute_c_frame,
    compute_n_frame,
    compute_prev_frame,
    compute_next_frame,
)

# Glycosidic frame (nucleotide-specific)
from .transforms import compute_glycosidic_frame

# Residue type detection
from .transforms import is_purine

# Residue linking
from .transforms import (
    position_residue,
    position_residue_fast,
)

# Frame computation with precomputed indices (fast path)
from .frames import (
    FrameIndices,
    compute_glycosidic_frame_indexed,
    align_to_frame,
    align_and_compute_transform,
    position_next_residue,
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
    # Frame computation (fast path with pre-resolved indices)
    "compute_frame_from_indices",
    # Frame computation (convenience wrappers)
    "compute_o3p_frame",
    "compute_p_frame",
    "compute_c_frame",
    "compute_n_frame",
    "compute_prev_frame",
    "compute_next_frame",
    # Glycosidic frame (nucleotide-specific)
    "compute_glycosidic_frame",
    # Residue type detection
    "is_purine",
    # Residue linking
    "position_residue",
    "position_residue_fast",
    # Frame computation with precomputed indices
    "FrameIndices",
    "compute_glycosidic_frame_indexed",
    "align_to_frame",
    "align_and_compute_transform",
    "position_next_residue",
    # Geometry projection
    "project_bond_lengths",
    # Geometry constraints
    "GeometryConstraints",
]
