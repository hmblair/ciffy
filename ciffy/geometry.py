"""
Backend-agnostic geometric primitives for molecular modeling.

This module provides pure geometric functions that work with both NumPy and PyTorch
backends. All functions are stateless and testable in isolation.

The key functions are:
- Vector operations: cross, dot, norm, normalize
- Rodrigues rotation: rotate points around an axis
- CCD optimization: find optimal rotation angle to minimize distance to target
- Circle-sphere intersection: for ring closure and constraint solving
"""

from __future__ import annotations

import numpy as np

from .backend import Array, is_torch

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
    # Ring closure
    "circle_sphere_intersect",
    "verify_closure_distance",
    # SE(3) transforms
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
]


# =============================================================================
# Backend-Agnostic Vector Operations
# =============================================================================

def cross(a: Array, b: Array) -> Array:
    """
    Cross product of 3D vectors.

    Args:
        a: (3,) first vector
        b: (3,) second vector

    Returns:
        (3,) cross product a × b
    """
    if is_torch(a):
        import torch
        return torch.linalg.cross(a, b)
    return np.cross(a, b)


def dot(a: Array, b: Array) -> float | Array:
    """
    Dot product of vectors.

    Args:
        a: (n,) first vector
        b: (n,) second vector

    Returns:
        Scalar dot product a · b
    """
    if is_torch(a):
        import torch
        return torch.dot(a, b)
    return np.dot(a, b)


def norm(v: Array) -> float | Array:
    """
    L2 norm of a vector.

    Args:
        v: (n,) vector

    Returns:
        Scalar norm |v|
    """
    if is_torch(v):
        import torch
        return torch.linalg.norm(v)
    return np.linalg.norm(v)


def normalize(v: Array, eps: float = 1e-10) -> Array:
    """
    Normalize a vector to unit length.

    Args:
        v: (n,) vector
        eps: Small constant to avoid division by zero

    Returns:
        (n,) unit vector v/|v|
    """
    n = norm(v)
    return v / (n + eps)


# =============================================================================
# Trigonometric Functions
# =============================================================================

def atan2(y: Array, x: Array) -> float | Array:
    """
    Two-argument arctangent.

    Args:
        y: Numerator (y-coordinate)
        x: Denominator (x-coordinate)

    Returns:
        Angle in radians in [-π, π]
    """
    if is_torch(y):
        import torch
        return torch.atan2(y, x)
    return np.arctan2(y, x)


def cos(x: float | Array) -> float | Array:
    """Cosine of angle in radians."""
    if is_torch(x):
        import torch
        return torch.cos(x)
    return np.cos(x)


def sin(x: float | Array) -> float | Array:
    """Sine of angle in radians."""
    if is_torch(x):
        import torch
        return torch.sin(x)
    return np.sin(x)


# =============================================================================
# Array Utilities
# =============================================================================

def zeros_like(shape: tuple, like: Array) -> Array:
    """
    Create zeros array matching backend of reference array.

    Args:
        shape: Shape of output array
        like: Reference array to match dtype/device

    Returns:
        Zeros array with same backend as `like`
    """
    if is_torch(like):
        import torch
        return torch.zeros(shape, dtype=like.dtype, device=like.device)
    return np.zeros(shape, dtype=like.dtype)


def clone(arr: Array) -> Array:
    """
    Clone/copy an array.

    Args:
        arr: Input array

    Returns:
        Deep copy of array
    """
    if is_torch(arr):
        return arr.clone()
    return arr.copy()


def to_scalar(x: Array) -> float:
    """
    Convert 0-d array to Python float.

    Args:
        x: Scalar array or Python number

    Returns:
        Python float
    """
    if is_torch(x):
        return float(x.item())
    if isinstance(x, np.ndarray):
        return float(x)
    return float(x)


# =============================================================================
# Rodrigues Rotation
# =============================================================================

def rodrigues_rotate(
    point: Array,
    axis: Array,
    origin: Array,
    angle: float | Array,
) -> Array:
    """
    Rotate a point around an axis using Rodrigues' rotation formula.

    The rotation follows the right-hand rule: when looking along the axis
    direction, positive angles rotate counterclockwise.

    Args:
        point: (3,) point to rotate
        axis: (3,) normalized rotation axis direction
        origin: (3,) a point on the rotation axis
        angle: Rotation angle in radians

    Returns:
        (3,) rotated point

    Example:
        >>> # Rotate (1, 0, 0) by 90° around z-axis at origin
        >>> p = np.array([1.0, 0.0, 0.0])
        >>> axis = np.array([0.0, 0.0, 1.0])
        >>> origin = np.array([0.0, 0.0, 0.0])
        >>> rotated = rodrigues_rotate(p, axis, origin, np.pi/2)
        >>> # Result is approximately (0, 1, 0)
    """
    # Translate to origin
    p = point - origin

    # Rodrigues formula: R(n,θ)p = p·cos(θ) + (n×p)·sin(θ) + n·(n·p)·(1-cos(θ))
    cos_a = cos(angle)
    sin_a = sin(angle)

    # Decompose p into parallel and perpendicular to axis
    p_dot_axis = dot(axis, p)
    p_parallel = axis * p_dot_axis
    p_perp = p - p_parallel
    p_cross = cross(axis, p)

    # Rotated vector (parallel component unchanged, perp component rotates)
    rotated = p_parallel + p_perp * cos_a + p_cross * sin_a

    return rotated + origin


# =============================================================================
# CCD Optimization
# =============================================================================

def optimal_rotation_to_target(
    moving_point: Array,
    target_point: Array,
    axis_origin: Array,
    axis_direction: Array,
) -> float:
    """
    Find the rotation angle that minimizes distance from moving_point to target_point.

    When rotating `moving_point` around the axis, it traces a circle in a plane
    perpendicular to the axis. This function finds the angle θ that minimizes:

        |rodrigues_rotate(moving_point, axis, origin, θ) - target_point|²

    Mathematical derivation:

    Let r = moving_point - axis_origin, f = target_point - axis_origin.
    Decompose into parallel and perpendicular to axis:
        r = r_para + r_perp, f = f_para + f_perp

    After rotation by θ:
        r'(θ) = r_para + r_perp·cos(θ) + (axis × r_perp)·sin(θ)

    The squared distance is:
        |r'(θ) - f|² = |d|² + R² + |f_perp|² - 2R·(α·cos(θ) + β·sin(θ))

    where:
        - d = r_para - f_para (parallel offset, fixed)
        - R = |r_perp| (radius of rotation circle)
        - α = f_perp · (r_perp/R) (projection of target onto moving direction)
        - β = f_perp · (axis × r_perp/R) (projection onto perpendicular)

    To minimize, we maximize α·cos(θ) + β·sin(θ), which gives θ = atan2(β, α).

    Args:
        moving_point: (3,) the point that will be rotated
        target_point: (3,) the target position
        axis_origin: (3,) a point on the rotation axis
        axis_direction: (3,) normalized direction of rotation axis

    Returns:
        Optimal rotation angle in radians

    Notes:
        - If moving_point is on the axis (degenerate), returns 0.
        - The returned angle minimizes squared distance even when the target
          is not on the rotation circle (finds closest approach).
    """
    # Vectors from axis origin
    r = moving_point - axis_origin
    f = target_point - axis_origin

    # Decompose r into parallel and perpendicular to axis
    r_para = axis_direction * dot(axis_direction, r)
    r_perp = r - r_para

    # Same for f
    f_para = axis_direction * dot(axis_direction, f)
    f_perp = f - f_para

    # Radius of rotation circle
    R = norm(r_perp)
    R_scalar = to_scalar(R)

    # Degenerate case: point is on the axis
    if R_scalar < 1e-10:
        return 0.0

    # Build orthonormal basis in the rotation plane
    r_perp_unit = r_perp / R  # u: direction of moving point
    r_cross_unit = cross(axis_direction, r_perp_unit)  # v: perpendicular in plane

    # Project f_perp onto this basis
    alpha = dot(f_perp, r_perp_unit)  # f_perp · u
    beta = dot(f_perp, r_cross_unit)  # f_perp · v

    # Optimal angle maximizes alpha·cos(θ) + beta·sin(θ)
    # This is equivalent to rotating r_perp_unit toward f_perp
    optimal_angle = atan2(beta, alpha)

    return to_scalar(optimal_angle)


def project_to_rotation_circle(
    point: Array,
    axis_origin: Array,
    axis_direction: Array,
    radius: float,
) -> Array:
    """
    Project a point onto the circle traced by rotating at a given radius.

    This finds the point on the rotation circle that is closest to the input point.
    Useful for understanding CCD behavior.

    Args:
        point: (3,) point to project
        axis_origin: (3,) point on rotation axis
        axis_direction: (3,) normalized axis direction
        radius: Radius of the rotation circle

    Returns:
        (3,) closest point on the rotation circle
    """
    v = point - axis_origin

    # Parallel component (along axis)
    v_para = axis_direction * dot(axis_direction, v)
    # Perpendicular component (in rotation plane)
    v_perp = v - v_para

    v_perp_norm = norm(v_perp)
    if to_scalar(v_perp_norm) < 1e-10:
        # Point is on axis, any point on circle is equidistant
        # Return a canonical point
        if is_torch(point):
            import torch
            perp = torch.tensor([1.0, 0.0, 0.0], dtype=point.dtype, device=point.device)
        else:
            perp = np.array([1.0, 0.0, 0.0], dtype=point.dtype)
        if abs(to_scalar(dot(axis_direction, perp))) > 0.9:
            if is_torch(point):
                perp = torch.tensor([0.0, 1.0, 0.0], dtype=point.dtype, device=point.device)
            else:
                perp = np.array([0.0, 1.0, 0.0], dtype=point.dtype)
        perp = normalize(perp - axis_direction * dot(axis_direction, perp))
        return axis_origin + v_para + perp * radius

    # Normalize and scale to radius
    v_perp_unit = v_perp / v_perp_norm
    return axis_origin + v_para + v_perp_unit * radius


# =============================================================================
# Ring Closure Geometry
# =============================================================================

def circle_sphere_intersect(
    circle_center: Array,
    circle_axis: Array,
    circle_radius: float,
    sphere_center: Array,
    sphere_radius: float,
    tol: float = 1e-10,
) -> list[tuple[Array, float]]:
    """
    Find intersection points of a circle and a sphere.

    The circle lies in a plane perpendicular to circle_axis, centered at
    circle_center with the given radius. Points on the circle are:

        P(θ) = circle_center + circle_radius * (cos(θ)*u + sin(θ)*v)

    where u, v are orthonormal vectors perpendicular to circle_axis.

    The sphere is centered at sphere_center with sphere_radius.

    This is the core geometric primitive for analytical ring closure.
    When closing a ring, the last atom must lie on both:
    - A circle (tracing out as we vary its dihedral angle)
    - A sphere (at fixed distance from the first ring atom)

    Args:
        circle_center: (3,) center of the circle (on the axis)
        circle_axis: (3,) unit vector normal to circle plane
        circle_radius: radius of the circle
        sphere_center: (3,) center of the sphere
        sphere_radius: radius of the sphere
        tol: numerical tolerance for discriminant

    Returns:
        List of (point, angle) tuples where:
        - point: (3,) intersection point
        - angle: the θ parameter on the circle
        Returns empty list if no intersection, 1 element if tangent,
        2 elements if proper intersection.

    Mathematical derivation:
        Substituting the circle parameterization into |P - sphere_center|² = r²:

        |c + R*(cos θ u + sin θ v) - s|² = r²

        where c = circle_center, s = sphere_center, R = circle_radius, r = sphere_radius.

        Let d = c - s. Then:
        |d|² + R² + 2R*(cos θ (d·u) + sin θ (d·v)) = r²

        This is: A*cos θ + B*sin θ = C

        where A = 2R*(d·u), B = 2R*(d·v), C = r² - |d|² - R²

        Solution: θ = atan2(B, A) ± acos(C / sqrt(A² + B²))
    """
    # Vector from sphere center to circle center
    d = circle_center - sphere_center

    # Build orthonormal basis in the circle plane
    # u is an arbitrary unit vector perpendicular to axis
    if is_torch(circle_axis):
        import torch
        if abs(circle_axis[0].item()) < 0.9:
            ref = torch.tensor([1.0, 0.0, 0.0], dtype=circle_axis.dtype, device=circle_axis.device)
        else:
            ref = torch.tensor([0.0, 1.0, 0.0], dtype=circle_axis.dtype, device=circle_axis.device)
    else:
        if abs(circle_axis[0]) < 0.9:
            ref = np.array([1.0, 0.0, 0.0], dtype=circle_axis.dtype)
        else:
            ref = np.array([0.0, 1.0, 0.0], dtype=circle_axis.dtype)

    u = ref - circle_axis * dot(circle_axis, ref)
    u = normalize(u)
    v = cross(circle_axis, u)  # v is perpendicular to both axis and u

    # Compute coefficients A, B, C
    d_dot_u = to_scalar(dot(d, u))
    d_dot_v = to_scalar(dot(d, v))
    d_norm_sq = to_scalar(dot(d, d))

    A = 2.0 * circle_radius * d_dot_u
    B = 2.0 * circle_radius * d_dot_v
    C = sphere_radius**2 - d_norm_sq - circle_radius**2

    # Solve A*cos(θ) + B*sin(θ) = C
    # Rewrite as: sqrt(A² + B²) * cos(θ - φ) = C where φ = atan2(B, A)
    AB_norm = np.sqrt(A**2 + B**2)

    if AB_norm < tol:
        # Degenerate case: circle center is on sphere center
        # Check if circle radius equals sphere radius
        if abs(circle_radius - sphere_radius) < tol:
            # Entire circle is on sphere - return two arbitrary points
            angle1 = 0.0
            angle2 = np.pi
            if is_torch(circle_center):
                import torch
                p1 = circle_center + u * torch.tensor(circle_radius, dtype=u.dtype, device=u.device)
                p2 = circle_center - u * torch.tensor(circle_radius, dtype=u.dtype, device=u.device)
            else:
                p1 = circle_center + u * circle_radius
                p2 = circle_center - u * circle_radius
            return [(p1, angle1), (p2, angle2)]
        else:
            # No intersection
            return []

    cos_val = C / AB_norm

    # Check if intersection exists
    if cos_val < -1.0 - tol or cos_val > 1.0 + tol:
        # No intersection - circle doesn't reach the sphere
        return []

    # Clamp for numerical stability
    cos_val = max(-1.0, min(1.0, cos_val))

    phi = np.arctan2(B, A)
    delta = np.arccos(cos_val)

    # Two solutions (or one if tangent)
    theta1 = phi + delta
    theta2 = phi - delta

    # Wrap to [-π, π]
    while theta1 > np.pi:
        theta1 -= 2 * np.pi
    while theta1 < -np.pi:
        theta1 += 2 * np.pi
    while theta2 > np.pi:
        theta2 -= 2 * np.pi
    while theta2 < -np.pi:
        theta2 += 2 * np.pi

    # Compute the intersection points
    cos_t1, sin_t1 = np.cos(theta1), np.sin(theta1)
    cos_t2, sin_t2 = np.cos(theta2), np.sin(theta2)

    if is_torch(circle_center):
        import torch
        R = torch.tensor(circle_radius, dtype=circle_center.dtype, device=circle_center.device)
        p1 = circle_center + R * (u * cos_t1 + v * sin_t1)
        p2 = circle_center + R * (u * cos_t2 + v * sin_t2)
    else:
        p1 = circle_center + circle_radius * (u * cos_t1 + v * sin_t1)
        p2 = circle_center + circle_radius * (u * cos_t2 + v * sin_t2)

    # If tangent (delta ≈ 0), return only one point
    if abs(delta) < tol:
        return [(p1, theta1)]

    return [(p1, theta1), (p2, theta2)]


def verify_closure_distance(
    coords: Array,
    atom_i: int,
    atom_j: int,
    expected_distance: float,
    tolerance: float = 0.01,
) -> tuple[bool, float]:
    """
    Verify that the closure bond has the expected distance.

    Args:
        coords: (N, 3) atomic coordinates
        atom_i: first atom index of closure bond
        atom_j: second atom index of closure bond
        expected_distance: expected bond length in Angstroms
        tolerance: acceptable error in Angstroms

    Returns:
        (satisfied, error) where:
        - satisfied: True if |actual - expected| < tolerance
        - error: the absolute distance error
    """
    actual = to_scalar(norm(coords[atom_j] - coords[atom_i]))
    error = abs(actual - expected_distance)
    return error < tolerance, error


# =============================================================================
# SE(3) Transform Operations
# =============================================================================


def _stack_columns(x: Array, y: Array, z: Array) -> Array:
    """Stack three vectors as columns of a matrix."""
    if is_torch(x):
        import torch
        return torch.stack([x, y, z], dim=1)
    return np.column_stack([x, y, z])


def _eye3(like: Array) -> Array:
    """3x3 identity matrix matching backend of reference array."""
    if is_torch(like):
        import torch
        return torch.eye(3, dtype=like.dtype, device=like.device)
    return np.eye(3, dtype=like.dtype)


def _trace(R: Array) -> Array:
    """Matrix trace."""
    if is_torch(R):
        import torch
        return torch.trace(R)
    return np.trace(R)


def _acos_safe(x: Array) -> Array:
    """Arccos with safe clamping."""
    if is_torch(x):
        import torch
        return torch.acos(torch.clamp(x, -1.0, 1.0))
    return np.arccos(np.clip(x, -1.0, 1.0))


def rotation_to_axis_angle(R: Array) -> Array:
    """
    Convert rotation matrix to axis-angle representation.

    Uses the Rodrigues formula inverse. Handles edge cases for identity
    and 180-degree rotations.

    Args:
        R: (3, 3) rotation matrix.

    Returns:
        (3,) axis-angle vector where direction is axis and magnitude is angle.
    """
    angle = _acos_safe((_trace(R) - 1) / 2)
    angle_scalar = to_scalar(angle)

    if angle_scalar < 1e-6:
        # Near identity - return zero vector
        if is_torch(R):
            import torch
            return torch.zeros(3, dtype=R.dtype, device=R.device)
        return np.zeros(3, dtype=np.float32)

    if np.pi - angle_scalar < 1e-6:
        # Near 180 degrees - extract axis from R + I
        M = R + _eye3(R)
        if is_torch(R):
            col_norms = M.norm(dim=0)
            k = col_norms.argmax().item()
            axis = M[:, k] / col_norms[k]
        else:
            col_norms = np.linalg.norm(M, axis=0)
            k = np.argmax(col_norms)
            axis = M[:, k] / col_norms[k]
        return axis * angle

    # Standard case
    if is_torch(R):
        import torch
        axis = torch.stack([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ])
        axis = axis / (2 * torch.sin(angle) + 1e-8)
    else:
        axis = np.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ])
        axis = axis / (2 * np.sin(angle) + 1e-8)

    return axis * angle


def axis_angle_to_rotation(axis_angle: Array) -> Array:
    """
    Convert axis-angle to rotation matrix (Rodrigues' formula).

    R = I + sin(θ)K + (1-cos(θ))K²

    where K is the skew-symmetric matrix of the unit axis.

    Args:
        axis_angle: (3,) axis-angle vector (direction is axis, magnitude is angle).

    Returns:
        (3, 3) rotation matrix.
    """
    angle = norm(axis_angle)
    angle_scalar = to_scalar(angle)

    if angle_scalar < 1e-8:
        return _eye3(axis_angle)

    axis = axis_angle / angle

    if is_torch(axis_angle):
        import torch
        K = torch.tensor([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ], dtype=axis_angle.dtype, device=axis_angle.device)
        I = torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device)
        return I + torch.sin(angle) * K + (1 - torch.cos(angle)) * (K @ K)
    else:
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ], dtype=np.float32)
        return np.eye(3, dtype=np.float32) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def compute_relative_transform(
    origin1: Array,
    R1: Array,
    origin2: Array,
    R2: Array,
) -> Array:
    """
    Compute relative SE(3) transform from frame 1 to frame 2.

    The transform encodes how to get from frame 1 to frame 2:
    - Rotation: R_rel = R1.T @ R2
    - Translation: expressed in frame 1's coordinate system

    Args:
        origin1: (3,) position of frame 1.
        R1: (3, 3) rotation matrix of frame 1.
        origin2: (3,) position of frame 2.
        R2: (3, 3) rotation matrix of frame 2.

    Returns:
        (6,) transform vector [axis-angle (3), translation in frame1 coords (3)].
    """
    R_rel = R1.T @ R2
    axis_angle = rotation_to_axis_angle(R_rel)
    t_world = origin2 - origin1
    t_local = R1.T @ t_world

    if is_torch(origin1):
        import torch
        return torch.cat([axis_angle, t_local])
    return np.concatenate([axis_angle, t_local]).astype(np.float32)


def apply_relative_transform(
    origin: Array,
    R: Array,
    transform: Array,
) -> tuple[Array, Array]:
    """
    Apply relative transform to get a new frame from a source frame.

    This is the inverse of compute_relative_transform.

    Args:
        origin: (3,) position of source frame.
        R: (3, 3) rotation matrix of source frame.
        transform: (6,) vector [axis-angle (3), translation in source coords (3)].

    Returns:
        (origin2, R2): Position and rotation of target frame.
    """
    axis_angle = transform[:3]
    t_local = transform[3:]
    R_rel = axis_angle_to_rotation(axis_angle)
    R2 = R @ R_rel
    t_world = R @ t_local
    origin2 = origin + t_world
    return origin2, R2


# =============================================================================
# Residue Frame Computation
# =============================================================================
# These functions compute reference frames at specific atoms for residue linking.
# All frames return (origin, R) where R is a 3x3 rotation matrix with orthonormal
# columns representing the local x, y, z axes.


def _arbitrary_perpendicular(z_axis: Array) -> Array:
    """
    Compute an arbitrary unit vector perpendicular to z_axis.

    Used when no perpendicular reference atom is available (e.g., N frame in proteins).

    Args:
        z_axis: (3,) or (..., 3) normalized vector(s).

    Returns:
        Unit vector(s) perpendicular to z_axis, same shape as input.
    """
    if is_torch(z_axis):
        import torch
        # Choose reference that's not parallel to z_axis
        ref = torch.zeros_like(z_axis)
        ref[..., 0] = 1.0
        # If too parallel, use y-axis instead
        parallel = (dot(z_axis, ref).abs() > 0.9) if z_axis.ndim == 1 else None
        if parallel is not None and parallel:
            ref = torch.zeros_like(z_axis)
            ref[..., 1] = 1.0
    else:
        ref = np.zeros_like(z_axis)
        ref[..., 0] = 1.0
        if abs(dot(z_axis, ref)) > 0.9:
            ref = np.zeros_like(z_axis)
            ref[..., 1] = 1.0

    return normalize(cross(ref, z_axis))


def compute_frame_from_indices(
    coords: Array,
    frame_cols: tuple[int, int, int | None],
    z_toward_origin: bool,
) -> tuple[Array, Array]:
    """
    Compute coordinate frame using pre-resolved column indices.

    This is the fast path for frame computation - pure tensor math with no
    Python attribute lookups. Supports both single residues and batches.

    Args:
        coords: (n_atoms, 3) or (batch, n_atoms, 3) coordinates.
        frame_cols: Tuple of (origin_col, z_ref_col, perp_ref_col) where
            perp_ref_col may be None for frames without a perpendicular reference.
        z_toward_origin: If True, Z points from z_ref toward origin.
            If False, Z points from origin toward z_ref.

    Returns:
        origin: (3,) or (batch, 3) frame origin position.
        R: (3, 3) or (batch, 3, 3) rotation matrix with [x, y, z] as columns.

    Example:
        >>> # Pre-resolve indices once at model init
        >>> frame_cols = link_def.prev_frame.resolve(residue, atom_to_col)
        >>> z_toward_origin = link_def.prev_frame.z_toward_origin
        >>>
        >>> # Fast frame computation at runtime
        >>> origin, R = compute_frame_from_indices(coords, frame_cols, z_toward_origin)
    """
    origin_col, z_ref_col, perp_ref_col = frame_cols

    # Direct indexing - works for (..., n_atoms, 3)
    origin_pos = coords[..., origin_col, :]
    z_ref_pos = coords[..., z_ref_col, :]

    # Compute Z-axis
    if z_toward_origin:
        z_axis = normalize(origin_pos - z_ref_pos)
    else:
        z_axis = normalize(z_ref_pos - origin_pos)

    # Compute X-axis (perpendicular to Z)
    if perp_ref_col is not None:
        perp_pos = coords[..., perp_ref_col, :]
        # Project perp direction onto plane perpendicular to Z
        perp_dir = perp_pos - z_ref_pos
        x_axis = normalize(cross(perp_dir, z_axis))
    else:
        x_axis = _arbitrary_perpendicular(z_axis)

    # Y-axis completes right-handed system
    y_axis = cross(z_axis, x_axis)

    # Stack into rotation matrix
    origin = clone(origin_pos)
    R = _stack_columns(x_axis, y_axis, z_axis)

    return origin, R


def compute_o3p_frame(
    coords: Array,
    atom_to_col: dict[int, int],
    residue: "Residue",
) -> tuple[Array, Array]:
    """
    Compute the O3' frame for a nucleotide residue.

    This frame is used as the outgoing link point for backbone connectivity.

    Frame definition:
    - Origin: O3' atom
    - Z-axis: Along C3'->O3' bond (outward direction)
    - X-axis: Perpendicular, in the C4'-C3'-O3' plane
    - Y-axis: Completes right-handed system

    Args:
        coords: (n_atoms, 3) residue coordinates.
        atom_to_col: Dict mapping atom type value to column index.
        residue: Residue enum (e.g., Residue.A).

    Returns:
        origin: (3,) O3' position.
        R: (3, 3) rotation matrix [x, y, z] as columns.
    """
    c4p = coords[atom_to_col[residue.C4p.value]]
    c3p = coords[atom_to_col[residue.C3p.value]]
    o3p = coords[atom_to_col[residue.O3p.value]]

    origin = clone(o3p)

    z_axis = normalize(o3p - c3p)
    y_temp = c4p - c3p
    x_axis = normalize(cross(y_temp, z_axis))
    y_axis = cross(z_axis, x_axis)

    R = _stack_columns(x_axis, y_axis, z_axis)
    return origin, R


def compute_p_frame(
    coords: Array,
    atom_to_col: dict[int, int],
    residue: "Residue",
) -> tuple[Array, Array]:
    """
    Compute the P frame for a nucleotide residue.

    This frame is used as the incoming link point for backbone connectivity.

    Frame definition:
    - Origin: P atom
    - Z-axis: Along O5'->P bond
    - X-axis: Perpendicular, toward OP1
    - Y-axis: Completes right-handed system

    Args:
        coords: (n_atoms, 3) residue coordinates.
        atom_to_col: Dict mapping atom type value to column index.
        residue: Residue enum (e.g., Residue.A).

    Returns:
        origin: (3,) P position.
        R: (3, 3) rotation matrix [x, y, z] as columns.
    """
    p = coords[atom_to_col[residue.P.value]]
    o5p = coords[atom_to_col[residue.O5p.value]]
    op1 = coords[atom_to_col[residue.OP1.value]]

    origin = clone(p)

    z_axis = normalize(p - o5p)
    y_temp = op1 - p
    x_axis = normalize(cross(y_temp, z_axis))
    y_axis = cross(z_axis, x_axis)

    R = _stack_columns(x_axis, y_axis, z_axis)
    return origin, R


def compute_c_frame(
    coords: Array,
    atom_to_col: dict[int, int],
    residue: "Residue",
) -> tuple[Array, Array]:
    """
    Compute the carbonyl C frame for a protein residue.

    This frame is used as the outgoing link point for peptide bonds.

    Frame definition:
    - Origin: C atom (carbonyl carbon)
    - Z-axis: Along CA->C bond (outward direction)
    - X-axis: Perpendicular, toward O
    - Y-axis: Completes right-handed system

    Args:
        coords: (n_atoms, 3) residue coordinates.
        atom_to_col: Dict mapping atom type value to column index.
        residue: Residue enum (e.g., Residue.ALA).

    Returns:
        origin: (3,) C position.
        R: (3, 3) rotation matrix [x, y, z] as columns.
    """
    ca = coords[atom_to_col[residue.CA.value]]
    c = coords[atom_to_col[residue.C.value]]
    o = coords[atom_to_col[residue.O.value]]

    origin = clone(c)

    z_axis = normalize(c - ca)
    y_temp = o - c
    x_axis = normalize(cross(y_temp, z_axis))
    y_axis = cross(z_axis, x_axis)

    R = _stack_columns(x_axis, y_axis, z_axis)
    return origin, R


def compute_n_frame(
    coords: Array,
    atom_to_col: dict[int, int],
    residue: "Residue",
) -> tuple[Array, Array]:
    """
    Compute the amide N frame for a protein residue.

    This frame is used as the incoming link point for peptide bonds.

    Frame definition:
    - Origin: N atom (amide nitrogen)
    - Z-axis: Along C(prev)->N bond direction
    - X-axis: Perpendicular, toward CA
    - Y-axis: Completes right-handed system

    Note: For the incoming frame, we use CA->N as the Z-axis
    (pointing toward the incoming bond).

    Args:
        coords: (n_atoms, 3) residue coordinates.
        atom_to_col: Dict mapping atom type value to column index.
        residue: Residue enum (e.g., Residue.ALA).

    Returns:
        origin: (3,) N position.
        R: (3, 3) rotation matrix [x, y, z] as columns.
    """
    n = coords[atom_to_col[residue.N.value]]
    ca = coords[atom_to_col[residue.CA.value]]

    origin = clone(n)

    # Z points from N toward CA (incoming direction from previous C)
    z_axis = normalize(ca - n)
    # Use CA for perpendicular direction
    y_temp = ca - n
    # Need a different reference - use the H on N if available, else use CA
    # For simplicity, construct perpendicular from arbitrary vector
    if is_torch(n):
        import torch
        ref = torch.tensor([1.0, 0.0, 0.0], dtype=n.dtype, device=n.device)
        if abs(dot(z_axis, ref)) > 0.9:
            ref = torch.tensor([0.0, 1.0, 0.0], dtype=n.dtype, device=n.device)
    else:
        ref = np.array([1.0, 0.0, 0.0], dtype=n.dtype)
        if abs(dot(z_axis, ref)) > 0.9:
            ref = np.array([0.0, 1.0, 0.0], dtype=n.dtype)

    x_axis = normalize(cross(ref, z_axis))
    y_axis = cross(z_axis, x_axis)

    R = _stack_columns(x_axis, y_axis, z_axis)
    return origin, R


def compute_prev_frame(
    coords: Array,
    atom_to_col: dict[int, int],
    residue: "Residue",
) -> tuple[Array, Array]:
    """
    Compute the outgoing (previous) frame for a residue.

    Uses LinkingDefinition to determine the frame atoms based on molecule type.
    This is the convenience wrapper - for performance-critical code, use
    compute_frame_from_indices() with pre-resolved column indices.

    Args:
        coords: (n_atoms, 3) residue coordinates.
        atom_to_col: Dict mapping atom type value to column index.
        residue: Residue enum.

    Returns:
        origin: (3,) frame origin position.
        R: (3, 3) rotation matrix.

    Raises:
        ValueError: If no linking definition exists for this molecule type.
    """
    from .biochemistry.linking import LINKING_BY_TYPE

    link_def = LINKING_BY_TYPE.get(residue.molecule_type)
    if link_def is None:
        raise ValueError(
            f"No linking definition for molecule type {residue.molecule_type}"
        )

    frame_cols = link_def.prev_frame.resolve(residue, atom_to_col)
    return compute_frame_from_indices(coords, frame_cols, link_def.prev_frame.z_toward_origin)


def compute_next_frame(
    coords: Array,
    atom_to_col: dict[int, int],
    residue: "Residue",
) -> tuple[Array, Array]:
    """
    Compute the incoming (next) frame for a residue.

    Uses LinkingDefinition to determine the frame atoms based on molecule type.
    This is the convenience wrapper - for performance-critical code, use
    compute_frame_from_indices() with pre-resolved column indices.

    Args:
        coords: (n_atoms, 3) residue coordinates.
        atom_to_col: Dict mapping atom type value to column index.
        residue: Residue enum.

    Returns:
        origin: (3,) frame origin position.
        R: (3, 3) rotation matrix.

    Raises:
        ValueError: If no linking definition exists for this molecule type.
    """
    from .biochemistry.linking import LINKING_BY_TYPE

    link_def = LINKING_BY_TYPE.get(residue.molecule_type)
    if link_def is None:
        raise ValueError(
            f"No linking definition for molecule type {residue.molecule_type}"
        )

    frame_cols = link_def.next_frame.resolve(residue, atom_to_col)
    return compute_frame_from_indices(coords, frame_cols, link_def.next_frame.z_toward_origin)


# =============================================================================
# Residue Positioning
# =============================================================================


def position_residue(
    prev_coords: Array,
    next_coords: Array,
    prev_atom_to_col: dict[int, int],
    next_atom_to_col: dict[int, int],
    prev_residue: "Residue",
    next_residue: "Residue",
    transform: Array | None = None,
) -> Array:
    """
    Position a residue relative to the previous residue.

    This function places next_coords so that the incoming link point of the
    next residue aligns with the outgoing link point of the previous residue.

    Works with both NumPy and PyTorch arrays (auto-detected from input).

    Args:
        prev_coords: (n_atoms, 3) positioned coordinates of previous residue.
        next_coords: (n_atoms, 3) coordinates of residue to position.
        prev_atom_to_col: Dict mapping atom type value to column index for prev residue.
        next_atom_to_col: Dict mapping atom type value to column index for next residue.
        prev_residue: Residue enum for previous residue.
        next_residue: Residue enum for next residue.
        transform: Optional (6,) SE(3) transform [axis-angle, translation].
            If None, uses linear extension along the Z-axis with standard bond length.
            If provided, applies the learned transform from flow models.

    Returns:
        (n_atoms, 3) positioned coordinates of the next residue.

    Example (linear extension for templates):
        >>> positioned = position_residue(
        ...     prev_coords, next_coords,
        ...     prev_atom_to_col, next_atom_to_col,
        ...     Residue.A, Residue.C,
        ...     transform=None,  # Linear extension
        ... )

    Example (SE(3) transform for flow models):
        >>> positioned = position_residue(
        ...     prev_coords, next_coords,
        ...     prev_atom_to_col, next_atom_to_col,
        ...     Residue.A, Residue.C,
        ...     transform=learned_transform,  # From flow model
        ... )
    """
    from .biochemistry.linking import LINKING_BY_TYPE

    # Compute outgoing frame from previous residue
    prev_origin, prev_R = compute_prev_frame(
        prev_coords, prev_atom_to_col, prev_residue
    )

    if transform is None:
        # Linear extension: extend along frame's Z-axis with standard bond length
        link_def = LINKING_BY_TYPE[prev_residue.molecule_type]
        target_origin = prev_origin + prev_R[:, 2] * link_def.bond_length
        target_R = prev_R  # Same orientation
    else:
        # SE(3) transform: apply learned transform
        target_origin, target_R = apply_relative_transform(
            prev_origin, prev_R, transform
        )

    # Compute incoming frame from next residue (current position)
    next_origin, next_R = compute_next_frame(
        next_coords, next_atom_to_col, next_residue
    )

    # Compute rigid transformation to align next frame to target frame
    R_correction = target_R @ next_R.T
    t_correction = target_origin - R_correction @ next_origin

    # Apply transformation
    return (R_correction @ next_coords.T).T + t_correction


# =============================================================================
# Residue Type Detection
# =============================================================================


def is_purine(residue: "Residue") -> bool:
    """
    Check if a residue is a purine (has N9 atom).

    Purines (A, G, DA, DG) have an N9 atom connecting the base to the sugar.
    Pyrimidines (C, U, DC, DT) have an N1 atom instead.

    Args:
        residue: Residue type to check.

    Returns:
        True if purine (has N9), False if pyrimidine (has N1).
    """
    return hasattr(residue, 'N9')


# =============================================================================
# Glycosidic Frame (for nucleotide bases)
# =============================================================================


def compute_glycosidic_frame(
    coords: Array,
    atom_to_col: dict[int, int],
    residue: "Residue",
) -> tuple[Array, Array]:
    """
    Compute the glycosidic frame for a nucleotide residue.

    Frame definition:
    - Origin: C1' atom
    - X-axis: Toward N9 (purines) or N1 (pyrimidines)
    - Z-axis: Normal to the C1'-N9-C4 plane
    - Y-axis: Completes right-handed system

    Args:
        coords: (n_atoms, 3) coordinates (numpy or torch).
        atom_to_col: Dict mapping atom type value to column index.
        residue: Residue type.

    Returns:
        origin: (3,) C1' position.
        R: (3, 3) rotation matrix [x, y, z] as columns.
    """
    c1p_idx = atom_to_col[residue.C1p.value]
    c4_idx = atom_to_col[residue.C4.value]

    # N9 for purines (A, G), N1 for pyrimidines (C, U)
    if is_purine(residue):
        n_idx = atom_to_col[residue.N9.value]
    else:
        n_idx = atom_to_col[residue.N1.value]

    origin = clone(coords[c1p_idx])
    n_pos = coords[n_idx]
    c4_pos = coords[c4_idx]

    x_axis = normalize(n_pos - origin)
    y_temp = c4_pos - origin
    z_axis = normalize(cross(x_axis, y_temp))
    y_axis = cross(z_axis, x_axis)

    # Build rotation matrix with columns [x, y, z]
    if is_torch(coords):
        import torch
        R = torch.stack([x_axis, y_axis, z_axis], dim=1)
    else:
        import numpy as np
        R = np.column_stack([x_axis, y_axis, z_axis]).astype(np.float32)
        origin = origin.astype(np.float32)

    return origin, R


# =============================================================================
# Fast Residue Positioning (pre-resolved indices)
# =============================================================================


def position_residue_fast(
    prev_coords: Array,
    next_coords: Array,
    transform: Array,
    prev_frame_cols: tuple[int, int, int | None],
    prev_z_toward_origin: bool,
    next_frame_cols: tuple[int, int, int | None],
    next_z_toward_origin: bool,
) -> Array:
    """
    Position residue 2 relative to residue 1 using pre-resolved frame indices.

    This is the fast path for residue positioning. Uses pre-resolved column
    indices to compute frames with pure tensor math (no Python attribute lookups).
    The frame indices should be computed once at model initialization.

    Works with both NumPy and PyTorch arrays (auto-detected from input).

    Args:
        prev_coords: (n_atoms, 3) coordinates of previous residue.
        next_coords: (n_atoms, 3) coordinates of next residue (in canonical frame).
        transform: (6,) SE(3) transform [axis-angle, translation].
        prev_frame_cols: Pre-resolved (origin, z_ref, perp_ref) column indices for
            outgoing frame of prev_coords (e.g., O3' frame for RNA).
        prev_z_toward_origin: Z-axis direction for prev frame.
        next_frame_cols: Pre-resolved column indices for incoming frame of next_coords
            (e.g., P frame for RNA).
        next_z_toward_origin: Z-axis direction for next frame.

    Returns:
        (n_atoms, 3) positioned coordinates of next residue.

    Example:
        >>> # Pre-resolve indices at model init
        >>> link_def = LINKING_BY_TYPE[residue.molecule_type]
        >>> prev_cols = link_def.prev_frame.resolve(residue, atom_to_col)
        >>> next_cols = link_def.next_frame.resolve(residue, atom_to_col)
        >>>
        >>> # Fast positioning at runtime
        >>> positioned = position_residue_fast(
        ...     prev_coords, next_coords, transform,
        ...     prev_cols, link_def.prev_frame.z_toward_origin,
        ...     next_cols, link_def.next_frame.z_toward_origin,
        ... )
    """
    # Compute outgoing frame from prev_coords using pre-resolved indices
    prev_origin, prev_R = compute_frame_from_indices(
        prev_coords, prev_frame_cols, prev_z_toward_origin
    )

    # Apply transform to get target incoming frame
    target_origin, target_R = apply_relative_transform(
        prev_origin, prev_R, transform
    )

    # Compute current incoming frame from next_coords using pre-resolved indices
    current_origin, current_R = compute_frame_from_indices(
        next_coords, next_frame_cols, next_z_toward_origin
    )

    # Compute rigid transformation to align current frame to target frame
    R_correction = target_R @ current_R.T
    t_correction = target_origin - R_correction @ current_origin

    # Apply transformation
    return (R_correction @ next_coords.T).T + t_correction
