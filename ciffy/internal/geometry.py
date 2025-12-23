"""
Backend-agnostic geometric primitives for internal coordinate operations.

This module provides pure geometric functions that work with both NumPy and PyTorch
backends. All functions are stateless and testable in isolation.

The key functions are:
- Vector operations: cross, dot, norm, normalize
- Rodrigues rotation: rotate points around an axis
- CCD optimization: find optimal rotation angle to minimize distance to target

These primitives are used by the ring closure solver and other internal
coordinate operations.
"""

from __future__ import annotations

import numpy as np

from ..backend import Array, is_torch

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
