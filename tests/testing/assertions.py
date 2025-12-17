"""Domain-specific assertion functions for ciffy tests.

These assertions encapsulate common validation patterns and provide
clear error messages when assertions fail.
"""

import numpy as np

from .tolerances import get_tolerances


def assert_valid_angles(
    angles,
    name: str = "angles",
    allow_nan: bool = False,
    skip_first: int = 0,
) -> None:
    """Assert angles are in valid range [0, pi].

    Args:
        angles: Array of angles (radians)
        name: Name for error messages
        allow_nan: If True, NaN values are skipped
        skip_first: Skip first N entries (e.g., root atoms without valid angles)
    """
    tol = get_tolerances()
    angles_np = np.asarray(angles)[skip_first:]

    if allow_nan:
        valid_angles = angles_np[~np.isnan(angles_np)]
    else:
        nan_count = np.sum(np.isnan(angles_np))
        assert nan_count == 0, f"{name} contains {nan_count} NaN values"
        valid_angles = angles_np

    if len(valid_angles) == 0:
        return

    min_val = float(valid_angles.min())
    max_val = float(valid_angles.max())

    assert min_val >= -tol.angle_range_epsilon, (
        f"{name} has value {min_val} < 0"
    )
    assert max_val <= np.pi + tol.angle_range_epsilon, (
        f"{name} has value {max_val} > pi"
    )


def assert_valid_dihedrals(
    dihedrals,
    name: str = "dihedrals",
    allow_nan: bool = False,
    skip_first: int = 0,
) -> None:
    """Assert dihedrals are in valid range [-pi, pi].

    Args:
        dihedrals: Array of dihedral angles (radians)
        name: Name for error messages
        allow_nan: If True, NaN values are skipped
        skip_first: Skip first N entries (e.g., root atoms without valid dihedrals)
    """
    tol = get_tolerances()
    dihedrals_np = np.asarray(dihedrals)[skip_first:]

    if allow_nan:
        valid = dihedrals_np[~np.isnan(dihedrals_np)]
    else:
        nan_count = np.sum(np.isnan(dihedrals_np))
        assert nan_count == 0, f"{name} contains {nan_count} NaN values"
        valid = dihedrals_np

    if len(valid) == 0:
        return

    min_val = float(valid.min())
    max_val = float(valid.max())

    assert min_val >= -np.pi - tol.angle_range_epsilon, (
        f"{name} has value {min_val} < -pi"
    )
    assert max_val <= np.pi + tol.angle_range_epsilon, (
        f"{name} has value {max_val} > pi"
    )


def assert_positive_distances(
    distances,
    name: str = "distances",
    skip_first: int = 1,
) -> None:
    """Assert distances are positive.

    Args:
        distances: Array of distances
        name: Name for error messages
        skip_first: Skip first N entries (default 1 for root atom with no distance)
    """
    distances_np = np.asarray(distances)[skip_first:]

    if len(distances_np) == 0:
        return

    non_positive = distances_np[distances_np <= 0]
    assert len(non_positive) == 0, (
        f"{name} has {len(non_positive)} non-positive values after skipping first {skip_first}"
    )


def assert_coordinates_finite(
    coords,
    name: str = "coordinates",
) -> None:
    """Assert all coordinates are finite (no NaN or inf).

    Args:
        coords: Array of coordinates
        name: Name for error messages
    """
    coords_np = np.asarray(coords)

    nan_count = np.sum(np.isnan(coords_np))
    inf_count = np.sum(np.isinf(coords_np))

    assert nan_count == 0, f"{name} contains {nan_count} NaN values"
    assert inf_count == 0, f"{name} contains {inf_count} inf values"


def assert_shapes_equal(
    actual,
    expected: tuple,
    name: str = "array",
) -> None:
    """Assert array shape matches expected shape.

    Args:
        actual: Array to check
        expected: Expected shape tuple
        name: Name for error messages
    """
    actual_shape = tuple(np.asarray(actual).shape)
    assert actual_shape == expected, (
        f"{name} shape mismatch: got {actual_shape}, expected {expected}"
    )


def assert_in_range(
    value: float,
    min_val: float,
    max_val: float,
    name: str = "value",
) -> None:
    """Assert value is within [min_val, max_val].

    Args:
        value: Value to check
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)
        name: Name for error messages
    """
    assert min_val <= value <= max_val, (
        f"{name}={value} not in [{min_val}, {max_val}]"
    )
