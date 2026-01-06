"""
Unit tests for ciffy.geometry primitives.

These tests verify the correctness of geometric operations in isolation.
"""

import numpy as np
import pytest

from ciffy.geometry import (
    cross,
    dot,
    norm,
    normalize,
    atan2,
    cos,
    sin,
    to_scalar,
    rodrigues_rotate,
    optimal_rotation_to_target,
    project_to_rotation_circle,
)
from ciffy.backend import clone


class TestVectorOperations:
    """Tests for basic vector operations."""

    def test_cross_product_orthogonal(self):
        """Cross product of orthogonal unit vectors."""
        x = np.array([1.0, 0.0, 0.0])
        y = np.array([0.0, 1.0, 0.0])
        z = np.array([0.0, 0.0, 1.0])

        assert np.allclose(cross(x, y), z)
        assert np.allclose(cross(y, z), x)
        assert np.allclose(cross(z, x), y)

    def test_cross_product_anticommutative(self):
        """Cross product is anticommutative: a × b = -(b × a)."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])

        assert np.allclose(cross(a, b), -cross(b, a))

    def test_dot_product_orthogonal(self):
        """Dot product of orthogonal vectors is zero."""
        x = np.array([1.0, 0.0, 0.0])
        y = np.array([0.0, 1.0, 0.0])

        assert np.isclose(dot(x, y), 0.0)

    def test_dot_product_parallel(self):
        """Dot product of parallel unit vectors is ±1."""
        v = np.array([1.0, 0.0, 0.0])

        assert np.isclose(dot(v, v), 1.0)
        assert np.isclose(dot(v, -v), -1.0)

    def test_norm_unit_vectors(self):
        """Norm of unit vectors is 1."""
        for vec in [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        ]:
            assert np.isclose(norm(vec), 1.0)

    def test_norm_scaled_vector(self):
        """Norm scales linearly."""
        v = np.array([1.0, 0.0, 0.0])
        assert np.isclose(norm(3.0 * v), 3.0)

    def test_normalize(self):
        """Normalize produces unit vector."""
        v = np.array([3.0, 4.0, 0.0])
        v_unit = normalize(v)

        assert np.isclose(norm(v_unit), 1.0)
        assert np.allclose(v_unit, np.array([0.6, 0.8, 0.0]))

    def test_normalize_already_unit(self):
        """Normalizing unit vector returns same vector."""
        v = np.array([1.0, 0.0, 0.0])
        assert np.allclose(normalize(v), v)


class TestRodriguesRotation:
    """Tests for Rodrigues rotation formula."""

    def test_rotate_90_degrees_around_z(self):
        """Rotate (1,0,0) by 90° around z-axis gives (0,1,0)."""
        point = np.array([1.0, 0.0, 0.0])
        axis = np.array([0.0, 0.0, 1.0])
        origin = np.array([0.0, 0.0, 0.0])

        rotated = rodrigues_rotate(point, axis, origin, np.pi / 2)

        assert np.allclose(rotated, np.array([0.0, 1.0, 0.0]), atol=1e-10)

    def test_rotate_180_degrees(self):
        """Rotate by 180° inverts the perpendicular component."""
        point = np.array([1.0, 0.0, 0.0])
        axis = np.array([0.0, 0.0, 1.0])
        origin = np.array([0.0, 0.0, 0.0])

        rotated = rodrigues_rotate(point, axis, origin, np.pi)

        assert np.allclose(rotated, np.array([-1.0, 0.0, 0.0]), atol=1e-10)

    def test_rotate_360_degrees_identity(self):
        """Rotate by 360° returns original point."""
        point = np.array([1.0, 2.0, 3.0])
        axis = normalize(np.array([1.0, 1.0, 1.0]))
        origin = np.array([0.0, 0.0, 0.0])

        rotated = rodrigues_rotate(point, axis, origin, 2 * np.pi)

        assert np.allclose(rotated, point, atol=1e-10)

    def test_rotate_zero_degrees_identity(self):
        """Rotate by 0° returns original point."""
        point = np.array([1.0, 2.0, 3.0])
        axis = np.array([0.0, 0.0, 1.0])
        origin = np.array([0.5, 0.5, 0.0])

        rotated = rodrigues_rotate(point, axis, origin, 0.0)

        assert np.allclose(rotated, point, atol=1e-10)

    def test_rotate_point_on_axis(self):
        """Point on axis doesn't move."""
        point = np.array([0.0, 0.0, 5.0])
        axis = np.array([0.0, 0.0, 1.0])
        origin = np.array([0.0, 0.0, 0.0])

        rotated = rodrigues_rotate(point, axis, origin, np.pi / 3)

        assert np.allclose(rotated, point, atol=1e-10)

    def test_rotate_preserves_distance_from_axis(self):
        """Rotation preserves distance from axis."""
        point = np.array([3.0, 4.0, 5.0])
        axis = np.array([0.0, 0.0, 1.0])
        origin = np.array([1.0, 1.0, 0.0])

        # Distance from axis before
        p_rel = point - origin
        p_perp = p_rel - axis * dot(axis, p_rel)
        dist_before = norm(p_perp)

        # Rotate by arbitrary angle
        rotated = rodrigues_rotate(point, axis, origin, 1.23)

        # Distance from axis after
        r_rel = rotated - origin
        r_perp = r_rel - axis * dot(axis, r_rel)
        dist_after = norm(r_perp)

        assert np.isclose(dist_before, dist_after, atol=1e-10)

    def test_rotate_preserves_parallel_component(self):
        """Rotation preserves component parallel to axis."""
        point = np.array([3.0, 4.0, 5.0])
        axis = np.array([0.0, 0.0, 1.0])
        origin = np.array([1.0, 1.0, 0.0])

        # Parallel component before
        p_rel = point - origin
        para_before = dot(axis, p_rel)

        # Rotate
        rotated = rodrigues_rotate(point, axis, origin, 0.789)

        # Parallel component after
        r_rel = rotated - origin
        para_after = dot(axis, r_rel)

        assert np.isclose(para_before, para_after, atol=1e-10)

    def test_rotate_with_nonzero_origin(self):
        """Rotation around offset axis."""
        # Point at (2, 0, 0), rotate 90° around z-axis through (1, 0, 0)
        point = np.array([2.0, 0.0, 0.0])
        axis = np.array([0.0, 0.0, 1.0])
        origin = np.array([1.0, 0.0, 0.0])

        rotated = rodrigues_rotate(point, axis, origin, np.pi / 2)

        # The point is at distance 1 from origin, should rotate to (1, 1, 0)
        assert np.allclose(rotated, np.array([1.0, 1.0, 0.0]), atol=1e-10)


class TestOptimalRotation:
    """Tests for optimal_rotation_to_target."""

    def test_target_on_rotation_circle(self):
        """When target is on the rotation circle, optimal angle brings point exactly to target."""
        # Point at (1, 0, 0), target at (0, 1, 0), both on unit circle around z-axis
        point = np.array([1.0, 0.0, 0.0])
        target = np.array([0.0, 1.0, 0.0])
        axis = np.array([0.0, 0.0, 1.0])
        origin = np.array([0.0, 0.0, 0.0])

        angle = optimal_rotation_to_target(point, target, origin, axis)

        # Optimal angle should be 90°
        assert np.isclose(angle, np.pi / 2, atol=1e-10)

        # Verify rotation reaches target
        rotated = rodrigues_rotate(point, axis, origin, angle)
        assert np.allclose(rotated, target, atol=1e-10)

    def test_target_not_on_circle(self):
        """When target is not on circle, find closest approach."""
        # Point at (1, 0, 0), target at (0, 2, 0) - target is farther from axis
        point = np.array([1.0, 0.0, 0.0])
        target = np.array([0.0, 2.0, 0.0])
        axis = np.array([0.0, 0.0, 1.0])
        origin = np.array([0.0, 0.0, 0.0])

        angle = optimal_rotation_to_target(point, target, origin, axis)

        # Optimal should still be 90° (closest approach is at (0, 1, 0))
        assert np.isclose(angle, np.pi / 2, atol=1e-10)

        # Verify this is closest approach
        rotated = rodrigues_rotate(point, axis, origin, angle)
        dist_optimal = norm(rotated - target)

        # Check nearby angles give larger distance
        for delta in [-0.1, 0.1]:
            rotated_nearby = rodrigues_rotate(point, axis, origin, angle + delta)
            dist_nearby = norm(rotated_nearby - target)
            assert dist_optimal <= dist_nearby + 1e-10

    def test_target_on_axis(self):
        """When target is on the axis, all rotations are equidistant."""
        point = np.array([1.0, 0.0, 0.0])
        target = np.array([0.0, 0.0, 5.0])  # On z-axis
        axis = np.array([0.0, 0.0, 1.0])
        origin = np.array([0.0, 0.0, 0.0])

        angle = optimal_rotation_to_target(point, target, origin, axis)

        # Any angle gives same distance (since target is on axis)
        dist_at_optimal = norm(rodrigues_rotate(point, axis, origin, angle) - target)
        dist_at_zero = norm(point - target)
        dist_at_pi = norm(rodrigues_rotate(point, axis, origin, np.pi) - target)

        assert np.isclose(dist_at_optimal, dist_at_zero, atol=1e-10)
        assert np.isclose(dist_at_optimal, dist_at_pi, atol=1e-10)

    def test_point_on_axis_degenerate(self):
        """When point is on axis, returns 0 (degenerate case)."""
        point = np.array([0.0, 0.0, 1.0])  # On z-axis
        target = np.array([1.0, 0.0, 0.0])
        axis = np.array([0.0, 0.0, 1.0])
        origin = np.array([0.0, 0.0, 0.0])

        angle = optimal_rotation_to_target(point, target, origin, axis)

        assert angle == 0.0  # Degenerate case

    def test_180_degree_case(self):
        """Target directly opposite should give 180° rotation."""
        point = np.array([1.0, 0.0, 0.0])
        target = np.array([-1.0, 0.0, 0.0])
        axis = np.array([0.0, 0.0, 1.0])
        origin = np.array([0.0, 0.0, 0.0])

        angle = optimal_rotation_to_target(point, target, origin, axis)

        assert np.isclose(abs(angle), np.pi, atol=1e-10)

    def test_target_with_parallel_offset(self):
        """Target at different height along axis."""
        # Point at (1, 0, 0), target at (0, 1, 5) - same circle position but z=5
        point = np.array([1.0, 0.0, 0.0])
        target = np.array([0.0, 1.0, 5.0])
        axis = np.array([0.0, 0.0, 1.0])
        origin = np.array([0.0, 0.0, 0.0])

        angle = optimal_rotation_to_target(point, target, origin, axis)

        # Should still rotate 90° to align perpendicular components
        assert np.isclose(angle, np.pi / 2, atol=1e-10)

    def test_minimizes_distance(self):
        """Verify returned angle actually minimizes distance."""
        point = np.array([1.0, 0.5, 0.0])
        target = np.array([0.3, 1.5, 0.2])
        axis = normalize(np.array([0.1, 0.2, 1.0]))
        origin = np.array([0.1, 0.1, 0.0])

        angle = optimal_rotation_to_target(point, target, origin, axis)

        # Distance at optimal angle
        rotated = rodrigues_rotate(point, axis, origin, angle)
        dist_optimal = norm(rotated - target)

        # Test that nearby angles give larger (or equal) distance
        for delta in np.linspace(-0.5, 0.5, 21):
            if abs(delta) < 1e-6:
                continue
            rotated_test = rodrigues_rotate(point, axis, origin, angle + delta)
            dist_test = norm(rotated_test - target)
            assert dist_optimal <= dist_test + 1e-8, (
                f"angle={angle:.4f}, delta={delta:.4f}, "
                f"dist_optimal={dist_optimal:.6f}, dist_test={dist_test:.6f}"
            )


class TestProjectToRotationCircle:
    """Tests for project_to_rotation_circle."""

    def test_point_on_circle_unchanged(self):
        """Point already on circle projects to itself."""
        origin = np.array([0.0, 0.0, 0.0])
        axis = np.array([0.0, 0.0, 1.0])
        radius = 2.0

        point = np.array([2.0, 0.0, 0.0])  # On circle
        projected = project_to_rotation_circle(point, origin, axis, radius)

        assert np.allclose(projected, point, atol=1e-10)

    def test_point_outside_circle(self):
        """Point outside circle projects to nearest point on circle."""
        origin = np.array([0.0, 0.0, 0.0])
        axis = np.array([0.0, 0.0, 1.0])
        radius = 1.0

        point = np.array([3.0, 0.0, 0.0])  # Outside circle
        projected = project_to_rotation_circle(point, origin, axis, radius)

        assert np.allclose(projected, np.array([1.0, 0.0, 0.0]), atol=1e-10)

    def test_point_inside_circle(self):
        """Point inside circle projects outward to circle."""
        origin = np.array([0.0, 0.0, 0.0])
        axis = np.array([0.0, 0.0, 1.0])
        radius = 2.0

        point = np.array([0.5, 0.0, 0.0])  # Inside circle
        projected = project_to_rotation_circle(point, origin, axis, radius)

        assert np.allclose(projected, np.array([2.0, 0.0, 0.0]), atol=1e-10)

    def test_preserves_parallel_component(self):
        """Projection preserves height along axis."""
        origin = np.array([0.0, 0.0, 0.0])
        axis = np.array([0.0, 0.0, 1.0])
        radius = 1.0

        point = np.array([2.0, 0.0, 5.0])
        projected = project_to_rotation_circle(point, origin, axis, radius)

        assert np.isclose(projected[2], 5.0, atol=1e-10)  # z preserved


class TestTorchCompatibility:
    """Tests for PyTorch backend compatibility."""

    @pytest.fixture
    def torch_available(self):
        """Check if PyTorch is available."""
        try:
            import torch
            return True
        except ImportError:
            pytest.skip("PyTorch not available")

    def test_rodrigues_rotate_torch(self, torch_available):
        """Rodrigues rotation with torch tensors."""
        import torch

        point = torch.tensor([1.0, 0.0, 0.0])
        axis = torch.tensor([0.0, 0.0, 1.0])
        origin = torch.tensor([0.0, 0.0, 0.0])

        rotated = rodrigues_rotate(point, axis, origin, np.pi / 2)

        assert torch.allclose(rotated, torch.tensor([0.0, 1.0, 0.0]), atol=1e-6)

    def test_optimal_rotation_torch(self, torch_available):
        """Optimal rotation with torch tensors."""
        import torch

        point = torch.tensor([1.0, 0.0, 0.0])
        target = torch.tensor([0.0, 1.0, 0.0])
        axis = torch.tensor([0.0, 0.0, 1.0])
        origin = torch.tensor([0.0, 0.0, 0.0])

        angle = optimal_rotation_to_target(point, target, origin, axis)

        assert np.isclose(angle, np.pi / 2, atol=1e-6)

    def test_clone_torch(self, torch_available):
        """Clone creates independent copy for torch."""
        import torch

        arr = torch.tensor([1.0, 2.0, 3.0])
        arr_copy = clone(arr)

        arr_copy[0] = 999.0
        assert arr[0] == 1.0  # Original unchanged

    def test_roundtrip_numpy_torch(self, torch_available):
        """Same results with numpy and torch."""
        import torch

        # Numpy version
        point_np = np.array([1.0, 2.0, 3.0])
        axis_np = normalize(np.array([1.0, 1.0, 0.0]))
        origin_np = np.array([0.5, 0.5, 0.0])
        angle = 0.7

        rotated_np = rodrigues_rotate(point_np, axis_np, origin_np, angle)

        # Torch version
        point_t = torch.tensor(point_np)
        axis_t = torch.tensor(axis_np)
        origin_t = torch.tensor(origin_np)

        rotated_t = rodrigues_rotate(point_t, axis_t, origin_t, angle)

        assert np.allclose(rotated_np, rotated_t.numpy(), atol=1e-6)
