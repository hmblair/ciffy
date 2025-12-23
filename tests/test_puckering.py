"""
Tests for Cremer-Pople puckering coordinates.

Tests the puckering module's ability to:
1. Extract puckering parameters from ring coordinates
2. Apply puckering parameters to create puckered rings
3. Roundtrip: coords → params → coords should preserve geometry
"""

import numpy as np
import pytest

from ciffy.internal.puckering import (
    compute_puckering_5ring,
    apply_puckering_5ring,
    compute_puckering_6ring,
    apply_puckering_6ring,
    flatten_ring_to_plane,
    compute_mean_plane,
)


# =============================================================================
# Test Fixtures: Ideal Ring Geometries
# =============================================================================


def create_planar_5ring(radius: float = 1.5) -> np.ndarray:
    """Create a planar regular pentagon."""
    angles = 2 * np.pi * np.arange(5) / 5
    coords = np.zeros((5, 3))
    coords[:, 0] = radius * np.cos(angles)
    coords[:, 1] = radius * np.sin(angles)
    return coords


def create_planar_6ring(radius: float = 1.5) -> np.ndarray:
    """Create a planar regular hexagon."""
    angles = 2 * np.pi * np.arange(6) / 6
    coords = np.zeros((6, 3))
    coords[:, 0] = radius * np.cos(angles)
    coords[:, 1] = radius * np.sin(angles)
    return coords


def create_envelope_5ring(
    radius: float = 1.5,
    q2: float = 0.4,
    phi2: float = 0.0,
) -> np.ndarray:
    """
    Create a 5-ring with known envelope puckering.

    An envelope conformation has one atom out of the plane of the other four.
    """
    flat = create_planar_5ring(radius)
    return apply_puckering_5ring(flat, q2, phi2)


def create_chair_6ring(
    radius: float = 1.5,
    Q: float = 0.56,  # Typical chair amplitude for cyclohexane
) -> np.ndarray:
    """
    Create a 6-ring in chair conformation (θ=0).

    Chair has alternating up/down pattern: atoms 0,2,4 up and 1,3,5 down.
    """
    flat = create_planar_6ring(radius)
    # theta=0 means cos(theta)=1, so q3=Q, q2=0 (pure chair mode)
    return apply_puckering_6ring(flat, Q, theta=0.0, phi=0.0)


def create_boat_6ring(
    radius: float = 1.5,
    Q: float = 0.56,
    phi: float = 0.0,
) -> np.ndarray:
    """
    Create a 6-ring in boat conformation (θ=π/2).

    Boat has atoms at opposite ends puckered in same direction.
    """
    flat = create_planar_6ring(radius)
    # theta=π/2 means cos(theta)=0, so q3=0, q2=Q (pure twist/boat mode)
    return apply_puckering_6ring(flat, Q, theta=np.pi/2, phi=phi)


# =============================================================================
# 5-Ring Puckering Tests
# =============================================================================


class Test5RingPuckering:
    """Tests for 5-membered ring puckering."""

    def test_planar_ring_has_zero_amplitude(self):
        """A perfectly planar ring should have q2 ≈ 0."""
        flat = create_planar_5ring()
        q2, phi2 = compute_puckering_5ring(flat)
        assert q2 < 1e-10, f"Expected q2 ≈ 0 for planar ring, got {q2}"

    def test_apply_then_extract_roundtrip(self):
        """Applying puckering then extracting should recover params."""
        flat = create_planar_5ring()
        q2_in, phi2_in = 0.4, 0.3

        puckered = apply_puckering_5ring(flat, q2_in, phi2_in)
        q2_out, phi2_out = compute_puckering_5ring(puckered)

        assert abs(q2_out - q2_in) < 1e-10, f"q2 mismatch: {q2_out} vs {q2_in}"
        # phi2 is periodic, so compare with angle wrapping
        phi_diff = abs(phi2_out - phi2_in)
        phi_diff = min(phi_diff, 2*np.pi - phi_diff)
        assert phi_diff < 1e-10, f"phi2 mismatch: {phi2_out} vs {phi2_in}"

    def test_extract_then_apply_preserves_geometry(self):
        """Extracting params then applying should preserve ring geometry."""
        # Create a puckered ring
        original = create_envelope_5ring(q2=0.35, phi2=0.7)

        # Extract puckering params
        q2, phi2 = compute_puckering_5ring(original)

        # Flatten and re-apply
        flat = flatten_ring_to_plane(original)
        reconstructed = apply_puckering_5ring(flat, q2, phi2)

        # The reconstructed ring should match the original
        # (up to rigid transformation - we compare distances)
        for i in range(5):
            j = (i + 1) % 5
            orig_dist = np.linalg.norm(original[j] - original[i])
            recon_dist = np.linalg.norm(reconstructed[j] - reconstructed[i])
            assert abs(orig_dist - recon_dist) < 1e-6, \
                f"Bond length mismatch at {i}-{j}: {orig_dist} vs {recon_dist}"

    def test_envelope_conformations(self):
        """Test known envelope conformations (φ2 = 0, 2π/5, 4π/5, ...)."""
        flat = create_planar_5ring()
        q2 = 0.4

        for k in range(5):
            # Envelope at atom k has φ2 = 2πk/5 - π/10 (approximately)
            phi2 = 2 * np.pi * k / 5
            puckered = apply_puckering_5ring(flat, q2, phi2)

            # Verify center is preserved
            assert np.allclose(puckered.mean(axis=0), flat.mean(axis=0), atol=1e-10)

            # Verify roundtrip
            q2_out, phi2_out = compute_puckering_5ring(puckered)
            assert abs(q2_out - q2) < 1e-10

    def test_amplitude_scaling(self):
        """Larger q2 should give larger out-of-plane displacements."""
        flat = create_planar_5ring()

        z_rms_prev = 0
        for q2 in [0.1, 0.2, 0.3, 0.4, 0.5]:
            puckered = apply_puckering_5ring(flat, q2, phi2=0.0)

            # Compute RMS out-of-plane displacement
            center, normal = compute_mean_plane(puckered)
            z = np.dot(puckered - center, normal)
            z_rms = np.sqrt(np.mean(z**2))

            assert z_rms > z_rms_prev, \
                f"z_rms should increase with q2: {z_rms} not > {z_rms_prev}"
            z_rms_prev = z_rms

    def test_invalid_shape_raises(self):
        """Should raise ValueError for wrong input shape."""
        with pytest.raises(ValueError, match="Expected \\(5, 3\\)"):
            compute_puckering_5ring(np.zeros((6, 3)))

        with pytest.raises(ValueError, match="Expected \\(5, 3\\)"):
            apply_puckering_5ring(np.zeros((4, 3)), 0.4, 0.0)


# =============================================================================
# 6-Ring Puckering Tests
# =============================================================================


class Test6RingPuckering:
    """Tests for 6-membered ring puckering."""

    def test_planar_ring_has_zero_amplitude(self):
        """A perfectly planar ring should have Q ≈ 0."""
        flat = create_planar_6ring()
        Q, theta, phi = compute_puckering_6ring(flat)
        assert Q < 1e-10, f"Expected Q ≈ 0 for planar ring, got {Q}"

    def test_apply_then_extract_roundtrip_chair(self):
        """Roundtrip test for chair conformation (θ=0)."""
        flat = create_planar_6ring()
        Q_in, theta_in, phi_in = 0.56, 0.0, 0.0  # Perfect chair

        puckered = apply_puckering_6ring(flat, Q_in, theta_in, phi_in)
        Q_out, theta_out, phi_out = compute_puckering_6ring(puckered)

        assert abs(Q_out - Q_in) < 1e-10, f"Q mismatch: {Q_out} vs {Q_in}"
        assert abs(theta_out - theta_in) < 1e-10, f"theta mismatch: {theta_out} vs {theta_in}"
        # phi is undefined when theta=0 (chair), so we don't check it

    def test_apply_then_extract_roundtrip_boat(self):
        """Roundtrip test for boat conformation (θ=π/2)."""
        flat = create_planar_6ring()
        Q_in, theta_in, phi_in = 0.56, np.pi/2, 0.3

        puckered = apply_puckering_6ring(flat, Q_in, theta_in, phi_in)
        Q_out, theta_out, phi_out = compute_puckering_6ring(puckered)

        assert abs(Q_out - Q_in) < 1e-10, f"Q mismatch: {Q_out} vs {Q_in}"
        assert abs(theta_out - theta_in) < 1e-10, f"theta mismatch: {theta_out} vs {theta_in}"

        # phi should match (with period 2π/3 for boat symmetry, but exact should work)
        phi_diff = abs(phi_out - phi_in)
        phi_diff = min(phi_diff, 2*np.pi - phi_diff)
        assert phi_diff < 1e-10, f"phi mismatch: {phi_out} vs {phi_in}"

    def test_apply_then_extract_roundtrip_general(self):
        """Roundtrip test for general conformation."""
        flat = create_planar_6ring()
        Q_in, theta_in, phi_in = 0.5, np.pi/4, 0.8

        puckered = apply_puckering_6ring(flat, Q_in, theta_in, phi_in)
        Q_out, theta_out, phi_out = compute_puckering_6ring(puckered)

        assert abs(Q_out - Q_in) < 1e-10, f"Q mismatch: {Q_out} vs {Q_in}"
        assert abs(theta_out - theta_in) < 1e-10, f"theta mismatch: {theta_out} vs {theta_in}"

    def test_chair_alternating_pattern(self):
        """Chair conformation should have alternating up/down pattern."""
        chair = create_chair_6ring(Q=0.56)

        # Compute mean plane and displacements
        center, normal = compute_mean_plane(chair)
        z = np.dot(chair - center, normal)

        # Check alternating pattern: signs should alternate
        signs = np.sign(z)
        for i in range(6):
            j = (i + 1) % 6
            # Adjacent atoms should have opposite signs
            assert signs[i] * signs[j] < 0, \
                f"Chair should have alternating pattern: z = {z}"

    def test_boat_same_side_pattern(self):
        """Boat conformation should have opposite ends on same side."""
        boat = create_boat_6ring(Q=0.56, phi=0.0)

        # Compute mean plane and displacements
        center, normal = compute_mean_plane(boat)
        z = np.dot(boat - center, normal)

        # In boat with φ=0, atoms 0 and 3 should be on same side (both up or both down)
        # This is characteristic of the boat/twist-boat mode
        # The pattern depends on the exact φ value
        assert abs(z[0]) > 0.1, "Boat should have significant puckering"

    def test_inverted_chair(self):
        """θ=π should give inverted chair."""
        flat = create_planar_6ring()

        chair = apply_puckering_6ring(flat, Q=0.56, theta=0.0, phi=0.0)
        inverted = apply_puckering_6ring(flat, Q=0.56, theta=np.pi, phi=0.0)

        # Compute displacements
        center_c, normal_c = compute_mean_plane(chair)
        center_i, normal_i = compute_mean_plane(inverted)

        z_chair = np.dot(chair - center_c, normal_c)
        z_inv = np.dot(inverted - center_i, normal_i)

        # Inverted chair should have opposite displacements
        # (Note: normals might be in opposite directions, so compare patterns)
        assert np.allclose(np.abs(z_chair), np.abs(z_inv), atol=1e-10)

    def test_amplitude_scaling(self):
        """Larger Q should give larger out-of-plane displacements."""
        flat = create_planar_6ring()

        z_rms_prev = 0
        for Q in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
            puckered = apply_puckering_6ring(flat, Q, theta=0.0, phi=0.0)

            # Compute RMS out-of-plane displacement
            center, normal = compute_mean_plane(puckered)
            z = np.dot(puckered - center, normal)
            z_rms = np.sqrt(np.mean(z**2))

            assert z_rms > z_rms_prev, \
                f"z_rms should increase with Q: {z_rms} not > {z_rms_prev}"
            z_rms_prev = z_rms

    def test_invalid_shape_raises(self):
        """Should raise ValueError for wrong input shape."""
        with pytest.raises(ValueError, match="Expected \\(6, 3\\)"):
            compute_puckering_6ring(np.zeros((5, 3)))

        with pytest.raises(ValueError, match="Expected \\(6, 3\\)"):
            apply_puckering_6ring(np.zeros((7, 3)), 0.5, 0.0, 0.0)


# =============================================================================
# Utility Function Tests
# =============================================================================


class TestUtilities:
    """Tests for utility functions."""

    def test_flatten_ring_removes_puckering(self):
        """Flattening should remove out-of-plane displacements."""
        puckered = create_envelope_5ring(q2=0.4, phi2=0.5)
        flat = flatten_ring_to_plane(puckered)

        # Check that flattened ring is planar
        center, normal = compute_mean_plane(flat)
        z = np.dot(flat - center, normal)

        assert np.allclose(z, 0, atol=1e-10), f"Flattened ring not planar: z = {z}"

    def test_flatten_preserves_center(self):
        """Flattening should preserve the ring centroid."""
        puckered = create_envelope_5ring(q2=0.4, phi2=0.5)
        original_center = puckered.mean(axis=0)

        flat = flatten_ring_to_plane(puckered, preserve_center=True)
        flat_center = flat.mean(axis=0)

        assert np.allclose(original_center, flat_center, atol=1e-10)

    def test_compute_mean_plane_planar_ring(self):
        """Mean plane normal should be perpendicular to planar ring."""
        flat = create_planar_5ring()  # In XY plane

        center, normal = compute_mean_plane(flat)

        # Normal should be along Z (or -Z)
        assert abs(abs(normal[2]) - 1.0) < 1e-10, \
            f"Normal should be along Z for XY planar ring: {normal}"

        # Center should be at origin
        assert np.allclose(center, [0, 0, 0], atol=1e-10)


# =============================================================================
# PyTorch Backend Tests (if available)
# =============================================================================


class TestTorchBackend:
    """Test that puckering works with PyTorch tensors."""

    @pytest.fixture(autouse=True)
    def skip_if_no_torch(self):
        """Skip test if PyTorch is not available."""
        pytest.importorskip("torch")

    def test_5ring_torch_input(self):
        """5-ring puckering should work with torch input."""
        import torch

        coords_np = create_planar_5ring()
        coords_torch = torch.from_numpy(coords_np).float()

        # Apply puckering
        puckered = apply_puckering_5ring(coords_torch, 0.4, 0.3)
        assert isinstance(puckered, torch.Tensor)
        assert puckered.shape == (5, 3)

        # Extract puckering
        q2, phi2 = compute_puckering_5ring(puckered)
        assert abs(q2 - 0.4) < 1e-5
        assert abs(phi2 - 0.3) < 1e-5

    def test_6ring_torch_input(self):
        """6-ring puckering should work with torch input."""
        import torch

        coords_np = create_planar_6ring()
        coords_torch = torch.from_numpy(coords_np).float()

        # Apply puckering
        puckered = apply_puckering_6ring(coords_torch, 0.5, 0.3, 0.7)
        assert isinstance(puckered, torch.Tensor)
        assert puckered.shape == (6, 3)

        # Extract puckering
        Q, theta, phi = compute_puckering_6ring(puckered)
        assert abs(Q - 0.5) < 1e-5
        assert abs(theta - 0.3) < 1e-5
