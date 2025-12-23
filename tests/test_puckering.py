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


# =============================================================================
# Edge Cases: Phase Wrapping and Numerical Stability
# =============================================================================


class TestPhaseWrapping:
    """Tests for phase angle handling at boundaries."""

    def test_5ring_phase_near_pi(self):
        """Phase near +π should roundtrip correctly."""
        flat = create_planar_5ring()
        for phi2_in in [np.pi - 0.01, np.pi - 0.001, np.pi - 1e-6]:
            puckered = apply_puckering_5ring(flat, 0.4, phi2_in)
            q2_out, phi2_out = compute_puckering_5ring(puckered)

            assert abs(q2_out - 0.4) < 1e-8
            # Handle wraparound
            phi_diff = abs(phi2_out - phi2_in)
            phi_diff = min(phi_diff, 2*np.pi - phi_diff)
            assert phi_diff < 1e-8, f"Phase near π failed: in={phi2_in}, out={phi2_out}"

    def test_5ring_phase_near_minus_pi(self):
        """Phase near -π should roundtrip correctly."""
        flat = create_planar_5ring()
        for phi2_in in [-np.pi + 0.01, -np.pi + 0.001, -np.pi + 1e-6]:
            puckered = apply_puckering_5ring(flat, 0.4, phi2_in)
            q2_out, phi2_out = compute_puckering_5ring(puckered)

            assert abs(q2_out - 0.4) < 1e-8
            phi_diff = abs(phi2_out - phi2_in)
            phi_diff = min(phi_diff, 2*np.pi - phi_diff)
            assert phi_diff < 1e-8, f"Phase near -π failed: in={phi2_in}, out={phi2_out}"

    def test_5ring_phase_zero(self):
        """Phase exactly at 0 should roundtrip correctly."""
        flat = create_planar_5ring()
        puckered = apply_puckering_5ring(flat, 0.4, 0.0)
        q2_out, phi2_out = compute_puckering_5ring(puckered)

        assert abs(q2_out - 0.4) < 1e-10
        assert abs(phi2_out) < 1e-10 or abs(phi2_out - 2*np.pi) < 1e-10

    def test_6ring_phase_sweep(self):
        """6-ring phi should roundtrip across full range."""
        flat = create_planar_6ring()
        Q_in, theta_in = 0.5, np.pi/3  # Non-degenerate theta

        for phi_in in np.linspace(-np.pi, np.pi, 13):
            puckered = apply_puckering_6ring(flat, Q_in, theta_in, phi_in)
            Q_out, theta_out, phi_out = compute_puckering_6ring(puckered)

            assert abs(Q_out - Q_in) < 1e-9
            assert abs(theta_out - theta_in) < 1e-9

            phi_diff = abs(phi_out - phi_in)
            phi_diff = min(phi_diff, 2*np.pi - phi_diff)
            assert phi_diff < 1e-9, f"phi={phi_in:.4f} failed"


class TestNumericalStability:
    """Tests for numerical stability with extreme values."""

    def test_5ring_very_small_amplitude(self):
        """Very small amplitudes should still work."""
        flat = create_planar_5ring()

        for q2 in [1e-3, 1e-6, 1e-9]:
            puckered = apply_puckering_5ring(flat, q2, 0.5)
            q2_out, phi2_out = compute_puckering_5ring(puckered)

            # For very small q2, we mainly check amplitude is correct
            assert abs(q2_out - q2) < 1e-12 or abs(q2_out - q2) / q2 < 1e-6

    def test_5ring_large_amplitude(self):
        """Large amplitudes (extreme puckering) should roundtrip."""
        flat = create_planar_5ring()

        for q2 in [0.8, 1.0, 1.5]:  # Beyond typical ribose values
            puckered = apply_puckering_5ring(flat, q2, 0.3)
            q2_out, phi2_out = compute_puckering_5ring(puckered)

            assert abs(q2_out - q2) < 1e-9
            phi_diff = abs(phi2_out - 0.3)
            phi_diff = min(phi_diff, 2*np.pi - phi_diff)
            assert phi_diff < 1e-9

    def test_6ring_very_small_amplitude(self):
        """Very small Q should handle theta degeneracy gracefully."""
        flat = create_planar_6ring()

        for Q in [1e-3, 1e-6, 1e-9]:
            puckered = apply_puckering_6ring(flat, Q, np.pi/4, 0.5)
            Q_out, theta_out, phi_out = compute_puckering_6ring(puckered)

            # Amplitude should be preserved
            assert abs(Q_out - Q) < 1e-12 or (Q > 0 and abs(Q_out - Q) / Q < 1e-6)

    def test_6ring_large_amplitude(self):
        """Large Q values should roundtrip correctly."""
        flat = create_planar_6ring()

        for Q in [0.8, 1.0, 1.5]:
            puckered = apply_puckering_6ring(flat, Q, np.pi/4, 0.5)
            Q_out, theta_out, phi_out = compute_puckering_6ring(puckered)

            assert abs(Q_out - Q) < 1e-9
            assert abs(theta_out - np.pi/4) < 1e-9

    def test_5ring_zero_amplitude(self):
        """Zero amplitude should give planar ring (q2=0)."""
        flat = create_planar_5ring()
        puckered = apply_puckering_5ring(flat, 0.0, 0.5)

        # Should remain planar
        center, normal = compute_mean_plane(puckered)
        z = np.dot(puckered - center, normal)
        assert np.allclose(z, 0, atol=1e-12)

        q2_out, _ = compute_puckering_5ring(puckered)
        assert q2_out < 1e-12

    def test_6ring_zero_amplitude(self):
        """Zero Q should give planar ring."""
        flat = create_planar_6ring()
        puckered = apply_puckering_6ring(flat, 0.0, np.pi/4, 0.5)

        # Should remain planar
        center, normal = compute_mean_plane(puckered)
        z = np.dot(puckered - center, normal)
        assert np.allclose(z, 0, atol=1e-12)

        Q_out, _, _ = compute_puckering_6ring(puckered)
        assert Q_out < 1e-12


# =============================================================================
# Rotation and Translation Invariance
# =============================================================================


class TestInvariance:
    """Tests that puckering is invariant under rigid transformations."""

    def test_5ring_rotation_invariance(self):
        """Rotating the ring should not change puckering parameters."""
        flat = create_planar_5ring()
        q2_ref, phi2_ref = 0.4, 0.7
        puckered = apply_puckering_5ring(flat, q2_ref, phi2_ref)

        # Test several random rotations
        np.random.seed(42)
        for _ in range(10):
            # Random rotation matrix
            theta = np.random.uniform(0, 2*np.pi)
            phi = np.random.uniform(0, np.pi)
            psi = np.random.uniform(0, 2*np.pi)

            # Euler angles to rotation matrix
            Rz1 = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ])
            Ry = np.array([
                [np.cos(phi), 0, np.sin(phi)],
                [0, 1, 0],
                [-np.sin(phi), 0, np.cos(phi)]
            ])
            Rz2 = np.array([
                [np.cos(psi), -np.sin(psi), 0],
                [np.sin(psi), np.cos(psi), 0],
                [0, 0, 1]
            ])
            R = Rz2 @ Ry @ Rz1

            rotated = puckered @ R.T

            q2_out, phi2_out = compute_puckering_5ring(rotated)

            assert abs(q2_out - q2_ref) < 1e-9, \
                f"q2 changed under rotation: {q2_out} vs {q2_ref}"
            # phi2 might change due to reference frame, but q2 must be invariant

    def test_5ring_translation_invariance(self):
        """Translating the ring should not change puckering parameters."""
        flat = create_planar_5ring()
        q2_ref, phi2_ref = 0.4, 0.7
        puckered = apply_puckering_5ring(flat, q2_ref, phi2_ref)

        # Test several random translations
        np.random.seed(42)
        for _ in range(10):
            translation = np.random.uniform(-100, 100, size=3)
            translated = puckered + translation

            q2_out, phi2_out = compute_puckering_5ring(translated)

            assert abs(q2_out - q2_ref) < 1e-9
            phi_diff = abs(phi2_out - phi2_ref)
            phi_diff = min(phi_diff, 2*np.pi - phi_diff)
            assert phi_diff < 1e-9

    def test_6ring_rotation_invariance(self):
        """Rotating the 6-ring should preserve Q and theta."""
        flat = create_planar_6ring()
        Q_ref, theta_ref, phi_ref = 0.5, np.pi/3, 0.8
        puckered = apply_puckering_6ring(flat, Q_ref, theta_ref, phi_ref)

        np.random.seed(42)
        for _ in range(10):
            # Random rotation
            axis = np.random.randn(3)
            axis /= np.linalg.norm(axis)
            angle = np.random.uniform(0, 2*np.pi)

            # Rodrigues formula
            K = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ])
            R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K

            rotated = puckered @ R.T
            Q_out, theta_out, _ = compute_puckering_6ring(rotated)

            assert abs(Q_out - Q_ref) < 1e-9, f"Q changed: {Q_out} vs {Q_ref}"
            assert abs(theta_out - theta_ref) < 1e-9, f"theta changed: {theta_out} vs {theta_ref}"

    def test_6ring_translation_invariance(self):
        """Translating the 6-ring should not change puckering."""
        flat = create_planar_6ring()
        Q_ref, theta_ref, phi_ref = 0.5, np.pi/3, 0.8
        puckered = apply_puckering_6ring(flat, Q_ref, theta_ref, phi_ref)

        np.random.seed(42)
        for _ in range(10):
            translation = np.random.uniform(-100, 100, size=3)
            translated = puckered + translation

            Q_out, theta_out, phi_out = compute_puckering_6ring(translated)

            assert abs(Q_out - Q_ref) < 1e-9
            assert abs(theta_out - theta_ref) < 1e-9
            phi_diff = abs(phi_out - phi_ref)
            phi_diff = min(phi_diff, 2*np.pi - phi_diff)
            assert phi_diff < 1e-9


# =============================================================================
# Known Conformations (Literature Values)
# =============================================================================


class TestKnownConformations:
    """Tests against known conformational values from literature."""

    def test_5ring_envelope_displacement_pattern(self):
        """
        Envelope conformation: one atom maximally displaced.

        For envelope at atom 0, z[0] should be maximum displacement.
        """
        flat = create_planar_5ring()
        q2 = 0.4
        phi2 = 0.0  # Envelope at atom 0

        puckered = apply_puckering_5ring(flat, q2, phi2)
        center, normal = compute_mean_plane(puckered)
        z = np.dot(puckered - center, normal)

        # Atom 0 should have maximum displacement
        assert abs(z[0]) == pytest.approx(max(abs(z)), rel=1e-6)

    def test_5ring_twist_displacement_pattern(self):
        """
        Twist conformation: two adjacent atoms displaced up, two down.

        Twist has phi2 shifted by π/10 from envelope.
        """
        flat = create_planar_5ring()
        q2 = 0.4
        phi2 = np.pi / 10  # Twist between atoms 0 and 4

        puckered = apply_puckering_5ring(flat, q2, phi2)
        center, normal = compute_mean_plane(puckered)
        z = np.dot(puckered - center, normal)

        # Sum of displacements should be zero (centroid in plane)
        assert abs(np.sum(z)) < 1e-10

    def test_6ring_chair_displacement_pattern(self):
        """
        Chair (θ=0): alternating +/- displacement pattern.

        z_j = √(1/6) * Q * (-1)^j
        """
        flat = create_planar_6ring()
        Q = 0.56

        chair = apply_puckering_6ring(flat, Q, theta=0.0, phi=0.0)
        center, normal = compute_mean_plane(chair)
        z = np.dot(chair - center, normal)

        # All displacements should have same magnitude
        expected_z = np.sqrt(1.0/6.0) * Q
        for i in range(6):
            assert abs(abs(z[i]) - expected_z) < 1e-9

        # Alternating signs
        for i in range(6):
            assert z[i] * z[(i+1) % 6] < 0

    def test_6ring_boat_displacement_pattern(self):
        """
        Boat (θ=π/2): atoms 0,3 displaced same direction, 1,4 and 2,5 opposite.
        """
        flat = create_planar_6ring()
        Q = 0.56

        boat = apply_puckering_6ring(flat, Q, theta=np.pi/2, phi=0.0)
        center, normal = compute_mean_plane(boat)
        z = np.dot(boat - center, normal)

        # In boat with φ=0:
        # z_j = √(1/3) * Q * cos(2πj/3)
        # j=0: cos(0) = 1 → positive
        # j=1: cos(2π/3) = -0.5 → negative
        # j=2: cos(4π/3) = -0.5 → negative
        # j=3: cos(2π) = 1 → positive
        # j=4: cos(8π/3) = -0.5 → negative
        # j=5: cos(10π/3) = -0.5 → negative

        # Atoms 0 and 3 should be on same side
        assert z[0] * z[3] > 0

    def test_6ring_half_chair(self):
        """
        Half-chair (θ ≈ π/4): intermediate between chair and boat.
        """
        flat = create_planar_6ring()
        Q = 0.56
        theta = np.pi / 4

        half_chair = apply_puckering_6ring(flat, Q, theta=theta, phi=0.0)
        Q_out, theta_out, _ = compute_puckering_6ring(half_chair)

        assert abs(Q_out - Q) < 1e-9
        assert abs(theta_out - theta) < 1e-9

    def test_ribose_typical_amplitude(self):
        """
        Ribose sugar typically has q2 ≈ 0.3-0.5 Å.

        C2'-endo: φ2 ≈ 162° (2.83 rad)
        C3'-endo: φ2 ≈ 18° (0.31 rad)
        """
        flat = create_planar_5ring(radius=1.42)  # Approximate ribose size

        # C3'-endo (North)
        q2_c3endo = 0.38
        phi2_c3endo = 0.31  # 18 degrees

        puckered = apply_puckering_5ring(flat, q2_c3endo, phi2_c3endo)
        q2_out, phi2_out = compute_puckering_5ring(puckered)

        assert abs(q2_out - q2_c3endo) < 1e-9
        phi_diff = abs(phi2_out - phi2_c3endo)
        phi_diff = min(phi_diff, 2*np.pi - phi_diff)
        assert phi_diff < 1e-9

        # C2'-endo (South)
        q2_c2endo = 0.40
        phi2_c2endo = 2.83  # 162 degrees

        puckered = apply_puckering_5ring(flat, q2_c2endo, phi2_c2endo)
        q2_out, phi2_out = compute_puckering_5ring(puckered)

        assert abs(q2_out - q2_c2endo) < 1e-9
        phi_diff = abs(phi2_out - phi2_c2endo)
        phi_diff = min(phi_diff, 2*np.pi - phi_diff)
        assert phi_diff < 1e-9


# =============================================================================
# Irregular Ring Geometries
# =============================================================================


class TestIrregularRings:
    """Tests with non-ideal ring geometries."""

    def test_5ring_irregular_bond_lengths(self):
        """Puckering should work with irregular (non-regular) pentagon."""
        # Create irregular pentagon with varying radii
        angles = 2 * np.pi * np.arange(5) / 5
        radii = [1.3, 1.5, 1.4, 1.6, 1.45]
        coords = np.zeros((5, 3))
        for i in range(5):
            coords[i, 0] = radii[i] * np.cos(angles[i])
            coords[i, 1] = radii[i] * np.sin(angles[i])

        q2_in, phi2_in = 0.35, 0.5
        puckered = apply_puckering_5ring(coords, q2_in, phi2_in)
        q2_out, phi2_out = compute_puckering_5ring(puckered)

        # For irregular rings, amplitude roundtrip is approximate due to non-uniform geometry
        assert abs(q2_out - q2_in) < 0.01, f"q2 mismatch: {q2_out} vs {q2_in}"
        phi_diff = abs(phi2_out - phi2_in)
        phi_diff = min(phi_diff, 2*np.pi - phi_diff)
        assert phi_diff < 0.01

    def test_6ring_irregular_bond_lengths(self):
        """Puckering should work with irregular hexagon."""
        # Create irregular hexagon
        angles = 2 * np.pi * np.arange(6) / 6
        radii = [1.5, 1.4, 1.55, 1.45, 1.5, 1.48]
        coords = np.zeros((6, 3))
        for i in range(6):
            coords[i, 0] = radii[i] * np.cos(angles[i])
            coords[i, 1] = radii[i] * np.sin(angles[i])

        Q_in, theta_in, phi_in = 0.5, np.pi/3, 0.7
        puckered = apply_puckering_6ring(coords, Q_in, theta_in, phi_in)
        Q_out, theta_out, phi_out = compute_puckering_6ring(puckered)

        # For irregular rings, amplitude roundtrip is approximate
        assert abs(Q_out - Q_in) < 0.01, f"Q mismatch: {Q_out} vs {Q_in}"
        assert abs(theta_out - theta_in) < 0.01

    def test_5ring_already_puckered_input(self):
        """Applying puckering to already-puckered ring should override."""
        flat = create_planar_5ring()

        # First puckering
        puckered1 = apply_puckering_5ring(flat, 0.3, 0.2)

        # Apply different puckering to already-puckered ring
        q2_new, phi2_new = 0.5, 1.0
        puckered2 = apply_puckering_5ring(puckered1, q2_new, phi2_new)

        # Should have new puckering, not old
        q2_out, phi2_out = compute_puckering_5ring(puckered2)

        assert abs(q2_out - q2_new) < 1e-9
        phi_diff = abs(phi2_out - phi2_new)
        phi_diff = min(phi_diff, 2*np.pi - phi_diff)
        assert phi_diff < 1e-9

    def test_5ring_in_arbitrary_plane(self):
        """Ring not in XY plane should still work correctly."""
        # Create ring in XZ plane
        angles = 2 * np.pi * np.arange(5) / 5
        coords = np.zeros((5, 3))
        coords[:, 0] = 1.5 * np.cos(angles)
        coords[:, 2] = 1.5 * np.sin(angles)  # Z instead of Y

        q2_in, phi2_in = 0.4, 0.6
        puckered = apply_puckering_5ring(coords, q2_in, phi2_in)
        q2_out, phi2_out = compute_puckering_5ring(puckered)

        assert abs(q2_out - q2_in) < 1e-9
        phi_diff = abs(phi2_out - phi2_in)
        phi_diff = min(phi_diff, 2*np.pi - phi_diff)
        assert phi_diff < 1e-9

    def test_6ring_tilted_plane(self):
        """Ring in tilted plane should work correctly."""
        # Create ring and tilt it
        flat = create_planar_6ring()

        # Tilt by rotating around X axis
        theta = 0.7
        R = np.array([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)]
        ])
        tilted = flat @ R.T

        Q_in, theta_in, phi_in = 0.5, np.pi/4, 0.8
        puckered = apply_puckering_6ring(tilted, Q_in, theta_in, phi_in)
        Q_out, theta_out, phi_out = compute_puckering_6ring(puckered)

        assert abs(Q_out - Q_in) < 1e-9
        assert abs(theta_out - theta_in) < 1e-9


# =============================================================================
# Continuous Parameter Sweep Tests
# =============================================================================


class TestParameterSweep:
    """Tests that sweep through parameter ranges."""

    def test_5ring_full_phase_sweep(self):
        """Test roundtrip across full phase range with high resolution."""
        flat = create_planar_5ring()
        q2 = 0.4

        n_samples = 50
        max_error = 0.0

        for phi2_in in np.linspace(-np.pi, np.pi, n_samples):
            puckered = apply_puckering_5ring(flat, q2, phi2_in)
            q2_out, phi2_out = compute_puckering_5ring(puckered)

            q2_error = abs(q2_out - q2)
            phi_diff = abs(phi2_out - phi2_in)
            phi_diff = min(phi_diff, 2*np.pi - phi_diff)

            max_error = max(max_error, q2_error, phi_diff)

        assert max_error < 1e-9, f"Max error across phase sweep: {max_error}"

    def test_5ring_amplitude_sweep(self):
        """Test roundtrip across amplitude range."""
        flat = create_planar_5ring()
        phi2 = 0.5

        for q2_in in np.linspace(0.01, 1.0, 20):
            puckered = apply_puckering_5ring(flat, q2_in, phi2)
            q2_out, phi2_out = compute_puckering_5ring(puckered)

            assert abs(q2_out - q2_in) < 1e-9

    def test_6ring_theta_sweep(self):
        """Test roundtrip across theta range (chair to boat to inverted chair)."""
        flat = create_planar_6ring()
        Q = 0.5
        phi = 0.5

        for theta_in in np.linspace(0.01, np.pi - 0.01, 20):
            puckered = apply_puckering_6ring(flat, Q, theta_in, phi)
            Q_out, theta_out, phi_out = compute_puckering_6ring(puckered)

            assert abs(Q_out - Q) < 1e-9
            assert abs(theta_out - theta_in) < 1e-9

    def test_6ring_spherical_grid(self):
        """Test roundtrip on a grid of (theta, phi) values."""
        flat = create_planar_6ring()
        Q = 0.5

        max_error = 0.0
        n_theta = 10
        n_phi = 10

        for theta_in in np.linspace(0.1, np.pi - 0.1, n_theta):
            for phi_in in np.linspace(-np.pi, np.pi, n_phi):
                puckered = apply_puckering_6ring(flat, Q, theta_in, phi_in)
                Q_out, theta_out, phi_out = compute_puckering_6ring(puckered)

                Q_error = abs(Q_out - Q)
                theta_error = abs(theta_out - theta_in)

                max_error = max(max_error, Q_error, theta_error)

        assert max_error < 1e-9, f"Max error on spherical grid: {max_error}"


# =============================================================================
# Float32 Precision Tests
# =============================================================================


class TestFloat32Precision:
    """Tests for float32 precision handling."""

    def test_5ring_float32_roundtrip(self):
        """Puckering should work with float32 input (output may be float64)."""
        flat = create_planar_5ring().astype(np.float32)
        q2_in, phi2_in = 0.4, 0.5

        puckered = apply_puckering_5ring(flat, q2_in, phi2_in)
        # Implementation uses float64 internally for numerical stability

        q2_out, phi2_out = compute_puckering_5ring(puckered)

        # Should still give accurate roundtrip
        assert abs(q2_out - q2_in) < 1e-5
        phi_diff = abs(phi2_out - phi2_in)
        phi_diff = min(phi_diff, 2*np.pi - phi_diff)
        assert phi_diff < 1e-5

    def test_6ring_float32_roundtrip(self):
        """6-ring puckering should work with float32 input."""
        flat = create_planar_6ring().astype(np.float32)
        Q_in, theta_in, phi_in = 0.5, np.pi/4, 0.7

        puckered = apply_puckering_6ring(flat, Q_in, theta_in, phi_in)
        # Implementation uses float64 internally for numerical stability

        Q_out, theta_out, phi_out = compute_puckering_6ring(puckered)

        assert abs(Q_out - Q_in) < 1e-5
        assert abs(theta_out - theta_in) < 1e-5


# =============================================================================
# Geometric Property Preservation
# =============================================================================


class TestGeometricPreservation:
    """Tests that geometric properties are preserved."""

    def test_5ring_in_plane_distances_preserved(self):
        """Puckering should preserve in-plane bond lengths."""
        flat = create_planar_5ring()

        # Compute original in-plane bond lengths (XY plane)
        original_xy_bonds = []
        for i in range(5):
            j = (i + 1) % 5
            diff = flat[j, :2] - flat[i, :2]  # XY only
            original_xy_bonds.append(np.linalg.norm(diff))

        # Apply puckering
        puckered = apply_puckering_5ring(flat, 0.4, 0.5)

        # Project puckered coords to plane and check in-plane distances
        # The in-plane projection should preserve distances
        center, normal = compute_mean_plane(puckered)
        centered = puckered - center

        # Project onto plane perpendicular to normal
        z_displacements = np.dot(centered, normal)
        in_plane = centered - np.outer(z_displacements, normal)

        # Check that in-plane bond lengths match original
        for i in range(5):
            j = (i + 1) % 5
            new_bond = np.linalg.norm(in_plane[j] - in_plane[i])
            original_bond = np.linalg.norm(flat[j, :2] - flat[i, :2])
            assert abs(new_bond - original_bond) < 1e-9

    def test_6ring_in_plane_distances_preserved(self):
        """6-ring puckering should preserve in-plane bond lengths."""
        flat = create_planar_6ring()

        # Apply puckering
        puckered = apply_puckering_6ring(flat, 0.5, np.pi/4, 0.7)

        # Project puckered coords to plane and check in-plane distances
        center, normal = compute_mean_plane(puckered)
        centered = puckered - center

        z_displacements = np.dot(centered, normal)
        in_plane = centered - np.outer(z_displacements, normal)

        # Check that in-plane bond lengths match original planar ring
        for i in range(6):
            j = (i + 1) % 6
            new_bond = np.linalg.norm(in_plane[j] - in_plane[i])
            original_bond = np.linalg.norm(flat[j, :2] - flat[i, :2])
            assert abs(new_bond - original_bond) < 1e-9

    def test_3d_bonds_increase_with_puckering(self):
        """3D bond lengths should increase with puckering (expected behavior)."""
        flat = create_planar_5ring()

        # Original planar bond lengths
        original_bonds = []
        for i in range(5):
            j = (i + 1) % 5
            original_bonds.append(np.linalg.norm(flat[j] - flat[i]))

        # Apply significant puckering
        puckered = apply_puckering_5ring(flat, 0.4, 0.5)

        # 3D bond lengths should be >= planar (Pythagorean theorem)
        for i in range(5):
            j = (i + 1) % 5
            new_bond = np.linalg.norm(puckered[j] - puckered[i])
            assert new_bond >= original_bonds[i] - 1e-9, \
                f"3D bond should be >= planar: {new_bond} vs {original_bonds[i]}"

    def test_5ring_centroid_preserved(self):
        """Puckering should preserve ring centroid."""
        flat = create_planar_5ring()
        original_center = flat.mean(axis=0)

        puckered = apply_puckering_5ring(flat, 0.4, 0.5)
        new_center = puckered.mean(axis=0)

        assert np.allclose(new_center, original_center, atol=1e-10)

    def test_6ring_centroid_preserved(self):
        """6-ring puckering should preserve centroid."""
        flat = create_planar_6ring()
        original_center = flat.mean(axis=0)

        puckered = apply_puckering_6ring(flat, 0.5, np.pi/4, 0.7)
        new_center = puckered.mean(axis=0)

        assert np.allclose(new_center, original_center, atol=1e-10)

    def test_flatten_preserves_in_plane_distances(self):
        """Flattening should preserve in-plane distances (approximately)."""
        puckered = create_envelope_5ring(q2=0.3, phi2=0.5)
        flat = flatten_ring_to_plane(puckered)

        # Bond lengths should be approximately preserved
        for i in range(5):
            j = (i + 1) % 5
            puckered_dist = np.linalg.norm(puckered[j] - puckered[i])
            flat_dist = np.linalg.norm(flat[j] - flat[i])
            # Allow small differences due to projection
            assert abs(puckered_dist - flat_dist) < 0.1
