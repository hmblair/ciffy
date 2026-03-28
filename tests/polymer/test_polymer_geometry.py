"""
Tests for geometry operations.

Tests pairwise_distances, knn, center, align, moment, bonded_distances, adjacency.
"""

import pytest
import numpy as np

from ciffy import operations
from ciffy.biochemistry import Scale
from ciffy.backend import ops
from tests.utils import get_test_cif, BACKENDS, random_coordinates, get_single_chain_poly, get_like
from tests.testing import get_tolerances


class TestPairwiseDistances:
    """Test operations.pairwise_distances() edge cases."""

    def test_pairwise_single_atom(self, backend):
        """pairwise_distances with 1 atom returns 1x1 zero matrix."""
        p = get_single_chain_poly(backend)
        single = p[:1]

        dists = operations.pairwise_distances(single)

        assert dists.shape == (1, 1)
        val = dists[0, 0].item() if hasattr(dists[0, 0], 'item') else dists[0, 0]
        assert val == 0.0

    def test_pairwise_two_atoms(self, backend):
        """pairwise_distances with 2 atoms returns 2x2 symmetric matrix."""
        p = get_single_chain_poly(backend)
        # Take first 2 atoms
        two = p[:2]

        dists = operations.pairwise_distances(two)

        assert dists.shape == (2, 2)
        # Diagonal should be zero
        assert dists[0, 0].item() == 0.0
        assert dists[1, 1].item() == 0.0
        # Should be symmetric
        d01 = dists[0, 1].item() if hasattr(dists[0, 1], 'item') else dists[0, 1]
        d10 = dists[1, 0].item() if hasattr(dists[1, 0], 'item') else dists[1, 0]
        tol = get_tolerances()
        assert abs(d01 - d10) < tol.symmetry

    def test_pairwise_at_residue_scale(self, backend):
        """pairwise_distances at residue scale computes centroid distances."""
        p = get_single_chain_poly(backend)
        n_res = p.size(Scale.RESIDUE)

        dists = operations.pairwise_distances(p, scale=Scale.RESIDUE)

        assert dists.shape == (n_res, n_res)
        # Diagonal should be zero (distance to self)
        for i in range(n_res):
            val = dists[i, i].item() if hasattr(dists[i, i], 'item') else dists[i, i]
            assert val == 0.0

    def test_pairwise_at_chain_scale(self, backend):
        """pairwise_distances at chain scale on multi-chain structure."""
        import ciffy

        p = ciffy.load(get_test_cif("9GCM"), backend=backend)
        n_chains = p.size(Scale.CHAIN)

        dists = operations.pairwise_distances(p, scale=Scale.CHAIN)

        assert dists.shape == (n_chains, n_chains)


class TestKNN:
    """Test operations.knn() edge cases."""

    def test_knn_k_equals_n_fails(self, backend):
        """knn raises ValueError when k >= n."""
        import ciffy

        p = get_single_chain_poly(backend)
        n = p.size()

        with pytest.raises(ValueError, match="must be less than"):
            operations.knn(p, k=n)

    def test_knn_k_greater_than_n_fails(self, backend):
        """knn raises ValueError when k > n."""
        import ciffy

        p = get_single_chain_poly(backend)
        n = p.size()

        with pytest.raises(ValueError, match="must be less than"):
            operations.knn(p, k=n + 10)

    def test_knn_single_atom_fails(self, backend):
        """knn on single atom raises ValueError."""
        import ciffy

        p = get_single_chain_poly(backend)[:1]

        with pytest.raises(ValueError):
            operations.knn(p, k=1)

    def test_knn_k_one(self, backend):
        """knn with k=1 returns single nearest neighbor per point."""
        import ciffy

        p = ciffy.load(get_test_cif("3SKW"), backend=backend)
        neighbors = operations.knn(p, k=1)

        # Shape should be (1, n_atoms)
        assert neighbors.shape[0] == 1
        assert neighbors.shape[1] == p.size()

    def test_knn_k_multiple(self, backend):
        """knn with k=5 returns correct shape."""
        import ciffy

        p = ciffy.load(get_test_cif("3SKW"), backend=backend)
        k = 5
        neighbors = operations.knn(p, k=k)

        # Shape should be (k, n_atoms)
        assert neighbors.shape[0] == k
        assert neighbors.shape[1] == p.size()

    def test_knn_at_residue_scale(self, backend):
        """knn at residue scale returns residue neighbors."""
        import ciffy
        from ciffy import Scale

        p = get_single_chain_poly(backend, "acguacgu")  # 8 residues for k=3
        n_res = p.size(Scale.RESIDUE)
        k = min(3, n_res - 1)

        neighbors = operations.knn(p, k=k, scale=Scale.RESIDUE)

        assert neighbors.shape[0] == k
        assert neighbors.shape[1] == n_res

    def test_knn_neighbors_are_valid_indices(self, backend):
        """knn returns valid atom indices."""
        import ciffy

        p = ciffy.load(get_test_cif("3SKW"), backend=backend)
        neighbors = operations.knn(p, k=3)

        neighbors_np = np.asarray(neighbors)
        n = p.size()

        # All indices should be in [0, n)
        assert np.all(neighbors_np >= 0)
        assert np.all(neighbors_np < n)


class TestCenter:
    """Test center() edge cases."""

    def test_center_molecule_scale(self, backend):
        """center at molecule scale centers entire structure."""
        import ciffy
        from ciffy import Scale

        p = ciffy.load(get_test_cif("3SKW"), backend=backend)
        centered, centroid = p.center(Scale.MOLECULE)

        # Centroid should have shape (1, 3)
        assert centroid.shape == (1, 3)

        # Centered structure's mean should be ~zero (within floating point tolerance)
        mean = centered.reduce(centered.coordinates, Scale.MOLECULE)
        mean_np = np.asarray(mean)
        tol = get_tolerances()
        assert np.allclose(mean_np, 0, atol=tol.center_origin)

    def test_center_chain_scale(self, backend):
        """center at chain scale centers each chain independently."""
        import ciffy
        from ciffy import Scale

        p = ciffy.load(get_test_cif("9GCM"), backend=backend)
        centered, centroids = p.center(Scale.CHAIN)

        # Should have one centroid per chain
        assert centroids.shape[0] == p.size(Scale.CHAIN)
        assert centroids.shape[1] == 3

    def test_center_returns_new_polymer(self, backend):
        """center returns new polymer, doesn't modify original."""
        import ciffy
        from ciffy import Scale

        p = ciffy.load(get_test_cif("3SKW"), backend=backend)
        original_coords = np.asarray(p.coordinates).copy()

        centered, _ = p.center(Scale.MOLECULE)

        # Original should be unchanged
        assert np.allclose(np.asarray(p.coordinates), original_coords)

    def test_center_single_residue(self, backend):
        """center on single-residue polymer."""
        import ciffy
        from ciffy import Scale

        p = get_single_chain_poly(backend)
        # Give non-zero coordinates
        p.coordinates = random_coordinates(p.size(), backend, scale=1.0)

        centered, centroid = p.center(Scale.MOLECULE)

        # Should center to mean ~0
        mean = np.asarray(centered.coordinates).mean(axis=0)
        tol = get_tolerances()
        assert np.allclose(mean, 0, atol=tol.allclose_atol)


class TestMoment:
    """Test operations.moment() edge cases."""

    def test_moment_first_order(self, backend):
        """moment(1) returns centroid (same as reduce MEAN)."""
        import ciffy
        from ciffy import Scale, Reduction

        p = ciffy.load(get_test_cif("3SKW"), backend=backend)

        m1 = operations.moment(p, 1, Scale.CHAIN)
        mean = p.reduce(p.coordinates, Scale.CHAIN, Reduction.MEAN)

        m1_np = np.asarray(m1)
        mean_np = np.asarray(mean)

        assert np.allclose(m1_np, mean_np)

    def test_moment_second_order(self, backend):
        """moment(2) returns second moment."""
        import ciffy
        from ciffy import Scale

        p = ciffy.load(get_test_cif("3SKW"), backend=backend)
        m2 = operations.moment(p, 2, Scale.CHAIN)

        # Second moment should be non-negative for coordinates
        # (squares are non-negative)
        m2_np = np.asarray(m2)
        assert np.all(m2_np >= 0)

    def test_moment_third_order(self, backend):
        """moment(3) returns skewness (can be negative)."""
        import ciffy
        from ciffy import Scale

        p = ciffy.load(get_test_cif("3SKW"), backend=backend)
        m3 = operations.moment(p, 3, Scale.CHAIN)

        # Third moment can be positive or negative
        assert m3.shape[0] == p.size(Scale.CHAIN)
        assert m3.shape[1] == 3


class TestAlign:
    """Test align() edge cases."""

    def test_align_single_chain(self, backend):
        """align at chain scale on single chain."""
        import ciffy
        from ciffy import Scale

        p = ciffy.load(get_test_cif("3SKW"), backend=backend)
        chain = p.chain(0)

        # Give varied coordinates for meaningful alignment
        np.random.seed(42)
        ref = get_like(backend)
        chain.coordinates = ops.randn((chain.size(), 3), like=ref) * 10

        aligned, Q = operations.pca(chain, Scale.CHAIN)

        # Rotation matrix should be 3x3
        assert Q.shape[-2:] == (3, 3)

        # Aligned structure should be centered
        mean = np.asarray(aligned.coordinates).mean(axis=0)
        tol = get_tolerances()
        assert np.allclose(mean, 0, atol=tol.center_origin)

    def test_pca_returns_rotation_matrix(self, backend):
        """pca returns valid rotation matrices."""
        import ciffy
        from ciffy import Scale

        p = ciffy.load(get_test_cif("3SKW"), backend=backend)

        # Give varied coordinates
        np.random.seed(42)
        ref = get_like(backend)
        p.coordinates = ops.randn((p.size(), 3), like=ref) * 10

        _, Q = operations.pca(p, Scale.MOLECULE)

        Q_np = np.asarray(Q).squeeze()

        # Should be orthogonal: Q @ Q.T ≈ I
        QQt = Q_np @ Q_np.T
        tol = get_tolerances()
        assert np.allclose(QQt, np.eye(3), atol=tol.orthogonality)


class TestCopyWithCoordinates:
    """Test copy(coordinates=...) edge cases."""

    def test_copy_coordinates_creates_copy(self, backend):
        """copy(coordinates=...) creates new polymer with new coords."""
        import ciffy

        p = get_single_chain_poly(backend)
        original_coords = np.asarray(p.coordinates).copy()

        np.random.seed(42)
        ref = get_like(backend)
        new_coords = ops.randn((p.size(), 3), like=ref)

        p2 = p.copy(coordinates=new_coords)

        # Original unchanged
        assert np.allclose(np.asarray(p.coordinates), original_coords)
        # New polymer has new coords
        assert np.allclose(np.asarray(p2.coordinates), np.asarray(new_coords))

    def test_copy_coordinates_preserves_structure(self, backend):
        """copy(coordinates=...) preserves other polymer attributes."""
        import ciffy
        from ciffy import Scale

        p = get_single_chain_poly(backend)

        np.random.seed(42)
        ref = get_like(backend)
        new_coords = ops.randn((p.size(), 3), like=ref)

        p2 = p.copy(coordinates=new_coords)

        # Structure should be preserved
        assert p2.size() == p.size()
        assert p2.size(Scale.RESIDUE) == p.size(Scale.RESIDUE)
        assert p2.size(Scale.CHAIN) == p.size(Scale.CHAIN)
        assert p2.pdb_id == p.pdb_id


class TestKabschAlignment:
    """Test kabsch_rotation and kabsch_align functions."""

    def test_kabsch_rotation_returns_3x3(self, backend):
        """kabsch_rotation returns a 3x3 rotation matrix."""
        from ciffy.operations.alignment import kabsch_rotation

        np.random.seed(42)
        ref = get_like(backend)
        coords1 = ops.randn((10, 3), like=ref)
        coords2 = ops.randn((10, 3), like=ref)

        R = kabsch_rotation(coords1, coords2)

        assert R.shape == (3, 3)

    def test_kabsch_rotation_is_orthogonal(self, backend):
        """kabsch_rotation returns orthogonal matrix (R @ R.T = I)."""
        from ciffy.operations.alignment import kabsch_rotation

        np.random.seed(42)
        ref = get_like(backend)
        coords1 = ops.randn((20, 3), like=ref)
        coords2 = ops.randn((20, 3), like=ref)

        R = kabsch_rotation(coords1, coords2)
        R_np = np.asarray(R)

        # R @ R.T should be identity
        RRt = R_np @ R_np.T
        tol = get_tolerances()
        assert np.allclose(RRt, np.eye(3), atol=tol.allclose_atol)

    def test_kabsch_rotation_det_positive(self, backend):
        """kabsch_rotation returns proper rotation (det = +1)."""
        from ciffy.operations.alignment import kabsch_rotation

        np.random.seed(42)
        ref = get_like(backend)
        coords1 = ops.randn((15, 3), like=ref)
        coords2 = ops.randn((15, 3), like=ref)

        R = kabsch_rotation(coords1, coords2)
        det = np.linalg.det(np.asarray(R))

        # Should be a proper rotation (not reflection)
        tol = get_tolerances()
        assert abs(det - 1.0) < tol.rotation_determinant

    def test_kabsch_align_returns_tuple(self, backend):
        """kabsch_align returns (aligned, rotation, translation)."""
        from ciffy.operations.alignment import kabsch_align

        np.random.seed(42)
        ref = get_like(backend)
        coords1 = ops.randn((10, 3), like=ref)
        coords2 = ops.randn((10, 3), like=ref)

        result = kabsch_align(coords1, coords2)

        assert len(result) == 3
        aligned, R, translation = result
        assert aligned.shape == coords1.shape
        assert R.shape == (3, 3)
        assert translation.shape == (3,)

    def test_kabsch_align_self_zero_rmsd(self, backend):
        """kabsch_align of coords to itself gives zero RMSD."""
        from ciffy.operations.alignment import kabsch_align

        np.random.seed(42)
        ref = get_like(backend)
        coords = ops.randn((20, 3), like=ref)

        aligned, R, _ = kabsch_align(coords, coords, center=True)
        aligned_np = np.asarray(aligned)
        coords_np = np.asarray(coords)

        # RMSD should be ~0
        rmsd = np.sqrt(((aligned_np - coords_np) ** 2).sum(axis=1).mean())
        tol = get_tolerances()
        assert rmsd < tol.allclose_atol

    def test_kabsch_align_rotation_only(self, backend):
        """kabsch_align recovers known rotation."""
        from ciffy.operations.alignment import kabsch_align

        # Create a known rotation (90 degrees around z-axis)
        theta = np.pi / 2
        R_true = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ], dtype=np.float32)

        np.random.seed(42)
        ref = get_like(backend)
        coords2 = ops.randn((30, 3), like=ref)
        coords1 = coords2 @ ops.to_backend(R_true.T, like=ref)  # Rotate coords2

        # Center both first
        coords1_c = coords1 - coords1.mean(0)
        coords2_c = coords2 - coords2.mean(0)

        aligned, R, _ = kabsch_align(coords1_c, coords2_c, center=False)

        # Aligned should match coords2_c closely
        aligned_np = np.asarray(aligned)
        coords2_c_np = np.asarray(coords2_c)
        rmsd = np.sqrt(((aligned_np - coords2_c_np) ** 2).sum(axis=1).mean())
        tol = get_tolerances()
        assert rmsd < tol.alignment_rmsd

    def test_kabsch_align_with_translation(self, backend):
        """kabsch_align handles translation correctly."""
        from ciffy.operations.alignment import kabsch_align

        np.random.seed(42)
        ref = get_like(backend)
        coords2 = ops.randn((20, 3), like=ref)
        translation = ops.to_backend(np.array([10.0, -5.0, 3.0], dtype=np.float32), like=ref)
        coords1 = coords2 + translation

        aligned, _, _ = kabsch_align(coords1, coords2, center=True)

        # After alignment, should match closely
        aligned_np = np.asarray(aligned)
        coords2_np = np.asarray(coords2)
        rmsd = np.sqrt(((aligned_np - coords2_np) ** 2).sum(axis=1).mean())
        tol = get_tolerances()
        assert rmsd < tol.alignment_rmsd


class TestScale:
    """Test scale() method.

    Note: scale() uses isotropic scaling - all axes scaled by the same factor.
    This preserves molecular geometry (bond angles, relative distances).
    The returned std is a single scalar per unit, not per-axis.
    """

    def test_scale_molecule_scale(self, backend):
        """scale at molecule scale scales entire structure."""
        import ciffy
        from ciffy import Scale

        p = ciffy.load(get_test_cif("3SKW"), backend=backend)
        scaled, stds = p.scale(Scale.MOLECULE)

        # std should have shape (1, 1) - single scalar for isotropic scaling
        assert stds.shape == (1, 1)

        # Scaled structure should have overall std ≈ 1.0
        # (isotropic: same factor applied to all axes)
        scaled_coords = np.asarray(scaled.coordinates)
        # Overall std = sqrt(mean(x^2 + y^2 + z^2))
        overall_std = np.sqrt((scaled_coords ** 2).mean())
        tol = get_tolerances()
        assert abs(overall_std - 1.0) < tol.allclose_atol

    def test_scale_chain_scale(self, backend):
        """scale at chain scale scales each chain independently."""
        import ciffy
        from ciffy import Scale

        p = ciffy.load(get_test_cif("9GCM"), backend=backend)
        n_chains = p.size(Scale.CHAIN)
        scaled, stds = p.scale(Scale.CHAIN)

        # stds should have one scalar per chain (isotropic scaling)
        assert stds.shape == (n_chains, 1)

    def test_scale_residue_scale(self, backend):
        """scale at residue scale scales each residue independently."""
        import ciffy
        from ciffy import Scale

        p = get_single_chain_poly(backend)
        n_res = p.size(Scale.RESIDUE)
        scaled, stds = p.scale(Scale.RESIDUE)

        # stds should have one scalar per residue (isotropic scaling)
        assert stds.shape == (n_res, 1)

    def test_scale_returns_std_before_scaling(self, backend):
        """scale returns standard deviations BEFORE scaling."""
        import ciffy
        from ciffy import Scale

        p = ciffy.load(get_test_cif("3SKW"), backend=backend)

        # Center first (scale() centers before computing std)
        centered, _ = p.center(Scale.MOLECULE)
        coords = np.asarray(centered.coordinates)
        # Original overall std = sqrt(mean(x^2 + y^2 + z^2)) for centered coords
        original_std = np.sqrt((coords ** 2).mean())

        # Scale the structure
        _, stds = p.scale(Scale.MOLECULE)

        # Returned std should match original (before scaling)
        std_val = float(np.asarray(stds).squeeze())
        tol = get_tolerances()
        assert abs(std_val - original_std) < tol.allclose_atol

    def test_scale_custom_size(self, backend):
        """scale with custom size parameter."""
        import ciffy
        from ciffy import Scale

        p = ciffy.load(get_test_cif("3SKW"), backend=backend)
        target_size = 2.5

        scaled, _ = p.scale(Scale.MOLECULE, size=target_size)

        # Scaled structure should have overall std ≈ target_size
        scaled_coords = np.asarray(scaled.coordinates)
        overall_std = np.sqrt((scaled_coords ** 2).mean())
        tol = get_tolerances()
        assert abs(overall_std - target_size) < tol.allclose_atol

    def test_scale_preserves_structure(self, backend):
        """scale preserves polymer attributes other than coordinates."""
        import ciffy
        from ciffy import Scale

        p = ciffy.load(get_test_cif("3SKW"), backend=backend)
        scaled, _ = p.scale(Scale.MOLECULE)

        # Structure should be preserved
        assert scaled.size() == p.size()
        assert scaled.size(Scale.RESIDUE) == p.size(Scale.RESIDUE)
        assert scaled.size(Scale.CHAIN) == p.size(Scale.CHAIN)
        assert scaled.pdb_id == p.pdb_id

    def test_scale_returns_new_polymer(self, backend):
        """scale returns new polymer, doesn't modify original."""
        import ciffy
        from ciffy import Scale

        p = ciffy.load(get_test_cif("3SKW"), backend=backend)
        original_coords = np.asarray(p.coordinates).copy()

        scaled, _ = p.scale(Scale.MOLECULE)

        # Original should be unchanged
        assert np.allclose(np.asarray(p.coordinates), original_coords)

    def test_scale_centered_result(self, backend):
        """scale produces centered coordinates."""
        import ciffy
        from ciffy import Scale

        p = ciffy.load(get_test_cif("3SKW"), backend=backend)
        scaled, _ = p.scale(Scale.MOLECULE)

        # Scaled coordinates should be centered (mean ≈ 0)
        mean = np.asarray(scaled.coordinates).mean(axis=0)
        tol = get_tolerances()
        assert np.allclose(mean, 0, atol=tol.center_origin)


class TestBondedDistances:
    """Test operations.bonded_distances() edge cases."""

    def test_bonded_distances_o3p_p(self, backend):
        """bonded_distances finds O3'-P phosphodiester bonds."""
        import ciffy
        from ciffy.biochemistry import Residue

        p = ciffy.load(get_test_cif("9MDS"), backend=backend)

        distances = operations.bonded_distances(p, Residue.A.O3p, Residue.A.P)

        # Should find many O3'-P bonds
        assert len(distances) > 0
        # Typical O3'-P distance is ~1.6 Å
        mean_dist = float(np.mean(np.asarray(distances)))
        assert 1.5 < mean_dist < 1.7

    def test_bonded_distances_no_matches(self, backend):
        """bonded_distances returns empty array when no matching bonds."""
        import ciffy

        p = ciffy.load(get_test_cif("9MDS"), backend=backend)

        # Use invalid atom type values that don't exist
        distances = operations.bonded_distances(p, 99999, 99998)

        assert len(distances) == 0

    def test_bonded_distances_c3p_o3p(self, backend):
        """bonded_distances finds C3'-O3' intra-residue bonds."""
        import ciffy
        from ciffy.biochemistry import Residue

        p = ciffy.load(get_test_cif("9MDS"), backend=backend)

        distances = operations.bonded_distances(p, Residue.A.C3p, Residue.A.O3p)

        assert len(distances) > 0
        # Typical C-O bond is ~1.4 Å
        mean_dist = float(np.mean(np.asarray(distances)))
        assert 1.3 < mean_dist < 1.5

    def test_bonded_distances_gradient_flow(self):
        """bonded_distances allows gradient flow to coordinates."""
        torch = pytest.importorskip("torch")
        import ciffy
        from ciffy.biochemistry import Residue

        p = ciffy.load(get_test_cif("9MDS"), backend="torch")

        # Enable gradients on coordinates
        coords = p.coordinates.clone().requires_grad_(True)
        p.coordinates = coords

        # Compute bond distances (O3'-P)
        distances = operations.bonded_distances(p, Residue.A.O3p, Residue.A.P)

        assert distances.requires_grad

        # Compute loss and backprop
        ideal = 1.6
        loss = ((distances - ideal) ** 2).mean()
        loss.backward()

        # Gradients should flow to coordinates
        assert coords.grad is not None
        assert coords.grad.shape == coords.shape
        # At least some gradients should be non-zero
        assert coords.grad.abs().max() > 0

    def test_bonded_distances_backend_consistency(self, backend):
        """bonded_distances gives same results for numpy and torch."""
        if backend == "numpy":
            pytest.skip("Cross-backend comparison only runs once")
        import ciffy
        from ciffy.biochemistry import Residue

        p_np = ciffy.load(get_test_cif("9MDS"), backend="numpy")
        p_torch = ciffy.load(get_test_cif("9MDS"), backend="torch")

        dist_np = operations.bonded_distances(p_np, Residue.A.O3p, Residue.A.P)
        dist_torch = operations.bonded_distances(p_torch, Residue.A.O3p, Residue.A.P)

        assert len(dist_np) == len(dist_torch)
        tol = get_tolerances()
        assert np.allclose(
            np.asarray(dist_np),
            np.asarray(dist_torch),
            atol=tol.allclose_atol
        )


class TestAlignFunction:
    """Test ciffy.align() function."""

    def test_align_returns_tuple_of_polymers(self, backend):
        """align returns (polymer1, aligned_polymer2)."""
        import ciffy

        p1 = ciffy.load(get_test_cif("3SKW"), backend=backend)
        p2 = ciffy.load(get_test_cif("3SKW"), backend=backend)

        ref, aligned = ciffy.align(p1, p2)

        assert isinstance(ref, ciffy.Polymer)
        assert isinstance(aligned, ciffy.Polymer)

    def test_align_reference_unchanged(self, backend):
        """align does not modify the reference polymer."""
        import ciffy

        p1 = ciffy.load(get_test_cif("3SKW"), backend=backend)
        p2 = ciffy.load(get_test_cif("3SKW"), backend=backend)

        original_coords = np.asarray(p1.coordinates).copy()

        ref, aligned = ciffy.align(p1, p2)

        # Reference should be unchanged
        assert np.allclose(np.asarray(ref.coordinates), original_coords)

    def test_align_minimizes_rmsd(self, backend):
        """align produces minimal RMSD between structures."""
        import ciffy

        p1 = ciffy.load(get_test_cif("3SKW"), backend=backend)
        p2 = ciffy.load(get_test_cif("3SKW"), backend=backend)

        # Apply rotation and translation to p2
        theta = np.pi / 3
        R = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ], dtype=np.float32)

        ref = get_like(backend)
        R_t = ops.to_backend(R.T, like=ref)
        translation = ops.to_backend(np.array([10.0, -5.0, 3.0], dtype=np.float32), like=ref)
        p2.coordinates = p2.coordinates @ R_t + translation

        # Before alignment, raw RMSD should be large
        raw_rmsd_before = np.sqrt(((np.asarray(p1.coordinates) - np.asarray(p2.coordinates)) ** 2).sum(axis=1).mean())
        assert raw_rmsd_before > 5.0

        # After alignment, raw RMSD should be minimal
        ref, aligned = ciffy.align(p1, p2)
        raw_rmsd_after = np.sqrt(((np.asarray(ref.coordinates) - np.asarray(aligned.coordinates)) ** 2).sum(axis=1).mean())
        tol = get_tolerances()
        assert raw_rmsd_after < tol.roundtrip_medium

    def test_align_self(self, backend):
        """align of structure with itself gives zero RMSD."""
        import ciffy

        p = ciffy.load(get_test_cif("3SKW"), backend=backend)

        ref, aligned = ciffy.align(p, p)

        # Should be essentially identical
        rmsd = np.sqrt(((np.asarray(ref.coordinates) - np.asarray(aligned.coordinates)) ** 2).sum(axis=1).mean())
        tol = get_tolerances()
        assert rmsd < tol.allclose_atol

    def test_align_size_mismatch_raises(self, backend):
        """align raises ValueError for different-sized polymers."""
        import ciffy

        p1 = get_single_chain_poly(backend)
        p2 = get_single_chain_poly(backend, "acguacgu")

        with pytest.raises(ValueError, match="same size"):
            ciffy.align(p1, p2)


# =============================================================================
# Gather Tests
# =============================================================================


class TestGather:
    """Test operations.gather() function."""

    def test_gather_returns_coordinates(self, backend):
        """gather() returns coordinate array."""
        from ciffy.biochemistry.constants import Sugar
        import ciffy
        from ciffy import operations

        p = ciffy.load(get_test_cif("3SKW"), backend=backend)
        rna = p.molecule_type(__import__('ciffy').Molecule.RNA)

        if rna.empty():
            pytest.skip("No RNA chains in structure")

        coords = operations.gather(rna, [Sugar.C1p])
        assert coords is not None
        assert len(coords.shape) == 3  # (n_residues, n_groups, 3)

    def test_gather_shape(self, backend):
        """gather() returns (n_residues, n_groups, 3) shape."""
        from ciffy.biochemistry.constants import Sugar
        from ciffy import Scale, operations
        import ciffy

        # Use helper which creates polymer with coordinates
        p = get_single_chain_poly(backend)

        groups = [Sugar.C1p]
        coords = operations.gather(p, groups)

        assert coords.shape == (p.size(Scale.RESIDUE), len(groups), 3)

    def test_gather_single_group(self, backend):
        """gather() with single group returns (n_residues, 1, 3)."""
        from ciffy.biochemistry.constants import Sugar
        from ciffy import Scale, operations
        import ciffy

        p = ciffy.load(get_test_cif("3SKW"), backend=backend)
        rna = p.molecule_type(__import__('ciffy').Molecule.RNA)

        if rna.empty():
            pytest.skip("No RNA chains in structure")

        coords = operations.gather(rna, [Sugar.C1p])
        assert coords.shape[0] == rna.size(Scale.RESIDUE)
        assert coords.shape[1] == 1
        assert coords.shape[2] == 3

    def test_gather_coordinates_match_polymer(self, backend):
        """gather() coordinates are valid 3D coordinates."""
        from ciffy.biochemistry.constants import Sugar
        from ciffy import operations
        import ciffy

        p = get_single_chain_poly(backend)
        coords = operations.gather(p, [Sugar.C1p])
        gathered_coords = np.asarray(coords)[:, 0, :]

        # Should have one coordinate per residue
        from ciffy import Scale
        assert gathered_coords.shape[0] == p.size(Scale.RESIDUE)

        # Coordinates should be finite
        assert np.isfinite(gathered_coords).all()

        # Coordinates should be within polymer bounds
        poly_coords = np.asarray(p.coordinates)
        coord_min = poly_coords.min(axis=0)
        coord_max = poly_coords.max(axis=0)
        assert (gathered_coords >= coord_min - 0.1).all()
        assert (gathered_coords <= coord_max + 0.1).all()

    def test_gather_multiple_groups(self, backend):
        """gather() with multiple groups extracts all correctly."""
        from ciffy.biochemistry.constants import Sugar, PurineBase
        from ciffy import operations
        import ciffy

        # Use a structure with purines
        p = ciffy.load(get_test_cif("3SKW"), backend=backend)
        rna = p.molecule_type(__import__('ciffy').Molecule.RNA)

        if rna.empty():
            pytest.skip("No RNA chains in structure")

        # Filter to just adenosine residues (purines with N9)
        from ciffy import Residue
        adenosine = rna.residue_type(Residue.A)

        if adenosine.empty():
            pytest.skip("No adenosine residues")

        groups = [Sugar.C1p, PurineBase.N9]
        coords = operations.gather(adenosine, groups)

        # Should have 2 groups
        assert coords.shape[1] == 2
        # Both coordinates should be different (different atoms)
        c1p = np.asarray(coords)[:, 0, :]
        n9 = np.asarray(coords)[:, 1, :]
        assert not np.allclose(c1p, n9)


# =============================================================================
# Adjacency Tests (Edge Cases)
# =============================================================================


class TestAdjacency:
    """Test operations.adjacency() edge cases."""

    def test_adjacency_dtype_bool(self, backend):
        """adjacency(dtype='bool') returns boolean matrix."""
        import ciffy

        p = get_single_chain_poly(backend)
        adj = operations.adjacency(p, dtype='bool')

        adj_np = np.asarray(adj)
        assert adj_np.dtype == np.bool_ or adj_np.dtype == bool

    def test_adjacency_dtype_float32(self, backend):
        """adjacency(dtype='float32') returns float matrix."""
        import ciffy

        p = get_single_chain_poly(backend)
        adj = operations.adjacency(p, dtype='float32')

        adj_np = np.asarray(adj)
        assert adj_np.dtype == np.float32

    def test_adjacency_shape(self, backend):
        """adjacency() returns (n_atoms, n_atoms) matrix."""
        import ciffy

        p = get_single_chain_poly(backend)
        adj = operations.adjacency(p)

        n = p.size()
        assert adj.shape == (n, n)

    def test_adjacency_empty_polymer(self, backend):
        """adjacency() on empty polymer returns empty matrix."""
        import ciffy

        _template = ciffy.template("a", backend=backend)
        empty = _template[_template.atoms < 0]  # Impossible mask

        adj = operations.adjacency(empty)
        assert adj.shape == (0, 0)


class TestDeprecatedMethods:
    """Test that deprecated Polymer methods still work and emit warnings."""

    def test_pairwise_distances_deprecated(self, backend):
        """polymer.pairwise_distances() emits DeprecationWarning."""
        import warnings

        p = get_single_chain_poly(backend)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = p.pairwise_distances()
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "pairwise_distances" in str(w[0].message)

        # Verify result matches operations
        expected = operations.pairwise_distances(p)
        assert result.shape == expected.shape

    def test_knn_deprecated(self, backend):
        """polymer.knn() emits DeprecationWarning."""
        import warnings
        import ciffy

        p = ciffy.load(get_test_cif("3SKW"), backend=backend)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = p.knn(k=3)
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "knn" in str(w[0].message)

        expected = operations.knn(p, k=3)
        assert result.shape == expected.shape

    def test_moment_deprecated(self, backend):
        """polymer.moment() emits DeprecationWarning."""
        import warnings
        import ciffy

        p = ciffy.load(get_test_cif("9GCM"), backend=backend)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = p.moment(1, Scale.CHAIN)
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "moment" in str(w[0].message)

        expected = operations.moment(p, 1, Scale.CHAIN)
        assert result.shape == expected.shape

    def test_pca_deprecated(self, backend):
        """polymer.pca() emits DeprecationWarning."""
        import warnings
        import ciffy

        p = ciffy.load(get_test_cif("3SKW"), backend=backend)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result, _ = p.pca(Scale.MOLECULE)
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "pca" in str(w[0].message)

        expected, _ = operations.pca(p, Scale.MOLECULE)
        assert result.coordinates.shape == expected.coordinates.shape

    def test_bonded_distances_deprecated(self, backend):
        """polymer.bonded_distances() emits DeprecationWarning."""
        import warnings
        from ciffy import Residue

        p = get_single_chain_poly(backend)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = p.bonded_distances(Residue.A.O3p, Residue.A.P)
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "bonded_distances" in str(w[0].message)

        expected = operations.bonded_distances(p, Residue.A.O3p, Residue.A.P)
        assert result.shape == expected.shape

    def test_adjacency_deprecated(self, backend):
        """polymer.adjacency() emits DeprecationWarning."""
        import warnings

        p = get_single_chain_poly(backend)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = p.adjacency()
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "adjacency" in str(w[0].message)

        expected = operations.adjacency(p)
        assert result.shape == expected.shape
