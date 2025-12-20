"""
Tests for Polymer geometric method edge cases.

Tests pairwise_distances, knn, center, align, and moment.
"""

import pytest
import numpy as np

from tests.utils import get_test_cif, BACKENDS, random_coordinates
from tests.testing import get_tolerances


class TestPairwiseDistances:
    """Test pairwise_distances() edge cases."""

    def test_pairwise_single_atom(self, backend):
        """pairwise_distances with 1 atom returns 1x1 zero matrix."""
        import ciffy

        p = ciffy.from_sequence("g", backend=backend)
        single = p[:1]

        dists = single.pairwise_distances()

        assert dists.shape == (1, 1)
        val = dists[0, 0].item() if hasattr(dists[0, 0], 'item') else dists[0, 0]
        assert val == 0.0

    def test_pairwise_two_atoms(self, backend):
        """pairwise_distances with 2 atoms returns 2x2 symmetric matrix."""
        import ciffy

        p = ciffy.from_sequence("ac", backend=backend)
        # Take first 2 atoms
        two = p[:2]

        dists = two.pairwise_distances()

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
        import ciffy
        from ciffy import Scale

        p = ciffy.from_sequence("acgu", backend=backend)
        n_res = p.size(Scale.RESIDUE)

        dists = p.pairwise_distances(scale=Scale.RESIDUE)

        assert dists.shape == (n_res, n_res)
        # Diagonal should be zero (distance to self)
        for i in range(n_res):
            val = dists[i, i].item() if hasattr(dists[i, i], 'item') else dists[i, i]
            assert val == 0.0

    def test_pairwise_at_chain_scale(self, backend):
        """pairwise_distances at chain scale on multi-chain structure."""
        import ciffy
        from ciffy import Scale

        p = ciffy.load(get_test_cif("9GCM"), backend=backend)
        n_chains = p.size(Scale.CHAIN)

        dists = p.pairwise_distances(scale=Scale.CHAIN)

        assert dists.shape == (n_chains, n_chains)


class TestKNN:
    """Test knn() edge cases."""

    def test_knn_k_equals_n_fails(self, backend):
        """knn raises ValueError when k >= n."""
        import ciffy

        p = ciffy.from_sequence("ac", backend=backend)
        n = p.size()

        with pytest.raises(ValueError, match="must be less than"):
            p.knn(k=n)

    def test_knn_k_greater_than_n_fails(self, backend):
        """knn raises ValueError when k > n."""
        import ciffy

        p = ciffy.from_sequence("acgu", backend=backend)
        n = p.size()

        with pytest.raises(ValueError, match="must be less than"):
            p.knn(k=n + 10)

    def test_knn_single_atom_fails(self, backend):
        """knn on single atom raises ValueError."""
        import ciffy

        p = ciffy.from_sequence("g", backend=backend)[:1]

        with pytest.raises(ValueError):
            p.knn(k=1)

    def test_knn_k_one(self, backend):
        """knn with k=1 returns single nearest neighbor per point."""
        import ciffy

        p = ciffy.load(get_test_cif("3SKW"), backend=backend)
        neighbors = p.knn(k=1)

        # Shape should be (1, n_atoms)
        assert neighbors.shape[0] == 1
        assert neighbors.shape[1] == p.size()

    def test_knn_k_multiple(self, backend):
        """knn with k=5 returns correct shape."""
        import ciffy

        p = ciffy.load(get_test_cif("3SKW"), backend=backend)
        k = 5
        neighbors = p.knn(k=k)

        # Shape should be (k, n_atoms)
        assert neighbors.shape[0] == k
        assert neighbors.shape[1] == p.size()

    def test_knn_at_residue_scale(self, backend):
        """knn at residue scale returns residue neighbors."""
        import ciffy
        from ciffy import Scale

        p = ciffy.from_sequence("acguacgu", backend=backend)
        n_res = p.size(Scale.RESIDUE)
        k = min(3, n_res - 1)

        neighbors = p.knn(k=k, scale=Scale.RESIDUE)

        assert neighbors.shape[0] == k
        assert neighbors.shape[1] == n_res

    def test_knn_neighbors_are_valid_indices(self, backend):
        """knn returns valid atom indices."""
        import ciffy

        p = ciffy.load(get_test_cif("3SKW"), backend=backend)
        neighbors = p.knn(k=3)

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

        p = ciffy.from_sequence("a", backend=backend)
        # Give non-zero coordinates
        p.coordinates = random_coordinates(p.size(), backend, scale=1.0)

        centered, centroid = p.center(Scale.MOLECULE)

        # Should center to mean ~0
        mean = np.asarray(centered.coordinates).mean(axis=0)
        tol = get_tolerances()
        assert np.allclose(mean, 0, atol=tol.allclose_atol)


class TestMoment:
    """Test moment() edge cases."""

    def test_moment_first_order(self, backend):
        """moment(1) returns centroid (same as reduce MEAN)."""
        import ciffy
        from ciffy import Scale, Reduction

        p = ciffy.load(get_test_cif("3SKW"), backend=backend)

        m1 = p.moment(1, Scale.CHAIN)
        mean = p.reduce(p.coordinates, Scale.CHAIN, Reduction.MEAN)

        m1_np = np.asarray(m1)
        mean_np = np.asarray(mean)

        assert np.allclose(m1_np, mean_np)

    def test_moment_second_order(self, backend):
        """moment(2) returns second moment."""
        import ciffy
        from ciffy import Scale

        p = ciffy.load(get_test_cif("3SKW"), backend=backend)
        m2 = p.moment(2, Scale.CHAIN)

        # Second moment should be non-negative for coordinates
        # (squares are non-negative)
        m2_np = np.asarray(m2)
        assert np.all(m2_np >= 0)

    def test_moment_third_order(self, backend):
        """moment(3) returns skewness (can be negative)."""
        import ciffy
        from ciffy import Scale

        p = ciffy.load(get_test_cif("3SKW"), backend=backend)
        m3 = p.moment(3, Scale.CHAIN)

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
        chain = p.by_index(0)

        # Give varied coordinates for meaningful alignment
        if backend == "torch":
            import torch
            chain.coordinates = torch.randn(chain.size(), 3) * 10
        else:
            chain.coordinates = np.random.randn(chain.size(), 3).astype(np.float32) * 10

        aligned, Q = chain.align(Scale.CHAIN)

        # Rotation matrix should be 3x3
        assert Q.shape[-2:] == (3, 3)

        # Aligned structure should be centered
        mean = np.asarray(aligned.coordinates).mean(axis=0)
        tol = get_tolerances()
        assert np.allclose(mean, 0, atol=tol.center_origin)

    def test_align_returns_rotation_matrix(self, backend):
        """align returns valid rotation matrices."""
        import ciffy
        from ciffy import Scale

        p = ciffy.load(get_test_cif("3SKW"), backend=backend)

        # Give varied coordinates
        if backend == "torch":
            import torch
            p.coordinates = torch.randn(p.size(), 3) * 10
        else:
            p.coordinates = np.random.randn(p.size(), 3).astype(np.float32) * 10

        _, Q = p.align(Scale.MOLECULE)

        Q_np = np.asarray(Q).squeeze()

        # Should be orthogonal: Q @ Q.T ≈ I
        QQt = Q_np @ Q_np.T
        tol = get_tolerances()
        assert np.allclose(QQt, np.eye(3), atol=tol.orthogonality)


class TestWithCoordinates:
    """Test with_coordinates() edge cases."""

    def test_with_coordinates_creates_copy(self, backend):
        """with_coordinates creates new polymer with new coords."""
        import ciffy

        p = ciffy.from_sequence("acgu", backend=backend)
        original_coords = np.asarray(p.coordinates).copy()

        if backend == "torch":
            import torch
            new_coords = torch.randn(p.size(), 3)
        else:
            new_coords = np.random.randn(p.size(), 3).astype(np.float32)

        p2 = p.with_coordinates(new_coords)

        # Original unchanged
        assert np.allclose(np.asarray(p.coordinates), original_coords)
        # New polymer has new coords
        assert np.allclose(np.asarray(p2.coordinates), np.asarray(new_coords))

    def test_with_coordinates_preserves_structure(self, backend):
        """with_coordinates preserves other polymer attributes."""
        import ciffy
        from ciffy import Scale

        p = ciffy.from_sequence("acgu", backend=backend)

        if backend == "torch":
            import torch
            new_coords = torch.randn(p.size(), 3)
        else:
            new_coords = np.random.randn(p.size(), 3).astype(np.float32)

        p2 = p.with_coordinates(new_coords)

        # Structure should be preserved
        assert p2.size() == p.size()
        assert p2.size(Scale.RESIDUE) == p.size(Scale.RESIDUE)
        assert p2.size(Scale.CHAIN) == p.size(Scale.CHAIN)
        assert p2.pdb_id == p.pdb_id


class TestKabschAlignment:
    """Test kabsch_rotation and kabsch_align functions."""

    def test_kabsch_rotation_returns_3x3(self, backend):
        """kabsch_rotation returns a 3x3 rotation matrix."""
        from ciffy import kabsch_rotation

        if backend == "torch":
            import torch
            coords1 = torch.randn(10, 3)
            coords2 = torch.randn(10, 3)
        else:
            coords1 = np.random.randn(10, 3).astype(np.float32)
            coords2 = np.random.randn(10, 3).astype(np.float32)

        R = kabsch_rotation(coords1, coords2)

        assert R.shape == (3, 3)

    def test_kabsch_rotation_is_orthogonal(self, backend):
        """kabsch_rotation returns orthogonal matrix (R @ R.T = I)."""
        from ciffy import kabsch_rotation

        if backend == "torch":
            import torch
            coords1 = torch.randn(20, 3)
            coords2 = torch.randn(20, 3)
        else:
            coords1 = np.random.randn(20, 3).astype(np.float32)
            coords2 = np.random.randn(20, 3).astype(np.float32)

        R = kabsch_rotation(coords1, coords2)
        R_np = np.asarray(R)

        # R @ R.T should be identity
        RRt = R_np @ R_np.T
        tol = get_tolerances()
        assert np.allclose(RRt, np.eye(3), atol=tol.allclose_atol)

    def test_kabsch_rotation_det_positive(self, backend):
        """kabsch_rotation returns proper rotation (det = +1)."""
        from ciffy import kabsch_rotation

        if backend == "torch":
            import torch
            coords1 = torch.randn(15, 3)
            coords2 = torch.randn(15, 3)
        else:
            coords1 = np.random.randn(15, 3).astype(np.float32)
            coords2 = np.random.randn(15, 3).astype(np.float32)

        R = kabsch_rotation(coords1, coords2)
        det = np.linalg.det(np.asarray(R))

        # Should be a proper rotation (not reflection)
        tol = get_tolerances()
        assert abs(det - 1.0) < tol.rotation_determinant

    def test_kabsch_align_returns_tuple(self, backend):
        """kabsch_align returns (aligned, rotation, translation)."""
        from ciffy import kabsch_align

        if backend == "torch":
            import torch
            coords1 = torch.randn(10, 3)
            coords2 = torch.randn(10, 3)
        else:
            coords1 = np.random.randn(10, 3).astype(np.float32)
            coords2 = np.random.randn(10, 3).astype(np.float32)

        result = kabsch_align(coords1, coords2)

        assert len(result) == 3
        aligned, R, translation = result
        assert aligned.shape == coords1.shape
        assert R.shape == (3, 3)
        assert translation.shape == (3,)

    def test_kabsch_align_self_zero_rmsd(self, backend):
        """kabsch_align of coords to itself gives zero RMSD."""
        from ciffy import kabsch_align

        if backend == "torch":
            import torch
            coords = torch.randn(20, 3)
        else:
            coords = np.random.randn(20, 3).astype(np.float32)

        aligned, R, _ = kabsch_align(coords, coords, center=True)
        aligned_np = np.asarray(aligned)
        coords_np = np.asarray(coords)

        # RMSD should be ~0
        rmsd = np.sqrt(((aligned_np - coords_np) ** 2).sum(axis=1).mean())
        tol = get_tolerances()
        assert rmsd < tol.allclose_atol

    def test_kabsch_align_rotation_only(self, backend):
        """kabsch_align recovers known rotation."""
        from ciffy import kabsch_align

        # Create a known rotation (90 degrees around z-axis)
        theta = np.pi / 2
        R_true = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ], dtype=np.float32)

        if backend == "torch":
            import torch
            coords2 = torch.randn(30, 3)
            coords1 = coords2 @ torch.from_numpy(R_true.T)  # Rotate coords2
        else:
            coords2 = np.random.randn(30, 3).astype(np.float32)
            coords1 = coords2 @ R_true.T  # Rotate coords2

        # Center both first
        if backend == "torch":
            coords1_c = coords1 - coords1.mean(dim=0)
            coords2_c = coords2 - coords2.mean(dim=0)
        else:
            coords1_c = coords1 - coords1.mean(axis=0)
            coords2_c = coords2 - coords2.mean(axis=0)

        aligned, R, _ = kabsch_align(coords1_c, coords2_c, center=False)

        # Aligned should match coords2_c closely
        aligned_np = np.asarray(aligned)
        coords2_c_np = np.asarray(coords2_c)
        rmsd = np.sqrt(((aligned_np - coords2_c_np) ** 2).sum(axis=1).mean())
        tol = get_tolerances()
        assert rmsd < tol.alignment_rmsd

    def test_kabsch_align_with_translation(self, backend):
        """kabsch_align handles translation correctly."""
        from ciffy import kabsch_align

        if backend == "torch":
            import torch
            coords2 = torch.randn(20, 3)
            # Translate coords1
            coords1 = coords2 + torch.tensor([10.0, -5.0, 3.0])
        else:
            coords2 = np.random.randn(20, 3).astype(np.float32)
            coords1 = coords2 + np.array([10.0, -5.0, 3.0], dtype=np.float32)

        aligned, _, _ = kabsch_align(coords1, coords2, center=True)

        # After alignment, should match closely
        aligned_np = np.asarray(aligned)
        coords2_np = np.asarray(coords2)
        rmsd = np.sqrt(((aligned_np - coords2_np) ** 2).sum(axis=1).mean())
        tol = get_tolerances()
        assert rmsd < tol.alignment_rmsd


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

        if backend == "torch":
            import torch
            p2.coordinates = p2.coordinates @ torch.from_numpy(R.T) + torch.tensor([10.0, -5.0, 3.0])
        else:
            p2.coordinates = p2.coordinates @ R.T + np.array([10.0, -5.0, 3.0], dtype=np.float32)

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

        p1 = ciffy.from_sequence("acgu", backend=backend)
        p2 = ciffy.from_sequence("acguacgu", backend=backend)

        with pytest.raises(ValueError, match="same size"):
            ciffy.align(p1, p2)
