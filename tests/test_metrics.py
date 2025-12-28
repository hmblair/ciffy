"""Tests for structure comparison metrics."""

import pytest
import numpy as np

import ciffy
from ciffy import Scale, Molecule, tm_score, lddt, rmsd

from tests.utils import (
    get_test_cif,
    TORCH_AVAILABLE,
    skip_if_no_torch,
    random_coordinates,
)
from tests.testing import get_tolerances


# =============================================================================
# Test TM-score
# =============================================================================

class TestTMScore:
    """Tests for tm_score function."""

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_tm_score_self(self, backend):
        """TM-score of structure with itself should be 1.0."""
        skip_if_no_torch(backend)

        p = ciffy.load(get_test_cif("3SKW"), backend=backend)
        score = tm_score(p, p, scale=Scale.RESIDUE)

        tol = get_tolerances()
        assert abs(score - 1.0) < tol.score_self

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_tm_score_range(self, backend):
        """TM-score should be between 0 and 1."""
        skip_if_no_torch(backend)

        p = ciffy.load(get_test_cif("3SKW"), backend=backend)
        score = tm_score(p, p, scale=Scale.RESIDUE)

        assert 0.0 <= score <= 1.0

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_tm_score_atom_scale(self, backend):
        """Test TM-score at atom scale."""
        skip_if_no_torch(backend)

        p = ciffy.load(get_test_cif("3SKW"), backend=backend)
        score = tm_score(p, p, scale=Scale.ATOM)

        tol = get_tolerances()
        assert abs(score - 1.0) < tol.score_self

    def test_tm_score_size_mismatch(self):
        """TM-score should raise error for mismatched sizes."""
        p1 = ciffy.load(get_test_cif("3SKW"), backend="numpy")
        p2 = ciffy.load(get_test_cif("9GCM"), backend="numpy")

        with pytest.raises(ValueError, match="sizes must match"):
            tm_score(p1, p2, scale=Scale.RESIDUE)


# =============================================================================
# Test lDDT
# =============================================================================

class TestLDDT:
    """Tests for lddt function."""

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_lddt_self(self, backend):
        """lDDT of structure with itself should be 1.0."""
        skip_if_no_torch(backend)

        p = ciffy.load(get_test_cif("3SKW"), backend=backend)
        global_score, per_res = lddt(p, p)

        tol = get_tolerances()
        assert abs(global_score - 1.0) < tol.score_self

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_lddt_range(self, backend):
        """lDDT should be between 0 and 1."""
        skip_if_no_torch(backend)

        p = ciffy.load(get_test_cif("3SKW"), backend=backend)
        global_score, per_res = lddt(p, p)

        assert 0.0 <= global_score <= 1.0

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_lddt_per_residue_shape(self, backend):
        """lDDT should return per-residue scores with correct shape."""
        skip_if_no_torch(backend)

        p = ciffy.load(get_test_cif("3SKW"), backend=backend)
        global_score, per_res = lddt(p, p)

        expected_shape = (p.size(Scale.RESIDUE),)
        assert per_res.shape == expected_shape

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_lddt_per_residue_self(self, backend):
        """Per-residue lDDT with itself should be mostly 1.0."""
        skip_if_no_torch(backend)

        p = ciffy.load(get_test_cif("3SKW"), backend=backend)
        global_score, per_res = lddt(p, p)

        if backend == "torch":
            per_res = per_res.numpy()

        # Most per-residue scores should be 1.0
        # (some terminal/isolated residues may have 0.0 due to no neighbors within cutoff)
        assert np.mean(per_res == 1.0) > 0.9, "Most residues should have lDDT=1.0"

    def test_lddt_custom_thresholds(self):
        """Test lDDT with custom thresholds."""
        p = ciffy.load(get_test_cif("3SKW"), backend="numpy")

        # Custom thresholds
        global_score, _ = lddt(p, p, thresholds=(0.5, 1.0))
        tol = get_tolerances()
        assert abs(global_score - 1.0) < tol.score_self

    def test_lddt_custom_cutoff(self):
        """Test lDDT with custom cutoff."""
        p = ciffy.load(get_test_cif("3SKW"), backend="numpy")

        # Very small cutoff should still work
        global_score, _ = lddt(p, p, cutoff=5.0)
        tol = get_tolerances()
        assert abs(global_score - 1.0) < tol.score_self

    def test_lddt_size_mismatch(self):
        """lDDT should raise error for mismatched sizes."""
        p1 = ciffy.load(get_test_cif("3SKW"), backend="numpy")
        p2 = ciffy.load(get_test_cif("9GCM"), backend="numpy")

        with pytest.raises(ValueError, match="sizes must match"):
            lddt(p1, p2)


# =============================================================================
# Edge Case Tests for Metrics
# =============================================================================

class TestTMScoreEdgeCases:
    """Edge case tests for tm_score."""

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_tm_score_very_small_structure(self, backend):
        """TM-score handles very small structures (d_0 edge case)."""
        skip_if_no_torch(backend)

        # Create 5-residue structure
        p = ciffy.from_sequence("acgua", backend=backend)

        # Attach random non-zero coordinates
        p.coordinates = random_coordinates(p.size(), backend)

        # Templates don't have molecule_types, so specify explicitly
        score = tm_score(p, p, scale=Scale.RESIDUE, molecule_type=Molecule.RNA)

        assert 0.0 <= score <= 1.0
        # Self-comparison should be ~1.0
        assert score > 0.99

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_tm_score_single_residue(self, backend):
        """TM-score handles single-residue structure (may return NaN for L<5)."""
        skip_if_no_torch(backend)

        p = ciffy.from_sequence("a", backend=backend)

        # Attach non-zero coordinates
        p.coordinates = random_coordinates(p.size(), backend)

        # Templates don't have molecule_types, so specify explicitly
        score = tm_score(p, p, scale=Scale.RESIDUE, molecule_type=Molecule.RNA)

        # TM-score should always return a valid float in [0, 1]
        assert 0.0 <= score <= 1.0

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_tm_score_two_residues(self, backend):
        """TM-score handles two-residue structure."""
        skip_if_no_torch(backend)

        p = ciffy.from_sequence("ac", backend=backend)

        # Attach non-zero coordinates
        p.coordinates = random_coordinates(p.size(), backend)

        # Templates don't have molecule_types, so specify explicitly
        score = tm_score(p, p, scale=Scale.RESIDUE, molecule_type=Molecule.RNA)

        # TM-score should always return a valid float in [0, 1]
        assert 0.0 <= score <= 1.0

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_tm_score_requires_molecule_types(self, backend):
        """TM-score raises ValueError when molecule_types not available and not specified."""
        skip_if_no_torch(backend)

        p = ciffy.from_sequence("acgu", backend=backend)
        p.coordinates = random_coordinates(p.size(), backend)

        with pytest.raises(ValueError, match="molecule_types not available"):
            tm_score(p, p, scale=Scale.RESIDUE)

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_tm_score_at_residue_scale(self, backend):
        """TM-score at residue scale on larger structure."""
        skip_if_no_torch(backend)

        p = ciffy.load(get_test_cif("3SKW"), backend=backend)
        score = tm_score(p, p, scale=Scale.RESIDUE)

        assert 0.0 <= score <= 1.0
        assert score > 0.99


class TestLDDTEdgeCases:
    """Edge case tests for lddt."""

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_lddt_very_small_cutoff(self, backend):
        """lDDT with very small cutoff (few/no pairs)."""
        skip_if_no_torch(backend)

        p = ciffy.from_sequence("acgu", backend=backend)

        # Place atoms very far apart
        n = p.size()
        coords = np.zeros((n, 3), dtype=np.float32)
        coords[:, 0] = np.arange(n) * 100  # 100 angstroms apart
        if backend == "torch":
            import torch
            p.coordinates = torch.from_numpy(coords)
        else:
            p.coordinates = coords

        # Very small cutoff = no pairs
        global_score, per_res = lddt(p, p, cutoff=1.0)

        # With no valid pairs, lDDT should return 0.0
        assert global_score == 0.0

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_lddt_single_threshold(self, backend):
        """lDDT with single threshold."""
        skip_if_no_torch(backend)

        p = ciffy.load(get_test_cif("3SKW"), backend=backend)
        global_score, per_res = lddt(p, p, thresholds=(1.0,))

        assert 0.0 <= global_score <= 1.0

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_lddt_many_thresholds(self, backend):
        """lDDT with many thresholds."""
        skip_if_no_torch(backend)

        p = ciffy.load(get_test_cif("3SKW"), backend=backend)
        thresholds = tuple(i * 0.1 for i in range(1, 21))  # 0.1 to 2.0
        global_score, per_res = lddt(p, p, thresholds=thresholds)

        assert 0.0 <= global_score <= 1.0

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_lddt_single_residue(self, backend):
        """lDDT on single-residue structure."""
        skip_if_no_torch(backend)

        p = ciffy.from_sequence("a", backend=backend)

        # Attach non-zero coordinates
        p.coordinates = random_coordinates(p.size(), backend, scale=1.0)

        global_score, per_res = lddt(p, p)

        # Single residue may have undefined lDDT (no pairs to compare)
        assert per_res.shape == (1,)

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_lddt_two_residues(self, backend):
        """lDDT on two-residue structure."""
        skip_if_no_torch(backend)

        p = ciffy.from_sequence("ac", backend=backend)

        # Attach coordinates close together
        p.coordinates = random_coordinates(p.size(), backend, scale=1.0)

        global_score, per_res = lddt(p, p)

        assert 0.0 <= global_score <= 1.0
        assert per_res.shape == (2,)

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_lddt_large_cutoff(self, backend):
        """lDDT with very large cutoff (all pairs included)."""
        skip_if_no_torch(backend)

        p = ciffy.load(get_test_cif("3SKW"), backend=backend)
        global_score, per_res = lddt(p, p, cutoff=1000.0)

        # Self comparison should be 1.0
        tol = get_tolerances()
        assert abs(global_score - 1.0) < tol.score_self


# =============================================================================
# Test RMSD Function
# =============================================================================

class TestRmsdFunction:
    """Tests for the rmsd() function."""

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_rmsd_self_is_zero(self, backend):
        """RMSD of structure with itself should be 0."""
        skip_if_no_torch(backend)

        p = ciffy.load(get_test_cif("3SKW"), backend=backend)
        rmsd_val = rmsd(p, p)

        tol = get_tolerances()
        rmsd_np = np.asarray(rmsd_val)
        assert np.all(rmsd_np < tol.allclose_atol)

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_rmsd_symmetric(self, backend):
        """RMSD(a, b) == RMSD(b, a)."""
        skip_if_no_torch(backend)

        p1 = ciffy.load(get_test_cif("3SKW"), backend=backend)
        # Create a perturbed copy
        p2 = p1.with_coordinates(p1.coordinates + 0.1)

        rmsd_ab = rmsd(p1, p2)
        rmsd_ba = rmsd(p2, p1)

        rmsd_ab_np = np.asarray(rmsd_ab)
        rmsd_ba_np = np.asarray(rmsd_ba)

        tol = get_tolerances()
        assert np.allclose(rmsd_ab_np, rmsd_ba_np, atol=tol.allclose_atol)

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_rmsd_at_molecule_scale(self, backend):
        """RMSD at molecule scale returns single value."""
        skip_if_no_torch(backend)

        p = ciffy.load(get_test_cif("3SKW"), backend=backend)
        rmsd_val = rmsd(p, p, scale=Scale.MOLECULE)

        # Should be shape (1,) for single molecule
        rmsd_np = np.asarray(rmsd_val)
        assert rmsd_np.shape == (1,)

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_rmsd_at_chain_scale(self, backend):
        """RMSD at chain scale returns one value per chain."""
        skip_if_no_torch(backend)

        p = ciffy.load(get_test_cif("9GCM"), backend=backend)
        n_chains = p.size(Scale.CHAIN)
        rmsd_val = rmsd(p, p, scale=Scale.CHAIN)

        rmsd_np = np.asarray(rmsd_val)
        assert rmsd_np.shape == (n_chains,)

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_rmsd_at_residue_scale(self, backend):
        """RMSD at residue scale returns one value per residue."""
        skip_if_no_torch(backend)

        # Use poly() to exclude hetero atoms (water, ions) that don't have residues
        p = ciffy.load(get_test_cif("3SKW"), backend=backend).poly()
        n_res = p.size(Scale.RESIDUE)
        rmsd_val = rmsd(p, p, scale=Scale.RESIDUE)

        rmsd_np = np.asarray(rmsd_val)
        assert rmsd_np.shape == (n_res,)

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_rmsd_nonnegative(self, backend):
        """RMSD values are always non-negative."""
        skip_if_no_torch(backend)

        # Use poly() to exclude hetero atoms
        p1 = ciffy.load(get_test_cif("3SKW"), backend=backend).poly()

        np.random.seed(42)
        noise = np.random.randn(p1.size(), 3).astype(np.float32) * 0.5
        if backend == "torch":
            import torch
            noise = torch.from_numpy(noise)
        p2 = p1.with_coordinates(p1.coordinates + noise)

        rmsd_val = rmsd(p1, p2, scale=Scale.RESIDUE)

        rmsd_np = np.asarray(rmsd_val)
        assert np.all(rmsd_np >= 0)

    def test_rmsd_size_mismatch_raises(self):
        """RMSD raises ValueError for mismatched sizes."""
        p1 = ciffy.from_sequence("acgu", backend="numpy")
        p2 = ciffy.from_sequence("acguacgu", backend="numpy")

        # Just check that it raises ValueError (message may vary)
        with pytest.raises(ValueError):
            rmsd(p1, p2)

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_rmsd_array_input(self, backend):
        """RMSD works with raw coordinate arrays."""
        skip_if_no_torch(backend)

        np.random.seed(42)
        coords1 = np.random.randn(20, 3).astype(np.float32)
        coords2 = coords1 + 0.1

        if backend == "torch":
            import torch
            coords1 = torch.from_numpy(coords1)
            coords2 = torch.from_numpy(coords2)

        rmsd_val = rmsd(coords1, coords2)

        # Should be a scalar or 0-d array
        if hasattr(rmsd_val, 'item'):
            rmsd_scalar = rmsd_val.item()
        else:
            rmsd_scalar = float(rmsd_val)

        assert rmsd_scalar >= 0
        assert rmsd_scalar < 1.0  # Should be small for small perturbation
