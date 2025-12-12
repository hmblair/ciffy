"""Tests for structure comparison metrics."""

import os
import pytest
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

import ciffy
from ciffy import Scale, tm_score, lddt

TESTS_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(TESTS_DIR, "data")


def get_cif(pdb_id: str) -> str:
    """Get path to a test CIF file."""
    return os.path.join(DATA_DIR, f"{pdb_id}.cif")


# =============================================================================
# Test TM-score
# =============================================================================

class TestTMScore:
    """Tests for tm_score function."""

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_tm_score_self(self, backend):
        """TM-score of structure with itself should be 1.0."""
        if backend == "torch" and not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")

        p = ciffy.load(get_cif("3SKW"), backend=backend)
        score = tm_score(p, p, scale=Scale.RESIDUE)

        assert abs(score - 1.0) < 1e-6

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_tm_score_range(self, backend):
        """TM-score should be between 0 and 1."""
        if backend == "torch" and not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")

        p = ciffy.load(get_cif("3SKW"), backend=backend)
        score = tm_score(p, p, scale=Scale.RESIDUE)

        assert 0.0 <= score <= 1.0

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_tm_score_atom_scale(self, backend):
        """Test TM-score at atom scale."""
        if backend == "torch" and not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")

        p = ciffy.load(get_cif("3SKW"), backend=backend)
        score = tm_score(p, p, scale=Scale.ATOM)

        assert abs(score - 1.0) < 1e-6

    def test_tm_score_size_mismatch(self):
        """TM-score should raise error for mismatched sizes."""
        p1 = ciffy.load(get_cif("3SKW"), backend="numpy")
        p2 = ciffy.load(get_cif("9GCM"), backend="numpy")

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
        if backend == "torch" and not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")

        p = ciffy.load(get_cif("3SKW"), backend=backend)
        global_score, per_res = lddt(p, p)

        assert abs(global_score - 1.0) < 1e-6

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_lddt_range(self, backend):
        """lDDT should be between 0 and 1."""
        if backend == "torch" and not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")

        p = ciffy.load(get_cif("3SKW"), backend=backend)
        global_score, per_res = lddt(p, p)

        assert 0.0 <= global_score <= 1.0

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_lddt_per_residue_shape(self, backend):
        """lDDT should return per-residue scores with correct shape."""
        if backend == "torch" and not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")

        p = ciffy.load(get_cif("3SKW"), backend=backend)
        global_score, per_res = lddt(p, p)

        expected_shape = (p.size(Scale.RESIDUE),)
        assert per_res.shape == expected_shape

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_lddt_per_residue_self(self, backend):
        """Per-residue lDDT with itself should be mostly 1.0."""
        if backend == "torch" and not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")

        p = ciffy.load(get_cif("3SKW"), backend=backend)
        global_score, per_res = lddt(p, p)

        if backend == "torch":
            per_res = per_res.numpy()

        # Most per-residue scores should be 1.0
        # (some terminal/isolated residues may have 0.0 due to no neighbors within cutoff)
        assert np.mean(per_res == 1.0) > 0.9, "Most residues should have lDDT=1.0"

    def test_lddt_custom_thresholds(self):
        """Test lDDT with custom thresholds."""
        p = ciffy.load(get_cif("3SKW"), backend="numpy")

        # Custom thresholds
        global_score, _ = lddt(p, p, thresholds=(0.5, 1.0))
        assert abs(global_score - 1.0) < 1e-6

    def test_lddt_custom_cutoff(self):
        """Test lDDT with custom cutoff."""
        p = ciffy.load(get_cif("3SKW"), backend="numpy")

        # Very small cutoff should still work
        global_score, _ = lddt(p, p, cutoff=5.0)
        assert abs(global_score - 1.0) < 1e-6

    def test_lddt_size_mismatch(self):
        """lDDT should raise error for mismatched sizes."""
        p1 = ciffy.load(get_cif("3SKW"), backend="numpy")
        p2 = ciffy.load(get_cif("9GCM"), backend="numpy")

        with pytest.raises(ValueError, match="sizes must match"):
            lddt(p1, p2)
