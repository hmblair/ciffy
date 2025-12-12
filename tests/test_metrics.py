"""Tests for structure comparison metrics."""

import pytest
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

import ciffy
from ciffy import Scale, tm_score, lddt


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

        p = ciffy.load("tests/data/1ZEW.cif", backend=backend)
        score = tm_score(p, p, scale=Scale.RESIDUE)

        assert abs(score - 1.0) < 1e-6

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_tm_score_range(self, backend):
        """TM-score should be between 0 and 1."""
        if backend == "torch" and not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")

        p = ciffy.load("tests/data/1ZEW.cif", backend=backend)
        score = tm_score(p, p, scale=Scale.RESIDUE)

        assert 0.0 <= score <= 1.0

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_tm_score_atom_scale(self, backend):
        """Test TM-score at atom scale."""
        if backend == "torch" and not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")

        p = ciffy.load("tests/data/1ZEW.cif", backend=backend)
        score = tm_score(p, p, scale=Scale.ATOM)

        assert abs(score - 1.0) < 1e-6

    def test_tm_score_size_mismatch(self):
        """TM-score should raise error for mismatched sizes."""
        p1 = ciffy.load("tests/data/1ZEW.cif", backend="numpy")
        p2 = ciffy.load("tests/data/9MDS.cif", backend="numpy")

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

        p = ciffy.load("tests/data/1ZEW.cif", backend=backend)
        global_score, per_res = lddt(p, p)

        assert abs(global_score - 1.0) < 1e-6

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_lddt_range(self, backend):
        """lDDT should be between 0 and 1."""
        if backend == "torch" and not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")

        p = ciffy.load("tests/data/1ZEW.cif", backend=backend)
        global_score, per_res = lddt(p, p)

        assert 0.0 <= global_score <= 1.0

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_lddt_per_residue_shape(self, backend):
        """lDDT should return per-residue scores with correct shape."""
        if backend == "torch" and not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")

        p = ciffy.load("tests/data/1ZEW.cif", backend=backend)
        global_score, per_res = lddt(p, p)

        expected_shape = (p.size(Scale.RESIDUE),)
        assert per_res.shape == expected_shape

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_lddt_per_residue_self(self, backend):
        """Per-residue lDDT with itself should be all 1.0."""
        if backend == "torch" and not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")

        p = ciffy.load("tests/data/1ZEW.cif", backend=backend)
        global_score, per_res = lddt(p, p)

        if backend == "torch":
            per_res = per_res.numpy()

        # All per-residue scores should be 1.0
        assert np.allclose(per_res, 1.0, atol=1e-6)

    def test_lddt_custom_thresholds(self):
        """Test lDDT with custom thresholds."""
        p = ciffy.load("tests/data/1ZEW.cif", backend="numpy")

        # Custom thresholds
        global_score, _ = lddt(p, p, thresholds=(0.5, 1.0))
        assert abs(global_score - 1.0) < 1e-6

    def test_lddt_custom_cutoff(self):
        """Test lDDT with custom cutoff."""
        p = ciffy.load("tests/data/1ZEW.cif", backend="numpy")

        # Very small cutoff should still work
        global_score, _ = lddt(p, p, cutoff=5.0)
        assert abs(global_score - 1.0) < 1e-6

    def test_lddt_size_mismatch(self):
        """lDDT should raise error for mismatched sizes."""
        p1 = ciffy.load("tests/data/1ZEW.cif", backend="numpy")
        p2 = ciffy.load("tests/data/9MDS.cif", backend="numpy")

        with pytest.raises(ValueError, match="sizes must match"):
            lddt(p1, p2)


