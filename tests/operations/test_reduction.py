"""Tests for reduction operations."""

import numpy as np


class TestReduction:
    """Test reduction operations."""

    def test_reductions_dict(self):
        from ciffy.operations import Reduction, REDUCTIONS
        assert Reduction.NONE in REDUCTIONS
        assert Reduction.MEAN in REDUCTIONS
        assert Reduction.SUM in REDUCTIONS

    def test_create_reduction_index(self, backend):
        """Test create_reduction_index with both backends."""
        from ciffy.operations.reduction import create_reduction_index
        from ciffy.backend import to_torch

        counts = np.array([2, 1, 3], dtype=np.int64)
        if backend == "torch":
            counts = to_torch(counts)

        result = create_reduction_index(3, counts)
        expected = np.array([0, 0, 1, 2, 2, 2])
        assert np.array_equal(np.asarray(result), expected)
