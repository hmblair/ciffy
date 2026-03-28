"""Tests for utility functions."""

import numpy as np


class TestUtilityFunctions:
    """Test utility functions."""

    def test_all_equal(self):
        from ciffy.utils import all_equal
        assert all_equal(1, 1, 1) is True
        assert all_equal(1, 2, 1) is False
        assert all_equal(1) is True
        assert all_equal() is True

    def test_filter_by_mask(self, backend):
        """Test filter_by_mask works with both numpy and torch masks."""
        from ciffy.utils import filter_by_mask
        from ciffy.backend import to_torch

        mask = np.array([True, False, True, False])
        if backend == "torch":
            mask = to_torch(mask)

        result = filter_by_mask(['a', 'b', 'c', 'd'], mask)
        assert result == ['a', 'c']
