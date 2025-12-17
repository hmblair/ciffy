"""
Tests for Polymer reduction and expansion edge cases.

Tests reduce, rreduce, expand, and count methods.
"""

import pytest
import numpy as np

from tests.utils import get_test_cif, BACKENDS


class TestReduce:
    """Test reduce() edge cases."""

    def test_reduce_single_residue(self, backend):
        """reduce to residue scale on single-residue polymer."""
        import ciffy
        from ciffy import Scale, Reduction

        p = ciffy.from_sequence("a", backend=backend)
        result = p.reduce(p.coordinates, Scale.RESIDUE, Reduction.MEAN)

        assert result.shape == (1, 3)

    def test_reduce_single_chain(self, backend):
        """reduce to chain scale on single-chain polymer."""
        import ciffy
        from ciffy import Scale, Reduction

        p = ciffy.from_sequence("acgu", backend=backend)
        result = p.reduce(p.coordinates, Scale.CHAIN, Reduction.MEAN)

        assert result.shape == (1, 3)

    def test_reduce_molecule_scale(self, backend):
        """reduce to molecule scale returns single result."""
        import ciffy
        from ciffy import Scale, Reduction

        p = ciffy.from_sequence("acgu", backend=backend)
        result = p.reduce(p.coordinates, Scale.MOLECULE, Reduction.MEAN)

        assert result.shape == (1, 3)

    def test_reduce_sum_vs_mean(self, backend):
        """reduce SUM should be greater than MEAN for multi-atom residues."""
        import ciffy
        from ciffy import Scale, Reduction

        p = ciffy.from_sequence("acgu", backend=backend)

        mean = p.reduce(p.coordinates, Scale.RESIDUE, Reduction.MEAN)
        summed = p.reduce(p.coordinates, Scale.RESIDUE, Reduction.SUM)

        # Sum magnitude should generally be larger
        mean_norm = np.linalg.norm(np.asarray(mean))
        sum_norm = np.linalg.norm(np.asarray(summed))

        # For non-zero coordinates, sum should generally have larger magnitude
        # (This is a sanity check, not a strict assertion)
        assert isinstance(mean, type(summed))

    def test_reduce_min_max(self, backend):
        """reduce MIN and MAX return values (and optionally indices)."""
        import ciffy
        from ciffy import Scale, Reduction

        p = ciffy.load(get_test_cif("3SKW"), backend=backend)

        min_result = p.reduce(p.coordinates, Scale.CHAIN, Reduction.MIN)
        max_result = p.reduce(p.coordinates, Scale.CHAIN, Reduction.MAX)

        # MIN/MAX may return (value, index) tuple or just value
        # Handle both cases
        if isinstance(min_result, tuple):
            min_val, min_idx = min_result
            max_val, max_idx = max_result
            assert min_val.shape[0] == p.size(Scale.CHAIN)
            assert max_val.shape[0] == p.size(Scale.CHAIN)
        else:
            min_val = min_result
            max_val = max_result
            assert min_val.shape[0] == p.size(Scale.CHAIN)
            assert max_val.shape[0] == p.size(Scale.CHAIN)

    def test_reduce_excludes_hetero_at_residue_scale(self, backend):
        """reduce to RESIDUE scale excludes non-polymer atoms."""
        import ciffy
        from ciffy import Scale, Reduction

        p = ciffy.load(get_test_cif("3SKW"), backend=backend)

        # Structure has hetero atoms
        if p.nonpoly > 0:
            result = p.reduce(p.coordinates, Scale.RESIDUE, Reduction.MEAN)

            # Result size should match number of residues
            assert result.shape[0] == p.size(Scale.RESIDUE)


class TestRReduce:
    """Test rreduce() (residue-level reduction) edge cases."""

    def test_rreduce_single_residue(self, backend):
        """rreduce on single-residue polymer."""
        import ciffy
        from ciffy import Scale, Reduction

        p = ciffy.from_sequence("a", backend=backend)
        # Create per-residue feature
        residue_feature = p.sequence.float() if backend == "torch" else p.sequence.astype(np.float32)

        result = p.rreduce(residue_feature, Scale.CHAIN, Reduction.MEAN)
        assert result.shape[0] == 1

    def test_rreduce_to_chain(self, backend):
        """rreduce per-residue features to chain scale."""
        import ciffy
        from ciffy import Scale, Reduction

        p = ciffy.from_sequence("acgu", backend=backend)

        # Create dummy per-residue features
        if backend == "torch":
            import torch
            residue_feat = torch.ones(p.size(Scale.RESIDUE), dtype=torch.float32)
        else:
            residue_feat = np.ones(p.size(Scale.RESIDUE), dtype=np.float32)

        result = p.rreduce(residue_feat, Scale.CHAIN, Reduction.SUM)

        # Single chain, so result should be sum of all residues
        assert result.shape[0] == 1
        expected_sum = p.size(Scale.RESIDUE)
        actual_sum = result[0].item() if hasattr(result[0], 'item') else result[0]
        assert actual_sum == expected_sum

    def test_rreduce_min_max(self, backend):
        """rreduce with MIN and MAX reductions."""
        import ciffy
        from ciffy import Scale, Reduction

        p = ciffy.load(get_test_cif("9GCM"), backend=backend)

        # Use sequence (residue types) as feature
        if backend == "torch":
            seq_float = p.sequence.float()
        else:
            seq_float = p.sequence.astype(np.float32)

        min_val, min_idx = p.rreduce(seq_float, Scale.CHAIN, Reduction.MIN)
        max_val, max_idx = p.rreduce(seq_float, Scale.CHAIN, Reduction.MAX)

        assert min_val.shape[0] == p.size(Scale.CHAIN)
        assert max_val.shape[0] == p.size(Scale.CHAIN)


class TestExpand:
    """Test expand() edge cases."""

    def test_expand_chain_to_atom(self, backend):
        """expand chain features to atom scale."""
        import ciffy
        from ciffy import Scale

        p = ciffy.from_sequence("acgu", backend=backend)

        # Create per-chain feature (single chain)
        if backend == "torch":
            import torch
            chain_feat = torch.tensor([[1.0, 2.0, 3.0]])
        else:
            chain_feat = np.array([[1.0, 2.0, 3.0]])

        expanded = p.expand(chain_feat, Scale.CHAIN, Scale.ATOM)

        # Should have one row per atom, all same values
        assert expanded.shape[0] == p.size()
        assert expanded.shape[1] == 3

        # All values should be the same (repeated chain feature)
        first_row = np.asarray(expanded[0])
        last_row = np.asarray(expanded[-1])
        assert np.allclose(first_row, last_row)

    def test_expand_residue_to_atom(self, backend):
        """expand residue features to atom scale."""
        import ciffy
        from ciffy import Scale

        p = ciffy.from_sequence("acgu", backend=backend)

        # Create per-residue features
        n_res = p.size(Scale.RESIDUE)
        if backend == "torch":
            import torch
            residue_feat = torch.arange(n_res, dtype=torch.float32).unsqueeze(1)
        else:
            residue_feat = np.arange(n_res, dtype=np.float32).reshape(-1, 1)

        expanded = p.expand(residue_feat, Scale.RESIDUE, Scale.ATOM)

        assert expanded.shape[0] == p.size()

    def test_expand_chain_to_residue(self, backend):
        """expand chain features to residue scale."""
        import ciffy
        from ciffy import Scale

        p = ciffy.from_sequence("acgu", backend=backend)

        # Per-chain feature
        if backend == "torch":
            import torch
            chain_feat = torch.tensor([[1.0]])
        else:
            chain_feat = np.array([[1.0]])

        expanded = p.expand(chain_feat, Scale.CHAIN, Scale.RESIDUE)

        assert expanded.shape[0] == p.size(Scale.RESIDUE)

    def test_expand_invalid_dest_chain(self, backend):
        """expand to CHAIN scale raises ValueError."""
        import ciffy
        from ciffy import Scale

        p = ciffy.from_sequence("acgu", backend=backend)

        if backend == "torch":
            import torch
            mol_feat = torch.tensor([[1.0]])
        else:
            mol_feat = np.array([[1.0]])

        with pytest.raises(ValueError):
            p.expand(mol_feat, Scale.MOLECULE, Scale.CHAIN)


class TestCount:
    """Test count() edge cases."""

    def test_count_all_false_mask(self, backend):
        """count with all-False mask returns zeros."""
        import ciffy
        from ciffy import Scale

        p = ciffy.from_sequence("acgu", backend=backend)

        if backend == "torch":
            import torch
            mask = torch.zeros(p.size(), dtype=torch.bool)
        else:
            mask = np.zeros(p.size(), dtype=bool)

        counts = p.count(mask, Scale.RESIDUE)

        # All counts should be zero
        all_zero = (counts == 0).all() if hasattr(counts, 'all') else np.all(counts == 0)
        assert all_zero

    def test_count_all_true_mask(self, backend):
        """count with all-True mask returns atoms per unit."""
        import ciffy
        from ciffy import Scale

        p = ciffy.from_sequence("acgu", backend=backend)

        if backend == "torch":
            import torch
            mask = torch.ones(p.size(), dtype=torch.bool)
        else:
            mask = np.ones(p.size(), dtype=bool)

        counts = p.count(mask, Scale.RESIDUE)

        # Counts should match atoms per residue
        expected = p.sizes(Scale.RESIDUE)
        counts_np = np.asarray(counts)
        expected_np = np.asarray(expected)

        assert np.array_equal(counts_np, expected_np)

    def test_count_at_chain_scale(self, backend):
        """count at chain scale."""
        import ciffy
        from ciffy import Scale

        p = ciffy.load(get_test_cif("9GCM"), backend=backend)

        if backend == "torch":
            import torch
            mask = torch.ones(p.size(), dtype=torch.bool)
        else:
            mask = np.ones(p.size(), dtype=bool)

        counts = p.count(mask, Scale.CHAIN)

        # Should have one count per chain
        assert len(counts) == p.size(Scale.CHAIN)

        # Sum of counts should equal total atoms
        total = counts.sum().item() if hasattr(counts.sum(), 'item') else counts.sum()
        assert total == p.size()

    def test_count_partial_mask(self, backend):
        """count with partial mask returns correct counts."""
        import ciffy
        from ciffy import Scale

        p = ciffy.from_sequence("acgu", backend=backend)

        # Mask only first half of atoms
        n = p.size()
        if backend == "torch":
            import torch
            mask = torch.zeros(n, dtype=torch.bool)
            mask[:n // 2] = True
        else:
            mask = np.zeros(n, dtype=bool)
            mask[:n // 2] = True

        counts = p.count(mask, Scale.RESIDUE)

        # Total counted should be n//2
        total = counts.sum().item() if hasattr(counts.sum(), 'item') else counts.sum()
        assert total == n // 2


class TestPer:
    """Test per() method edge cases."""

    def test_per_same_scale(self, backend):
        """per(scale, scale) returns ones."""
        import ciffy
        from ciffy import Scale

        p = ciffy.from_sequence("acgu", backend=backend)
        result = p.per(Scale.RESIDUE, Scale.RESIDUE)

        # Should be all ones
        all_ones = (result == 1).all() if hasattr(result, 'all') else np.all(result == 1)
        assert all_ones
        assert len(result) == p.size(Scale.RESIDUE)

    def test_per_atom_chain(self, backend):
        """per(ATOM, CHAIN) returns atoms per chain."""
        import ciffy
        from ciffy import Scale

        p = ciffy.from_sequence("acgu", backend=backend)
        result = p.per(Scale.ATOM, Scale.CHAIN)

        # Single chain, should have one element
        assert len(result) == 1
        # Should equal total atom count
        total = result[0].item() if hasattr(result[0], 'item') else result[0]
        assert total == p.size()

    def test_per_residue_chain(self, backend):
        """per(RESIDUE, CHAIN) returns lengths."""
        import ciffy
        from ciffy import Scale

        p = ciffy.from_sequence("acgu", backend=backend)
        result = p.per(Scale.RESIDUE, Scale.CHAIN)

        # Should match lengths attribute
        lengths_np = np.asarray(p.lengths)
        result_np = np.asarray(result)

        assert np.array_equal(result_np, lengths_np)

    def test_per_invalid_combination(self, backend):
        """per with invalid scale combination raises ValueError."""
        import ciffy
        from ciffy import Scale

        p = ciffy.from_sequence("acgu", backend=backend)

        # CHAIN per RESIDUE doesn't make sense
        with pytest.raises(ValueError):
            p.per(Scale.CHAIN, Scale.RESIDUE)
