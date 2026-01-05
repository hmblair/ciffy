"""Tests for the reactivity module."""

import numpy as np
import pytest

import ciffy
from ciffy import ReactivityIndex, ReactivityMatch
from ciffy.biochemistry import Scale


@pytest.fixture
def polymer():
    """Load a test polymer chain."""
    return ciffy.load("tests/data/9MDS.cif").chain(0).strip()


class TestReactivityIndex:
    """Test ReactivityIndex basic operations."""

    def test_add_and_len(self):
        """Can add entries and get count."""
        index = ReactivityIndex()
        assert len(index) == 0

        index.add("entry1", "ACGU", np.array([0.1, 0.2, 0.3, 0.4]))
        assert len(index) == 1

        index.add("entry2", "GCUA", np.array([0.5, 0.6, 0.7, 0.8]))
        assert len(index) == 2

    def test_add_validates_length(self):
        """add() raises ValueError if lengths don't match."""
        index = ReactivityIndex()

        with pytest.raises(ValueError, match="doesn't match"):
            index.add("bad", "ACGU", np.array([0.1, 0.2, 0.3]))

    def test_add_normalizes_sequence(self):
        """add() normalizes sequence to uppercase and U."""
        index = ReactivityIndex()
        index.add("test", "acgt", np.array([1, 2, 3, 4]))

        # Should match when we search with uppercase
        # (internal implementation detail, but verifiable via match)


class TestReactivityMatch:
    """Test reactivity matching logic."""

    def test_exact_match(self, polymer):
        """Exact sequence match returns correct reactivity."""
        seq = polymer.sequence_str().upper()
        n_res = len(seq)
        reactivity = np.arange(n_res, dtype=np.float32)

        index = ReactivityIndex()
        index.add("exact", seq, reactivity)

        result = index.match(polymer)

        assert result is not None
        assert result.name == "exact"
        assert result.span == 1.0
        assert result.offset == 0
        np.testing.assert_array_equal(result.reactivity, reactivity)

    def test_structure_substring_of_probed(self, polymer):
        """Structure sequence is substring of probed sequence."""
        seq = polymer.sequence_str().upper()
        n_res = len(seq)

        # Add padding to probed sequence
        padded_seq = "GGGAAA" + seq + "AAAGGG"
        padded_reactivity = np.arange(len(padded_seq), dtype=np.float32)

        index = ReactivityIndex()
        index.add("padded", padded_seq, padded_reactivity)

        result = index.match(polymer)

        assert result is not None
        assert result.name == "padded"
        assert result.span == 1.0
        assert result.offset == 6  # Length of "GGGAAA"
        assert result.reactivity.shape == (n_res,)
        # Reactivity should be sliced from padded array
        np.testing.assert_array_equal(
            result.reactivity, padded_reactivity[6 : 6 + n_res]
        )

    def test_probed_substring_of_structure(self, polymer):
        """Probed sequence is substring of structure sequence."""
        seq = polymer.sequence_str().upper()
        n_res = len(seq)

        # Use middle portion of structure sequence
        start = 5
        length = min(10, n_res - 10)
        if length <= 0:
            pytest.skip("Structure too short for this test")

        partial_seq = seq[start : start + length]
        partial_reactivity = np.arange(length, dtype=np.float32) + 100

        index = ReactivityIndex()
        index.add("partial", partial_seq, partial_reactivity)

        result = index.match(polymer)

        assert result is not None
        assert result.name == "partial"
        assert result.offset == start
        assert result.span == length / n_res
        assert result.reactivity.shape == (n_res,)

        # Check NaN padding
        assert np.isnan(result.reactivity[:start]).all()
        assert np.isnan(result.reactivity[start + length :]).all()

        # Check actual values
        np.testing.assert_array_equal(
            result.reactivity[start : start + length], partial_reactivity
        )

    def test_no_match_returns_none(self, polymer):
        """No matching sequence returns None."""
        index = ReactivityIndex()
        index.add("unrelated", "XXXXXX", np.zeros(6))

        result = index.match(polymer)

        assert result is None

    def test_multiple_matches_returns_none(self, polymer):
        """Multiple matches returns None."""
        seq = polymer.sequence_str().upper()
        n_res = len(seq)
        reactivity = np.zeros(n_res)

        index = ReactivityIndex()
        # Add same sequence twice with different names
        index.add("match1", seq, reactivity)
        index.add("match2", seq, reactivity + 1)

        result = index.match(polymer)

        assert result is None

    def test_match_all_returns_all_matches(self, polymer):
        """match_all() returns all matching entries."""
        seq = polymer.sequence_str().upper()
        n_res = len(seq)

        index = ReactivityIndex()
        index.add("match1", seq, np.zeros(n_res))
        index.add("match2", seq, np.ones(n_res))
        index.add("nomatch", "XXXXXX", np.zeros(6))

        results = index.match_all(polymer)

        assert len(results) == 2
        names = {r.name for r in results}
        assert names == {"match1", "match2"}


class TestMultiChannel:
    """Test multi-channel reactivity data (e.g., SHAPE + DMS)."""

    def test_multichannel_exact_match(self, polymer):
        """Multi-channel data is handled correctly."""
        seq = polymer.sequence_str().upper()
        n_res = len(seq)

        # 2-channel data (e.g., SHAPE and DMS)
        reactivity = np.random.rand(n_res, 2).astype(np.float32)

        index = ReactivityIndex()
        index.add("multichannel", seq, reactivity)

        result = index.match(polymer)

        assert result is not None
        assert result.reactivity.shape == (n_res, 2)
        np.testing.assert_array_equal(result.reactivity, reactivity)

    def test_multichannel_with_padding(self, polymer):
        """Multi-channel data with NaN padding."""
        seq = polymer.sequence_str().upper()
        n_res = len(seq)

        # Use partial sequence
        start = 3
        length = min(8, n_res - 6)
        if length <= 0:
            pytest.skip("Structure too short for this test")

        partial_seq = seq[start : start + length]
        partial_reactivity = np.random.rand(length, 2).astype(np.float32)

        index = ReactivityIndex()
        index.add("partial_multi", partial_seq, partial_reactivity)

        result = index.match(polymer)

        assert result is not None
        assert result.reactivity.shape == (n_res, 2)

        # Check NaN padding for both channels
        assert np.isnan(result.reactivity[:start]).all()
        assert np.isnan(result.reactivity[start + length :]).all()

        # Check actual values
        np.testing.assert_array_equal(
            result.reactivity[start : start + length], partial_reactivity
        )


class TestAnnotateIntegration:
    """Test integration with polymer.annotate()."""

    def test_annotate_from_match(self, polymer):
        """Can annotate polymer with matched reactivity."""
        seq = polymer.sequence_str().upper()
        n_res = len(seq)
        reactivity = np.random.rand(n_res).astype(np.float32)

        index = ReactivityIndex()
        index.add("test", seq, reactivity)

        result = index.match(polymer)
        assert result is not None

        polymer.annotate("shape", result.reactivity)

        assert hasattr(polymer, "shape")
        np.testing.assert_array_equal(polymer.shape, reactivity)

    def test_annotate_propagates_through_selection(self, polymer):
        """Annotated reactivity propagates through residue selection."""
        seq = polymer.sequence_str().upper()
        n_res = len(seq)
        reactivity = np.arange(n_res, dtype=np.float32)

        index = ReactivityIndex()
        index.add("test", seq, reactivity)

        result = index.match(polymer)
        polymer.annotate("shape", result.reactivity)

        # Select subset
        selected = polymer.residue(slice(5, 10))

        assert selected.size(Scale.RESIDUE) == 5
        np.testing.assert_array_equal(selected.shape, reactivity[5:10])


class TestBackendConversion:
    """Test that reactivity matches polymer backend."""

    def test_numpy_polymer_returns_numpy(self, polymer):
        """Numpy polymer returns numpy reactivity."""
        seq = polymer.sequence_str().upper()
        n_res = len(seq)
        reactivity = np.random.rand(n_res).astype(np.float32)

        index = ReactivityIndex()
        index.add("test", seq, reactivity)

        result = index.match(polymer)
        assert result is not None
        assert isinstance(result.reactivity, np.ndarray)

    def test_torch_polymer_returns_torch(self, polymer):
        """Torch polymer returns torch reactivity."""
        torch = pytest.importorskip("torch")

        seq = polymer.sequence_str().upper()
        n_res = len(seq)
        reactivity = np.random.rand(n_res).astype(np.float32)

        index = ReactivityIndex()
        index.add("test", seq, reactivity)

        # Convert polymer to torch
        torch_polymer = polymer.torch()
        result = index.match(torch_polymer)

        assert result is not None
        assert isinstance(result.reactivity, torch.Tensor)

        # Should be directly usable with annotate
        torch_polymer.annotate("shape", result.reactivity)
        assert isinstance(torch_polymer.shape, torch.Tensor)

    def test_gpu_polymer_returns_gpu_tensor(self, polymer):
        """GPU polymer returns GPU tensor reactivity."""
        torch = pytest.importorskip("torch")

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        seq = polymer.sequence_str().upper()
        n_res = len(seq)
        reactivity = np.random.rand(n_res).astype(np.float32)

        index = ReactivityIndex()
        index.add("test", seq, reactivity)

        # Convert polymer to GPU
        gpu_polymer = polymer.torch().to("cuda")
        result = index.match(gpu_polymer)

        assert result is not None
        assert isinstance(result.reactivity, torch.Tensor)
        assert result.reactivity.device.type == "cuda"

        # Should be directly usable with annotate
        gpu_polymer.annotate("shape", result.reactivity)
        assert gpu_polymer.shape.device.type == "cuda"

    def test_match_all_converts_backend(self, polymer):
        """match_all() also converts to polymer backend."""
        torch = pytest.importorskip("torch")

        seq = polymer.sequence_str().upper()
        n_res = len(seq)

        index = ReactivityIndex()
        index.add("test1", seq, np.zeros(n_res, dtype=np.float32))
        index.add("test2", seq, np.ones(n_res, dtype=np.float32))

        torch_polymer = polymer.torch()
        results = index.match_all(torch_polymer)

        assert len(results) == 2
        for result in results:
            assert isinstance(result.reactivity, torch.Tensor)
