"""Tests for RNA secondary structure utilities."""

import numpy as np
import pytest

from ciffy.rna import dotbracket_to_pairs, pairs_to_dotbracket


class TestDotbracketToPairs:
    """Test dotbracket_to_pairs function."""

    def test_simple_hairpin(self):
        pairs = dotbracket_to_pairs("((...))")
        expected = np.array([[0, 6], [1, 5]])
        np.testing.assert_array_equal(pairs, expected)

    def test_nested_structure(self):
        pairs = dotbracket_to_pairs("(((..)))")
        expected = np.array([[0, 7], [1, 6], [2, 5]])
        np.testing.assert_array_equal(pairs, expected)

    def test_multiple_stems(self):
        pairs = dotbracket_to_pairs("((.)).((..))")
        expected = np.array([[0, 4], [1, 3], [6, 11], [7, 10]])
        np.testing.assert_array_equal(pairs, expected)

    def test_empty_structure(self):
        pairs = dotbracket_to_pairs(".....")
        assert pairs.shape == (0, 2)

    def test_pseudoknot_square(self):
        pairs = dotbracket_to_pairs("([)]")
        # ( at 0 pairs with ) at 2, [ at 1 pairs with ] at 3
        expected = np.array([[0, 2], [1, 3]])
        np.testing.assert_array_equal(pairs, expected)

    def test_pseudoknot_curly(self):
        pairs = dotbracket_to_pairs("({)}")
        expected = np.array([[0, 2], [1, 3]])
        np.testing.assert_array_equal(pairs, expected)

    def test_unbalanced_raises(self):
        with pytest.raises(ValueError, match="Unclosed"):
            dotbracket_to_pairs("((..)")

    def test_unclosed_raises(self):
        with pytest.raises(ValueError, match="Unclosed"):
            dotbracket_to_pairs("((..))(")

    def test_invalid_char_raises(self):
        with pytest.raises(ValueError, match="Invalid character"):
            dotbracket_to_pairs("((.x.))")


class TestPairsToDotbracket:
    """Test pairs_to_dotbracket function."""

    def test_simple_hairpin(self):
        pairs = np.array([[0, 6], [1, 5]])
        result = pairs_to_dotbracket(pairs, 7)
        assert result == "((...))"

    def test_nested_structure(self):
        pairs = np.array([[0, 7], [1, 6], [2, 5]])
        result = pairs_to_dotbracket(pairs, 8)
        assert result == "(((..)))"

    def test_empty_pairs(self):
        pairs = np.empty((0, 2), dtype=np.int64)
        result = pairs_to_dotbracket(pairs, 5)
        assert result == "....."

    def test_pseudoknot_detected(self):
        # Crossing pairs: a < c < b < d
        # (0, 4) and (2, 5): 0 < 2 < 4 < 5 -> crosses!
        pairs = np.array([[0, 4], [2, 5]])
        result = pairs_to_dotbracket(pairs, 6)
        # Should use different bracket types
        assert "(" in result and "[" in result

    def test_invalid_pair_order_raises(self):
        pairs = np.array([[5, 0]])  # i > j
        with pytest.raises(ValueError, match="i < j"):
            pairs_to_dotbracket(pairs, 6)

    def test_out_of_bounds_raises(self):
        pairs = np.array([[0, 10]])
        with pytest.raises(ValueError, match="must be in"):
            pairs_to_dotbracket(pairs, 5)


class TestRoundTrip:
    """Test round-trip conversion."""

    @pytest.mark.parametrize("dotbracket", [
        "((...))",
        "(((..)))",
        "...((.))...",
        "((.)).((..))",
        ".....",
        "",
    ])
    def test_round_trip(self, dotbracket):
        pairs = dotbracket_to_pairs(dotbracket)
        result = pairs_to_dotbracket(pairs, len(dotbracket))
        assert result == dotbracket

    @pytest.mark.parametrize("dotbracket", [
        "([)]",
        "(([.)])",
        "((.[.)])",
    ])
    def test_round_trip_pseudoknot(self, dotbracket):
        pairs = dotbracket_to_pairs(dotbracket)
        result = pairs_to_dotbracket(pairs, len(dotbracket))
        assert result == dotbracket
