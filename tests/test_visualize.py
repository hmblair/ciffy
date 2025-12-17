"""
Tests for ciffy.visualize module.

Tests defattr generation, plotting, and sequence alignment.
"""

import pytest
import numpy as np

import ciffy
from ciffy import Scale
from ciffy.visualize import to_defattr, plot_profile, contact_map, align_values
from ciffy.visualize.alignment import needleman_wunsch, _map_values_through_alignment


# =============================================================================
# Check for optional dependencies
# =============================================================================

try:
    import matplotlib
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# =============================================================================
# Test defattr generation
# =============================================================================

class TestDefattr:
    """Tests for ChimeraX defattr file generation."""

    def test_to_defattr_residue_basic(self, single_chain_polymer, tmp_path):
        """Test basic residue-level defattr generation."""
        polymer = single_chain_polymer
        n_res = polymer.size(Scale.RESIDUE)
        values = np.ones(n_res)

        output = tmp_path / "test.defattr"
        to_defattr(polymer, values, str(output), scale=Scale.RESIDUE)

        content = output.read_text()
        assert "attribute: value" in content
        assert "recipient: residues" in content
        # Check first residue present
        assert ":1\t" in content

    def test_to_defattr_atom_basic(self, single_chain_polymer, tmp_path):
        """Test basic atom-level defattr generation."""
        polymer = single_chain_polymer
        n_atoms = polymer.size()
        values = np.ones(n_atoms)

        output = tmp_path / "test.defattr"
        to_defattr(polymer, values, str(output), scale=Scale.ATOM)

        content = output.read_text()
        assert "attribute: value" in content
        assert "recipient: atoms" in content
        # Check atom specification format
        assert "@" in content  # Atom specifier

    def test_to_defattr_custom_attr_name(self, single_chain_polymer, tmp_path):
        """Test custom attribute name."""
        polymer = single_chain_polymer
        values = np.ones(polymer.size(Scale.RESIDUE))

        output = tmp_path / "test.defattr"
        to_defattr(polymer, values, str(output), attr_name="reactivity")

        content = output.read_text()
        assert "attribute: reactivity" in content

    def test_to_defattr_handles_nan(self, single_chain_polymer, tmp_path):
        """Test that NaN values are converted to 0."""
        polymer = single_chain_polymer
        values = np.array([1.0, np.nan, 2.0, 3.0])

        output = tmp_path / "test.defattr"
        to_defattr(polymer, values, str(output))

        content = output.read_text()
        # NaN should be converted to 0
        lines = [l for l in content.split('\n') if ':2\t' in l]
        assert len(lines) == 1
        assert '0.0' in lines[0] or '0\n' in lines[0]

    def test_to_defattr_wrong_size(self, single_chain_polymer, tmp_path):
        """Test error when values don't match polymer size."""
        polymer = single_chain_polymer
        wrong_size = np.ones(100)  # Wrong size

        output = tmp_path / "test.defattr"
        with pytest.raises(ValueError, match="must match"):
            to_defattr(polymer, wrong_size, str(output))

    def test_to_defattr_invalid_scale(self, single_chain_polymer, tmp_path):
        """Test error for invalid scale."""
        polymer = single_chain_polymer
        values = np.ones(polymer.size(Scale.CHAIN))

        output = tmp_path / "test.defattr"
        with pytest.raises(ValueError, match="RESIDUE or ATOM"):
            to_defattr(polymer, values, str(output), scale=Scale.CHAIN)


# =============================================================================
# Test matplotlib plots
# =============================================================================

class TestPlots:
    """Tests for matplotlib plotting functions."""

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
    def test_plot_profile_basic(self, single_chain_polymer):
        """Test basic profile plot creation."""
        import matplotlib.pyplot as plt

        polymer = single_chain_polymer
        values = np.random.rand(polymer.size(Scale.RESIDUE))

        ax = plot_profile(polymer, values)
        assert ax is not None

        plt.close('all')

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
    def test_plot_profile_with_ax(self, single_chain_polymer):
        """Test profile plot on existing axes."""
        import matplotlib.pyplot as plt

        polymer = single_chain_polymer
        values = np.random.rand(polymer.size(Scale.RESIDUE))

        fig, ax = plt.subplots()
        result_ax = plot_profile(polymer, values, ax=ax)
        assert result_ax is ax

        plt.close('all')

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
    def test_plot_profile_no_fill(self, single_chain_polymer):
        """Test profile plot without fill."""
        import matplotlib.pyplot as plt

        polymer = single_chain_polymer
        values = np.random.rand(polymer.size(Scale.RESIDUE))

        ax = plot_profile(polymer, values, fill=False)
        assert ax is not None

        plt.close('all')

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
    def test_plot_profile_custom_labels(self, single_chain_polymer):
        """Test profile plot with custom labels."""
        import matplotlib.pyplot as plt

        polymer = single_chain_polymer
        values = np.random.rand(polymer.size(Scale.RESIDUE))

        ax = plot_profile(
            polymer, values,
            xlabel="Position",
            ylabel="Reactivity",
            title="Test Profile"
        )
        assert ax.get_xlabel() == "Position"
        assert ax.get_ylabel() == "Reactivity"
        assert ax.get_title() == "Test Profile"

        plt.close('all')

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
    def test_plot_profile_wrong_size(self, single_chain_polymer):
        """Test error when values don't match polymer size."""
        polymer = single_chain_polymer
        wrong_size = np.ones(100)

        with pytest.raises(ValueError, match="must match"):
            plot_profile(polymer, wrong_size)

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
    def test_contact_map_basic(self, single_chain_polymer):
        """Test basic contact map creation."""
        import matplotlib.pyplot as plt

        polymer = single_chain_polymer
        ax = contact_map(polymer, scale=Scale.RESIDUE)
        assert ax is not None

        plt.close('all')

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
    def test_contact_map_power_values(self, single_chain_polymer):
        """Test contact map with different power values."""
        import matplotlib.pyplot as plt

        polymer = single_chain_polymer

        for power in [1.0, 2.0, 6.0]:
            ax = contact_map(polymer, power=power)
            assert ax is not None
            plt.close('all')

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
    def test_contact_map_no_colorbar(self, single_chain_polymer):
        """Test contact map without colorbar."""
        import matplotlib.pyplot as plt

        polymer = single_chain_polymer
        ax = contact_map(polymer, colorbar=False)
        assert ax is not None

        plt.close('all')

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
    def test_contact_map_atom_scale(self, single_residue_polymer):
        """Test contact map at atom scale."""
        import matplotlib.pyplot as plt

        polymer = single_residue_polymer
        ax = contact_map(polymer, scale=Scale.ATOM)
        assert ax is not None

        plt.close('all')


# =============================================================================
# Test sequence alignment
# =============================================================================

class TestAlignment:
    """Tests for sequence alignment utilities."""

    def test_needleman_wunsch_identical(self):
        """Test alignment of identical sequences."""
        aln1, aln2 = needleman_wunsch("ACGU", "ACGU")
        assert aln1 == "ACGU"
        assert aln2 == "ACGU"

    def test_needleman_wunsch_insertion(self):
        """Test alignment with insertion."""
        aln1, aln2 = needleman_wunsch("ACGU", "ACU")
        # seq1 has extra G
        assert "-" in aln2 or "-" in aln1

    def test_needleman_wunsch_deletion(self):
        """Test alignment with deletion."""
        aln1, aln2 = needleman_wunsch("ACU", "ACGU")
        # seq2 has extra G
        assert "-" in aln1 or "-" in aln2

    def test_needleman_wunsch_empty(self):
        """Test alignment with empty sequence."""
        aln1, aln2 = needleman_wunsch("", "ACGU")
        assert aln1 == "----"
        assert aln2 == "ACGU"

    def test_map_values_identical(self):
        """Test mapping values through identical alignment."""
        values = np.array([1.0, 2.0, 3.0, 4.0])
        result = _map_values_through_alignment(values, "ACGU", "ACGU")
        np.testing.assert_array_equal(result, values)

    def test_map_values_with_gap_in_seq2(self):
        """Test mapping when seq2 has a gap."""
        values = np.array([1.0, 2.0, 3.0, 4.0])
        result = _map_values_through_alignment(values, "ACGU", "AC-U")
        # Gap in seq2 means position is skipped
        np.testing.assert_array_equal(result, [1.0, 2.0, 4.0])

    def test_map_values_with_gap_in_seq1(self):
        """Test mapping when seq1 has a gap."""
        values = np.array([1.0, 2.0, 3.0])
        result = _map_values_through_alignment(values, "AC-U", "ACGU", gap_value=0.0)
        # Gap in seq1 means we insert gap_value for that position in seq2
        np.testing.assert_array_equal(result, [1.0, 2.0, 0.0, 3.0])

    def test_align_values_identical(self, single_chain_polymer):
        """Test align_values with identical sequences."""
        polymer = single_chain_polymer
        seq = polymer.sequence_str()
        values = np.random.rand(len(seq))

        result = align_values(values, seq, polymer, chain=0)
        np.testing.assert_array_almost_equal(result, values)

    def test_align_values_case_insensitive(self, single_chain_polymer):
        """Test align_values is case insensitive."""
        polymer = single_chain_polymer
        seq = polymer.sequence_str().lower()
        values = np.random.rand(len(seq))

        result = align_values(values, seq, polymer, chain=0)
        np.testing.assert_array_almost_equal(result, values)

    def test_align_values_t_to_u(self, single_chain_polymer):
        """Test align_values converts T to U."""
        polymer = single_chain_polymer
        seq = polymer.sequence_str().replace('u', 't')
        values = np.random.rand(len(seq))

        result = align_values(values, seq, polymer, chain=0)
        np.testing.assert_array_almost_equal(result, values)

    def test_align_values_wrong_data_length(self, single_chain_polymer):
        """Test error when data doesn't match data_seq."""
        polymer = single_chain_polymer
        seq = polymer.sequence_str()
        wrong_values = np.random.rand(len(seq) + 5)

        with pytest.raises(ValueError, match="must match"):
            align_values(wrong_values, seq, polymer, chain=0)


# =============================================================================
# Integration tests
# =============================================================================

class TestIntegration:
    """Integration tests combining multiple components."""

    def test_defattr_round_trip(self, single_chain_polymer, tmp_path):
        """Test writing and reading back defattr values."""
        polymer = single_chain_polymer
        values = np.arange(polymer.size(Scale.RESIDUE), dtype=float)

        output = tmp_path / "test.defattr"
        to_defattr(polymer, values, str(output))

        # Parse the file and verify values
        content = output.read_text()
        lines = [l.strip() for l in content.split('\n') if '\t' in l]

        for i, line in enumerate(lines):
            parts = line.split('\t')
            assert len(parts) == 2
            parsed_value = float(parts[1])
            assert parsed_value == values[i]

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
    def test_profile_and_contact_same_polymer(self, single_chain_polymer):
        """Test that both plots work on the same polymer."""
        import matplotlib.pyplot as plt

        polymer = single_chain_polymer
        values = np.random.rand(polymer.size(Scale.RESIDUE))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        plot_profile(polymer, values, ax=ax1)
        contact_map(polymer, ax=ax2)

        plt.close('all')
