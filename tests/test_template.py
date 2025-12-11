"""Tests for template polymer generation from sequences."""

import pytest
import numpy as np
import warnings


class TestFromSequence:
    """Test the from_sequence function."""

    def test_rna_sequence(self):
        """Test RNA sequence generates correct polymer."""
        from ciffy import from_sequence, Scale

        polymer = from_sequence("acgu")

        # Check structure counts
        assert polymer.size(Scale.RESIDUE) == 4
        assert polymer.size(Scale.CHAIN) == 1

        # Check sequence values (A=0, C=1, G=2, U=3)
        assert list(polymer.sequence) == [0, 1, 2, 3]

        # Check coordinates are zeros
        assert np.allclose(polymer.coordinates, 0.0)

        # Check all atoms have valid indices (> 0)
        assert (polymer.atoms > 0).all()

        # Check elements are valid (H=1, C=6, N=7, O=8, P=15)
        valid_elements = {1, 6, 7, 8, 15}
        unique_elements = set(polymer.elements.tolist())
        assert unique_elements.issubset(valid_elements)

    def test_protein_sequence(self):
        """Test protein sequence generates correct polymer."""
        from ciffy import from_sequence, Scale

        polymer = from_sequence("MGKLF")

        # Check structure counts
        assert polymer.size(Scale.RESIDUE) == 5
        assert polymer.size(Scale.CHAIN) == 1

        # Check sequence values (M=15, G=10, K=13, L=14, F=9)
        assert list(polymer.sequence) == [15, 10, 13, 14, 9]

        # Check coordinates are zeros
        assert np.allclose(polymer.coordinates, 0.0)

    def test_single_residue_rna(self):
        """Test single nucleotide."""
        from ciffy import from_sequence, Scale

        polymer = from_sequence("a")

        assert polymer.size(Scale.RESIDUE) == 1
        assert list(polymer.sequence) == [0]  # Adenosine

    def test_single_residue_protein(self):
        """Test single amino acid."""
        from ciffy import from_sequence, Scale

        polymer = from_sequence("M")  # Methionine (not ambiguous with nucleotides)

        assert polymer.size(Scale.RESIDUE) == 1
        assert list(polymer.sequence) == [15]  # MET

    def test_warning_uppercase_nucleotides(self):
        """Test warning when uppercase looks like nucleotides."""
        from ciffy import from_sequence

        # Note: "ACGU" would fail because U is not a valid protein letter
        # Use "ACG" which is A=Ala, C=Cys, G=Gly (all valid protein)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            polymer = from_sequence("ACG")

            assert len(w) == 1
            assert "nucleotide characters" in str(w[0].message)
            assert "Did you mean lowercase" in str(w[0].message)

        # Should work as protein (A=Ala, C=Cys, G=Gly)
        assert list(polymer.sequence) == [5, 6, 10]

    def test_uppercase_acgt_warning(self):
        """Test warning for ACGT (valid protein letters)."""
        from ciffy import from_sequence

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # A=Ala, C=Cys, G=Gly, T=Thr - all valid protein letters
            polymer = from_sequence("ACGT")

            assert len(w) == 1
            assert "nucleotide characters" in str(w[0].message)

        # Sequence values: A=5, C=6, G=10, T=21
        assert list(polymer.sequence) == [5, 6, 10, 21]

    def test_invalid_character_rna(self):
        """Test invalid character in RNA sequence raises ValueError."""
        from ciffy import from_sequence

        with pytest.raises(ValueError, match="Unknown RNA residue 'x'"):
            from_sequence("acgx")

    def test_invalid_character_protein(self):
        """Test invalid character in protein sequence raises ValueError."""
        from ciffy import from_sequence

        with pytest.raises(ValueError, match="Unknown protein residue 'X'"):
            from_sequence("MGXLF")

    def test_dna_not_supported(self):
        """Test DNA (thymine) raises ValueError."""
        from ciffy import from_sequence

        with pytest.raises(ValueError, match="DNA.*not supported"):
            from_sequence("acgt")

    def test_empty_sequence(self):
        """Test empty sequence raises ValueError."""
        from ciffy import from_sequence

        with pytest.raises(ValueError, match="Empty sequence"):
            from_sequence("")

    def test_mixed_case(self):
        """Test mixed case raises ValueError."""
        from ciffy import from_sequence

        with pytest.raises(ValueError, match="Mixed case not supported"):
            from_sequence("AcGu")

    def test_backend_numpy(self):
        """Test numpy backend."""
        from ciffy import from_sequence

        polymer = from_sequence("acgu", backend="numpy")

        assert isinstance(polymer.coordinates, np.ndarray)
        assert polymer.backend == "numpy"

    def test_backend_torch(self):
        """Test torch backend."""
        import torch
        from ciffy import from_sequence

        polymer = from_sequence("acgu", backend="torch")

        assert isinstance(polymer.coordinates, torch.Tensor)
        assert polymer.backend == "torch"

    def test_custom_id(self):
        """Test custom ID."""
        from ciffy import from_sequence

        polymer = from_sequence("acgu", id="my_rna")

        assert polymer.id() == "my_rna"

    def test_default_id(self):
        """Test default ID is 'template'."""
        from ciffy import from_sequence

        polymer = from_sequence("acgu")

        assert polymer.id() == "template"

    def test_chain_name(self):
        """Test chain name is 'A'."""
        from ciffy import from_sequence

        polymer = from_sequence("acgu")

        assert polymer.names == ["A"]

    def test_atoms_per_residue_vary(self):
        """Test atoms per residue varies by residue type."""
        from ciffy import from_sequence, Scale

        # Different residues have different numbers of atoms
        polymer = from_sequence("acgu")

        atoms_per_res = polymer.per(Scale.ATOM, Scale.RESIDUE)
        # Each nucleotide should have atoms (varies by type)
        assert len(atoms_per_res) == 4
        assert all(count > 0 for count in atoms_per_res)

    def test_polymer_count_equals_total(self):
        """Test all atoms are polymer atoms (no HETATM)."""
        from ciffy import from_sequence

        polymer = from_sequence("acgu")

        # polymer_count should equal total atoms
        assert polymer.polymer_count == polymer.size()

    def test_all_20_amino_acids(self):
        """Test all 20 standard amino acids work."""
        from ciffy import from_sequence, Scale

        # All 20 standard amino acid one-letter codes
        all_aa = "ACDEFGHIKLMNPQRSTVWY"

        polymer = from_sequence(all_aa)

        assert polymer.size(Scale.RESIDUE) == 20
        # Each should have at least backbone atoms
        assert polymer.size() > 20 * 4  # At least N, CA, C, O per residue

    def test_all_4_rna_nucleotides(self):
        """Test all 4 RNA nucleotides work."""
        from ciffy import from_sequence, Scale

        polymer = from_sequence("acgu")

        assert polymer.size(Scale.RESIDUE) == 4

    def test_elements_are_atomic_numbers(self):
        """Test elements are atomic numbers."""
        from ciffy import from_sequence

        polymer = from_sequence("M")  # Methionine - has S for variety

        # Methionine has: N, CA, C, O, CB, CG, SD, CE, plus hydrogens
        # N=7, C=6, O=8, S=16, H=1
        elements = set(polymer.elements.tolist())
        expected = {1, 6, 7, 8, 16}  # H, C, N, O, S
        assert elements == expected


class TestFromSequenceIntegration:
    """Integration tests for template polymers."""

    def test_can_write_to_cif(self, tmp_path):
        """Test template polymer can be written to CIF."""
        from ciffy import from_sequence

        polymer = from_sequence("acgu", id="test_rna")
        output_path = tmp_path / "test.cif"

        # Should not raise
        polymer.write(str(output_path))

        # File should exist
        assert output_path.exists()

    def test_can_attach_coordinates(self):
        """Test coordinates can be replaced."""
        from ciffy import from_sequence

        polymer = from_sequence("acgu")
        n_atoms = polymer.size()

        # Generate random coordinates
        new_coords = np.random.randn(n_atoms, 3).astype(np.float32)

        # Attach coordinates
        polymer.coordinates = new_coords

        # Verify
        assert np.allclose(polymer.coordinates, new_coords)

    def test_round_trip_with_coordinates(self, tmp_path):
        """Test write and reload with attached coordinates."""
        from ciffy import from_sequence, load

        polymer = from_sequence("acgu", id="test_rna")
        n_atoms = polymer.size()

        # Attach non-zero coordinates
        polymer.coordinates = np.random.randn(n_atoms, 3).astype(np.float32) * 10

        # Write
        output_path = tmp_path / "test.cif"
        polymer.write(str(output_path))

        # Reload
        reloaded = load(str(output_path))

        # Check structure preserved
        assert reloaded.size() == polymer.size()
        assert np.allclose(reloaded.coordinates, polymer.coordinates, atol=0.001)
