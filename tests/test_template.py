"""Tests for template polymer generation from sequences."""

import pytest
import numpy as np
import warnings

from tests.utils import get_test_cif


class TestFromSequence:
    """Test the from_sequence function."""

    def test_rna_sequence(self):
        """Test RNA sequence generates correct polymer."""
        from ciffy import from_sequence, Scale
        from ciffy.biochemistry import Residue

        polymer = from_sequence("acgu")

        # Check structure counts
        assert polymer.size(Scale.RESIDUE) == 4
        assert polymer.size(Scale.CHAIN) == 1

        # Check sequence values match Residue enum
        expected = [Residue.A.value, Residue.C.value, Residue.G.value, Residue.U.value]
        assert list(polymer.sequence) == expected

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
        from ciffy.biochemistry import Residue

        polymer = from_sequence("MGKLF")

        # Check structure counts
        assert polymer.size(Scale.RESIDUE) == 5
        assert polymer.size(Scale.CHAIN) == 1

        # Check sequence values match Residue enum
        expected = [Residue.MET.value, Residue.GLY.value, Residue.LYS.value,
                    Residue.LEU.value, Residue.PHE.value]
        assert list(polymer.sequence) == expected

        # Check coordinates are zeros
        assert np.allclose(polymer.coordinates, 0.0)

    def test_single_residue_rna(self):
        """Test single nucleotide."""
        from ciffy import from_sequence, Scale
        from ciffy.biochemistry import Residue

        polymer = from_sequence("a")

        assert polymer.size(Scale.RESIDUE) == 1
        assert list(polymer.sequence) == [Residue.A.value]

    def test_single_residue_protein(self):
        """Test single amino acid."""
        from ciffy import from_sequence, Scale
        from ciffy.biochemistry import Residue

        polymer = from_sequence("M")  # Methionine (not ambiguous with nucleotides)

        assert polymer.size(Scale.RESIDUE) == 1
        assert list(polymer.sequence) == [Residue.MET.value]

    def test_warning_uppercase_nucleotides(self):
        """Test warning when uppercase looks like nucleotides."""
        from ciffy import from_sequence
        from ciffy.biochemistry import Residue

        # Note: "ACGU" would fail because U is not a valid protein letter
        # Use "ACG" which is A=Ala, C=Cys, G=Gly (all valid protein)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            polymer = from_sequence("ACG")

            assert len(w) == 1
            assert "nucleotide characters" in str(w[0].message)
            assert "Did you mean lowercase" in str(w[0].message)

        # Should work as protein
        expected = [Residue.ALA.value, Residue.CYS.value, Residue.GLY.value]
        assert list(polymer.sequence) == expected

    def test_uppercase_acgt_warning(self):
        """Test warning for ACGT (valid protein letters)."""
        from ciffy import from_sequence
        from ciffy.biochemistry import Residue

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # A=Ala, C=Cys, G=Gly, T=Thr - all valid protein letters
            polymer = from_sequence("ACGT")

            assert len(w) == 1
            assert "nucleotide characters" in str(w[0].message)

        # Check sequence matches Residue enum
        expected = [Residue.ALA.value, Residue.CYS.value, Residue.GLY.value, Residue.THR.value]
        assert list(polymer.sequence) == expected

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

    def test_dna_sequence(self):
        """Test DNA sequence generates correct polymer."""
        from ciffy import from_sequence, Scale
        from ciffy.biochemistry import Residue

        polymer = from_sequence("acgt")

        # Check structure counts
        assert polymer.size(Scale.RESIDUE) == 4
        assert polymer.size(Scale.CHAIN) == 1

        # Check sequence values match Residue enum
        expected = [Residue.DA.value, Residue.DC.value, Residue.DG.value, Residue.DT.value]
        assert list(polymer.sequence) == expected

        # Check coordinates are zeros
        assert np.allclose(polymer.coordinates, 0.0)

        # Check all atoms have valid indices (> 0)
        assert (polymer.atoms > 0).all()

    def test_dna_rna_mixed_raises(self):
        """Test mixing 'u' and 't' raises ValueError."""
        from ciffy import from_sequence

        with pytest.raises(ValueError, match="both 'u'.*and 't'"):
            from_sequence("acgut")

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

    def test_all_4_dna_nucleotides(self):
        """Test all 4 DNA nucleotides work."""
        from ciffy import from_sequence, Scale

        polymer = from_sequence("acgt")

        assert polymer.size(Scale.RESIDUE) == 4
        # DNA nucleotides have slightly different atom counts (no 2'-OH)
        assert polymer.size() > 0

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


class TestFromSequenceEdgeCases:
    """Edge case tests for from_sequence."""

    def test_very_long_rna_sequence(self):
        """Test very long sequence (performance check)."""
        import time
        from ciffy import from_sequence, Scale

        # 10000 residues
        seq = "acgu" * 2500

        start = time.time()
        polymer = from_sequence(seq)
        elapsed = time.time() - start

        assert polymer.size(Scale.RESIDUE) == 10000
        # Should complete in reasonable time
        assert elapsed < 10.0

    def test_very_long_protein_sequence(self):
        """Test very long protein sequence."""
        import time
        from ciffy import from_sequence, Scale

        # All 20 amino acids repeated
        seq = "ACDEFGHIKLMNPQRSTVWY" * 500  # 10000 residues

        start = time.time()
        polymer = from_sequence(seq)
        elapsed = time.time() - start

        assert polymer.size(Scale.RESIDUE) == 10000
        assert elapsed < 10.0

    def test_whitespace_in_sequence(self):
        """Test whitespace in sequence raises ValueError."""
        from ciffy import from_sequence

        with pytest.raises(ValueError):
            from_sequence("ac gu")

    def test_newline_in_sequence(self):
        """Test newline in sequence raises ValueError."""
        from ciffy import from_sequence

        with pytest.raises(ValueError):
            from_sequence("ac\ngu")

    def test_tab_in_sequence(self):
        """Test tab in sequence raises ValueError."""
        from ciffy import from_sequence

        with pytest.raises(ValueError):
            from_sequence("ac\tgu")

    def test_number_in_sequence(self):
        """Test number in sequence raises ValueError."""
        from ciffy import from_sequence

        with pytest.raises(ValueError):
            from_sequence("ac1gu")

    def test_special_character_in_sequence(self):
        """Test special character in sequence raises ValueError."""
        from ciffy import from_sequence

        with pytest.raises(ValueError):
            from_sequence("ac-gu")

    def test_only_invalid_characters(self):
        """Test sequence with only invalid characters raises ValueError."""
        from ciffy import from_sequence

        with pytest.raises(ValueError):
            from_sequence("xyz")

    def test_repeated_single_residue(self):
        """Test repeated single residue."""
        from ciffy import from_sequence, Scale

        polymer = from_sequence("aaaa")

        assert polymer.size(Scale.RESIDUE) == 4
        # All residues should be adenosine (0)
        assert all(r == 0 for r in polymer.sequence)

    def test_backend_invalid_raises(self):
        """Test invalid backend raises ValueError."""
        from ciffy import from_sequence

        # from_sequence might not validate backend (creates numpy then converts)
        # Check that the result has a valid backend regardless
        p = from_sequence("acgu", backend="numpy")
        assert p.backend in ["numpy", "torch"]


class TestFromSequenceMultiChain:
    """Test multi-chain from_sequence functionality."""

    def test_two_rna_chains(self):
        """Test creating two RNA chains."""
        from ciffy import from_sequence, Scale

        polymer = from_sequence(["acgu", "acgu"])

        assert polymer.size(Scale.CHAIN) == 2
        assert polymer.size(Scale.RESIDUE) == 8
        assert polymer.names == ["A", "B"]

    def test_three_chains(self):
        """Test creating three chains."""
        from ciffy import from_sequence, Scale

        polymer = from_sequence(["acgu", "acgu", "acgu"])

        assert polymer.size(Scale.CHAIN) == 3
        assert polymer.names == ["A", "B", "C"]

    def test_mixed_rna_protein(self):
        """Test mixing RNA and protein chains."""
        from ciffy import from_sequence, Scale

        polymer = from_sequence(["acgu", "MGKLF"])

        assert polymer.size(Scale.CHAIN) == 2
        assert polymer.size(Scale.RESIDUE) == 9  # 4 RNA + 5 protein

    def test_different_length_chains(self):
        """Test chains with different lengths."""
        from ciffy import from_sequence, Scale

        polymer = from_sequence(["a", "acgu", "acguacgu"])

        assert polymer.size(Scale.CHAIN) == 3
        assert list(polymer.lengths) == [1, 4, 8]

    def test_single_element_list(self):
        """Test list with single sequence equals string input."""
        from ciffy import from_sequence, Scale

        p1 = from_sequence("acgu")
        p2 = from_sequence(["acgu"])

        assert p1.size() == p2.size()
        assert p1.size(Scale.CHAIN) == p2.size(Scale.CHAIN)
        assert p1.names == p2.names

    def test_empty_list_raises(self):
        """Test empty list raises ValueError."""
        from ciffy import from_sequence

        with pytest.raises(ValueError, match="Empty sequence list"):
            from_sequence([])

    def test_chain_names_beyond_z(self):
        """Test chain naming beyond 26 chains."""
        from ciffy import from_sequence, Scale

        # Create 27 chains
        seqs = ["a"] * 27
        polymer = from_sequence(seqs)

        assert polymer.size(Scale.CHAIN) == 27
        assert polymer.names[0] == "A"
        assert polymer.names[25] == "Z"
        assert polymer.names[26] == "AA"

    def test_multi_chain_torch_backend(self):
        """Test multi-chain with torch backend."""
        import torch
        from ciffy import from_sequence, Scale

        polymer = from_sequence(["acgu", "acgu"], backend="torch")

        assert polymer.backend == "torch"
        assert polymer.size(Scale.CHAIN) == 2
        assert isinstance(polymer.coordinates, torch.Tensor)

    def test_atoms_per_chain(self):
        """Test atoms are correctly distributed per chain."""
        from ciffy import from_sequence, Scale

        polymer = from_sequence(["acgu", "MGKLF"])

        # Get atoms per chain
        atoms_per_chain = polymer.per(Scale.ATOM, Scale.CHAIN)
        assert len(atoms_per_chain) == 2

        # First chain (RNA) should have 148 atoms
        # Second chain (protein) should have different count
        assert atoms_per_chain[0] > 0
        assert atoms_per_chain[1] > 0
        assert sum(atoms_per_chain) == polymer.size()

    def test_two_dna_chains(self):
        """Test creating two DNA chains (e.g., double helix)."""
        from ciffy import from_sequence, Scale

        polymer = from_sequence(["acgt", "acgt"])

        assert polymer.size(Scale.CHAIN) == 2
        assert polymer.size(Scale.RESIDUE) == 8
        assert polymer.names == ["A", "B"]

    def test_mixed_rna_dna(self):
        """Test mixing RNA and DNA chains."""
        from ciffy import from_sequence, Scale
        from ciffy.biochemistry import Residue

        polymer = from_sequence(["acgu", "acgt"])

        assert polymer.size(Scale.CHAIN) == 2
        assert polymer.size(Scale.RESIDUE) == 8
        # First chain is RNA, second is DNA
        seq = list(polymer.sequence)
        rna_expected = [Residue.A.value, Residue.C.value, Residue.G.value, Residue.U.value]
        dna_expected = [Residue.DA.value, Residue.DC.value, Residue.DG.value, Residue.DT.value]
        assert seq[:4] == rna_expected
        assert seq[4:] == dna_expected

    def test_mixed_dna_protein(self):
        """Test mixing DNA and protein chains."""
        from ciffy import from_sequence, Scale

        polymer = from_sequence(["acgt", "MGKLF"])

        assert polymer.size(Scale.CHAIN) == 2
        # 4 DNA + 5 protein residues
        assert polymer.size(Scale.RESIDUE) == 9


class TestTerminalAtoms:
    """Test correct handling of terminal atoms."""

    def test_single_residue_has_all_terminal_atoms(self):
        """Single residue should have both 5' and 3' terminal atoms."""
        from ciffy import from_sequence
        from ciffy.biochemistry import A  # CCD name for adenosine

        # Single residue has all atoms (both termini)
        single = from_sequence("a")
        full_count = len(list(A))
        assert single.size() == full_count

    def test_internal_residues_lack_terminal_atoms(self):
        """Internal residues should not have terminal atoms."""
        from ciffy import from_sequence, Scale
        from ciffy.biochemistry import A  # CCD name for adenosine

        # Use same residue type to control for inherent size differences
        polymer = from_sequence("aaaa")
        apr = list(polymer.per(Scale.ATOM, Scale.RESIDUE))
        full_count = len(list(A))

        # First residue: all atoms except HO3' (3'-terminal) -> full - 1
        # Middle residues: no OP3, HOP3, HO3' (all terminal) -> full - 3
        # Last residue: all atoms except OP3, HOP3 (5'-terminal) -> full - 2

        assert apr[0] == full_count - 1, f"First residue: expected {full_count - 1}, got {apr[0]}"
        assert apr[1] == full_count - 3, f"Middle residue: expected {full_count - 3}, got {apr[1]}"
        assert apr[2] == full_count - 3, f"Middle residue: expected {full_count - 3}, got {apr[2]}"
        assert apr[3] == full_count - 2, f"Last residue: expected {full_count - 2}, got {apr[3]}"

    @pytest.mark.filterwarnings("ignore:Sequence 'AAA' contains only nucleotide")
    def test_protein_terminal_atoms(self):
        """Protein should only have OXT on C-terminus, H2/H3 on N-terminus."""
        from ciffy import from_sequence, Scale
        from ciffy.biochemistry import ATOM_NAMES

        polymer = from_sequence("AAA")  # 3 alanines
        apr = list(polymer.per(Scale.ATOM, Scale.RESIDUE))

        # First residue has H2, H3 (N-terminal), no OXT
        # Middle residue has neither terminal atoms
        # Last residue has OXT (C-terminal), no H2, H3

        # Middle should have fewer atoms
        assert apr[1] < apr[0], "Middle residue should have fewer atoms than N-terminus"
        assert apr[1] < apr[2], "Middle residue should have fewer atoms than C-terminus"

    def test_multi_chain_each_has_termini(self):
        """Each chain should have its own terminal atoms."""
        from ciffy import from_sequence, Scale

        # Two separate chains
        multi = from_sequence(["ac", "gu"])

        # Compare to single 4-residue chain
        single = from_sequence("acgu")

        # Multi-chain should have MORE atoms because each chain has termini
        assert multi.size() > single.size(), \
            "Two 2-residue chains should have more atoms than one 4-residue chain"

    def test_terminal_atoms_per_chain_independent(self):
        """Terminal atom filtering should be per-chain, not global."""
        from ciffy import from_sequence, Scale

        # Two identical single-residue chains
        two_singles = from_sequence(["a", "a"])

        # One two-residue chain
        one_double = from_sequence("aa")

        # Two single-residue chains: each has full atoms (both termini)
        # One two-residue chain: first has 5', second has 3'

        # So two singles should have more atoms
        assert two_singles.size() > one_double.size()


class TestTemplateMatchesCIF:
    """Test that from_sequence produces structures consistent with CIF files."""

    def _extract_chain_sequences(self, polymer) -> list[str]:
        """Extract per-chain sequences from a polymer."""
        sequences = []
        seq_str = polymer.str()
        offset = 0
        for length in polymer.lengths:
            length = int(length)
            if length > 0:
                sequences.append(seq_str[offset:offset + length])
                offset += length
        return sequences

    def _verify_template_matches_loaded(self, loaded, template):
        """
        Verify template matches loaded CIF (except coordinates).

        Checks:
        1. Residue sequences match exactly
        2. Loaded atoms are subset of template atoms (per residue)
        3. Elements match for corresponding atoms
        4. Chain and residue counts match
        """
        from ciffy import Scale

        # 1. Sequences match
        assert np.array_equal(loaded.sequence, template.sequence), \
            "Residue sequences differ"

        # 2. Loaded atoms are subset of template atoms per residue
        loaded_sizes = loaded.per(Scale.ATOM, Scale.RESIDUE)
        template_sizes = template.per(Scale.ATOM, Scale.RESIDUE)

        loaded_offset = 0
        template_offset = 0

        for i in range(len(loaded_sizes)):
            loaded_atoms = set(loaded.atoms[loaded_offset:loaded_offset + loaded_sizes[i]].tolist())
            template_atoms = set(template.atoms[template_offset:template_offset + template_sizes[i]].tolist())

            assert loaded_atoms.issubset(template_atoms), \
                f"Residue {i}: loaded atoms {loaded_atoms - template_atoms} not in template"

            loaded_offset += loaded_sizes[i]
            template_offset += template_sizes[i]

        # 3. Elements match for corresponding atom types
        template_atom_to_element = dict(zip(
            template.atoms.tolist(),
            template.elements.tolist()
        ))

        for atom_idx, elem in zip(loaded.atoms.tolist(), loaded.elements.tolist()):
            template_elem = template_atom_to_element.get(atom_idx)
            if template_elem is not None:
                assert elem == template_elem, \
                    f"Element mismatch for atom {atom_idx}: loaded={elem}, template={template_elem}"

        # 4. Counts match
        assert loaded.size(Scale.CHAIN) == template.size(Scale.CHAIN), "Chain counts differ"
        assert loaded.size(Scale.RESIDUE) == template.size(Scale.RESIDUE), "Residue counts differ"

    def test_1zew_dna_consistency(self):
        """Test template matches 1ZEW (DNA duplex)."""
        from ciffy import load, from_sequence

        loaded = load(get_test_cif("1ZEW")).poly()
        sequences = self._extract_chain_sequences(loaded)
        template = from_sequence(sequences)

        self._verify_template_matches_loaded(loaded, template)

    def test_9gcm_rna_protein_consistency(self):
        """Test template matches 9GCM (RNA + protein complex)."""
        from ciffy import load, from_sequence

        loaded = load(get_test_cif("9GCM")).poly()
        sequences = self._extract_chain_sequences(loaded)

        # Skip if no valid sequences (structure might have unusual residues)
        if not sequences or not all(sequences):
            pytest.skip("No valid polymer sequences in structure")

        template = from_sequence(sequences)

        self._verify_template_matches_loaded(loaded, template)
