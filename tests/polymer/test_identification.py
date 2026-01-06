"""
Tests for Polymer identification methods: chain_id, istype.
"""

import pytest

from tests.utils import get_test_cif


class TestChainId:
    """Test chain_id() method."""

    def test_chain_id_format(self, backend):
        """chain_id returns PDB_chainname format."""
        from ciffy import load

        cif = get_test_cif("9GCM")
        polymer = load(cif, backend=backend)

        chain_id = polymer.chain_id(0)

        # Format should be PDBID_chainname
        assert "_" in chain_id
        assert chain_id.startswith(polymer.pdb_id)
        assert chain_id.endswith(polymer.names[0])

    def test_chain_id_first_chain(self, backend):
        """chain_id(0) returns correct ID for first chain."""
        from ciffy import load

        cif = get_test_cif("9GCM")
        polymer = load(cif, backend=backend)

        chain_id = polymer.chain_id(0)
        expected = f"{polymer.pdb_id}_{polymer.names[0]}"

        assert chain_id == expected

    def test_chain_id_last_chain(self, backend):
        """chain_id with last valid index works."""
        from ciffy import load, Scale

        cif = get_test_cif("9GCM")
        polymer = load(cif, backend=backend)
        last_idx = polymer.size(Scale.CHAIN) - 1

        chain_id = polymer.chain_id(last_idx)
        expected = f"{polymer.pdb_id}_{polymer.names[last_idx]}"

        assert chain_id == expected

    def test_chain_id_out_of_bounds_raises(self, backend):
        """chain_id with out-of-bounds index raises IndexError."""
        from ciffy import load, Scale

        cif = get_test_cif("9GCM")
        polymer = load(cif, backend=backend)
        invalid_idx = polymer.size(Scale.CHAIN) + 10

        with pytest.raises(IndexError):
            polymer.chain_id(invalid_idx)

    def test_chain_id_negative_index(self, backend):
        """chain_id with negative index uses Python semantics."""
        from ciffy import load

        cif = get_test_cif("9GCM")
        polymer = load(cif, backend=backend)

        # Negative indexing should work like Python lists
        chain_id = polymer.chain_id(-1)
        expected = f"{polymer.pdb_id}_{polymer.names[-1]}"

        assert chain_id == expected

    def test_chain_id_template_polymer(self, backend):
        """chain_id works on template-generated polymers."""
        from ciffy import from_sequence

        polymer = from_sequence("acgu", backend=backend)

        chain_id = polymer.chain_id(0)

        assert isinstance(chain_id, str)
        assert "_" in chain_id


class TestIstype:
    """Test istype() method."""

    def test_istype_single_rna_chain(self, backend):
        """istype returns True for single RNA chain."""
        from ciffy import load, Molecule, Scale

        cif = get_test_cif("9GCM")  # Has RNA and protein
        polymer = load(cif, backend=backend)

        # Get a single RNA chain
        rna = polymer.molecule_type(Molecule.RNA)
        if rna.size(Scale.CHAIN) > 0:
            single_rna = rna.chain(0)
            assert single_rna.istype(Molecule.RNA)

    def test_istype_single_protein_chain(self, backend):
        """istype returns True for single protein chain."""
        from ciffy import load, Molecule, Scale

        cif = get_test_cif("9GCM")  # Has RNA and protein
        polymer = load(cif, backend=backend)

        # Get a single protein chain
        protein = polymer.molecule_type(Molecule.PROTEIN)
        if protein.size(Scale.CHAIN) > 0:
            single_protein = protein.chain(0)
            assert single_protein.istype(Molecule.PROTEIN)

    def test_istype_returns_false_for_multi_chain(self, backend):
        """istype returns False for multi-chain polymers."""
        from ciffy import load, Molecule, Scale

        cif = get_test_cif("9GCM")
        polymer = load(cif, backend=backend)

        # Multi-chain should return False
        if polymer.size(Scale.CHAIN) > 1:
            assert not polymer.istype(Molecule.RNA)
            assert not polymer.istype(Molecule.PROTEIN)

    def test_istype_returns_false_for_wrong_type(self, backend):
        """istype returns False when molecule type doesn't match."""
        from ciffy import load, Molecule, Scale

        cif = get_test_cif("9GCM")
        polymer = load(cif, backend=backend)

        # Get a single RNA chain and check if it's protein (should be False)
        rna = polymer.molecule_type(Molecule.RNA)
        if rna.size(Scale.CHAIN) > 0:
            single_rna = rna.chain(0)
            assert not single_rna.istype(Molecule.PROTEIN)
            assert not single_rna.istype(Molecule.DNA)

    def test_istype_works_for_templates(self, backend):
        """istype works for template-generated polymers."""
        from ciffy import from_sequence, Molecule

        # Template polymers have molecule_types
        polymer = from_sequence("acgu", backend=backend)

        # Should work and return True for RNA template
        assert polymer.istype(Molecule.RNA)
        assert not polymer.istype(Molecule.PROTEIN)

    def test_istype_all_molecule_types(self, backend):
        """istype works with all molecule types."""
        from ciffy import load, Molecule, Scale

        cif = get_test_cif("9GCM")  # RNA + protein complex (no empty chains)
        polymer = load(cif, backend=backend)

        # Test that istype doesn't crash for any molecule type
        for i in range(polymer.size(Scale.CHAIN)):
            chain = polymer.chain(i)
            if chain.empty():
                continue  # Skip empty chains
            # Just verify it returns bool without error
            for mol_type in [Molecule.RNA, Molecule.PROTEIN, Molecule.DNA,
                             Molecule.ION, Molecule.WATER, Molecule.LIGAND]:
                result = chain.istype(mol_type)
                assert isinstance(result, bool)
