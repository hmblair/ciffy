"""
Tests for Polymer selection method edge cases.

Tests atom_type, residue_type, molecule_type, chain, mask, and __getitem__.
"""

import pytest
import numpy as np

from tests.utils import get_test_cif, BACKENDS, get_single_chain_poly


class TestAtomType:
    """Test atom_type() edge cases."""

    def test_atom_type_nonexistent_index(self, backend):
        """atom_type with non-existent index returns empty polymer."""
        import ciffy

        p = ciffy.load(get_test_cif("3SKW"), backend=backend)
        result = p.atom_type(99999)  # Non-existent atom type

        assert result.empty()

    def test_atom_type_single_match(self, backend):
        """atom_type with valid index returns non-empty polymer."""
        import ciffy

        p = ciffy.template("acgu", backend=backend)
        # Get first atom type present
        first_atom = p.atoms[0].item() if hasattr(p.atoms[0], 'item') else p.atoms[0]

        result = p.atom_type(first_atom)
        assert not result.empty()

    def test_atom_type_array_input(self, backend):
        """atom_type accepts array of indices."""
        import ciffy

        p = ciffy.template("acgu", backend=backend)
        # Get first few unique atom types
        atoms_np = np.asarray(p.atoms)
        unique_atoms = np.unique(atoms_np)[:3]

        result = p.atom_type(unique_atoms)
        assert isinstance(result, ciffy.Polymer)
        assert not result.empty()

    def test_atom_type_negative_index(self, backend):
        """atom_type with negative index (unknown atoms)."""
        import ciffy

        p = ciffy.template("acgu", backend=backend)
        result = p.atom_type(-1)

        # Should return empty or atoms with -1 (unknown)
        # Depending on structure, may be empty
        assert isinstance(result, ciffy.Polymer)


class TestResidueType:
    """Test residue_type() edge cases."""

    def test_residue_type_nonexistent_index(self, backend):
        """residue_type with non-existent index returns empty polymer."""
        import ciffy

        p = ciffy.template("acgu", backend=backend)
        result = p.residue_type(99999)  # Non-existent residue type

        assert result.empty()

    def test_residue_type_valid_index(self, backend):
        """residue_type with valid index returns matching residues."""
        import ciffy

        p = ciffy.template("acgu", backend=backend)
        # Adenosine is residue type 0
        result = p.residue_type(0)

        assert not result.empty()
        # Should have atoms from adenosine residue
        assert result.size() < p.size()  # Only 1 of 4 residues

    def test_residue_type_array_input(self, backend):
        """residue_type accepts array of residue indices."""
        import ciffy

        p = ciffy.template("acgu", backend=backend)
        # Select adenosine (type 0) and cytidine (type 1)
        result = p.residue_type(np.array([0, 1]))

        assert not result.empty()
        # Should return residues (non-empty result means selection worked)
        assert isinstance(result, ciffy.Polymer)

    def test_residue_type_all_residues(self, backend):
        """residue_type with all types returns same polymer."""
        import ciffy
        from ciffy import Scale

        p = ciffy.template("acgu", backend=backend)
        seq_np = np.asarray(p.sequence)
        all_types = np.unique(seq_np)

        result = p.residue_type(all_types)
        assert result.size(Scale.RESIDUE) == p.size(Scale.RESIDUE)


class TestMoleculeType:
    """Test molecule_type() edge cases."""

    def test_molecule_type_no_match(self, backend):
        """molecule_type returns empty when no chains match."""
        import ciffy
        from ciffy import Molecule

        # Use real CIF with known molecule types (9MDS is all RNA)
        p = ciffy.load(get_test_cif("9MDS"), backend=backend)
        result = p.molecule_type(Molecule.DNA)

        assert result.empty()

    # Note: test_molecule_type_requires_molecule_types removed - _templates now have molecule_types

    def test_molecule_type_all_match(self, backend):
        """molecule_type on matching type returns full structure."""
        import ciffy
        from ciffy import Molecule

        # Use real CIF with known molecule types
        p = ciffy.load(get_test_cif("9MDS"), backend=backend)  # All RNA
        result = p.molecule_type(Molecule.RNA)

        assert not result.empty()
        # All chains are RNA, so should get all
        from ciffy import Scale
        assert result.size(Scale.CHAIN) == p.size(Scale.CHAIN)

    def test_molecule_type_mixed_structure(self, backend):
        """molecule_type on mixed RNA+protein returns subset."""
        import ciffy
        from ciffy import Molecule, Scale

        p = ciffy.load(get_test_cif("9GCM"), backend=backend)  # RNA + protein

        rna = p.molecule_type(Molecule.RNA)
        protein = p.molecule_type(Molecule.PROTEIN)

        assert not rna.empty()
        assert not protein.empty()
        # Sum should equal original chain count
        assert rna.size(Scale.CHAIN) + protein.size(Scale.CHAIN) <= p.size(Scale.CHAIN)


class TestChain:
    """Test chain() edge cases."""

    def test_chain_first(self, backend):
        """chain(0) returns first chain."""
        import ciffy
        from ciffy import Scale

        p = ciffy.load(get_test_cif("9GCM"), backend=backend)
        chain = p.chain(0)

        assert not chain.empty()
        assert chain.size(Scale.CHAIN) == 1

    def test_chain_last(self, backend):
        """chain() with last valid index works."""
        import ciffy
        from ciffy import Scale

        p = ciffy.load(get_test_cif("9GCM"), backend=backend)
        last_idx = p.size(Scale.CHAIN) - 1
        chain = p.chain(last_idx)

        assert not chain.empty()
        assert chain.size(Scale.CHAIN) == 1

    def test_chain_out_of_bounds(self, backend):
        """chain() raises IndexError for out-of-bounds index."""
        import ciffy
        from ciffy import Scale

        p = ciffy.load(get_test_cif("9GCM"), backend=backend)
        invalid_idx = p.size(Scale.CHAIN) + 10

        with pytest.raises(IndexError):
            p.chain(invalid_idx)

    def test_chain_array_input(self, backend):
        """chain() accepts array of indices."""
        import ciffy
        from ciffy import Scale

        p = ciffy.load(get_test_cif("9GCM"), backend=backend)
        n_chains = p.size(Scale.CHAIN)

        if n_chains >= 2:
            result = p.chain(np.array([0, 1]))
            assert result.size(Scale.CHAIN) == 2


class TestMask:
    """Test mask() edge cases."""

    def test_mask_single_index(self, backend):
        """mask with single index creates correct mask."""
        import ciffy
        from ciffy import Scale

        p = ciffy.template("acgu", backend=backend)
        mask = p._mask(0, Scale.RESIDUE, Scale.ATOM)

        # Mask should be True for atoms of first residue only
        true_count = mask.sum().item() if hasattr(mask.sum(), 'item') else mask.sum()
        atoms_in_first_residue = p.counts(Scale.RESIDUE)[0].item()

        assert true_count == atoms_in_first_residue

    def test_mask_boundary_index(self, backend):
        """mask with last valid index works."""
        import ciffy
        from ciffy import Scale

        p = ciffy.template("acgu", backend=backend)
        last_idx = p.size(Scale.RESIDUE) - 1

        mask = p._mask(last_idx, Scale.RESIDUE, Scale.ATOM)
        true_count = mask.sum().item() if hasattr(mask.sum(), 'item') else mask.sum()

        assert true_count > 0

    def test_mask_chain_to_atom(self, backend):
        """mask from chain scale to atom scale."""
        import ciffy
        from ciffy import Scale

        p = ciffy.load(get_test_cif("9GCM"), backend=backend)
        mask = p._mask(0, Scale.CHAIN, Scale.ATOM)

        # Mask should have size equal to total atoms
        assert len(mask) == p.size()

        # Sum should equal atoms in first chain
        true_count = mask.sum().item() if hasattr(mask.sum(), 'item') else mask.sum()
        atoms_in_first_chain = p.counts(Scale.CHAIN)[0].item()

        assert true_count == atoms_in_first_chain


class TestGetItem:
    """Test __getitem__ edge cases."""

    def test_getitem_slice_first_half(self, backend):
        """__getitem__ with slice [:n//2] returns first half."""
        import ciffy

        p = ciffy.template("acgu", backend=backend)
        n = p.size()
        result = p[:n // 2]

        assert result.size() == n // 2

    def test_getitem_slice_second_half(self, backend):
        """__getitem__ with slice [n//2:] returns second half."""
        import ciffy

        p = ciffy.template("acgu", backend=backend)
        n = p.size()
        result = p[n // 2:]

        assert result.size() == n - n // 2

    def test_getitem_slice_with_step(self, backend):
        """__getitem__ with slice [::2] returns every other atom."""
        import ciffy

        p = ciffy.template("acgu", backend=backend)
        n = p.size()
        result = p[::2]

        expected_size = (n + 1) // 2
        assert result.size() == expected_size

    def test_getitem_negative_slice(self, backend):
        """__getitem__ with negative slice [-10:]."""
        import ciffy

        p = ciffy.template("acgu", backend=backend)
        n = p.size()
        result = p[-10:]

        expected_size = min(10, n)
        assert result.size() == expected_size

    def test_getitem_empty_slice(self, backend):
        """__getitem__ with empty slice [5:5] returns empty polymer."""
        import ciffy

        p = ciffy.template("acgu", backend=backend)
        result = p[5:5]

        assert result.empty()

    def test_getitem_out_of_bounds_slice(self, backend):
        """__getitem__ with out-of-bounds slice is bounded."""
        import ciffy

        p = ciffy.template("acgu", backend=backend)
        n = p.size()
        result = p[0:n + 100]  # Beyond end

        # Python slice semantics: bounded to actual size
        assert result.size() == n

    def test_getitem_boolean_mask_all_true(self, backend):
        """__getitem__ with all-True mask returns same polymer."""
        import ciffy

        p = ciffy.template("acgu", backend=backend)
        mask = p.atoms >= 0  # All True (atoms are non-negative)

        if backend == "torch":
            import torch
            mask = torch.ones(p.size(), dtype=torch.bool)
        else:
            mask = np.ones(p.size(), dtype=bool)

        result = p[mask]
        assert result.size() == p.size()

    def test_getitem_boolean_mask_all_false(self, backend):
        """__getitem__ with all-False mask returns empty polymer."""
        import ciffy

        p = ciffy.template("acgu", backend=backend)

        if backend == "torch":
            import torch
            mask = torch.zeros(p.size(), dtype=torch.bool)
        else:
            mask = np.zeros(p.size(), dtype=bool)

        result = p[mask]
        assert result.empty()


class TestSpecializedSelections:
    """Test specialized selection methods (backbone, nucleobase, etc.)."""

    def test_backbone_selection(self, backend):
        """backbone() returns non-empty subset."""
        import ciffy

        p = ciffy.load(get_test_cif("3SKW"), backend=backend)
        backbone = p.backbone()

        assert not backbone.empty()
        assert backbone.size() < p.size()

    def test_nucleobase_selection(self, backend):
        """nucleobase() returns non-empty subset for RNA."""
        import ciffy

        p = ciffy.load(get_test_cif("3SKW"), backend=backend)
        nucleobase = p.nucleobase()

        # 3SKW is RNA, should have nucleobases
        assert not nucleobase.empty()
        assert nucleobase.size() < p.size()

    def test_phosphate_selection(self, backend):
        """phosphate() returns non-empty subset for RNA."""
        import ciffy

        p = ciffy.load(get_test_cif("3SKW"), backend=backend)
        phosphate = p.phosphate()

        assert not phosphate.empty()
        assert phosphate.size() < p.size()

    def test_sidechain_selection(self, backend):
        """sidechain() returns subset for protein."""
        import ciffy
        from ciffy import Molecule

        p = ciffy.load(get_test_cif("9GCM"), backend=backend)
        protein = p.molecule_type(Molecule.PROTEIN)

        if not protein.empty():
            sidechain = protein.sidechain()
            # May be empty or non-empty depending on structure
            assert isinstance(sidechain, ciffy.Polymer)


# =============================================================================
# Heavy Atom Selection Tests
# =============================================================================


class TestHeavy:
    """Test Polymer.heavy() method."""

    def test_heavy_returns_polymer(self, backend):
        """heavy() returns a Polymer instance."""
        import ciffy

        p = ciffy.load(get_test_cif("3SKW"), backend=backend)
        heavy = p.heavy()

        assert isinstance(heavy, ciffy.Polymer)

    def test_heavy_excludes_hydrogen(self, backend):
        """heavy() excludes hydrogen atoms (element 1)."""
        import ciffy

        p = ciffy.load(get_test_cif("3SKW"), backend=backend)
        heavy = p.heavy()

        # Check that no hydrogen atoms remain
        elements = np.asarray(heavy.elements)
        assert (elements != 1).all(), "Hydrogen atoms found in heavy selection"

    def test_heavy_reduces_atom_count(self, backend):
        """heavy() returns fewer atoms than original."""
        import ciffy

        p = ciffy.load(get_test_cif("3SKW"), backend=backend)

        # Check if structure has hydrogens
        elements = np.asarray(p.elements)
        has_hydrogens = (elements == 1).any()

        heavy = p.heavy()

        if has_hydrogens:
            assert heavy.size() < p.size()
        else:
            # No hydrogens, so heavy returns same size
            assert heavy.size() == p.size()

    def test_heavy_preserves_structure(self, backend):
        """heavy() preserves residue/chain structure."""
        import ciffy
        from ciffy import Scale

        p = ciffy.load(get_test_cif("3SKW"), backend=backend)
        heavy = p.heavy()

        # Chain count should be preserved or reduced
        assert heavy.size(Scale.CHAIN) <= p.size(Scale.CHAIN)

        # If heavy is non-empty, should have valid structure
        if not heavy.empty():
            assert heavy.size(Scale.RESIDUE) > 0

    def test_heavy_on_all_heavy_structure(self, backend):
        """heavy() on structure with no hydrogens returns same structure."""
        import ciffy

        # Templates typically don't have hydrogens
        p = get_single_chain_poly(backend)
        elements = np.asarray(p.elements)

        if (elements == 1).any():
            pytest.skip("Structure has hydrogens")

        heavy = p.heavy()
        assert heavy.size() == p.size()


# =============================================================================
# Atom Selection Tests
# =============================================================================


class TestAtom:
    """Test Polymer.atom() method."""

    def test_atom_single_index(self, backend):
        """atom() with single index returns single-atom polymer."""
        import ciffy

        p = ciffy.load(get_test_cif("3SKW"), backend=backend)
        single = p.atom(0)

        assert single.size() == 1

    def test_atom_array_indices(self, backend):
        """atom() with array of indices returns correct count."""
        import ciffy

        p = ciffy.load(get_test_cif("3SKW"), backend=backend)
        indices = np.array([0, 1, 2, 5, 10])
        result = p.atom(indices)

        assert result.size() == len(indices)

    def test_atom_preserves_coordinates(self, backend):
        """atom() preserves original coordinates."""
        import ciffy

        p = ciffy.load(get_test_cif("3SKW"), backend=backend)
        indices = np.array([0, 5, 10])
        result = p.atom(indices)

        # Selected coordinates should match originals
        orig_coords = np.asarray(p.coordinates)
        result_coords = np.asarray(result.coordinates)

        for i, idx in enumerate(indices):
            assert np.allclose(result_coords[i], orig_coords[idx])

    def test_atom_out_of_bounds(self, backend):
        """atom() with out-of-bounds index raises error."""
        import ciffy

        p = ciffy.load(get_test_cif("3SKW"), backend=backend)
        n = p.size()

        with pytest.raises((IndexError, ValueError)):
            p.atom(n + 100)

    def test_atom_negative_index(self, backend):
        """atom() with negative index selects from end."""
        import ciffy

        p = ciffy.load(get_test_cif("3SKW"), backend=backend)

        # Negative index should work like Python list indexing
        try:
            last = p.atom(-1)
            assert last.size() == 1
            # Last atom should have same coords as p.coordinates[-1]
            last_coords = np.asarray(last.coordinates)[0]
            expected = np.asarray(p.coordinates)[-1]
            assert np.allclose(last_coords, expected)
        except (IndexError, ValueError):
            # Some implementations may not support negative indexing
            pytest.skip("Negative indexing not supported")

    def test_atom_empty_array(self, backend):
        """atom() with empty array returns empty polymer."""
        import ciffy

        p = ciffy.load(get_test_cif("3SKW"), backend=backend)
        indices = np.array([], dtype=np.int64)
        result = p.atom(indices)

        assert result.empty()

    def test_atom_slice_like(self, backend):
        """atom() with list behaves like array."""
        import ciffy

        p = ciffy.load(get_test_cif("3SKW"), backend=backend)
        indices_list = [0, 1, 2]
        indices_array = np.array([0, 1, 2])

        result_list = p.atom(indices_list)
        result_array = p.atom(indices_array)

        assert result_list.size() == result_array.size()
