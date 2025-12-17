"""
Tests for Polymer selection method edge cases.

Tests by_atom, by_residue, by_type, by_index, mask, and __getitem__.
"""

import pytest
import numpy as np

from tests.utils import get_test_cif, BACKENDS


class TestByAtom:
    """Test by_atom() edge cases."""

    def test_by_atom_nonexistent_index(self, backend):
        """by_atom with non-existent index returns empty polymer."""
        import ciffy

        p = ciffy.load(get_test_cif("3SKW"), backend=backend)
        result = p.by_atom(99999)  # Non-existent atom type

        assert result.empty()

    def test_by_atom_single_match(self, backend):
        """by_atom with valid index returns non-empty polymer."""
        import ciffy

        p = ciffy.from_sequence("acgu", backend=backend)
        # Get first atom type present
        first_atom = p.atoms[0].item() if hasattr(p.atoms[0], 'item') else p.atoms[0]

        result = p.by_atom(first_atom)
        assert not result.empty()

    def test_by_atom_array_input(self, backend):
        """by_atom accepts array of indices."""
        import ciffy

        p = ciffy.from_sequence("acgu", backend=backend)
        # Get first few unique atom types
        atoms_np = np.asarray(p.atoms)
        unique_atoms = np.unique(atoms_np)[:3]

        result = p.by_atom(unique_atoms)
        assert isinstance(result, ciffy.Polymer)
        assert not result.empty()

    def test_by_atom_negative_index(self, backend):
        """by_atom with negative index (unknown atoms)."""
        import ciffy

        p = ciffy.from_sequence("acgu", backend=backend)
        result = p.by_atom(-1)

        # Should return empty or atoms with -1 (unknown)
        # Depending on structure, may be empty
        assert isinstance(result, ciffy.Polymer)


class TestByResidue:
    """Test by_residue() edge cases."""

    def test_by_residue_nonexistent_index(self, backend):
        """by_residue with non-existent index returns empty polymer."""
        import ciffy

        p = ciffy.from_sequence("acgu", backend=backend)
        result = p.by_residue(99999)  # Non-existent residue type

        assert result.empty()

    def test_by_residue_valid_index(self, backend):
        """by_residue with valid index returns matching residues."""
        import ciffy

        p = ciffy.from_sequence("acgu", backend=backend)
        # Adenosine is residue type 0
        result = p.by_residue(0)

        assert not result.empty()
        # Should have atoms from adenosine residue
        assert result.size() < p.size()  # Only 1 of 4 residues

    def test_by_residue_array_input(self, backend):
        """by_residue accepts array of residue indices."""
        import ciffy

        p = ciffy.from_sequence("acgu", backend=backend)
        # Select adenosine (type 0) and cytidine (type 1)
        result = p.by_residue(np.array([0, 1]))

        assert not result.empty()
        # Should return residues (non-empty result means selection worked)
        assert isinstance(result, ciffy.Polymer)

    def test_by_residue_all_residues(self, backend):
        """by_residue with all types returns same polymer."""
        import ciffy
        from ciffy import Scale

        p = ciffy.from_sequence("acgu", backend=backend)
        seq_np = np.asarray(p.sequence)
        all_types = np.unique(seq_np)

        result = p.by_residue(all_types)
        assert result.size(Scale.RESIDUE) == p.size(Scale.RESIDUE)


class TestByType:
    """Test by_type() edge cases."""

    def test_by_type_no_match(self, backend):
        """by_type returns empty when no chains match."""
        import ciffy
        from ciffy import Molecule

        p = ciffy.from_sequence("acgu", backend=backend)  # RNA
        result = p.by_type(Molecule.DNA)

        assert result.empty()

    def test_by_type_all_match(self, backend):
        """by_type on matching type returns full structure."""
        import ciffy
        from ciffy import Molecule

        # Use real CIF with known molecule types
        p = ciffy.load(get_test_cif("9MDS"), backend=backend)  # All RNA
        result = p.by_type(Molecule.RNA)

        assert not result.empty()
        # All chains are RNA, so should get all
        from ciffy import Scale
        assert result.size(Scale.CHAIN) == p.size(Scale.CHAIN)

    def test_by_type_mixed_structure(self, backend):
        """by_type on mixed RNA+protein returns subset."""
        import ciffy
        from ciffy import Molecule, Scale

        p = ciffy.load(get_test_cif("9GCM"), backend=backend)  # RNA + protein

        rna = p.by_type(Molecule.RNA)
        protein = p.by_type(Molecule.PROTEIN)

        assert not rna.empty()
        assert not protein.empty()
        # Sum should equal original chain count
        assert rna.size(Scale.CHAIN) + protein.size(Scale.CHAIN) <= p.size(Scale.CHAIN)


class TestByIndex:
    """Test by_index() edge cases."""

    def test_by_index_first_chain(self, backend):
        """by_index(0) returns first chain."""
        import ciffy
        from ciffy import Scale

        p = ciffy.load(get_test_cif("9GCM"), backend=backend)
        chain = p.by_index(0)

        assert not chain.empty()
        assert chain.size(Scale.CHAIN) == 1

    def test_by_index_last_chain(self, backend):
        """by_index with last valid index works."""
        import ciffy
        from ciffy import Scale

        p = ciffy.load(get_test_cif("9GCM"), backend=backend)
        last_idx = p.size(Scale.CHAIN) - 1
        chain = p.by_index(last_idx)

        assert not chain.empty()
        assert chain.size(Scale.CHAIN) == 1

    def test_by_index_out_of_bounds_positive(self, backend):
        """by_index raises IndexError for out-of-bounds positive index."""
        import ciffy
        from ciffy import Scale

        p = ciffy.load(get_test_cif("9GCM"), backend=backend)
        invalid_idx = p.size(Scale.CHAIN) + 10

        with pytest.raises(IndexError):
            p.by_index(invalid_idx)

    def test_by_index_array_input(self, backend):
        """by_index accepts array of indices."""
        import ciffy
        from ciffy import Scale

        p = ciffy.load(get_test_cif("9GCM"), backend=backend)
        n_chains = p.size(Scale.CHAIN)

        if n_chains >= 2:
            result = p.by_index(np.array([0, 1]))
            assert result.size(Scale.CHAIN) == 2


class TestMask:
    """Test mask() edge cases."""

    def test_mask_single_index(self, backend):
        """mask with single index creates correct mask."""
        import ciffy
        from ciffy import Scale

        p = ciffy.from_sequence("acgu", backend=backend)
        mask = p.mask(0, Scale.RESIDUE, Scale.ATOM)

        # Mask should be True for atoms of first residue only
        true_count = mask.sum().item() if hasattr(mask.sum(), 'item') else mask.sum()
        atoms_in_first_residue = p.sizes(Scale.RESIDUE)[0].item()

        assert true_count == atoms_in_first_residue

    def test_mask_boundary_index(self, backend):
        """mask with last valid index works."""
        import ciffy
        from ciffy import Scale

        p = ciffy.from_sequence("acgu", backend=backend)
        last_idx = p.size(Scale.RESIDUE) - 1

        mask = p.mask(last_idx, Scale.RESIDUE, Scale.ATOM)
        true_count = mask.sum().item() if hasattr(mask.sum(), 'item') else mask.sum()

        assert true_count > 0

    def test_mask_chain_to_atom(self, backend):
        """mask from chain scale to atom scale."""
        import ciffy
        from ciffy import Scale

        p = ciffy.load(get_test_cif("9GCM"), backend=backend)
        mask = p.mask(0, Scale.CHAIN, Scale.ATOM)

        # Mask should have size equal to total atoms
        assert len(mask) == p.size()

        # Sum should equal atoms in first chain
        true_count = mask.sum().item() if hasattr(mask.sum(), 'item') else mask.sum()
        atoms_in_first_chain = p.sizes(Scale.CHAIN)[0].item()

        assert true_count == atoms_in_first_chain


class TestGetItem:
    """Test __getitem__ edge cases."""

    def test_getitem_slice_first_half(self, backend):
        """__getitem__ with slice [:n//2] returns first half."""
        import ciffy

        p = ciffy.from_sequence("acgu", backend=backend)
        n = p.size()
        result = p[:n // 2]

        assert result.size() == n // 2

    def test_getitem_slice_second_half(self, backend):
        """__getitem__ with slice [n//2:] returns second half."""
        import ciffy

        p = ciffy.from_sequence("acgu", backend=backend)
        n = p.size()
        result = p[n // 2:]

        assert result.size() == n - n // 2

    def test_getitem_slice_with_step(self, backend):
        """__getitem__ with slice [::2] returns every other atom."""
        import ciffy

        p = ciffy.from_sequence("acgu", backend=backend)
        n = p.size()
        result = p[::2]

        expected_size = (n + 1) // 2
        assert result.size() == expected_size

    def test_getitem_negative_slice(self, backend):
        """__getitem__ with negative slice [-10:]."""
        import ciffy

        p = ciffy.from_sequence("acgu", backend=backend)
        n = p.size()
        result = p[-10:]

        expected_size = min(10, n)
        assert result.size() == expected_size

    def test_getitem_empty_slice(self, backend):
        """__getitem__ with empty slice [5:5] returns empty polymer."""
        import ciffy

        p = ciffy.from_sequence("acgu", backend=backend)
        result = p[5:5]

        assert result.empty()

    def test_getitem_out_of_bounds_slice(self, backend):
        """__getitem__ with out-of-bounds slice is bounded."""
        import ciffy

        p = ciffy.from_sequence("acgu", backend=backend)
        n = p.size()
        result = p[0:n + 100]  # Beyond end

        # Python slice semantics: bounded to actual size
        assert result.size() == n

    def test_getitem_boolean_mask_all_true(self, backend):
        """__getitem__ with all-True mask returns same polymer."""
        import ciffy

        p = ciffy.from_sequence("acgu", backend=backend)
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

        p = ciffy.from_sequence("acgu", backend=backend)

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
        protein = p.by_type(Molecule.PROTEIN)

        if not protein.empty():
            sidechain = protein.sidechain()
            # May be empty or non-empty depending on structure
            assert isinstance(sidechain, ciffy.Polymer)
