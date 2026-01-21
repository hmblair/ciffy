"""
Tests for Polymer.copy() - field deletion and deep copying.
"""

import numpy as np

import ciffy
from tests.utils import get_test_cif
from ciffy import Scale
from ciffy.biochemistry import Residue


class TestCopyFieldDeletion:
    """Tests for Polymer.copy(field=None) to delete fields."""

    def test_copy_removes_bfactors(self):
        """copy(bfactors=None) removes bfactors field."""
        p = ciffy.load(get_test_cif("9MDS")).chain(0).residue(0)

        # Verify bfactors exist
        assert hasattr(p, 'bfactors')

        # Remove bfactors
        p2 = p.copy(bfactors=None)

        # Original unchanged
        assert hasattr(p, 'bfactors')

        # Copy has no bfactors
        assert not hasattr(p2, 'bfactors')

        # Other fields preserved
        assert hasattr(p2, 'coordinates')
        assert hasattr(p2, 'atoms')
        assert p2.size() == p.size()

    def test_copy_without_field_allows_extend(self):
        """Polymer without bfactors can extend without providing bfactors."""
        p = ciffy.load(get_test_cif("9MDS")).chain(0).residue(0)
        p = p.copy(bfactors=None)

        # Get new residue data (no bfactors)
        atom_group = Residue.G.terminal(start=False, end=False)
        atoms, elements, coords = atom_group.index(), atom_group.elements(), atom_group.ideal
        transform = np.array([1, 0, 0, 0, 0, 0, 6.0], dtype=np.float32)

        # Should work without providing bfactors
        extended = p._append(
            coordinates=coords,
            atoms=atoms,
            elements=elements,
            transform=transform,
            residue=Residue.G
        )

        assert extended.size(Scale.RESIDUE) == 2
        assert not hasattr(extended, 'bfactors')

    def test_copy_none_preserves_other_fields(self):
        """Removing one field preserves all other fields."""
        p = ciffy.load(get_test_cif("9MDS")).chain(0).residue(0)
        original_coords = np.asarray(p.coordinates).copy()
        original_atoms = np.asarray(p.atoms).copy()
        original_seq = p.sequence_str()

        p2 = p.copy(bfactors=None)

        np.testing.assert_array_equal(np.asarray(p2.coordinates), original_coords)
        np.testing.assert_array_equal(np.asarray(p2.atoms), original_atoms)
        assert p2.sequence_str() == original_seq

    def test_copy_deep_copies_arrays(self):
        """copy() creates independent array copies."""
        p = ciffy.load(get_test_cif("9MDS")).chain(0).residue(0)
        p2 = p.copy()

        # Modify original
        p.coordinates[0, 0] = 999.0

        # Copy should be unchanged
        assert p2.coordinates[0, 0] != 999.0
