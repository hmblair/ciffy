"""
Tests for AtomGroup.elements() method.
"""

import numpy as np

from ciffy.biochemistry import Residue


class TestAtomGroupElements:
    """Tests for AtomGroup.elements() method."""

    def test_elements_returns_array(self):
        """elements() returns numpy array of atomic numbers."""
        elems = Residue.A.elements()
        assert isinstance(elems, np.ndarray)
        assert elems.dtype == np.int64

    def test_elements_shape_matches_n_atoms(self):
        """elements() shape matches number of atoms."""
        for res in [Residue.A, Residue.C, Residue.G, Residue.U]:
            elems = res.elements()
            assert elems.shape == (res.n_atoms,)

    def test_elements_has_expected_values(self):
        """elements() returns correct atomic numbers for RNA."""
        elems = Residue.A.elements()
        # RNA has P (15), O (8), C (6), N (7), H (1)
        unique = set(elems)
        assert unique <= {1, 6, 7, 8, 15}, f"Unexpected elements: {unique}"
        # Must have at least C, N, O, P
        assert 6 in unique  # Carbon
        assert 7 in unique  # Nitrogen
        assert 8 in unique  # Oxygen
        assert 15 in unique  # Phosphorus

    def test_elements_on_subset(self):
        """elements() works on AtomGroup subsets."""
        subset = Residue.A.subset({2, 3, 5, 6, 7})  # 5 atoms
        elems = subset.elements()
        assert elems.shape == (5,)
