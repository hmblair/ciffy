"""
Tests for operations.sort_atoms() - canonical atom ordering.
"""

import numpy as np
import pytest

import ciffy
from ciffy import Scale, operations


class TestSortAtoms:
    """Tests for operations.sort_atoms() - canonical atom ordering."""

    def test_sort_atoms_returns_polymer(self, backend):
        """sort_atoms() returns a Polymer."""
        p = ciffy.load("tests/data/9MDS.cif", backend=backend).chain(0).residue([0, 1, 2])
        sorted_p = operations.sort_atoms(p)

        assert isinstance(sorted_p, ciffy.Polymer)
        assert sorted_p.size(Scale.RESIDUE) == 3
        assert sorted_p.backend == backend

    def test_sort_atoms_preserves_size(self, backend):
        """sort_atoms() preserves atom count."""
        p = ciffy.load("tests/data/9MDS.cif", backend=backend).chain(0).residue([0, 1, 2])
        sorted_p = operations.sort_atoms(p)

        assert sorted_p.size() == p.size()
        np.testing.assert_array_equal(
            np.asarray(sorted_p.counts(Scale.RESIDUE)),
            np.asarray(p.counts(Scale.RESIDUE))
        )

    def test_sort_atoms_orders_by_enum_value(self, backend):
        """Atoms within each residue are sorted by enum value."""
        p = ciffy.load("tests/data/9MDS.cif", backend=backend).chain(0).residue([0, 1, 2])
        sorted_p = operations.sort_atoms(p)

        for i in range(sorted_p.size(Scale.RESIDUE)):
            res = sorted_p.residue(i)
            atoms = np.asarray(res.atoms)
            # Check atoms are sorted
            assert np.all(atoms[:-1] <= atoms[1:]), f"Residue {i} not sorted"

    def test_sort_atoms_sorts_all_fields(self, backend):
        """sort_atoms() reorders all atom-level fields."""
        p = ciffy.load("tests/data/9MDS.cif", backend=backend).chain(0).residue(0)

        # Get original atoms and their corresponding coords
        orig_atoms = np.asarray(p.atoms).copy()
        orig_coords = np.asarray(p.coordinates).copy()

        sorted_p = operations.sort_atoms(p)
        sorted_atoms = np.asarray(sorted_p.atoms)
        sorted_coords = np.asarray(sorted_p.coordinates)

        # Verify atoms are sorted
        expected_order = np.argsort(orig_atoms)
        np.testing.assert_array_equal(sorted_atoms, orig_atoms[expected_order])
        np.testing.assert_array_equal(sorted_coords, orig_coords[expected_order])

    def test_sort_atoms_idempotent(self, backend):
        """Calling sort_atoms() twice gives same result."""
        p = ciffy.load("tests/data/9MDS.cif", backend=backend).chain(0).residue([0, 1, 2])
        sorted_once = operations.sort_atoms(p)
        sorted_twice = operations.sort_atoms(sorted_once)

        np.testing.assert_array_equal(
            np.asarray(sorted_once.atoms),
            np.asarray(sorted_twice.atoms)
        )
        np.testing.assert_array_equal(
            np.asarray(sorted_once.coordinates),
            np.asarray(sorted_twice.coordinates)
        )

