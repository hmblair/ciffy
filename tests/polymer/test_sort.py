"""
Tests for Polymer.sort_atoms() - canonical atom ordering.
"""

import numpy as np
import pytest

import ciffy
from ciffy import Scale

from tests.utils import BACKENDS


class TestSortAtoms:
    """Tests for Polymer.sort_atoms() - canonical atom ordering."""

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_sort_atoms_returns_polymer(self, backend):
        """sort_atoms() returns a Polymer."""
        if backend == "torch":
            pytest.importorskip("torch")
        p = ciffy.load("tests/data/9MDS.cif").chain(0).residue([0, 1, 2])
        if backend == "torch":
            p = p.torch()
        sorted_p = p.sort_atoms()

        assert isinstance(sorted_p, ciffy.Polymer)
        assert sorted_p.size(Scale.RESIDUE) == 3
        assert sorted_p.backend == backend

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_sort_atoms_preserves_size(self, backend):
        """sort_atoms() preserves atom count."""
        if backend == "torch":
            pytest.importorskip("torch")
        p = ciffy.load("tests/data/9MDS.cif").chain(0).residue([0, 1, 2])
        if backend == "torch":
            p = p.torch()
        sorted_p = p.sort_atoms()

        assert sorted_p.size() == p.size()
        np.testing.assert_array_equal(
            np.asarray(sorted_p.counts(Scale.RESIDUE)),
            np.asarray(p.counts(Scale.RESIDUE))
        )

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_sort_atoms_orders_by_enum_value(self, backend):
        """Atoms within each residue are sorted by enum value."""
        if backend == "torch":
            pytest.importorskip("torch")
        p = ciffy.load("tests/data/9MDS.cif").chain(0).residue([0, 1, 2])
        if backend == "torch":
            p = p.torch()
        sorted_p = p.sort_atoms()

        for i in range(sorted_p.size(Scale.RESIDUE)):
            res = sorted_p.residue(i)
            atoms = np.asarray(res.atoms)
            # Check atoms are sorted
            assert np.all(atoms[:-1] <= atoms[1:]), f"Residue {i} not sorted"

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_sort_atoms_sorts_all_fields(self, backend):
        """sort_atoms() reorders all atom-level fields."""
        if backend == "torch":
            pytest.importorskip("torch")
        p = ciffy.load("tests/data/9MDS.cif").chain(0).residue(0)
        if backend == "torch":
            p = p.torch()

        # Get original atoms and their corresponding coords
        orig_atoms = np.asarray(p.atoms).copy()
        orig_coords = np.asarray(p.coordinates).copy()

        sorted_p = p.sort_atoms()
        sorted_atoms = np.asarray(sorted_p.atoms)
        sorted_coords = np.asarray(sorted_p.coordinates)

        # Verify atoms are sorted
        expected_order = np.argsort(orig_atoms)
        np.testing.assert_array_equal(sorted_atoms, orig_atoms[expected_order])
        np.testing.assert_array_equal(sorted_coords, orig_coords[expected_order])

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_sort_atoms_idempotent(self, backend):
        """Calling sort_atoms() twice gives same result."""
        if backend == "torch":
            pytest.importorskip("torch")
        p = ciffy.load("tests/data/9MDS.cif").chain(0).residue([0, 1, 2])
        if backend == "torch":
            p = p.torch()
        sorted_once = p.sort_atoms()
        sorted_twice = sorted_once.sort_atoms()

        np.testing.assert_array_equal(
            np.asarray(sorted_once.atoms),
            np.asarray(sorted_twice.atoms)
        )
        np.testing.assert_array_equal(
            np.asarray(sorted_once.coordinates),
            np.asarray(sorted_twice.coordinates)
        )

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_align_then_sort_canonical(self, backend):
        """align().sort_atoms() produces canonical representation."""
        if backend == "torch":
            pytest.importorskip("torch")
        p = ciffy.load("tests/data/9MDS.cif").chain(0).residue([0, 1, 2])
        if backend == "torch":
            p = p.torch()
        aligned, _ = p.align()
        canonical = aligned.sort_atoms()

        # Each residue should have sorted atoms and aligned coords
        for i in range(canonical.size(Scale.RESIDUE)):
            res = canonical.residue(i)
            atoms = np.asarray(res.atoms)
            coords = np.asarray(res.coordinates)

            # Atoms sorted
            assert np.all(atoms[:-1] <= atoms[1:])

            # Coords are centered (roughly, since origin is C1' not centroid)
            centroid = coords.mean(axis=0)
            assert np.linalg.norm(centroid) < 5.0
