"""
Tests for Polymer.align() and related methods.

Tests residue-to-local-frame alignment, field deletion via copy(),
and edge cases for extend().
"""

import numpy as np
import pytest

import ciffy
from ciffy import Scale, from_sequence
from ciffy.biochemistry import Residue

from tests.utils import BACKENDS

# Identity rotation + Z-axis translation for ideal backbone spacing
LINEAR_EXTEND_TRANSFORM = np.array([0, 0, 0, 0, 0, 6.0], dtype=np.float32)


def template_with_coords(sequence: str, backend: str = "numpy") -> ciffy.Polymer:
    """Create a template with ideal coordinates for testing."""
    template = from_sequence(sequence, backend=backend)
    if template.empty():
        return template

    poly = ciffy.Polymer()
    if backend == "torch":
        poly = poly.torch()

    residue_indices = list(template.sequence[:len(sequence)])
    for i, res_idx in enumerate(residue_indices):
        residue = Residue.from_index(int(res_idx))
        is_first = (i == 0)
        is_last = (i == len(residue_indices) - 1)

        atom_group = residue.terminal(start=is_first, end=is_last)
        atoms = atom_group.index()
        elements = atom_group.elements()
        coords = atom_group.ideal

        if backend == "torch":
            import torch
            coords = torch.from_numpy(coords)
            atoms = torch.from_numpy(atoms)
            elements = torch.from_numpy(elements)

        if poly.empty():
            poly = poly.extend(residue, coords, atoms=atoms, elements=elements)
        else:
            poly = poly.extend(
                residue, coords, LINEAR_EXTEND_TRANSFORM, atoms=atoms, elements=elements
            )

    return poly


# =============================================================================
# Polymer.align() Tests
# =============================================================================

class TestResidueAlign:
    """Tests for Polymer.align() - residue-to-local-frame alignment."""

    def test_align_rna_returns_polymer(self):
        """align() returns a Polymer with aligned coordinates and Rs."""
        p = ciffy.load("tests/data/9MDS.cif").chain(0).residue([0, 1, 2])
        aligned, Rs = p.align()

        assert isinstance(aligned, ciffy.Polymer)
        assert aligned.size(Scale.RESIDUE) == 3
        assert aligned.coordinates.ndim == 2
        assert aligned.coordinates.shape[1] == 3
        assert Rs.shape == (3, 3, 3)  # (n_residues, 3, 3)

    def test_align_rna_preserves_atom_count(self):
        """Each aligned residue has correct number of atoms."""
        p = ciffy.load("tests/data/9MDS.cif").chain(0).residue([0, 1, 2])
        atom_counts = p.counts(Scale.RESIDUE)
        aligned, _ = p.align()

        for i in range(aligned.size(Scale.RESIDUE)):
            res = aligned.residue(i)
            assert res.size() == atom_counts[i]

    def test_align_rna_centers_origin(self):
        """Aligned residues have frame origin near coordinate origin."""
        p = ciffy.load("tests/data/9MDS.cif").chain(0).residue([0, 1, 2])
        aligned, _ = p.align()

        # The C1' atom (origin of glycosidic frame) should be at origin
        for i in range(aligned.size(Scale.RESIDUE)):
            coords = aligned.residue(i).coordinates
            # Check that coordinates are centered (mean should be near origin)
            centroid = coords.mean(axis=0)
            # Centroid won't be exactly at origin since origin is C1', not centroid
            # But it should be reasonably close (within ~5A)
            assert np.linalg.norm(centroid) < 5.0

    def test_align_single_residue(self):
        """align() works on single residue."""
        p = ciffy.load("tests/data/9MDS.cif").chain(0).residue(0)
        aligned, Rs = p.align()

        assert aligned.size(Scale.RESIDUE) == 1
        assert aligned.size() == p.size()
        assert Rs.shape == (1, 3, 3)

    def test_align_purine_and_pyrimidine(self):
        """align() handles both purines (A, G) and pyrimidines (C, U)."""
        p = ciffy.load("tests/data/9MDS.cif").chain(0)

        # Get first 10 residues to ensure we have both types
        p = p.residue(list(range(min(10, p.size(Scale.RESIDUE)))))
        aligned, Rs = p.align()

        # Should succeed for all residues
        assert aligned.size(Scale.RESIDUE) == p.size(Scale.RESIDUE)
        assert Rs.shape[0] == p.size(Scale.RESIDUE)

    def test_align_consistent_frame(self):
        """Same residue type aligned multiple times gives consistent frames."""
        p = ciffy.load("tests/data/9MDS.cif").chain(0)

        # Find two adenine residues
        seq = p.sequence_str()
        a_indices = [i for i, c in enumerate(seq) if c.upper() == 'A']

        if len(a_indices) < 2:
            pytest.skip("Need at least 2 adenines for this test")

        # Align them separately
        p1 = p.residue(a_indices[0])
        p2 = p.residue(a_indices[1])

        aligned1, _ = p1.align()
        aligned2, _ = p2.align()

        # Both should have same shape (same residue type)
        assert aligned1.residue(0).coordinates.shape == aligned2.residue(0).coordinates.shape

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_align_backend_preserved(self, backend):
        """align() works with both numpy and torch backends."""
        if backend == "torch":
            pytest.importorskip("torch")

        p = ciffy.load("tests/data/9MDS.cif").chain(0).residue([0, 1, 2])
        if backend == "torch":
            p = p.torch()

        aligned, Rs = p.align()

        # Check output type matches backend
        if backend == "torch":
            import torch
            assert isinstance(aligned.coordinates, torch.Tensor)
            assert isinstance(Rs, torch.Tensor)
        else:
            assert isinstance(aligned.coordinates, np.ndarray)
            assert isinstance(Rs, np.ndarray)


# =============================================================================
# Polymer.copy() Field Deletion Tests
# =============================================================================

class TestCopyFieldDeletion:
    """Tests for Polymer.copy(field=None) to delete fields."""

    def test_copy_removes_bfactors(self):
        """copy(bfactors=None) removes bfactors field."""
        p = ciffy.load("tests/data/9MDS.cif").chain(0).residue(0)

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
        p = ciffy.load("tests/data/9MDS.cif").chain(0).residue(0)
        p = p.copy(bfactors=None)

        # Get new residue data (no bfactors)
        atom_group = Residue.G.terminal(start=False, end=False)
        atoms, elements, coords = atom_group.index(), atom_group.elements(), atom_group.ideal
        transform = np.array([0, 0, 0, 0, 0, 6.0], dtype=np.float32)

        # Should work without providing bfactors
        extended = p.extend(
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
        p = ciffy.load("tests/data/9MDS.cif").chain(0).residue(0)
        original_coords = np.asarray(p.coordinates).copy()
        original_atoms = np.asarray(p.atoms).copy()
        original_seq = p.sequence_str()

        p2 = p.copy(bfactors=None)

        np.testing.assert_array_equal(np.asarray(p2.coordinates), original_coords)
        np.testing.assert_array_equal(np.asarray(p2.atoms), original_atoms)
        assert p2.sequence_str() == original_seq

    def test_copy_deep_copies_arrays(self):
        """copy() creates independent array copies."""
        p = ciffy.load("tests/data/9MDS.cif").chain(0).residue(0)
        p2 = p.copy()

        # Modify original
        p.coordinates[0, 0] = 999.0

        # Copy should be unchanged
        assert p2.coordinates[0, 0] != 999.0


# =============================================================================
# Polymer.sort_atoms() Tests
# =============================================================================

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


# =============================================================================
# Polymer.extend() Edge Cases
# =============================================================================

class TestExtendEdgeCases:
    """Edge case tests for Polymer.extend()."""

    def test_extend_with_identity_transform(self):
        """Extend with zero rotation places residue along backbone."""
        p = template_with_coords("a")

        atom_group = Residue.C.terminal(start=False, end=False)
        atoms, elements, coords = atom_group.index(), atom_group.elements(), atom_group.ideal
        # Identity rotation, translate along Z
        transform = np.array([0, 0, 0, 0, 0, 6.0], dtype=np.float32)

        extended = p.extend(
            coordinates=coords,
            atoms=atoms,
            elements=elements,
            transform=transform,
            residue=Residue.C
        )

        assert extended.size(Scale.RESIDUE) == 2

    def test_extend_with_absolute_coords(self):
        """Extend with absolute coordinates (no transform)."""
        p = template_with_coords("a")

        atom_group = Residue.C.terminal(start=False, end=False)
        atoms, elements = atom_group.index(), atom_group.elements()
        # Absolute coordinates - offset from origin
        abs_coords = atom_group.ideal + np.array([10.0, 0.0, 0.0], dtype=np.float32)

        extended = p.extend(
            coordinates=abs_coords,
            atoms=atoms,
            elements=elements,
            residue=Residue.C
        )

        assert extended.size(Scale.RESIDUE) == 2
        # New residue coords should be exactly what we passed (offset by 10 in X)
        new_res_coords = extended.coordinates[-len(atoms):]
        assert np.allclose(new_res_coords, abs_coords, atol=1e-5)

    def test_extend_sequence_updated(self):
        """Extended polymer has correct sequence."""
        p = template_with_coords("acg")

        atom_group = Residue.U.terminal(start=False, end=False)
        atoms, elements, coords = atom_group.index(), atom_group.elements(), atom_group.ideal
        transform = np.array([0, 0, 0, 0, 0, 6.0], dtype=np.float32)

        extended = p.extend(
            coordinates=coords,
            atoms=atoms,
            elements=elements,
            transform=transform,
            residue=Residue.U
        )

        assert extended.sequence_str() == "acgu"

    def test_extend_chain_count_unchanged(self):
        """Extend adds residue to existing chain, not new chain."""
        p = template_with_coords("ac")
        assert p.size(Scale.CHAIN) == 1

        atom_group = Residue.G.terminal(start=False, end=False)
        atoms, elements, coords = atom_group.index(), atom_group.elements(), atom_group.ideal
        transform = np.array([0, 0, 0, 0, 0, 6.0], dtype=np.float32)

        extended = p.extend(
            coordinates=coords,
            atoms=atoms,
            elements=elements,
            transform=transform,
            residue=Residue.G
        )

        assert extended.size(Scale.CHAIN) == 1

    def test_extend_from_different_residue_types(self):
        """Can extend from any RNA residue type."""
        for start_res in ['a', 'c', 'g', 'u']:
            p = template_with_coords(start_res)

            atom_group = Residue.A.terminal(start=False, end=False)
            atoms, elements, coords = atom_group.index(), atom_group.elements(), atom_group.ideal
            transform = np.array([0, 0, 0, 0, 0, 6.0], dtype=np.float32)

            extended = p.extend(
                coordinates=coords,
                atoms=atoms,
                elements=elements,
                transform=transform,
                residue=Residue.A
            )

            assert extended.size(Scale.RESIDUE) == 2


# =============================================================================
# AtomGroup.elements() Tests
# =============================================================================

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


# =============================================================================
# Polymer.extend_new() Tests
# =============================================================================

class TestExtendNew:
    """Tests for Polymer.extend_new() method."""

    def test_extend_new_template_mode(self):
        """extend_new() creates template without coordinates."""
        p = ciffy.Polymer()
        p = p.extend_new(Residue.A)

        assert p.size() == Residue.A.n_atoms
        assert p.size(Scale.RESIDUE) == 1
        assert hasattr(p, 'atoms')
        assert hasattr(p, 'elements')
        # Template mode: no coordinates
        assert not hasattr(p, '_coordinates') or p._coordinates is None

    def test_extend_new_with_coordinates(self):
        """extend_new() with coordinates creates polymer with coords."""
        p = ciffy.Polymer()
        coords = Residue.A.ideal
        p = p.extend_new(Residue.A, coords)

        assert p.size() == Residue.A.n_atoms
        assert p.coordinates is not None
        assert p.coordinates.shape == coords.shape

    def test_extend_new_multi_residue_template(self):
        """extend_new() builds multi-residue template."""
        p = ciffy.Polymer()
        for res in [Residue.A, Residue.C, Residue.G, Residue.U]:
            p = p.extend_new(res)

        assert p.size(Scale.RESIDUE) == 4
        expected_atoms = sum(r.n_atoms for r in [Residue.A, Residue.C, Residue.G, Residue.U])
        assert p.size() == expected_atoms
        assert p.sequence_str() == "acgu"

    def test_extend_new_with_transform(self):
        """extend_new() positions residue using transform."""
        p = ciffy.Polymer()
        p = p.extend_new(Residue.A, Residue.A.ideal)
        p = p.extend_new(Residue.C, Residue.C.ideal, np.zeros(6))

        assert p.size(Scale.RESIDUE) == 2
        assert p.coordinates is not None
        expected = Residue.A.n_atoms + Residue.C.n_atoms
        assert p.coordinates.shape == (expected, 3)

    def test_extend_new_with_subset_requires_residue(self):
        """extend_new() with subset requires explicit residue parameter."""
        subset = Residue.A.subset({2, 3, 5, 6, 7})
        p = ciffy.Polymer()

        # Should raise without residue=
        with pytest.raises(ValueError, match="residue"):
            p.extend_new(subset)

        # Should work with residue=
        p = p.extend_new(subset, residue=Residue.A)
        assert p.size() == 5

    def test_extend_new_preserves_sequence(self):
        """extend_new() correctly sets sequence field."""
        p = ciffy.Polymer()
        p = p.extend_new(Residue.G)
        p = p.extend_new(Residue.C)
        p = p.extend_new(Residue.A)
        p = p.extend_new(Residue.U)

        assert p.sequence_str() == "gcau"

    def test_extend_new_atoms_match_atomgroup(self):
        """extend_new() atoms field matches AtomGroup.index()."""
        p = ciffy.Polymer()
        p = p.extend_new(Residue.A)

        np.testing.assert_array_equal(
            np.asarray(p.atoms),
            Residue.A.index()
        )

    def test_extend_new_elements_match_atomgroup(self):
        """extend_new() elements field matches AtomGroup.elements()."""
        p = ciffy.Polymer()
        p = p.extend_new(Residue.A)

        np.testing.assert_array_equal(
            np.asarray(p.elements),
            Residue.A.elements()
        )
