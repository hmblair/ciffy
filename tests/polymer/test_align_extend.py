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
from ciffy.polymer.builder import expand_residue, linear_extend_transform

from tests.utils import BACKENDS


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

        atoms, elements, coords = expand_residue(
            residue, start_terminal=is_first, end_terminal=is_last
        )

        if backend == "torch":
            import torch
            coords = torch.from_numpy(coords)
            atoms = torch.from_numpy(atoms)
            elements = torch.from_numpy(elements)

        if poly.empty():
            poly = poly.extend(residue, coords, atoms=atoms, elements=elements)
        else:
            last_coords, last_atoms, last_res = poly._residue_coords(-1)
            transform = linear_extend_transform(
                last_coords, last_atoms, last_res, atoms, residue
            )
            poly = poly.extend(
                residue, coords, transform, atoms=atoms, elements=elements
            )

    return poly


# =============================================================================
# Polymer.align() Tests
# =============================================================================

class TestResidueAlign:
    """Tests for Polymer.align() - residue-to-local-frame alignment."""

    def test_align_rna_returns_polymer(self):
        """align() returns a Polymer with aligned coordinates."""
        p = ciffy.load("tests/data/9MDS.cif").chain(0).residue([0, 1, 2])
        aligned = p.align()

        assert isinstance(aligned, ciffy.Polymer)
        assert aligned.size(Scale.RESIDUE) == 3
        assert aligned.coordinates.ndim == 2
        assert aligned.coordinates.shape[1] == 3

    def test_align_rna_preserves_atom_count(self):
        """Each aligned residue has correct number of atoms."""
        p = ciffy.load("tests/data/9MDS.cif").chain(0).residue([0, 1, 2])
        atom_counts = p.counts(Scale.RESIDUE)
        aligned = p.align()

        for i in range(aligned.size(Scale.RESIDUE)):
            res = aligned.residue(i)
            assert res.size() == atom_counts[i]

    def test_align_rna_centers_origin(self):
        """Aligned residues have frame origin near coordinate origin."""
        p = ciffy.load("tests/data/9MDS.cif").chain(0).residue([0, 1, 2])
        aligned = p.align()

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
        aligned = p.align()

        assert aligned.size(Scale.RESIDUE) == 1
        assert aligned.size() == p.size()

    def test_align_purine_and_pyrimidine(self):
        """align() handles both purines (A, G) and pyrimidines (C, U)."""
        p = ciffy.load("tests/data/9MDS.cif").chain(0)

        # Get first 10 residues to ensure we have both types
        p = p.residue(list(range(min(10, p.size(Scale.RESIDUE)))))
        aligned = p.align()

        # Should succeed for all residues
        assert aligned.size(Scale.RESIDUE) == p.size(Scale.RESIDUE)

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

        aligned1 = p1.align().residue(0).coordinates
        aligned2 = p2.align().residue(0).coordinates

        # Both should have same shape (same residue type)
        assert aligned1.shape == aligned2.shape

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_align_backend_preserved(self, backend):
        """align() works with both numpy and torch backends."""
        if backend == "torch":
            pytest.importorskip("torch")

        p = ciffy.load("tests/data/9MDS.cif").chain(0).residue([0, 1, 2])
        if backend == "torch":
            p = p.torch()

        aligned = p.align()

        # Check output type matches backend
        if backend == "torch":
            import torch
            assert isinstance(aligned.coordinates, torch.Tensor)
        else:
            assert isinstance(aligned.coordinates, np.ndarray)


class TestAlignBatch:
    """Tests for Polymer.align_batch() - batched alignment."""

    def test_align_batch_returns_padded_arrays(self):
        """align_batch() returns padded coordinate and mask arrays."""
        p = ciffy.load("tests/data/9MDS.cif").chain(0).residue([0, 1, 2])
        coords, mask = p.align_batch()

        n_residues = p.size(Scale.RESIDUE)
        assert coords.ndim == 3
        assert coords.shape[0] == n_residues
        assert coords.shape[2] == 3

        assert mask.ndim == 2
        assert mask.shape[0] == n_residues

    def test_align_batch_mask_matches_atoms(self):
        """Mask correctly indicates valid atoms per residue."""
        p = ciffy.load("tests/data/9MDS.cif").chain(0).residue([0, 1, 2])
        atom_counts = p.counts(Scale.RESIDUE)
        coords, mask = p.align_batch()

        for i, count in enumerate(atom_counts):
            # Number of True values in mask should equal atom count
            assert mask[i].sum() == count


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
        atoms, elements, coords = expand_residue(Residue.G, start_terminal=False)
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

    def test_sort_atoms_returns_polymer(self):
        """sort_atoms() returns a Polymer."""
        p = ciffy.load("tests/data/9MDS.cif").chain(0).residue([0, 1, 2])
        sorted_p = p.sort_atoms()

        assert isinstance(sorted_p, ciffy.Polymer)
        assert sorted_p.size(Scale.RESIDUE) == 3

    def test_sort_atoms_preserves_size(self):
        """sort_atoms() preserves atom count."""
        p = ciffy.load("tests/data/9MDS.cif").chain(0).residue([0, 1, 2])
        sorted_p = p.sort_atoms()

        assert sorted_p.size() == p.size()
        np.testing.assert_array_equal(
            np.asarray(sorted_p.counts(Scale.RESIDUE)),
            np.asarray(p.counts(Scale.RESIDUE))
        )

    def test_sort_atoms_orders_by_enum_value(self):
        """Atoms within each residue are sorted by enum value."""
        p = ciffy.load("tests/data/9MDS.cif").chain(0).residue([0, 1, 2])
        sorted_p = p.sort_atoms()

        for i in range(sorted_p.size(Scale.RESIDUE)):
            res = sorted_p.residue(i)
            atoms = np.asarray(res.atoms)
            # Check atoms are sorted
            assert np.all(atoms[:-1] <= atoms[1:]), f"Residue {i} not sorted"

    def test_sort_atoms_sorts_all_fields(self):
        """sort_atoms() reorders all atom-level fields."""
        p = ciffy.load("tests/data/9MDS.cif").chain(0).residue(0)

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

    def test_sort_atoms_idempotent(self):
        """Calling sort_atoms() twice gives same result."""
        p = ciffy.load("tests/data/9MDS.cif").chain(0).residue([0, 1, 2])
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

    def test_align_then_sort_canonical(self):
        """align().sort_atoms() produces canonical representation."""
        p = ciffy.load("tests/data/9MDS.cif").chain(0).residue([0, 1, 2])
        canonical = p.align().sort_atoms()

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

        atoms, elements, coords = expand_residue(Residue.C, start_terminal=False)
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

    def test_extend_sequence_updated(self):
        """Extended polymer has correct sequence."""
        p = template_with_coords("acg")

        atoms, elements, coords = expand_residue(Residue.U, start_terminal=False)
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

        atoms, elements, coords = expand_residue(Residue.G, start_terminal=False)
        transform = np.array([0, 0, 0, 0, 0, 6.0], dtype=np.float32)

        extended = p.extend(
            coordinates=coords,
            atoms=atoms,
            elements=elements,
            transform=transform,
            residue=Residue.G
        )

        assert extended.size(Scale.CHAIN) == 1

    def test_extend_linear_transform_helper(self):
        """linear_extend_transform produces valid spacing."""
        p = template_with_coords("a")

        atoms, elements, coords = expand_residue(Residue.C, start_terminal=False)

        # Get last residue info
        last_coords, last_atoms, last_res = p._residue_coords(-1)

        transform = linear_extend_transform(
            last_coords, last_atoms, last_res,
            atoms, Residue.C
        )

        # Transform should have identity rotation (first 3 zeros)
        assert transform[0] == 0
        assert transform[1] == 0
        assert transform[2] == 0

        # Translation should be positive (extending chain)
        assert transform[5] > 0

    def test_extend_from_different_residue_types(self):
        """Can extend from any RNA residue type."""
        for start_res in ['a', 'c', 'g', 'u']:
            p = template_with_coords(start_res)

            atoms, elements, coords = expand_residue(Residue.A, start_terminal=False)
            transform = np.array([0, 0, 0, 0, 0, 6.0], dtype=np.float32)

            extended = p.extend(
                coordinates=coords,
                atoms=atoms,
                elements=elements,
                transform=transform,
                residue=Residue.A
            )

            assert extended.size(Scale.RESIDUE) == 2
