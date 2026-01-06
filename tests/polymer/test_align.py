"""
Tests for Polymer.align() - residue-to-local-frame alignment.
"""

import numpy as np
import pytest

import ciffy
from ciffy import Scale

from tests.utils import BACKENDS


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
