"""
Tests for degenerate/edge case polymers.

Tests operations on empty, single-atom, and single-residue polymers.
"""

import pytest
import numpy as np

from tests.utils import get_test_cif, BACKENDS


class TestEmptyPolymer:
    """Test operations on empty (0-atom) polymers."""

    def test_empty_polymer_is_empty(self, backend):
        """Empty polymer reports empty() as True."""
        import ciffy

        template = ciffy.from_sequence("a", backend=backend)
        empty = template[template.atoms < 0]  # Impossible mask

        assert empty.empty()
        assert empty.size() == 0

    def test_empty_polymer_sizes(self, backend):
        """Empty polymer has zero counts at all scales."""
        import ciffy
        from ciffy import Scale

        template = ciffy.from_sequence("a", backend=backend)
        empty = template[template.atoms < 0]

        assert empty.size() == 0
        assert empty.size(Scale.RESIDUE) == 0
        assert empty.size(Scale.CHAIN) == 0

    def test_empty_polymer_coordinates_shape(self, backend):
        """Empty polymer has (0, 3) coordinate shape."""
        import ciffy

        template = ciffy.from_sequence("a", backend=backend)
        empty = template[template.atoms < 0]

        assert empty.coordinates.shape == (0, 3)

    def test_empty_polymer_repr(self, backend):
        """Empty polymer __repr__ doesn't crash."""
        import ciffy

        template = ciffy.from_sequence("a", backend=backend)
        empty = template[template.atoms < 0]

        repr_str = repr(empty)
        assert isinstance(repr_str, str)

    def test_empty_polymer_str(self, backend):
        """Empty polymer str() returns empty string."""
        import ciffy

        template = ciffy.from_sequence("a", backend=backend)
        empty = template[template.atoms < 0]

        assert empty.sequence_str() == ""


class TestSingleAtomPolymer:
    """Test operations on single-atom polymers."""

    def test_single_atom_not_empty(self, backend):
        """Single atom polymer is not empty."""
        import ciffy

        template = ciffy.from_sequence("g", backend=backend)  # Glycine
        single = template[:1]

        assert not single.empty()
        assert single.size() == 1

    def test_single_atom_coordinates(self, backend):
        """Single atom polymer has (1, 3) coordinates."""
        import ciffy

        template = ciffy.from_sequence("g", backend=backend)
        single = template[:1]

        assert single.coordinates.shape == (1, 3)

    def test_single_atom_pairwise_distances(self, backend):
        """Single atom pairwise_distances returns 1x1 zero matrix."""
        import ciffy

        template = ciffy.from_sequence("g", backend=backend)
        single = template[:1]

        dists = single.pairwise_distances()
        assert dists.shape == (1, 1)

        dist_val = dists[0, 0].item() if hasattr(dists[0, 0], 'item') else dists[0, 0]
        assert dist_val == 0.0

    def test_single_atom_knn_fails(self, backend):
        """Single atom knn raises ValueError (need at least 2 points)."""
        import ciffy

        template = ciffy.from_sequence("g", backend=backend)
        single = template[:1]

        with pytest.raises(ValueError):
            single.knn(k=1)


class TestSingleResiduePolymer:
    """Test operations on single-residue polymers."""

    def test_single_residue_size(self, backend):
        """Single residue polymer has correct residue count."""
        import ciffy
        from ciffy import Scale

        p = ciffy.from_sequence("a", backend=backend)

        assert p.size(Scale.RESIDUE) == 1
        assert p.size(Scale.CHAIN) == 1

    def test_single_residue_sequence(self, backend):
        """Single residue polymer has length-1 sequence."""
        import ciffy

        p = ciffy.from_sequence("a", backend=backend)

        assert len(p.sequence) == 1

    def test_single_residue_reduce(self, backend):
        """Reduce to residue scale works on single residue."""
        import ciffy
        from ciffy import Scale, Reduction

        p = ciffy.from_sequence("a", backend=backend)
        result = p.reduce(p.coordinates, Scale.RESIDUE, Reduction.MEAN)

        assert result.shape == (1, 3)

    def test_single_residue_str(self, backend):
        """Single residue str() returns single character."""
        import ciffy

        p = ciffy.from_sequence("a", backend=backend)

        assert p.sequence_str() == "a"


class TestSingleChainPolymer:
    """Test operations on single-chain polymers."""

    def test_single_chain_by_index(self, backend):
        """by_index(0) returns same structure on single chain."""
        import ciffy

        p = ciffy.from_sequence("acgu", backend=backend)
        chain = p.by_index(0)

        assert chain.size() == p.size()

    def test_single_chain_chains_generator(self, backend):
        """chains() generator yields once for single chain."""
        import ciffy

        p = ciffy.from_sequence("acgu", backend=backend)
        chains = list(p.chains())

        assert len(chains) == 1

    def test_single_chain_out_of_bounds(self, backend):
        """by_index(1) raises IndexError on single chain."""
        import ciffy

        p = ciffy.from_sequence("acgu", backend=backend)

        with pytest.raises(IndexError):
            p.by_index(1)


class TestPolyHeteroPartition:
    """Test poly() and hetero() partitioning."""

    def test_poly_on_all_polymer(self, backend):
        """poly() on all-polymer structure returns same size."""
        import ciffy

        p = ciffy.from_sequence("acgu", backend=backend)
        poly = p.poly()

        assert poly.size() == p.size()

    def test_hetero_on_all_polymer(self, backend):
        """hetero() on all-polymer structure returns empty."""
        import ciffy

        p = ciffy.from_sequence("acgu", backend=backend)
        hetero = p.hetero()

        assert hetero.empty()

    def test_poly_hetero_sum(self, backend):
        """poly() + hetero() atom counts sum to total."""
        import ciffy

        p = ciffy.load(get_test_cif("3SKW"), backend=backend)

        poly_count = p.poly().size()
        hetero_count = p.hetero().size()

        assert poly_count + hetero_count == p.size()

    def test_poly_matches_polymer_count(self, backend):
        """poly() size matches polymer_count attribute."""
        import ciffy

        p = ciffy.load(get_test_cif("3SKW"), backend=backend)

        assert p.poly().size() == p.polymer_count


class TestChainsGenerator:
    """Test chains() generator edge cases."""

    def test_chains_single_chain(self, backend):
        """chains() on single-chain polymer yields once."""
        import ciffy

        p = ciffy.from_sequence("acgu", backend=backend)
        chains = list(p.chains())

        assert len(chains) == 1
        assert chains[0].size() == p.size()

    def test_chains_multi_chain(self, backend):
        """chains() yields correct count on multi-chain."""
        import ciffy
        from ciffy import Scale

        p = ciffy.load(get_test_cif("9GCM"), backend=backend)
        chains = list(p.chains())

        assert len(chains) == p.size(Scale.CHAIN)

    def test_chains_with_filter_rna(self, backend):
        """chains(mol=RNA) yields only RNA chains."""
        import ciffy
        from ciffy import Molecule

        p = ciffy.load(get_test_cif("9GCM"), backend=backend)  # RNA + protein
        rna_chains = list(p.chains(mol=Molecule.RNA))
        all_chains = list(p.chains())

        assert len(rna_chains) <= len(all_chains)
        # Each yielded chain should be RNA
        for chain in rna_chains:
            assert chain.istype(Molecule.RNA)

    def test_chains_filter_no_match(self, backend):
        """chains() with non-matching filter yields nothing."""
        import ciffy
        from ciffy import Molecule

        p = ciffy.from_sequence("acgu", backend=backend)  # RNA only
        dna_chains = list(p.chains(mol=Molecule.DNA))

        assert len(dna_chains) == 0


class TestResolvedStrip:
    """Test resolved() and strip() edge cases."""

    def test_resolved_all_resolved(self, backend):
        """resolved() on fully resolved structure returns all True."""
        import ciffy
        from ciffy import Scale

        p = ciffy.from_sequence("acgu", backend=backend)
        resolved = p.resolved(Scale.RESIDUE)

        # All residues should be resolved (have atoms)
        all_true = resolved.all() if hasattr(resolved, 'all') else np.all(resolved)
        assert all_true

    def test_strip_all_resolved(self, backend):
        """strip() on fully resolved structure returns same size."""
        import ciffy
        from ciffy import Scale

        p = ciffy.from_sequence("acgu", backend=backend)
        stripped = p.strip(Scale.RESIDUE)

        assert stripped.size(Scale.RESIDUE) == p.size(Scale.RESIDUE)


class TestBackendConversion:
    """Test backend conversion edge cases."""

    def test_numpy_to_numpy(self):
        """numpy() on numpy polymer returns same object."""
        import ciffy

        p = ciffy.from_sequence("acgu", backend="numpy")
        p2 = p.numpy()

        # Should return self (or equivalent)
        assert p2.backend == "numpy"

    def test_torch_to_torch(self):
        """torch() on torch polymer returns same object."""
        import ciffy

        p = ciffy.from_sequence("acgu", backend="torch")
        p2 = p.torch()

        assert p2.backend == "torch"

    def test_numpy_to_torch_and_back(self):
        """Round-trip numpy -> torch -> numpy preserves data."""
        import ciffy

        p = ciffy.from_sequence("acgu", backend="numpy")
        coords_orig = p.coordinates.copy()

        p_torch = p.torch()
        assert p_torch.backend == "torch"

        p_back = p_torch.numpy()
        assert p_back.backend == "numpy"

        assert np.allclose(p_back.coordinates, coords_orig)

    def test_to_requires_torch_backend(self):
        """to() raises ValueError on numpy backend."""
        import ciffy

        p = ciffy.from_sequence("acgu", backend="numpy")

        with pytest.raises(ValueError, match="torch backend"):
            p.to("cpu")

    def test_to_no_args_returns_self(self):
        """to() with no args returns same object."""
        import ciffy

        p = ciffy.from_sequence("acgu", backend="torch")
        p2 = p.to()

        assert p2 is p


class TestArraySetterValidation:
    """Test backend/device validation on array property setters."""

    def test_coordinates_rejects_wrong_backend(self):
        """Setting coordinates with wrong backend raises TypeError."""
        import ciffy
        import torch

        p_numpy = ciffy.from_sequence("acgu", backend="numpy")
        torch_coords = torch.randn(p_numpy.size(), 3)

        with pytest.raises(TypeError, match="Cannot assign torch"):
            p_numpy.coordinates = torch_coords

    def test_coordinates_rejects_wrong_backend_reverse(self):
        """Setting torch coordinates with numpy raises TypeError."""
        import ciffy

        p_torch = ciffy.from_sequence("acgu", backend="torch")
        numpy_coords = np.random.randn(p_torch.size(), 3).astype(np.float32)

        with pytest.raises(TypeError, match="Cannot assign numpy"):
            p_torch.coordinates = numpy_coords

    def test_atoms_rejects_wrong_backend(self):
        """Setting atoms with wrong backend raises TypeError."""
        import ciffy
        import torch

        p_numpy = ciffy.from_sequence("acgu", backend="numpy")
        torch_atoms = torch.zeros(p_numpy.size(), dtype=torch.long)

        with pytest.raises(TypeError, match="Cannot assign torch"):
            p_numpy.atoms = torch_atoms

    def test_elements_rejects_wrong_backend(self):
        """Setting elements with wrong backend raises TypeError."""
        import ciffy
        import torch

        p_numpy = ciffy.from_sequence("acgu", backend="numpy")
        torch_elements = torch.zeros(p_numpy.size(), dtype=torch.long)

        with pytest.raises(TypeError, match="Cannot assign torch"):
            p_numpy.elements = torch_elements

    def test_sequence_rejects_wrong_backend(self):
        """Setting sequence with wrong backend raises TypeError."""
        import ciffy
        import torch
        from ciffy import Scale

        p_numpy = ciffy.from_sequence("acgu", backend="numpy")
        torch_seq = torch.zeros(p_numpy.size(Scale.RESIDUE), dtype=torch.long)

        with pytest.raises(TypeError, match="Cannot assign torch"):
            p_numpy.sequence = torch_seq

    def test_lengths_rejects_wrong_backend(self):
        """Setting lengths with wrong backend raises TypeError."""
        import ciffy
        import torch
        from ciffy import Scale

        p_numpy = ciffy.from_sequence("acgu", backend="numpy")
        torch_lengths = torch.zeros(p_numpy.size(Scale.CHAIN), dtype=torch.long)

        with pytest.raises(TypeError, match="Cannot assign torch"):
            p_numpy.lengths = torch_lengths

    def test_coordinates_accepts_same_backend(self):
        """Setting coordinates with same backend works."""
        import ciffy

        p = ciffy.from_sequence("acgu", backend="numpy")
        new_coords = np.random.randn(p.size(), 3).astype(np.float32)

        p.coordinates = new_coords
        assert np.allclose(p.coordinates, new_coords)

    def test_coordinates_accepts_same_backend_torch(self):
        """Setting torch coordinates on torch polymer works."""
        import ciffy
        import torch

        p = ciffy.from_sequence("acgu", backend="torch")
        new_coords = torch.randn(p.size(), 3)

        p.coordinates = new_coords
        assert torch.allclose(p.coordinates, new_coords)


class TestDeviceProperty:
    """Test device property on Polymer."""

    def test_device_numpy_returns_none(self):
        """device property returns None for numpy backend."""
        import ciffy

        p = ciffy.from_sequence("acgu", backend="numpy")
        assert p.device is None

    def test_device_torch_returns_cpu(self):
        """device property returns 'cpu' for torch CPU tensor."""
        import ciffy

        p = ciffy.from_sequence("acgu", backend="torch")
        assert p.device == "cpu"

    def test_device_after_backend_conversion(self):
        """device property updates after backend conversion."""
        import ciffy

        p = ciffy.from_sequence("acgu", backend="numpy")
        assert p.device is None

        p_torch = p.torch()
        assert p_torch.device == "cpu"


class TestIndexMethod:
    """Test the index(scale) method."""

    def test_index_residue_shape(self, backend):
        """index(RESIDUE) returns array with shape (num_atoms,)."""
        import ciffy
        from ciffy import Scale

        p = ciffy.from_sequence("acgu", backend=backend)
        idx = p.index(Scale.RESIDUE)

        assert idx.shape == (p.size(),)

    def test_index_residue_values(self, backend):
        """index(RESIDUE) returns values in [0, num_residues)."""
        import ciffy
        from ciffy import Scale

        p = ciffy.from_sequence("acgu", backend=backend)
        idx = p.index(Scale.RESIDUE)

        # Convert to numpy for checking
        idx_np = idx.numpy() if hasattr(idx, 'numpy') else idx
        assert idx_np.min() == 0
        assert idx_np.max() == p.size(Scale.RESIDUE) - 1

    def test_index_residue_unique_count(self, backend):
        """index(RESIDUE) has num_residues unique values."""
        import ciffy
        from ciffy import Scale

        p = ciffy.from_sequence("acgu", backend=backend)
        idx = p.index(Scale.RESIDUE)

        idx_np = idx.numpy() if hasattr(idx, 'numpy') else idx
        assert len(set(idx_np)) == p.size(Scale.RESIDUE)

    def test_index_chain_single_chain(self, backend):
        """index(CHAIN) returns all zeros for single chain."""
        import ciffy
        from ciffy import Scale

        p = ciffy.from_sequence("acgu", backend=backend)
        idx = p.index(Scale.CHAIN)

        idx_np = idx.numpy() if hasattr(idx, 'numpy') else idx
        assert (idx_np == 0).all()

    def test_index_chain_multi_chain(self, backend):
        """index(CHAIN) returns correct indices for multi-chain."""
        import ciffy
        from ciffy import Scale

        p = ciffy.from_sequence(["acgu", "MGKLF"], backend=backend)
        idx = p.index(Scale.CHAIN)

        idx_np = idx.numpy() if hasattr(idx, 'numpy') else idx
        assert len(set(idx_np)) == 2
        assert idx_np.min() == 0
        assert idx_np.max() == 1

    def test_index_molecule_all_zeros(self, backend):
        """index(MOLECULE) returns all zeros."""
        import ciffy
        from ciffy import Scale

        p = ciffy.from_sequence("acgu", backend=backend)
        idx = p.index(Scale.MOLECULE)

        idx_np = idx.numpy() if hasattr(idx, 'numpy') else idx
        assert (idx_np == 0).all()

    def test_index_dtype(self, backend):
        """index() returns integer dtype."""
        import ciffy
        from ciffy import Scale

        p = ciffy.from_sequence("acgu", backend=backend)
        idx = p.index(Scale.RESIDUE)

        if backend == "torch":
            import torch
            assert idx.dtype == torch.int64
        else:
            assert idx.dtype == np.int64

    def test_index_consistency_with_sizes(self, backend):
        """index() is consistent with sizes()."""
        import ciffy
        from ciffy import Scale

        p = ciffy.from_sequence("acgu", backend=backend)
        idx = p.index(Scale.RESIDUE)
        sizes = p.sizes(Scale.RESIDUE)

        # Count atoms per residue from index
        idx_np = idx.numpy() if hasattr(idx, 'numpy') else idx
        sizes_np = sizes.numpy() if hasattr(sizes, 'numpy') else sizes

        for i, expected_size in enumerate(sizes_np):
            actual_count = (idx_np == i).sum()
            assert actual_count == expected_size
