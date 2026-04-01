"""
Tests for degenerate/edge case polymers.

Tests operations on empty, single-atom, and single-residue polymers.
"""

import pytest
import numpy as np

from ciffy import operations
from ciffy.backend import ops
from tests.utils import get_test_cif, get_like, BACKENDS, skip_if_no_torch, TORCH_AVAILABLE


class TestEmptyPolymer:
    """Test operations on empty (0-atom) polymers."""

    def test_empty_polymer_is_empty(self, backend):
        """Empty polymer reports empty() as True."""
        import ciffy

        _template = ciffy.template("a", backend=backend)
        empty = _template[_template.atoms < 0]  # Impossible mask

        assert empty.empty()
        assert empty.size() == 0

    def test_empty_polymer_sizes(self, backend):
        """Empty polymer has zero counts at all scales."""
        import ciffy
        from ciffy import Scale

        _template = ciffy.template("a", backend=backend)
        empty = _template[_template.atoms < 0]

        assert empty.size() == 0
        assert empty.size(Scale.RESIDUE) == 0
        assert empty.size(Scale.CHAIN) == 0

    def test_empty_polymer_coordinates_shape(self, backend):
        """Empty polymer has (0, 3) coordinate shape."""
        import ciffy

        polymer = ciffy.load(get_test_cif("3SKW"), backend=backend)
        empty = polymer[polymer.atoms < 0]

        assert empty.coordinates.shape == (0, 3)

    def test_empty_polymer_repr(self, backend):
        """Empty polymer __repr__ doesn't crash."""
        import ciffy

        _template = ciffy.template("a", backend=backend)
        empty = _template[_template.atoms < 0]

        repr_str = repr(empty)
        assert isinstance(repr_str, str)

    def test_empty_polymer_str(self, backend):
        """Empty polymer str() returns empty string."""
        import ciffy

        _template = ciffy.template("a", backend=backend)
        empty = _template[_template.atoms < 0]

        assert empty.sequence_str() == ""


class TestSingleAtomPolymer:
    """Test operations on single-atom polymers."""

    def test_single_atom_not_empty(self, backend):
        """Single atom polymer is not empty."""
        import ciffy

        _template = ciffy.template("g", backend=backend)  # Glycine
        single = _template[:1]

        assert not single.empty()
        assert single.size() == 1

    def test_single_atom_coordinates(self, backend):
        """Single atom polymer has (1, 3) coordinates."""
        import ciffy

        polymer = ciffy.load(get_test_cif("3SKW"), backend=backend)
        single = polymer[:1]

        assert single.coordinates.shape == (1, 3)

    def test_single_atom_pairwise_distances(self, backend):
        """Single atom pairwise_distances returns 1x1 zero matrix."""
        import ciffy

        polymer = ciffy.load(get_test_cif("3SKW"), backend=backend)
        single = polymer[:1]

        dists = operations.pairwise_distances(single)
        assert dists.shape == (1, 1)

        dist_val = dists[0, 0].item() if hasattr(dists[0, 0], 'item') else dists[0, 0]
        assert dist_val == 0.0

    def test_single_atom_knn_fails(self, backend):
        """Single atom knn raises ValueError (need at least 2 points)."""
        import ciffy

        polymer = ciffy.load(get_test_cif("3SKW"), backend=backend)
        single = polymer[:1]

        with pytest.raises(ValueError):
            operations.knn(single, k=1)


class TestSingleResiduePolymer:
    """Test operations on single-residue polymers."""

    def test_single_residue_size(self, backend):
        """Single residue polymer has correct residue count."""
        import ciffy
        from ciffy import Scale

        p = ciffy.template("a", backend=backend)

        assert p.size(Scale.RESIDUE) == 1
        assert p.size(Scale.CHAIN) == 1

    def test_single_residue_sequence(self, backend):
        """Single residue polymer has length-1 sequence."""
        import ciffy

        p = ciffy.template("a", backend=backend)

        assert len(p.sequence) == 1

    def test_single_residue_reduce(self, backend):
        """Reduce to residue scale works on single residue."""
        import ciffy
        from ciffy import Scale, Reduction

        full = ciffy.load(get_test_cif("3SKW"), backend=backend)
        p = full.select(0, Scale.RESIDUE)  # Select first residue
        result = p.reduce(p.coordinates, Scale.RESIDUE, Reduction.MEAN)

        assert result.shape == (1, 3)

    def test_single_residue_str(self, backend):
        """Single residue str() returns single character."""
        import ciffy

        p = ciffy.template("a", backend=backend)

        assert p.sequence_str() == "a"


class TestSingleChainPolymer:
    """Test operations on single-chain polymers."""

    def test_single_chain_select(self, backend):
        """chain(0) returns same structure on single chain."""
        import ciffy

        p = ciffy.template("acgu", backend=backend)
        chain = p.chain(0)

        assert chain.size() == p.size()

    def test_single_chain_chains_generator(self, backend):
        """chains() generator yields once for single chain."""
        import ciffy

        p = ciffy.template("acgu", backend=backend)
        chains = list(p.chains())

        assert len(chains) == 1

    def test_single_chain_out_of_bounds(self, backend):
        """chain(1) raises IndexError on single chain."""
        import ciffy

        p = ciffy.template("acgu", backend=backend)

        with pytest.raises(IndexError):
            p.chain(1)


class TestPolyHeteroPartition:
    """Test poly() and hetero() partitioning."""

    def test_poly_on_all_polymer(self, backend):
        """poly() on all-polymer structure returns same size."""
        import ciffy

        p = ciffy.template("acgu", backend=backend)
        poly = p

        assert poly.size() == p.size()

    def test_hetero_on_all_polymer(self, backend):
        """hetero() on all-polymer structure returns empty."""
        import ciffy

        p = ciffy.template("acgu", backend=backend)
        hetero = p.hetero()

        assert hetero.empty()

    def test_poly_returns_self(self, backend):
        """poly() returns self since Polymer only contains polymer atoms."""
        import ciffy

        p = ciffy.load(get_test_cif("3SKW"), backend=backend)

        # poly() returns self - all atoms are polymer atoms
        poly = p
        assert poly.size() == p.size()

    def test_hetero_separate(self, backend):
        """hetero() returns separate HETATM container."""
        import ciffy

        p = ciffy.load(get_test_cif("3SKW"), backend=backend)
        hetero = p.hetero()

        # HETATM atoms are separate from polymer
        assert hetero.size() == 48  # 3SKW has 48 HETATM atoms
        assert p.size() == 2826  # Polymer only has polymer atoms

    def test_poly_returns_self(self, backend):
        """poly() returns self since Polymer only contains polymer atoms."""
        import ciffy

        p = ciffy.load(get_test_cif("3SKW"), backend=backend)

        # poly() is now a no-op that returns self
        assert p.size() == p.size()


class TestChainsGenerator:
    """Test chains() generator edge cases."""

    def test_chains_single_chain(self, backend):
        """chains() on single-chain polymer yields once."""
        import ciffy

        p = ciffy.template("acgu", backend=backend)
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
        """molecule_type(RNA).chains() yields only RNA chains."""
        import ciffy
        from ciffy import Molecule

        p = ciffy.load(get_test_cif("9GCM"), backend=backend)  # RNA + protein
        rna_chains = list(p.molecule_type(Molecule.RNA).chains())
        all_chains = list(p.chains())

        assert len(rna_chains) <= len(all_chains)
        # Each yielded chain should be RNA
        for chain in rna_chains:
            assert chain.istype(Molecule.RNA)

    def test_chains_filter_no_match(self, backend):
        """molecule_type() with non-matching filter yields nothing."""
        import ciffy
        from ciffy import Molecule

        # Use real CIF with known molecule types (9MDS is all RNA)
        p = ciffy.load(get_test_cif("9MDS"), backend=backend)
        dna_chains = list(p.molecule_type(Molecule.DNA).chains())

        assert len(dna_chains) == 0

class TestResolvedStrip:
    """Test resolved() and strip() edge cases."""

    def test_resolved_all_resolved(self, backend):
        """resolved() on fully resolved structure returns all True."""
        import ciffy
        from ciffy import Scale

        p = ciffy.template("acgu", backend=backend)
        resolved = p.resolved(Scale.RESIDUE)

        # All residues should be resolved (have atoms)
        all_true = resolved.all() if hasattr(resolved, 'all') else np.all(resolved)
        assert all_true

    def test_strip_all_resolved(self, backend):
        """strip() on fully resolved structure returns same size."""
        import ciffy
        from ciffy import Scale

        p = ciffy.template("acgu", backend=backend)
        stripped = p.strip(Scale.RESIDUE)

        assert stripped.size(Scale.RESIDUE) == p.size(Scale.RESIDUE)


class TestBackendConversion:
    """Test backend conversion edge cases."""

    def test_same_backend_roundtrip(self, backend):
        """Converting to same backend returns same backend."""
        import ciffy

        p = ciffy.template("acgu", backend=backend)
        if backend == "numpy":
            p2 = p.numpy()
        else:
            p2 = p.torch()

        assert p2.backend == backend

    def test_roundtrip_preserves_data(self, backend):
        """Round-trip through the other backend preserves data."""
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")
        import ciffy

        p = ciffy.load(get_test_cif("3SKW"), backend=backend)
        coords_orig = np.asarray(p.coordinates).copy()

        if backend == "numpy":
            p2 = p.torch().numpy()
        else:
            p2 = p.numpy().torch()

        assert p2.backend == backend
        assert np.allclose(np.asarray(p2.coordinates), coords_orig)

    def test_to_requires_torch_backend(self):
        """to() raises ValueError on numpy backend."""
        import ciffy

        p = ciffy.load(get_test_cif("3SKW"), backend="numpy")

        with pytest.raises(ValueError, match="torch backend"):
            p.to("cpu")

    def test_to_no_args_returns_self(self, backend):
        """to() with no args returns self (torch) or raises (numpy)."""
        import ciffy

        p = ciffy.load(get_test_cif("3SKW"), backend=backend)

        if backend == "numpy":
            with pytest.raises(ValueError, match="torch backend"):
                p.to()
        else:
            p2 = p.to()
            assert p2 is p


class TestArraySetterValidation:
    """Test backend/device validation on array property setters."""

    def test_coordinates_rejects_wrong_backend(self, backend):
        """Setting coordinates with wrong backend raises TypeError."""
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")
        import ciffy

        other = "torch" if backend == "numpy" else "numpy"
        p = ciffy.load(get_test_cif("3SKW"), backend=backend)
        ref = get_like(other)
        wrong_coords = ops.randn((p.size(), 3), like=ref)

        with pytest.raises(TypeError, match=f"Cannot assign {other}"):
            p.coordinates = wrong_coords

    def test_atoms_rejects_wrong_backend(self, backend):
        """Setting atoms with wrong backend raises TypeError."""
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")
        import ciffy

        other = "torch" if backend == "numpy" else "numpy"
        p = ciffy.load(get_test_cif("3SKW"), backend=backend)
        ref = get_like(other)
        wrong_atoms = ops.zeros(p.size(), like=ref, dtype='int64')

        with pytest.raises(TypeError, match=f"Cannot assign {other}"):
            p.atoms = wrong_atoms

    def test_elements_rejects_wrong_backend(self, backend):
        """Setting elements with wrong backend raises TypeError."""
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")
        import ciffy

        other = "torch" if backend == "numpy" else "numpy"
        p = ciffy.load(get_test_cif("3SKW"), backend=backend)
        ref = get_like(other)
        wrong_elements = ops.zeros(p.size(), like=ref, dtype='int64')

        with pytest.raises(TypeError, match=f"Cannot assign {other}"):
            p.elements = wrong_elements

    def test_sequence_rejects_wrong_backend(self, backend):
        """Setting sequence with wrong backend raises TypeError."""
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")
        import ciffy
        from ciffy import Scale

        other = "torch" if backend == "numpy" else "numpy"
        p = ciffy.load(get_test_cif("3SKW"), backend=backend)
        ref = get_like(other)
        wrong_seq = ops.zeros(p.size(Scale.RESIDUE), like=ref, dtype='int64')

        with pytest.raises(TypeError, match=f"Cannot assign {other}"):
            p.sequence = wrong_seq

    def test_lengths_is_readonly(self, backend):
        """lengths is a read-only property delegating to hierarchy."""
        import ciffy

        p = ciffy.template("acgu", backend=backend)

        # Python 3.10: "can't set attribute", Python 3.11+: "has no setter"
        with pytest.raises(AttributeError, match="can't set attribute|has no setter"):
            p.lengths = p.lengths

    def test_coordinates_accepts_same_backend(self, backend):
        """Setting coordinates with same backend works."""
        import ciffy

        p = ciffy.template("acgu", backend=backend)
        new_coords = ops.randn((p.size(), 3), like=get_like(backend))

        p.coordinates = new_coords
        assert ops.allclose(p.coordinates, new_coords)


class TestDeviceProperty:
    """Test device property on Polymer."""

    def test_device_numpy_returns_none(self):
        """device property returns None for numpy backend."""
        import ciffy

        p = ciffy.template("acgu", backend="numpy")
        assert p.device is None

    def test_device_torch_returns_cpu(self, backend):
        """device property returns 'cpu' for torch, None for numpy."""
        import ciffy

        p = ciffy.template("acgu", backend=backend)
        if backend == "torch":
            assert p.device == "cpu"
        else:
            assert p.device is None

    def test_device_after_backend_conversion(self, backend):
        """device property updates after backend conversion."""
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")
        import ciffy

        p = ciffy.template("acgu", backend=backend)

        if backend == "numpy":
            assert p.device is None
            p2 = p.torch()
            assert p2.device == "cpu"
        else:
            assert p.device == "cpu"
            p2 = p.numpy()
            assert p2.device is None


class TestMembershipMethod:
    """Test the membership(scale) method."""

    def test_membership_residue_shape(self, backend):
        """membership(RESIDUE) returns array with shape (num_atoms,)."""
        import ciffy
        from ciffy import Scale

        p = ciffy.template("acgu", backend=backend)
        idx = p.membership(Scale.RESIDUE)

        assert idx.shape == (p.size(),)

    def test_membership_residue_values(self, backend):
        """membership(RESIDUE) returns values in [0, num_residues)."""
        import ciffy
        from ciffy import Scale

        p = ciffy.template("acgu", backend=backend)
        idx = p.membership(Scale.RESIDUE)

        # Convert to numpy for checking
        idx_np = idx.numpy() if hasattr(idx, 'numpy') else idx
        assert idx_np.min() == 0
        assert idx_np.max() == p.size(Scale.RESIDUE) - 1

    def test_membership_residue_unique_count(self, backend):
        """membership(RESIDUE) has num_residues unique values."""
        import ciffy
        from ciffy import Scale

        p = ciffy.template("acgu", backend=backend)
        idx = p.membership(Scale.RESIDUE)

        idx_np = idx.numpy() if hasattr(idx, 'numpy') else idx
        assert len(set(idx_np)) == p.size(Scale.RESIDUE)

    def test_membership_chain_single_chain(self, backend):
        """membership(CHAIN) returns all zeros for single chain."""
        import ciffy
        from ciffy import Scale

        p = ciffy.template("acgu", backend=backend)
        idx = p.membership(Scale.CHAIN)

        idx_np = idx.numpy() if hasattr(idx, 'numpy') else idx
        assert (idx_np == 0).all()

    def test_membership_chain_multi_chain(self, backend):
        """membership(CHAIN) returns correct indices for multi-chain."""
        import ciffy
        from ciffy import Scale

        p = ciffy.template(["acgu", "MGKLF"], backend=backend)
        idx = p.membership(Scale.CHAIN)

        idx_np = idx.numpy() if hasattr(idx, 'numpy') else idx
        assert len(set(idx_np)) == 2
        assert idx_np.min() == 0
        assert idx_np.max() == 1

    def test_membership_molecule_all_zeros(self, backend):
        """membership(MOLECULE) returns all zeros."""
        import ciffy
        from ciffy import Scale

        p = ciffy.template("acgu", backend=backend)
        idx = p.membership(Scale.MOLECULE)

        idx_np = idx.numpy() if hasattr(idx, 'numpy') else idx
        assert (idx_np == 0).all()

    def test_membership_dtype(self, backend):
        """membership() returns integer dtype."""
        import ciffy
        from ciffy import Scale

        p = ciffy.template("acgu", backend=backend)
        idx = p.membership(Scale.RESIDUE)

        idx_np = np.asarray(idx)
        assert idx_np.dtype == np.int64

    def test_membership_consistency_with_counts(self, backend):
        """membership() is consistent with counts()."""
        import ciffy
        from ciffy import Scale

        p = ciffy.template("acgu", backend=backend)
        idx = p.membership(Scale.RESIDUE)
        sizes = p.counts(Scale.RESIDUE)

        # Count atoms per residue from index
        idx_np = idx.numpy() if hasattr(idx, 'numpy') else idx
        sizes_np = sizes.numpy() if hasattr(sizes, 'numpy') else sizes

        for i, expected_size in enumerate(sizes_np):
            actual_count = (idx_np == i).sum()
            assert actual_count == expected_size


class TestAtomNames:
    """Test atom_names() method."""

    def test_atom_names_returns_list(self, backend):
        """atom_names() returns a list."""
        import ciffy

        p = ciffy.template("acgu", backend=backend)
        names = p.atom_names()

        assert isinstance(names, list)

    def test_atom_names_length_matches_size(self, backend):
        """atom_names() length equals number of atoms."""
        import ciffy

        p = ciffy.template("acgu", backend=backend)
        names = p.atom_names()

        assert len(names) == p.size()

    def test_atom_names_are_strings(self, backend):
        """atom_names() entries are strings."""
        import ciffy

        p = ciffy.template("acgu", backend=backend)
        names = p.atom_names()

        for name in names:
            assert isinstance(name, str)

    def test_atom_names_common_atoms(self, backend):
        """atom_names() includes known atom names."""
        import ciffy

        p = ciffy.template("a", backend=backend)
        names = p.atom_names()

        # Adenine should have phosphate atoms
        assert any("P" in name or "O" in name or "C" in name for name in names)

    def test_atom_names_empty_polymer(self, backend):
        """atom_names() on empty polymer returns empty list."""
        import ciffy

        _template = ciffy.template("a", backend=backend)
        empty = _template[_template.atoms < 0]
        names = empty.atom_names()

        assert names == []

    def test_atom_names_unknown_returns_question_mark(self, backend):
        """atom_names() returns '?' for unknown atom types."""
        import ciffy

        p = ciffy.template("a", backend=backend)

        # Set invalid atom value
        p.atoms = ops.full((p.size(),), 99999, like=p.atoms)

        names = p.atom_names()
        assert all(name == "?" for name in names)


class TestChainInfo:
    """Test chain_info() method."""

    def test_chain_info_returns_list(self, backend):
        """chain_info() returns a list."""
        import ciffy

        p = ciffy.load(get_test_cif("3SKW"), backend=backend)
        info = p.chain_info()

        assert isinstance(info, list)

    def test_chain_info_length_matches_chains(self, backend):
        """chain_info() has one entry per chain."""
        import ciffy
        from ciffy import Scale

        p = ciffy.load(get_test_cif("3SKW"), backend=backend)
        info = p.chain_info()

        assert len(info) == p.size(Scale.CHAIN)

    def test_chain_info_has_required_keys(self, backend):
        """chain_info() entries have required keys."""
        import ciffy

        p = ciffy.load(get_test_cif("3SKW"), backend=backend)
        info = p.chain_info()

        required_keys = ['chain', 'type', 'res', 'atoms']
        for entry in info:
            for key in required_keys:
                assert key in entry, f"Missing key: {key}"

    def test_chain_info_values_types(self, backend):
        """chain_info() values have correct types."""
        import ciffy

        p = ciffy.load(get_test_cif("3SKW"), backend=backend)
        info = p.chain_info()

        for entry in info:
            assert isinstance(entry['chain'], str)
            assert isinstance(entry['type'], str)
            assert isinstance(entry['res'], int)
            assert isinstance(entry['atoms'], int)

    def test_chain_info_atoms_sum(self, backend):
        """chain_info() atom counts sum to total atoms."""
        import ciffy

        p = ciffy.load(get_test_cif("3SKW"), backend=backend)
        info = p.chain_info()

        total = sum(entry['atoms'] for entry in info)
        assert total == p.size()

    def test_chain_info__template_polymer(self, backend):
        """chain_info() works on _template polymers."""
        import ciffy

        p = ciffy.template("acgu", backend=backend)
        info = p.chain_info()

        assert len(info) == 1
        assert info[0]['chain'] is not None
        assert info[0]['res'] == 4


class TestDetach:
    """Test detach() method."""

    def test_detach_returns_self(self, backend):
        """detach() returns self for method chaining."""
        import ciffy

        p = ciffy.template("acgu", backend=backend)
        result = p.detach()

        assert result is p

    def test_detach_noop_numpy(self):
        """detach() is a no-op on numpy backend."""
        import ciffy

        p = ciffy.load(get_test_cif("3SKW"), backend="numpy")
        original_coords = p.coordinates.copy()

        p.detach()

        assert np.allclose(p.coordinates, original_coords)

    @pytest.mark.parametrize("backend", ["torch"])
    def test_detach_removes_grad(self, backend):
        """detach() removes gradient tracking on torch tensors."""
        import ciffy

        p = ciffy.load(get_test_cif("3SKW"), backend=backend)
        p.coordinates = p.coordinates.clone().requires_grad_(True)

        assert p.coordinates.requires_grad

        p.detach()

        assert not p.coordinates.requires_grad

    @pytest.mark.parametrize("backend", ["torch"])
    def test_detach_preserves_values(self, backend):
        """detach() preserves coordinate values."""
        import ciffy

        p = ciffy.load(get_test_cif("3SKW"), backend=backend)
        original_coords = p.coordinates.clone()

        p.coordinates = p.coordinates.requires_grad_(True)
        p.detach()

        assert ops.allclose(p.coordinates, original_coords)


class TestCanonical:
    """Test canonical() method."""

    def test_canonical_returns_polymer(self, backend):
        """canonical() returns a Polymer."""
        import ciffy

        p = ciffy.template("acgu", backend=backend)
        result = p.canonical()

        assert isinstance(result, ciffy.Polymer)

    def test_canonical_preserves_standard_residues(self, backend):
        """canonical() preserves standard RNA residues."""
        import ciffy
        from ciffy import Scale

        p = ciffy.template("acgu", backend=backend)
        canonical = p.canonical()

        # All standard, should be preserved
        assert canonical.size(Scale.RESIDUE) == p.size(Scale.RESIDUE)
        assert canonical.size() == p.size()

    def test_canonical_preserves_standard_protein(self, backend):
        """canonical() preserves standard amino acids."""
        import ciffy
        from ciffy import Scale

        p = ciffy.template("MGKLF", backend=backend)
        canonical = p.canonical()

        # All standard amino acids, should be preserved
        assert canonical.size(Scale.RESIDUE) == p.size(Scale.RESIDUE)

    def test_canonical_filters_modified_residues(self, backend):
        """canonical() filters out modified residues from real CIF."""
        import ciffy
        from ciffy import Scale

        # Load a structure that may have modified residues
        p = ciffy.load(get_test_cif("3SKW"), backend=backend)
        canonical = p.canonical()

        # canonical should have same or fewer residues
        assert canonical.size(Scale.RESIDUE) <= p.size(Scale.RESIDUE)

    def test_canonical_empty_on_all_noncanonical(self, backend):
        """canonical() on all non-canonical residues returns empty."""
        import ciffy
        from ciffy.biochemistry import Residue

        p = ciffy.template("a", backend=backend)

        # Set to a non-canonical (modified) residue type - H2U is dihydrouridine
        p.sequence = ops.array([Residue.H2U.value], like=p.sequence)

        canonical = p.canonical()
        assert canonical.empty()

    def test_canonical_sequence_preserved(self, backend):
        """canonical() preserves sequence string for standard residues."""
        import ciffy

        p = ciffy.template("acgu", backend=backend)
        canonical = p.canonical()

        assert canonical.sequence_str() == p.sequence_str()


# =============================================================================
# VdW Radii Tests
# =============================================================================


class TestVdwRadii:
    """Test Polymer.vdw_radii property."""

    def test_vdw_radii_shape(self, backend):
        """vdw_radii returns array with shape (n_atoms,)."""
        import ciffy

        p = ciffy.load(get_test_cif("3SKW"), backend=backend)
        radii = p.vdw_radii

        assert radii.shape == (p.size(),)

    def test_vdw_radii_positive(self, backend):
        """vdw_radii values are all positive."""
        import ciffy

        p = ciffy.load(get_test_cif("3SKW"), backend=backend)
        radii = np.asarray(p.vdw_radii)

        assert (radii > 0).all()

    def test_vdw_radii_reasonable_range(self, backend):
        """vdw_radii values are in reasonable range (0.5-3.0 Angstroms)."""
        import ciffy

        p = ciffy.load(get_test_cif("3SKW"), backend=backend)
        radii = np.asarray(p.vdw_radii)

        assert radii.min() >= 0.5
        assert radii.max() <= 3.0

    def test_vdw_radii_known_values(self, backend):
        """vdw_radii has expected values for common elements."""
        import ciffy

        p = ciffy.load(get_test_cif("3SKW"), backend=backend)
        radii = np.asarray(p.vdw_radii)
        elements = np.asarray(p.elements)

        # Carbon (element 6) should have ~1.7 A radius
        carbon_mask = elements == 6
        if carbon_mask.any():
            carbon_radii = radii[carbon_mask]
            assert np.allclose(carbon_radii, 1.7, atol=0.1)

        # Nitrogen (element 7) should have ~1.55 A radius
        nitrogen_mask = elements == 7
        if nitrogen_mask.any():
            nitrogen_radii = radii[nitrogen_mask]
            assert np.allclose(nitrogen_radii, 1.55, atol=0.1)

    def test_vdw_radii_empty_polymer(self, backend):
        """vdw_radii on empty polymer returns empty array."""
        import ciffy

        _template = ciffy.template("a", backend=backend)
        empty = _template[_template.atoms < 0]  # Impossible mask

        radii = empty.vdw_radii
        assert radii.shape == (0,)


# =============================================================================
# Extended Resolved/Strip Tests
# =============================================================================


class TestResolvedExtended:
    """Extended tests for resolved() method."""

    def test_resolved_chain_scale(self, backend):
        """resolved() at CHAIN scale returns per-chain mask."""
        import ciffy
        from ciffy import Scale

        p = ciffy.load(get_test_cif("9GCM"), backend=backend)
        resolved = p.resolved(Scale.CHAIN)

        # Should have one entry per chain
        assert len(resolved) == p.size(Scale.CHAIN)

    def test_resolved_with_unresolved(self, backend):
        """resolved() correctly identifies unresolved residues."""
        from tests.utils import get_test_cif
        import ciffy
        from ciffy import Scale

        # 2F8S has 4 unresolved residues
        p = ciffy.load(get_test_cif("2F8S"), backend=backend)
        resolved = np.asarray(p.resolved(Scale.RESIDUE))

        # Not all residues are resolved
        n_resolved = resolved.sum()
        n_total = len(resolved)
        assert n_resolved < n_total

    def test_resolved_empty_polymer(self, backend):
        """resolved() on empty polymer returns empty mask."""
        import ciffy
        from ciffy import Scale

        _template = ciffy.template("a", backend=backend)
        empty = _template[_template.atoms < 0]

        resolved = empty.resolved(Scale.RESIDUE)
        assert len(resolved) == 0


class TestStripExtended:
    """Extended tests for strip() method."""

    def test_strip_with_unresolved(self, backend):
        """strip() removes unresolved residues."""
        from tests.utils import get_test_cif
        import ciffy
        from ciffy import Scale

        # 2F8S has 4 unresolved residues
        p = ciffy.load(get_test_cif("2F8S"), backend=backend)
        n_before = p.size(Scale.RESIDUE)

        stripped = p.strip(Scale.RESIDUE)
        n_after = stripped.size(Scale.RESIDUE)

        assert n_after < n_before
        assert n_after == 1452  # Known value from earlier test

    def test_strip_chain_scale(self, backend):
        """strip() at CHAIN scale removes empty chains."""
        import ciffy
        from ciffy import Scale

        p = ciffy.load(get_test_cif("3SKW"), backend=backend)
        n_chains_before = p.size(Scale.CHAIN)

        stripped = p.strip(Scale.CHAIN)
        n_chains_after = stripped.size(Scale.CHAIN)

        # 3SKW should have no empty chains, so size unchanged
        assert n_chains_after == n_chains_before

    def test_strip_empty_polymer(self, backend):
        """strip() on empty polymer returns empty polymer."""
        import ciffy
        from ciffy import Scale

        _template = ciffy.template("a", backend=backend)
        empty = _template[_template.atoms < 0]

        stripped = empty.strip(Scale.RESIDUE)
        assert stripped.empty()

    def test_strip_preserves_resolved_coordinates(self, backend):
        """strip() preserves coordinates of resolved residues."""
        from tests.utils import get_test_cif
        import ciffy
        from ciffy import Scale

        p = ciffy.load(get_test_cif("2F8S"), backend=backend)

        # Get coordinates of first chain (fully resolved)
        chain0_before = p.chain(0)
        coords_before = np.asarray(chain0_before.coordinates).copy()

        stripped = p.strip(Scale.RESIDUE)
        chain0_after = stripped.chain(0)
        coords_after = np.asarray(chain0_after.coordinates)

        assert np.allclose(coords_before, coords_after)
