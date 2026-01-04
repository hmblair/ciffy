"""
Tests for Polymer properties: bfactors, resolution, bonds, nonpoly.

These are optional/computed properties that need dedicated test coverage.
"""

import glob
import pytest
import numpy as np

from tests.utils import DATA_DIR, get_test_cif


CIF_FILES = sorted(glob.glob(str(DATA_DIR / "*.cif")))


class TestBfactors:
    """Test bfactors property."""

    @pytest.mark.parametrize("cif_file", CIF_FILES)
    def test_bfactors_from_cif_file(self, cif_file, backend):
        """bfactors loaded from CIF file is not None."""
        from ciffy import load

        polymer = load(cif_file, backend=backend)

        # CIF files should have B-factors
        assert polymer.bfactors is not None

    @pytest.mark.parametrize("cif_file", CIF_FILES)
    def test_bfactors_shape_matches_atoms(self, cif_file, backend):
        """bfactors array has same length as atoms."""
        from ciffy import load

        polymer = load(cif_file, backend=backend)

        assert polymer.bfactors.shape == (polymer.size(),)

    def test_bfactors_raises_for_template(self, backend):
        """bfactors raises AttributeError for template-generated polymers."""
        from ciffy import from_sequence

        polymer = from_sequence("acgu", backend=backend)

        # Templates don't have bfactors - accessing raises AttributeError
        with pytest.raises(AttributeError, match="bfactors"):
            _ = polymer.bfactors

        # Use _get_field_data to check availability without raising
        assert polymer._get_field_data('bfactors') is None

    @pytest.mark.parametrize("cif_file", CIF_FILES)
    def test_bfactors_preserved_after_selection(self, cif_file, backend):
        """bfactors preserved after selecting a subset."""
        from ciffy import load

        polymer = load(cif_file, backend=backend)

        # Select first half of atoms
        n = polymer.size()
        subset = polymer[:n // 2]

        if not subset.empty():
            assert subset.bfactors is not None
            assert subset.bfactors.shape == (subset.size(),)

    @pytest.mark.parametrize("cif_file", CIF_FILES)
    def test_bfactors_are_nonnegative(self, cif_file, backend):
        """bfactors values are non-negative (physical constraint)."""
        from ciffy import load

        polymer = load(cif_file, backend=backend)
        bfactors = np.asarray(polymer.bfactors)

        # B-factors should be non-negative
        assert np.all(bfactors >= 0), "B-factors should be non-negative"

    def test_bfactors_preserved_after_backbone_selection(self, backend):
        """bfactors preserved after backbone() selection."""
        from ciffy import load

        cif = get_test_cif("3SKW")
        polymer = load(cif, backend=backend)
        backbone = polymer.backbone()

        if not backbone.empty():
            assert backbone.bfactors is not None
            assert backbone.bfactors.shape == (backbone.size(),)


class TestResolution:
    """Test resolution property."""

    @pytest.mark.parametrize("cif_file", CIF_FILES)
    def test_resolution_type(self, cif_file, backend):
        """resolution is float or None."""
        from ciffy import load

        polymer = load(cif_file, backend=backend)

        assert polymer.resolution is None or isinstance(polymer.resolution, float)

    def test_resolution_none_for_template(self, backend):
        """resolution is None for template-generated polymers."""
        from ciffy import from_sequence

        polymer = from_sequence("acgu", backend=backend)

        assert polymer.resolution is None

    @pytest.mark.parametrize("cif_file", CIF_FILES)
    def test_resolution_positive_when_present(self, cif_file, backend):
        """resolution is positive when present (None if unavailable)."""
        from ciffy import load

        polymer = load(cif_file, backend=backend)

        if polymer.resolution is not None:
            assert polymer.resolution > 0, \
                f"Resolution should be positive, got {polymer.resolution}"

    @pytest.mark.parametrize("cif_file", CIF_FILES)
    def test_resolution_preserved_after_selection(self, cif_file, backend):
        """resolution preserved after chain selection."""
        from ciffy import load, Scale

        polymer = load(cif_file, backend=backend)
        original_resolution = polymer.resolution

        if polymer.size(Scale.CHAIN) > 0:
            chain = polymer.chain(0)
            assert chain.resolution == original_resolution


class TestBonds:
    """Test bonds computed property."""

    @pytest.mark.parametrize("cif_file", CIF_FILES)
    def test_bonds_returns_2d_array(self, cif_file, backend):
        """bonds property returns 2D array matching backend."""
        from ciffy import load

        polymer = load(cif_file, backend=backend)
        bonds = polymer.bonds

        if backend == "torch":
            import torch
            assert isinstance(bonds, torch.Tensor)
        else:
            assert isinstance(bonds, np.ndarray)
        assert bonds.ndim == 2

    @pytest.mark.parametrize("cif_file", CIF_FILES)
    def test_bonds_shape_is_n_by_2(self, cif_file, backend):
        """bonds array has shape (n_bonds, 2)."""
        from ciffy import load

        polymer = load(cif_file, backend=backend)
        bonds = polymer.bonds

        assert bonds.shape[1] == 2

    @pytest.mark.parametrize("cif_file", CIF_FILES)
    def test_bonds_indices_valid(self, cif_file, backend):
        """bonds indices are within valid range."""
        from ciffy import load

        polymer = load(cif_file, backend=backend)
        bonds = polymer.bonds
        n_atoms = polymer.size()

        # All indices should be valid atom indices
        assert (bonds >= 0).all()
        assert (bonds < n_atoms).all()

    @pytest.mark.parametrize("cif_file", CIF_FILES)
    def test_bonds_are_unique_pairs(self, cif_file, backend):
        """bonds are stored with i < j (no duplicates)."""
        from ciffy import load
        from ciffy.backend import to_numpy

        polymer = load(cif_file, backend=backend)
        bonds = polymer.bonds

        # i < j for all bonds (as documented)
        assert (bonds[:, 0] < bonds[:, 1]).all()

        # No duplicate pairs
        bonds_np = to_numpy(bonds)
        bond_tuples = set(map(tuple, bonds_np))
        assert len(bond_tuples) == len(bonds)

    def test_bonds_cached(self, backend):
        """bonds property is cached (same object returned)."""
        from ciffy import load

        cif = get_test_cif("3SKW")
        polymer = load(cif, backend=backend)

        bonds1 = polymer.bonds
        bonds2 = polymer.bonds

        # Should be the exact same object (cached)
        assert bonds1 is bonds2

    def test_bonds_cleared_after_center(self, backend):
        """bonds cache cleared after center() operation."""
        from ciffy import load, Scale

        cif = get_test_cif("3SKW")
        polymer = load(cif, backend=backend)

        # Access bonds to cache them
        bonds1 = polymer.bonds

        # Center the structure
        centered, _ = polymer.center(Scale.MOLECULE)

        # New polymer should have fresh bond calculation
        # (we can't directly test cache clearing, but we can verify
        # the centered polymer has valid bonds)
        bonds2 = centered.bonds
        assert bonds2 is not bonds1  # Different objects

    def test_bonds_cleared_after_copy_coordinates(self, backend):
        """bonds cache cleared after copy(coordinates=...)."""
        from ciffy import load

        cif = get_test_cif("3SKW")
        polymer = load(cif, backend=backend)

        # Access bonds to cache them
        bonds1 = polymer.bonds

        # Create new polymer with same coordinates
        new_polymer = polymer.copy(coordinates=polymer.coordinates)

        # New polymer should have fresh bond calculation
        bonds2 = new_polymer.bonds
        assert bonds2 is not bonds1  # Different objects

    def test_bonds_template_polymer(self, backend):
        """bonds works on template-generated polymers."""
        from ciffy import from_sequence

        polymer = from_sequence("acgu", backend=backend)
        bonds = polymer.bonds

        # Should have bonds (nucleotides have internal bonds)
        if backend == "torch":
            import torch
            assert isinstance(bonds, torch.Tensor)
        else:
            assert isinstance(bonds, np.ndarray)
        assert len(bonds) > 0

    def test_bonds_empty_polymer(self, backend):
        """bonds on empty polymer returns empty array."""
        from ciffy import from_sequence

        template = from_sequence("a", backend=backend)
        empty = template[template.atoms < 0]

        bonds = empty.bonds
        if backend == "torch":
            import torch
            assert isinstance(bonds, torch.Tensor)
        else:
            assert isinstance(bonds, np.ndarray)
        assert len(bonds) == 0


class TestNonpoly:
    """Test nonpoly computed property."""

    @pytest.mark.parametrize("cif_file", CIF_FILES)
    def test_nonpoly_is_nonnegative(self, cif_file, backend):
        """nonpoly count is non-negative."""
        from ciffy import load

        polymer = load(cif_file, backend=backend)
        assert polymer.nonpoly() >= 0

    @pytest.mark.parametrize("cif_file", CIF_FILES)
    def test_nonpoly_plus_polymer_equals_total(self, cif_file, backend):
        """polymer_count + nonpoly == total atoms."""
        from ciffy import load

        polymer = load(cif_file, backend=backend)
        assert polymer.polymer_count + polymer.nonpoly() == polymer.size()

    @pytest.mark.parametrize("cif_file", CIF_FILES)
    def test_nonpoly_consistent_with_poly_size(self, cif_file, backend):
        """nonpoly == size - poly().size()."""
        from ciffy import load

        polymer = load(cif_file, backend=backend)
        poly_size = polymer.poly().size()

        assert polymer.nonpoly() == polymer.size() - poly_size

    def test_nonpoly_zero_for_template(self, backend):
        """nonpoly is 0 for template-generated polymers."""
        from ciffy import from_sequence

        polymer = from_sequence("acgu", backend=backend)

        assert polymer.nonpoly() == 0
        assert polymer.polymer_count == polymer.size()

    @pytest.mark.parametrize("cif_file", CIF_FILES)
    def test_nonpoly_consistent_with_hetero_size(self, cif_file, backend):
        """nonpoly == hetero().size()."""
        from ciffy import load

        polymer = load(cif_file, backend=backend)
        hetero_size = polymer.hetero().size()

        assert polymer.nonpoly() == hetero_size
