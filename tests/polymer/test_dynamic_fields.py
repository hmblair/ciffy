"""Tests for the dynamic field (annotate) system."""

import numpy as np
import pytest

import ciffy
from tests.utils import get_test_cif
from ciffy.biochemistry import Scale


@pytest.fixture
def polymer():
    """Load a test polymer."""
    return ciffy.load(get_test_cif("9MDS")).chain(0)


class TestAnnotateBasic:
    """Test basic annotate operations."""

    def test_annotate_residue(self, polymer):
        """Can annotate with residue-level data."""
        n_res = polymer.size(Scale.RESIDUE)
        data = np.random.randn(n_res).astype(np.float32)

        polymer = polymer.annotate('reactivity', data, Scale.RESIDUE)

        np.testing.assert_array_equal(polymer.reactivity, data)

    def test_annotate_atom(self, polymer):
        """Can annotate with atom-level data."""
        n_atoms = polymer.size(Scale.ATOM)
        data = np.random.randn(n_atoms, 16).astype(np.float32)

        polymer = polymer.annotate('embeddings', data, Scale.ATOM)

        np.testing.assert_array_equal(polymer.embeddings, data)

    def test_annotate_chain(self, polymer):
        """Can annotate with chain-level data."""
        n_chains = polymer.size(Scale.CHAIN)
        data = np.random.randn(n_chains).astype(np.float32)

        polymer = polymer.annotate('expression', data, Scale.CHAIN)

        np.testing.assert_array_equal(polymer.expression, data)

    def test_annotate_chaining(self, polymer):
        """annotate() supports method chaining."""
        n_res = polymer.size(Scale.RESIDUE)
        data1 = np.random.randn(n_res).astype(np.float32)
        data2 = np.random.randn(n_res).astype(np.float32)

        result = polymer.annotate('a', data1, Scale.RESIDUE).annotate('b', data2, Scale.RESIDUE)

        # annotate() returns a new container, not self
        assert result is not polymer
        np.testing.assert_array_equal(result.a, data1)
        np.testing.assert_array_equal(result.b, data2)

    def test_missing_field_raises_attributeerror(self, polymer):
        """Accessing non-existent field raises AttributeError."""
        with pytest.raises(AttributeError):
            _ = polymer.nonexistent

    def test_annotate_existing_raises(self, polymer):
        """annotate() fails if field already exists."""
        n_res = polymer.size(Scale.RESIDUE)
        data = np.zeros(n_res)
        polymer = polymer.annotate('temp', data, Scale.RESIDUE)

        with pytest.raises(ValueError, match="already exists"):
            polymer.annotate('temp', data, Scale.RESIDUE)

    def test_annotate_builtin_raises(self, polymer):
        """annotate() fails for built-in field names."""
        n_atoms = polymer.size(Scale.ATOM)
        data = np.zeros((n_atoms, 3))

        with pytest.raises(ValueError, match="already exists"):
            polymer.annotate('coordinates', data, Scale.ATOM)

    def test_annotate_underscore_raises(self, polymer):
        """annotate() fails for names starting with underscore."""
        n_res = polymer.size(Scale.RESIDUE)
        data = np.zeros(n_res)

        with pytest.raises(ValueError, match="underscore"):
            polymer.annotate('_private', data, Scale.RESIDUE)


class TestAnnotateValidation:
    """Test annotate validation."""

    def test_wrong_size_raises(self, polymer):
        """Wrong size raises ValueError."""
        n_res = polymer.size(Scale.RESIDUE)
        wrong_size = np.zeros(n_res + 10)

        with pytest.raises(ValueError, match="Shape mismatch"):
            polymer.annotate('bad', wrong_size, Scale.RESIDUE)

    def test_multidimensional_data(self, polymer):
        """Multi-dimensional data works if first dim matches."""
        n_res = polymer.size(Scale.RESIDUE)
        data = np.random.randn(n_res, 5, 3).astype(np.float32)

        polymer = polymer.annotate('multi', data, Scale.RESIDUE)

        assert polymer.multi.shape == (n_res, 5, 3)


class TestFieldSlicing:
    """Test dynamic field propagation through selections."""

    def test_residue_selection_slices_fields(self, polymer):
        """Residue selection slices residue-level fields."""
        n_res = polymer.size(Scale.RESIDUE)
        data = np.arange(n_res).astype(np.float32)
        polymer = polymer.annotate('idx', data, Scale.RESIDUE)

        # Select first 5 residues
        selected = polymer.residue(slice(0, 5))

        assert selected.size(Scale.RESIDUE) == 5
        np.testing.assert_array_equal(selected.idx, data[:5])

    def test_chain_selection_slices_all_scales(self):
        """Chain selection slices fields at all scales."""
        polymer = ciffy.load(get_test_cif("9MDS"))
        n_chains = polymer.size(Scale.CHAIN)
        if n_chains < 2:
            pytest.skip("Need multi-chain structure")

        # Add fields at all scales
        n_atoms = polymer.size(Scale.ATOM)
        n_res = polymer.size(Scale.RESIDUE)
        polymer = polymer.annotate('atom_data', np.arange(n_atoms).astype(np.float32), Scale.ATOM)
        polymer = polymer.annotate('res_data', np.arange(n_res).astype(np.float32), Scale.RESIDUE)
        polymer = polymer.annotate('chain_data', np.arange(n_chains).astype(np.float32), Scale.CHAIN)

        # Select first chain
        chain = polymer.chain(0)

        # All fields should be sliced
        assert hasattr(chain, 'atom_data')
        assert hasattr(chain, 'res_data')
        assert hasattr(chain, 'chain_data')
        assert chain.chain_data.shape[0] == 1


class TestFieldCloning:
    """Test field propagation through copy()."""

    def test_copy_clones_fields(self, polymer):
        """copy() creates independent field copies."""
        n_res = polymer.size(Scale.RESIDUE)
        data = np.arange(n_res).astype(np.float32)
        polymer = polymer.annotate('data', data, Scale.RESIDUE)

        copy = polymer.copy()

        # Verify data is copied
        np.testing.assert_array_equal(copy.data, data)

        # Verify independence
        copy.data[0] = 999
        assert polymer.data[0] != 999


class TestFieldBackendConversion:
    """Test field propagation through backend conversion."""

    def test_numpy_to_torch_converts_fields(self, polymer):
        """numpy() to torch() converts fields."""
        pytest.importorskip("torch")
        from ciffy.backend import is_torch

        n_res = polymer.size(Scale.RESIDUE)
        data = np.arange(n_res).astype(np.float32)
        polymer = polymer.annotate('data', data, Scale.RESIDUE)

        torch_polymer = polymer.torch()

        assert is_torch(torch_polymer.data)
        np.testing.assert_array_equal(
            torch_polymer.data.numpy(),
            data
        )

    def test_torch_to_numpy_converts_fields(self, polymer):
        """torch() to numpy() converts fields."""
        pytest.importorskip("torch")
        from ciffy.backend import ops

        n_res = polymer.size(Scale.RESIDUE)
        data = np.arange(n_res).astype(np.float32)
        polymer = polymer.annotate('data', data, Scale.RESIDUE)

        # Convert to torch, add field, convert back
        torch_polymer = polymer.torch()
        torch_polymer = torch_polymer.annotate('extra', ops.zeros(n_res, like=torch_polymer.coordinates, dtype='float32'), Scale.RESIDUE)

        numpy_polymer = torch_polymer.numpy()

        assert isinstance(numpy_polymer.data, np.ndarray)
        assert isinstance(numpy_polymer.extra, np.ndarray)


class TestFieldSetattr:
    """Test updating field values via setattr."""

    def test_update_builtin_field(self, polymer):
        """Can update built-in fields via attribute access."""
        n_atoms = polymer.size(Scale.ATOM)
        new_coords = np.zeros((n_atoms, 3), dtype=np.float32)

        polymer.coordinates = new_coords

        np.testing.assert_array_equal(polymer.coordinates, new_coords)

    def test_update_dynamic_field(self, polymer):
        """Can update dynamic fields via attribute access."""
        n_res = polymer.size(Scale.RESIDUE)
        data1 = np.zeros(n_res).astype(np.float32)
        data2 = np.ones(n_res).astype(np.float32)

        polymer = polymer.annotate('data', data1, Scale.RESIDUE)
        polymer.data = data2

        np.testing.assert_array_equal(polymer.data, data2)

    def test_update_wrong_size_raises(self, polymer):
        """Updating with wrong size raises ValueError."""
        n_res = polymer.size(Scale.RESIDUE)
        polymer = polymer.annotate('data', np.zeros(n_res).astype(np.float32), Scale.RESIDUE)

        with pytest.raises(ValueError, match="Shape mismatch"):
            polymer.data = np.zeros(n_res + 10)


class TestGetFields:
    """Test _get_fields() method."""

    def test_get_fields_includes_builtin(self, polymer):
        """_get_fields() includes built-in fields."""
        fields = polymer._get_fields()

        assert 'coordinates' in fields
        assert 'atoms' in fields
        assert 'sequence' in fields

    def test_get_fields_includes_dynamic(self, polymer):
        """_get_fields() includes dynamic fields."""
        n_res = polymer.size(Scale.RESIDUE)
        polymer = polymer.annotate('reactivity', np.zeros(n_res), Scale.RESIDUE)

        fields = polymer._get_fields()

        assert 'reactivity' in fields
        assert fields['reactivity'].scale == Scale.RESIDUE
