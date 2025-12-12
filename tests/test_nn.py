"""Tests for ciffy.nn module."""

import os
import pytest
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

import ciffy
from ciffy import Scale

TESTS_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(TESTS_DIR, "data")


def get_cif(pdb_id: str) -> str:
    """Get path to a test CIF file."""
    return os.path.join(DATA_DIR, f"{pdb_id}.cif")


# =============================================================================
# Test KNN
# =============================================================================

class TestKNN:
    """Tests for Polymer.knn() method."""

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_knn_shape(self, backend):
        """Test that knn returns correct shape."""
        if backend == "torch" and not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")

        p = ciffy.load(get_cif("3SKW"), backend=backend)
        k = 5
        neighbors = p.knn(k=k, scale=Scale.ATOM)

        assert neighbors.shape == (k, p.size())

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_knn_residue_scale(self, backend):
        """Test KNN at residue scale."""
        if backend == "torch" and not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")

        p = ciffy.load(get_cif("3SKW"), backend=backend)
        k = 3
        neighbors = p.knn(k=k, scale=Scale.RESIDUE)

        assert neighbors.shape == (k, p.size(Scale.RESIDUE))

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_knn_excludes_self(self, backend):
        """Test that knn excludes self (no point is its own neighbor)."""
        if backend == "torch" and not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")

        p = ciffy.load(get_cif("3SKW"), backend=backend)
        neighbors = p.knn(k=3, scale=Scale.ATOM)

        # Check that no atom is its own neighbor
        n_atoms = p.size()
        for i in range(n_atoms):
            neighbor_list = neighbors[:, i]
            if backend == "torch":
                neighbor_list = neighbor_list.numpy()
            assert i not in neighbor_list

    def test_knn_k_too_large(self):
        """Test that knn raises error when k >= n."""
        p = ciffy.load(get_cif("3SKW"), backend="numpy")
        with pytest.raises(ValueError, match="k=.* must be less than"):
            p.knn(k=p.size(), scale=Scale.ATOM)


# =============================================================================
# Test PolymerDataset
# =============================================================================

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestPolymerDataset:
    """Tests for PolymerDataset class."""

    def test_dataset_molecule_scale(self):
        """Test dataset at molecule scale."""
        from ciffy.nn import PolymerDataset

        dataset = PolymerDataset(DATA_DIR, scale=Scale.MOLECULE)
        assert len(dataset) > 0

        # Check that items are Polymers
        p = dataset[0]
        assert isinstance(p, ciffy.Polymer)

    def test_dataset_chain_scale(self):
        """Test dataset at chain scale."""
        from ciffy.nn import PolymerDataset

        dataset = PolymerDataset(DATA_DIR, scale=Scale.CHAIN)
        # Chain scale should have more items than molecule scale
        mol_dataset = PolymerDataset(DATA_DIR, scale=Scale.MOLECULE)
        assert len(dataset) >= len(mol_dataset)

    def test_dataset_max_atoms_filter(self):
        """Test that max_atoms filters correctly."""
        from ciffy.nn import PolymerDataset

        # Very small max_atoms should filter out most/all
        dataset_small = PolymerDataset(DATA_DIR, max_atoms=10)
        dataset_large = PolymerDataset(DATA_DIR, max_atoms=100000)

        assert len(dataset_small) <= len(dataset_large)

    def test_dataset_invalid_scale(self):
        """Test that invalid scale raises error."""
        from ciffy.nn import PolymerDataset

        with pytest.raises(ValueError, match="scale must be MOLECULE or CHAIN"):
            PolymerDataset(DATA_DIR, scale=Scale.ATOM)

    def test_dataset_invalid_directory(self):
        """Test that invalid directory raises error."""
        from ciffy.nn import PolymerDataset

        with pytest.raises(FileNotFoundError):
            PolymerDataset("/nonexistent/path/")


# =============================================================================
# Test PolymerEmbedding
# =============================================================================

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestPolymerEmbedding:
    """Tests for PolymerEmbedding class."""

    def test_embedding_atom_scale(self):
        """Test embedding at atom scale."""
        from ciffy.nn import PolymerEmbedding

        embed = PolymerEmbedding(
            scale=Scale.ATOM,
            atom_dim=64,
            residue_dim=32,
            element_dim=16,
        )

        p = ciffy.load(get_cif("3SKW"), backend="torch")
        features = embed(p)

        assert features.shape == (p.size(), embed.output_dim)
        assert embed.output_dim == 64 + 32 + 16

    def test_embedding_residue_scale(self):
        """Test embedding at residue scale."""
        from ciffy.nn import PolymerEmbedding

        embed = PolymerEmbedding(
            scale=Scale.RESIDUE,
            residue_dim=64,
        )

        p = ciffy.load(get_cif("3SKW"), backend="torch")
        features = embed(p)

        assert features.shape == (p.size(Scale.RESIDUE), 64)

    def test_embedding_invalid_scale_residue_with_atom(self):
        """Test that atom_dim with RESIDUE scale raises error."""
        from ciffy.nn import PolymerEmbedding

        with pytest.raises(ValueError, match="atom_dim cannot be used"):
            PolymerEmbedding(scale=Scale.RESIDUE, atom_dim=64)

    def test_embedding_invalid_scale_residue_with_element(self):
        """Test that element_dim with RESIDUE scale raises error."""
        from ciffy.nn import PolymerEmbedding

        with pytest.raises(ValueError, match="element_dim cannot be used"):
            PolymerEmbedding(scale=Scale.RESIDUE, element_dim=64)

    def test_embedding_no_dims_raises(self):
        """Test that no embedding dims raises error."""
        from ciffy.nn import PolymerEmbedding

        with pytest.raises(ValueError, match="At least one embedding"):
            PolymerEmbedding(scale=Scale.ATOM)

    def test_embedding_gradients(self):
        """Test that embeddings have gradients."""
        from ciffy.nn import PolymerEmbedding

        embed = PolymerEmbedding(scale=Scale.ATOM, atom_dim=32)
        p = ciffy.load(get_cif("3SKW"), backend="torch")

        features = embed(p)
        loss = features.sum()
        loss.backward()

        # Check that embedding weights have gradients
        assert embed.atom_embedding.weight.grad is not None
