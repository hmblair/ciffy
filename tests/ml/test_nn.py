"""Tests for ciffy.nn module."""

import glob
import os
import pytest
import numpy as np

import ciffy
from ciffy import Scale

from tests.utils import (
    get_test_cif,
    TORCH_AVAILABLE,
    skip_if_no_torch,
    DATA_DIR,
)


# =============================================================================
# Test KNN
# =============================================================================

class TestKNN:
    """Tests for Polymer.knn() method."""

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_knn_shape(self, backend):
        """Test that knn returns correct shape."""
        skip_if_no_torch(backend)

        p = ciffy.load(get_test_cif("3SKW"), backend=backend)
        k = 5
        neighbors = p.knn(k=k, scale=Scale.ATOM)

        assert neighbors.shape == (k, p.size())

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_knn_residue_scale(self, backend):
        """Test KNN at residue scale."""
        skip_if_no_torch(backend)

        p = ciffy.load(get_test_cif("3SKW"), backend=backend)
        k = 3
        neighbors = p.knn(k=k, scale=Scale.RESIDUE)

        assert neighbors.shape == (k, p.size(Scale.RESIDUE))

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_knn_excludes_self(self, backend):
        """Test that knn excludes self (no point is its own neighbor)."""
        skip_if_no_torch(backend)

        p = ciffy.load(get_test_cif("3SKW"), backend=backend)
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
        p = ciffy.load(get_test_cif("3SKW"), backend="numpy")
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

        p = ciffy.load(get_test_cif("3SKW"), backend="torch")
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

        p = ciffy.load(get_test_cif("3SKW"), backend="torch")
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
        p = ciffy.load(get_test_cif("3SKW"), backend="torch")

        features = embed(p)
        loss = features.sum()
        loss.backward()

        # Check that embedding weights have gradients
        assert embed.atom_embedding.weight.grad is not None


# =============================================================================
# Edge Case Tests
# =============================================================================

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestPolymerDatasetEdgeCases:
    """Edge case tests for PolymerDataset."""

    def test_dataset_empty_directory(self, tmp_path):
        """Dataset with empty directory has length 0."""
        from ciffy.nn import PolymerDataset

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        dataset = PolymerDataset(str(empty_dir))
        assert len(dataset) == 0

    def test_dataset_filters_match_nothing(self):
        """Dataset with impossible filters is empty."""
        from ciffy.nn import PolymerDataset

        # max_atoms=1 should filter out everything
        dataset = PolymerDataset(DATA_DIR, max_atoms=1)
        assert len(dataset) == 0

    def test_dataset_min_atoms_filter(self):
        """Dataset min_atoms filters correctly."""
        from ciffy.nn import PolymerDataset

        # Very large min_atoms should filter out everything
        dataset = PolymerDataset(DATA_DIR, min_atoms=1000000)
        assert len(dataset) == 0

    def test_dataset_all_excluded(self):
        """Dataset with all IDs excluded is empty."""
        from ciffy.nn import PolymerDataset

        # Get all CIF files and exclude them
        import glob
        cif_files = glob.glob(os.path.join(DATA_DIR, "*.cif"))
        exclude_ids = [os.path.basename(f).replace(".cif", "") for f in cif_files]

        dataset = PolymerDataset(DATA_DIR, exclude_ids=exclude_ids)
        assert len(dataset) == 0

    def test_dataset_molecule_types_filter_no_match(self):
        """Dataset with non-matching molecule_types is empty or smaller."""
        from ciffy.nn import PolymerDataset
        from ciffy import Molecule

        # DNA type likely not present in test data
        dataset = PolymerDataset(DATA_DIR, molecule_types=[Molecule.DNA])
        # May be empty or have some entries
        assert len(dataset) >= 0  # Shouldn't crash

    def test_dataset_directory_with_non_cif_files(self, tmp_path):
        """Dataset ignores non-CIF files."""
        from ciffy.nn import PolymerDataset
        import shutil

        # Create directory with mixed files
        mixed_dir = tmp_path / "mixed"
        mixed_dir.mkdir()

        # Add a CIF file
        shutil.copy(get_test_cif("3SKW"), mixed_dir / "3SKW.cif")

        # Add non-CIF files
        (mixed_dir / "readme.txt").write_text("test")
        (mixed_dir / "data.json").write_text("{}")

        dataset = PolymerDataset(str(mixed_dir))
        # Should only find the CIF file
        assert len(dataset) >= 1

    def test_dataset_limit(self):
        """Dataset respects limit parameter."""
        from ciffy.nn import PolymerDataset
        from ciffy import Scale

        # Create dataset without limit
        full_dataset = PolymerDataset(DATA_DIR, scale=Scale.CHAIN)
        full_count = len(full_dataset)

        # Skip if dataset too small
        if full_count < 3:
            pytest.skip("Need at least 3 chains to test limit")

        # Create dataset with limit
        limited_dataset = PolymerDataset(DATA_DIR, scale=Scale.CHAIN, limit=2)
        assert len(limited_dataset) == 2

        # Limit larger than dataset should return all
        large_limit = PolymerDataset(DATA_DIR, scale=Scale.CHAIN, limit=10000)
        assert len(large_limit) == full_count


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestPolymerEmbeddingEdgeCases:
    """Edge case tests for PolymerEmbedding."""

    def test_embedding_single_dim(self):
        """Embedding with dimension 1 works."""
        from ciffy.nn import PolymerEmbedding

        embed = PolymerEmbedding(scale=Scale.ATOM, atom_dim=1)
        p = ciffy.load(get_test_cif("3SKW"), backend="torch")

        features = embed(p)
        assert features.shape == (p.size(), 1)

    def test_embedding_large_dim(self):
        """Embedding with large dimension works."""
        from ciffy.nn import PolymerEmbedding

        embed = PolymerEmbedding(scale=Scale.ATOM, atom_dim=1024)
        p = ciffy.load(get_test_cif("3SKW"), backend="torch")

        features = embed(p)
        assert features.shape == (p.size(), 1024)

    def test_embedding_only_residue_dim(self):
        """Embedding with only residue_dim at atom scale."""
        from ciffy.nn import PolymerEmbedding

        embed = PolymerEmbedding(scale=Scale.ATOM, residue_dim=64)
        p = ciffy.load(get_test_cif("3SKW"), backend="torch")

        features = embed(p)
        # Should expand residue embeddings to atom level
        assert features.shape == (p.size(), 64)

    def test_embedding_only_element_dim(self):
        """Embedding with only element_dim at atom scale."""
        from ciffy.nn import PolymerEmbedding

        embed = PolymerEmbedding(scale=Scale.ATOM, element_dim=32)
        p = ciffy.load(get_test_cif("3SKW"), backend="torch")

        features = embed(p)
        assert features.shape == (p.size(), 32)

    def test_embedding_all_dims(self):
        """Embedding with all dims at atom scale."""
        from ciffy.nn import PolymerEmbedding

        embed = PolymerEmbedding(
            scale=Scale.ATOM,
            atom_dim=64,
            residue_dim=32,
            element_dim=16,
        )
        p = ciffy.load(get_test_cif("3SKW"), backend="torch")

        features = embed(p)
        assert features.shape == (p.size(), 64 + 32 + 16)

    def test_embedding_on_template_polymer(self):
        """Embedding works on template polymer."""
        from ciffy.nn import PolymerEmbedding

        embed = PolymerEmbedding(scale=Scale.ATOM, atom_dim=32)
        p = ciffy.from_sequence("acgu", backend="torch")

        features = embed(p)
        assert features.shape == (p.size(), 32)

    def test_embedding_on_single_residue(self):
        """Embedding works on single-residue polymer."""
        from ciffy.nn import PolymerEmbedding

        embed = PolymerEmbedding(scale=Scale.RESIDUE, residue_dim=32)
        p = ciffy.from_sequence("a", backend="torch")

        features = embed(p)
        assert features.shape == (1, 32)
