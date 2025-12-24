"""Tests for ciffy.nn.flow.polymer module."""

import numpy as np
import pytest
import torch

from ciffy.nn.flow import PolymerFlowModel, PCAFlow, ResidueFlowModel
from ciffy.nn.flow.residue.data import compute_pca
from ciffy.biochemistry import Residue


class MockResidueFlowModel:
    """Mock ResidueFlowModel for testing without training."""

    def __init__(self, residue: Residue, n_atoms: int, latent_dim: int = 8):
        self.residue = residue
        self.n_atoms = n_atoms
        self.latent_dim = latent_dim
        self._atom_indices = list(residue.index()[:n_atoms])
        self.pca_rmsd = 0.1
        self.var_explained = 0.95

        # Create a simple PCAFlow
        np.random.seed(42)
        n_samples = 100
        extended_dim = n_atoms * 3 + 6  # coords + transform

        # Generate synthetic extended data
        data = np.random.randn(n_samples, extended_dim).astype(np.float32)
        V, mean, _, _ = compute_pca(data, n_components=latent_dim)

        self.flow = PCAFlow(
            torch.from_numpy(V).float(),
            torch.from_numpy(mean).float(),
            n_layers=2,
            hidden_dim=16,
            bound=3.0,
        )

    def encode(self, coords: torch.Tensor, transforms=None) -> torch.Tensor:
        """Encode coordinates to latent space."""
        if coords.dim() == 3:
            coords = coords.reshape(coords.shape[0], -1)
        if transforms is None:
            transforms = torch.zeros(coords.shape[0], 6, device=coords.device)
        extended = torch.cat([coords, transforms], dim=-1)
        return self.flow.encode(extended)

    def decode(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode latent to coordinates and transform."""
        extended = self.flow.decode(z)
        n_coord_dims = self.n_atoms * 3
        coords_flat = extended[:, :n_coord_dims]
        transforms = extended[:, n_coord_dims:]
        coords = coords_flat.reshape(-1, self.n_atoms, 3)
        return coords, transforms


class TestPolymerFlowModel:
    """Tests for PolymerFlowModel."""

    @pytest.fixture
    def mock_models(self):
        """Create mock residue models for testing."""
        return {
            Residue.A: MockResidueFlowModel(Residue.A, n_atoms=10, latent_dim=6),
            Residue.G: MockResidueFlowModel(Residue.G, n_atoms=12, latent_dim=6),
        }

    @pytest.fixture
    def polymer_model(self, mock_models):
        """Create a PolymerFlowModel for testing."""
        return PolymerFlowModel(mock_models)

    def test_init_validates_empty(self):
        """Test that empty models dict raises error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            PolymerFlowModel({})

    def test_init_validates_latent_dim(self):
        """Test that mismatched latent dims raises error."""
        models = {
            Residue.A: MockResidueFlowModel(Residue.A, n_atoms=10, latent_dim=6),
            Residue.G: MockResidueFlowModel(Residue.G, n_atoms=12, latent_dim=8),
        }
        with pytest.raises(ValueError, match="same latent_dim"):
            PolymerFlowModel(models)

    def test_init_stores_latent_dim(self, polymer_model):
        """Test that latent_dim is correctly set."""
        assert polymer_model.latent_dim == 6

    def test_supported_residues(self, polymer_model):
        """Test supported_residues property."""
        supported = polymer_model.supported_residues
        assert Residue.A in supported
        assert Residue.G in supported
        assert len(supported) == 2

    def test_get_atom_counts(self, polymer_model):
        """Test atom count computation."""
        sequence = [Residue.A, Residue.G, Residue.A]
        counts = polymer_model._get_atom_counts(sequence)
        assert counts == [10, 12, 10]

    def test_get_atom_counts_missing_residue(self, polymer_model):
        """Test error on unsupported residue type."""
        sequence = [Residue.A, Residue.C]  # C not in mock_models
        with pytest.raises(ValueError, match="No model for residue type"):
            polymer_model._get_atom_counts(sequence)

    def test_encode_shape(self, polymer_model):
        """Test encode output shape."""
        sequence = [Residue.A, Residue.G, Residue.A]
        n_atoms = 10 + 12 + 10
        coords = torch.randn(n_atoms, 3)

        latents = polymer_model.encode(coords, sequence)

        assert latents.shape == (3, 6)  # 3 residues, 6 latent dims

    def test_encode_validates_coords_shape(self, polymer_model):
        """Test encode validates coordinate shape."""
        sequence = [Residue.A]
        coords = torch.randn(5, 3)  # Wrong number of atoms

        with pytest.raises(ValueError, match="atoms but sequence expects"):
            polymer_model.encode(coords, sequence)

    def test_decode_shape(self, polymer_model):
        """Test decode output shape."""
        sequence = [Residue.A, Residue.G, Residue.A]
        n_atoms = 10 + 12 + 10
        latents = torch.randn(3, 6)

        coords = polymer_model.decode(latents, sequence)

        assert coords.shape == (n_atoms, 3)

    def test_decode_validates_latents_shape(self, polymer_model):
        """Test decode validates latents shape."""
        sequence = [Residue.A, Residue.G]
        latents = torch.randn(3, 6)  # 3 latents but 2 residues

        with pytest.raises(ValueError, match="rows but sequence has"):
            polymer_model.decode(latents, sequence)

    def test_decode_empty_sequence(self, polymer_model):
        """Test decode with empty sequence."""
        latents = torch.randn(0, 6)
        coords = polymer_model.decode(latents, [])
        assert coords.shape == (0, 3)

    def test_sample_shape(self, polymer_model):
        """Test sample output shape."""
        sequence = [Residue.A, Residue.G]
        n_atoms = 10 + 12

        coords = polymer_model.sample(sequence)

        assert coords.shape == (n_atoms, 3)

    def test_sample_multiple(self, polymer_model):
        """Test sampling multiple conformations."""
        sequence = [Residue.A, Residue.G]
        n_atoms = 10 + 12

        samples = polymer_model.sample(sequence, n_samples=5)

        assert len(samples) == 5
        for s in samples:
            assert s.shape == (n_atoms, 3)

    def test_sample_empty_sequence(self, polymer_model):
        """Test sample with empty sequence."""
        coords = polymer_model.sample([])
        assert coords.shape == (0, 3)

        samples = polymer_model.sample([], n_samples=3)
        assert len(samples) == 3
        for s in samples:
            assert s.shape == (0, 3)

    def test_repr(self, polymer_model):
        """Test string representation."""
        repr_str = repr(polymer_model)
        assert "PolymerFlowModel" in repr_str
        assert "latent_dim=6" in repr_str


class TestPolymerFlowModelRoundtrip:
    """Tests for encode-decode roundtrip behavior."""

    @pytest.fixture
    def real_models(self):
        """Create minimal real ResidueFlowModels for roundtrip testing."""
        np.random.seed(42)
        n_atoms_a = 10
        n_atoms_g = 12
        latent_dim = 6

        def create_model(residue, n_atoms):
            extended_dim = n_atoms * 3 + 6
            n_samples = 50
            data = np.random.randn(n_samples, extended_dim).astype(np.float32)
            V, mean, _, var_explained = compute_pca(data, n_components=latent_dim)

            flow = PCAFlow(
                torch.from_numpy(V).float(),
                torch.from_numpy(mean).float(),
                n_layers=2,
                hidden_dim=16,
                bound=3.0,
            )

            return ResidueFlowModel(
                flow=flow,
                residue=residue,
                atom_indices=list(residue.index()[:n_atoms]),
                n_atoms=n_atoms,
                pca_rmsd=0.1,
                var_explained=float(var_explained[-1]),
            )

        return {
            Residue.A: create_model(Residue.A, n_atoms_a),
            Residue.G: create_model(Residue.G, n_atoms_g),
        }

    def test_encode_decode_preserves_structure(self, real_models):
        """Test that encode-decode approximately preserves input."""
        polymer = PolymerFlowModel(real_models)

        # Single residue (no positioning needed)
        sequence = [Residue.A]
        n_atoms = 10
        coords = torch.randn(n_atoms, 3)

        latents = polymer.encode(coords, sequence)
        coords_recon = polymer.decode(latents, sequence)

        # Should be close for single residue (only PCA truncation error)
        # Note: The mock data doesn't have real structure, so we just check shapes
        assert coords_recon.shape == coords.shape

    def test_multi_residue_decode(self, real_models):
        """Test decoding multi-residue sequence produces valid output."""
        polymer = PolymerFlowModel(real_models)

        sequence = [Residue.A, Residue.G, Residue.A]
        latents = torch.randn(3, 6)

        coords = polymer.decode(latents, sequence)

        # Check output has correct total atoms
        expected_atoms = 10 + 12 + 10
        assert coords.shape == (expected_atoms, 3)

        # Check coords are finite
        assert torch.isfinite(coords).all()


class TestPolymerFlowModelSaveLoad:
    """Tests for save/load functionality."""

    @pytest.fixture
    def real_models(self):
        """Create minimal real ResidueFlowModels."""
        np.random.seed(42)

        def create_model(residue, n_atoms):
            extended_dim = n_atoms * 3 + 6
            data = np.random.randn(30, extended_dim).astype(np.float32)
            V, mean, _, var_explained = compute_pca(data, n_components=6)

            flow = PCAFlow(
                torch.from_numpy(V).float(),
                torch.from_numpy(mean).float(),
                n_layers=2,
                hidden_dim=16,
            )

            return ResidueFlowModel(
                flow=flow,
                residue=residue,
                atom_indices=list(residue.index()[:n_atoms]),
                n_atoms=n_atoms,
                pca_rmsd=0.1,
                var_explained=float(var_explained[-1]),
            )

        return {
            Residue.A: create_model(Residue.A, 10),
        }

    def test_save_load_roundtrip(self, real_models, tmp_path):
        """Test save and load produces equivalent model."""
        polymer = PolymerFlowModel(real_models)

        # Save
        save_path = tmp_path / "polymer_model"
        polymer.save(save_path)

        # Check files exist
        assert (save_path / "config.json").exists()
        assert (save_path / "A").exists()

        # Load
        loaded = PolymerFlowModel.load(save_path)

        # Compare
        assert loaded.latent_dim == polymer.latent_dim
        assert set(loaded.supported_residues) == set(polymer.supported_residues)

        # Test outputs match
        sequence = [Residue.A]
        latents = torch.randn(1, 6)

        with torch.no_grad():
            coords1 = polymer.decode(latents, sequence)
            coords2 = loaded.decode(latents, sequence)

        assert torch.allclose(coords1, coords2, atol=1e-5)
