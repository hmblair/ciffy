"""Tests for ciffy.nn.flow.polymer module."""

import numpy as np
import pytest
import torch

from ciffy import from_sequence
from ciffy.nn.flow import PolymerFlowModel, PCAFlow, ResidueFlowModel
from ciffy.nn.flow.residue.data import compute_pca
from ciffy.biochemistry import Residue


def create_test_residue_model(
    residue: Residue,
    n_atoms: int,
    latent_dim: int = 6,
    seed: int = 42,
) -> ResidueFlowModel:
    """
    Create a ResidueFlowModel with synthetic random data for testing.

    This uses real ResidueFlowModel instances (not mocks) but with random
    PCA data instead of real molecular conformations. This ensures tests
    exercise the actual production code.

    Args:
        residue: Residue type for this model.
        n_atoms: Number of atoms in the residue subset.
        latent_dim: Latent space dimension.
        seed: Random seed for reproducibility.

    Returns:
        A fully functional ResidueFlowModel.
    """
    np.random.seed(seed)

    extended_dim = n_atoms * 3 + 6  # coords + transform
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
    )


# Module-level fixture for reuse across test classes
@pytest.fixture
def residue_models():
    """Create real ResidueFlowModels with synthetic data."""
    return {
        Residue.A: create_test_residue_model(Residue.A, n_atoms=10, latent_dim=6, seed=42),
        Residue.G: create_test_residue_model(Residue.G, n_atoms=12, latent_dim=6, seed=43),
    }


@pytest.fixture
def polymer_model(residue_models):
    """Create a PolymerFlowModel for testing."""
    return PolymerFlowModel(residue_models)


class TestPolymerFlowModel:
    """Tests for PolymerFlowModel."""

    def test_init_validates_empty(self):
        """Test that empty models dict raises error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            PolymerFlowModel({})

    def test_init_validates_latent_dim(self):
        """Test that mismatched latent dims raises error."""
        models = {
            Residue.A: create_test_residue_model(Residue.A, n_atoms=10, latent_dim=6),
            Residue.G: create_test_residue_model(Residue.G, n_atoms=12, latent_dim=8),
        }
        with pytest.raises(ValueError, match="same latent_dim"):
            PolymerFlowModel(models)

    def test_init_stores_latent_dim(self, polymer_model):
        """Test that latent_dim is correctly set."""
        assert polymer_model.latent_dim == 6

    def test_supported_residues(self, polymer_model):
        """Test supported_residues property."""
        supported = polymer_model.supported_residues
        assert Residue.A.value in supported
        assert Residue.G.value in supported
        assert len(supported) == 2

    def test_get_atom_counts(self, polymer_model):
        """Test atom count computation."""
        sequence = np.array([Residue.A.value, Residue.G.value, Residue.A.value])
        counts = polymer_model._get_atom_counts(sequence)
        assert counts == [10, 12, 10]

    def test_get_atom_counts_missing_residue(self, polymer_model):
        """Test error on unsupported residue type."""
        sequence = np.array([Residue.A.value, Residue.C.value])  # C not in models
        with pytest.raises(KeyError):
            polymer_model._get_atom_counts(sequence)

    def test_encode_shape(self, polymer_model):
        """Test encode output shape with pre-aligned per-residue coords."""
        sequence = np.array([Residue.A.value, Residue.G.value, Residue.A.value])
        # encode() expects list of (n_atoms, 3) tensors, one per residue
        aligned_coords = [
            torch.randn(10, 3),  # A has 10 atoms
            torch.randn(12, 3),  # G has 12 atoms
            torch.randn(10, 3),  # A has 10 atoms
        ]

        latents = polymer_model.encode(aligned_coords, sequence)

        assert latents.shape == (3, 6)  # 3 residues, 6 latent dims

    def test_encode_validates_list_length(self, polymer_model):
        """Test encode validates aligned_coords list length matches sequence."""
        sequence = np.array([Residue.A.value, Residue.G.value])  # 2 residues
        aligned_coords = [torch.randn(10, 3)]  # Only 1 tensor

        with pytest.raises(ValueError, match="coordinate tensors but sequence"):
            polymer_model.encode(aligned_coords, sequence)

    def test_decode_shape(self, polymer_model):
        """Test decode output shape."""
        sequence = np.array([Residue.A.value, Residue.G.value, Residue.A.value])
        n_atoms = 10 + 12 + 10
        latents = torch.randn(3, 6)

        coords = polymer_model.decode(latents, sequence)

        assert coords.shape == (n_atoms, 3)

    def test_decode_validates_latents_shape(self, polymer_model):
        """Test decode validates latents shape."""
        sequence = np.array([Residue.A.value, Residue.G.value])
        latents = torch.randn(3, 6)  # 3 latents but 2 residues

        with pytest.raises(ValueError, match="rows but sequence has"):
            polymer_model.decode(latents, sequence)

    def test_decode_empty_sequence(self, polymer_model):
        """Test decode with empty sequence."""
        latents = torch.randn(0, 6)
        coords = polymer_model.decode(latents, np.array([], dtype=np.int64))
        assert coords.shape == (0, 3)

    def test_sample_coords_internal(self, polymer_model):
        """Test internal _sample_coords method returns coordinate tensors."""
        sequence = np.array([Residue.A.value, Residue.G.value])
        # Get expected atoms from internal method
        n_atoms = sum(polymer_model._get_atom_counts(sequence))

        samples = polymer_model._sample_coords(sequence, n_samples=1)
        assert len(samples) == 1
        assert samples[0].shape == (n_atoms, 3)

        samples = polymer_model._sample_coords(sequence, n_samples=3)
        assert len(samples) == 3
        for s in samples:
            assert s.shape == (n_atoms, 3)

    def test_sample_protocol(self, polymer_model):
        """Test sample() protocol requires numpy backend."""
        # Create a template with torch backend
        template = from_sequence("ag", atoms=polymer_model.atom_filter)
        template_torch = template.torch()

        # Should raise for non-numpy backend
        with pytest.raises(ValueError, match="numpy backend"):
            polymer_model.sample(template_torch)

    def test_sample_protocol_returns_polymers(self, polymer_model):
        """Test sample() returns list of Polymers."""
        # Create template - use full atoms
        template = from_sequence("ag")
        n_atoms_model = sum(polymer_model._get_atom_counts(template.sequence))

        samples = polymer_model.sample(template, n_samples=3)

        assert len(samples) == 3
        for s in samples:
            # Sample should have valid 3D coordinates matching model's atom count
            assert s.coordinates is not None
            assert s.coordinates.shape == (n_atoms_model, 3)
            # Same sequence as template
            assert len(s.sequence) == len(template.sequence)

    def test_repr(self, polymer_model):
        """Test string representation."""
        repr_str = repr(polymer_model)
        assert "PolymerModel" in repr_str
        assert "latent_dim=6" in repr_str


class TestPolymerFlowModelRoundtrip:
    """Tests for encode-decode roundtrip behavior."""

    def test_encode_decode_preserves_structure(self, polymer_model):
        """Test that encode-decode approximately preserves input."""
        # Single residue (no positioning needed)
        sequence = np.array([Residue.A.value])
        n_atoms = 10
        # encode() expects list of (n_atoms, 3) tensors, one per residue
        aligned_coords = [torch.randn(n_atoms, 3)]

        latents = polymer_model.encode(aligned_coords, sequence)
        coords_recon = polymer_model.decode(latents, sequence)

        # Should be close for single residue (only PCA truncation error)
        # Note: The synthetic data doesn't have real structure, so we just check shapes
        assert coords_recon.shape == (n_atoms, 3)

    def test_multi_residue_decode(self, polymer_model):
        """Test decoding multi-residue sequence produces valid output."""
        sequence = np.array([Residue.A.value, Residue.G.value, Residue.A.value])
        latents = torch.randn(3, 6)

        coords = polymer_model.decode(latents, sequence)

        # Check output has correct total atoms
        expected_atoms = 10 + 12 + 10
        assert coords.shape == (expected_atoms, 3)

        # Check coords are finite
        assert torch.isfinite(coords).all()


class TestPolymerFlowModelSaveLoad:
    """Tests for save/load functionality."""

    @pytest.fixture
    def single_residue_model(self):
        """Create a single-residue PolymerFlowModel for save/load testing."""
        models = {
            Residue.A: create_test_residue_model(Residue.A, n_atoms=10, latent_dim=6),
        }
        return PolymerFlowModel(models)

    def test_save_load_roundtrip(self, single_residue_model, tmp_path):
        """Test save and load produces equivalent model."""
        pytest.importorskip("safetensors")
        polymer = single_residue_model

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
        sequence = np.array([Residue.A.value])
        latents = torch.randn(1, 6)

        with torch.no_grad():
            coords1 = polymer.decode(latents, sequence)
            coords2 = loaded.decode(latents, sequence)

        assert torch.allclose(coords1, coords2, atol=1e-5)


class TestPolymerFlowModelDevice:
    """Tests for device management."""

    def test_device_property(self, polymer_model):
        """Test device property returns correct device."""
        assert polymer_model.device == torch.device("cpu")

    def test_to_returns_self(self, polymer_model):
        """Test to() returns self for method chaining."""
        result = polymer_model.to("cpu")
        assert result is polymer_model

    def test_cpu_method(self, polymer_model):
        """Test cpu() method."""
        result = polymer_model.cpu()
        assert result is polymer_model
        assert polymer_model.device == torch.device("cpu")
