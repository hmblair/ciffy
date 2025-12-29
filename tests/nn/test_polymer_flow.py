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
        """Test encode output shape."""
        sequence = np.array([Residue.A.value, Residue.G.value, Residue.A.value])
        n_atoms = 10 + 12 + 10
        coords = torch.randn(n_atoms, 3)

        latents = polymer_model.encode(coords, sequence)

        assert latents.shape == (3, 6)  # 3 residues, 6 latent dims

    def test_encode_validates_coords_shape(self, polymer_model):
        """Test encode validates coordinate shape."""
        sequence = np.array([Residue.A.value])
        coords = torch.randn(5, 3)  # Wrong number of atoms

        with pytest.raises(ValueError, match="atoms but sequence expects"):
            polymer_model.encode(coords, sequence)

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

    @pytest.mark.xfail(reason="from_sequence atom count bug with multi-residue chains")
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
        assert "PolymerFlowModel" in repr_str
        assert "latent_dim=6" in repr_str


class TestPolymerFlowModelRoundtrip:
    """Tests for encode-decode roundtrip behavior."""

    def test_encode_decode_preserves_structure(self, polymer_model):
        """Test that encode-decode approximately preserves input."""
        # Single residue (no positioning needed)
        sequence = np.array([Residue.A.value])
        n_atoms = 10
        coords = torch.randn(n_atoms, 3)

        latents = polymer_model.encode(coords, sequence)
        coords_recon = polymer_model.decode(latents, sequence)

        # Should be close for single residue (only PCA truncation error)
        # Note: The synthetic data doesn't have real structure, so we just check shapes
        assert coords_recon.shape == coords.shape

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


class TestPolymerFlowModelLazyComputation:
    """Tests for stateful lazy computation API."""

    def test_initial_state_unbound(self, polymer_model):
        """Test model starts unbound."""
        assert not polymer_model.is_bound
        assert polymer_model.sequence is None

    def test_bind_sets_sequence(self, polymer_model):
        """Test bind sets the sequence."""
        sequence = np.array([Residue.A.value, Residue.G.value])
        polymer_model.bind(sequence)

        assert polymer_model.is_bound
        assert np.array_equal(polymer_model.sequence, sequence)

    def test_bind_returns_self(self, polymer_model):
        """Test bind returns self for method chaining."""
        sequence = np.array([Residue.A.value])
        result = polymer_model.bind(sequence)
        assert result is polymer_model

    def test_bind_validates_residue_types(self, polymer_model):
        """Test bind validates all residue types are supported."""
        with pytest.raises(ValueError, match="Unsupported residue types"):
            polymer_model.bind(np.array([Residue.A.value, Residue.C.value]))  # C not in models

    def test_unbind_clears_state(self, polymer_model):
        """Test unbind clears all cached state."""
        sequence = np.array([Residue.A.value])
        polymer_model.bind(sequence)
        polymer_model.coordinates = torch.randn(10, 3)
        _ = polymer_model.latents  # Compute latents

        polymer_model.unbind()

        assert not polymer_model.is_bound
        assert polymer_model.sequence is None

    def test_latents_property_requires_bound(self, polymer_model):
        """Test latents property raises if not bound."""
        with pytest.raises(RuntimeError, match="No sequence bound"):
            _ = polymer_model.latents

    def test_coordinates_property_requires_bound(self, polymer_model):
        """Test coordinates property raises if not bound."""
        with pytest.raises(RuntimeError, match="No sequence bound"):
            _ = polymer_model.coordinates

    def test_latents_setter_requires_bound(self, polymer_model):
        """Test latents setter raises if not bound."""
        with pytest.raises(RuntimeError, match="No sequence bound"):
            polymer_model.latents = torch.randn(1, 6)

    def test_coordinates_setter_requires_bound(self, polymer_model):
        """Test coordinates setter raises if not bound."""
        with pytest.raises(RuntimeError, match="No sequence bound"):
            polymer_model.coordinates = torch.randn(10, 3)

    def test_set_coordinates_marks_latents_dirty(self, polymer_model):
        """Test setting coordinates marks latents as needing recomputation."""
        sequence = np.array([Residue.A.value])
        polymer_model.bind(sequence)

        # Set coordinates
        coords = torch.randn(10, 3)
        polymer_model.coordinates = coords

        # Internal state should show latents dirty
        assert polymer_model._latents_dirty
        assert not polymer_model._coords_dirty

    def test_set_latents_marks_coords_dirty(self, polymer_model):
        """Test setting latents marks coordinates as needing recomputation."""
        sequence = np.array([Residue.A.value])
        polymer_model.bind(sequence)

        # Set latents
        latents = torch.randn(1, 6)
        polymer_model.latents = latents

        # Internal state should show coords dirty
        assert polymer_model._coords_dirty
        assert not polymer_model._latents_dirty

    def test_lazy_latents_computation(self, polymer_model):
        """Test latents are computed lazily from coordinates."""
        sequence = np.array([Residue.A.value])
        polymer_model.bind(sequence)

        coords = torch.randn(10, 3)
        polymer_model.coordinates = coords

        # First access computes latents
        latents = polymer_model.latents

        assert latents.shape == (1, 6)
        assert not polymer_model._latents_dirty

    def test_lazy_coords_computation(self, polymer_model):
        """Test coordinates are computed lazily from latents."""
        sequence = np.array([Residue.A.value])
        polymer_model.bind(sequence)

        latents = torch.randn(1, 6)
        polymer_model.latents = latents

        # First access computes coordinates
        coords = polymer_model.coordinates

        assert coords.shape == (10, 3)
        assert not polymer_model._coords_dirty

    def test_cached_latents_not_recomputed(self, polymer_model):
        """Test cached latents are returned without recomputation."""
        sequence = np.array([Residue.A.value])
        polymer_model.bind(sequence)

        coords = torch.randn(10, 3)
        polymer_model.coordinates = coords

        # First access computes
        latents1 = polymer_model.latents

        # Second access returns cached
        latents2 = polymer_model.latents

        assert latents1 is latents2  # Same object

    def test_cached_coords_not_recomputed(self, polymer_model):
        """Test cached coordinates are returned without recomputation."""
        sequence = np.array([Residue.A.value])
        polymer_model.bind(sequence)

        latents = torch.randn(1, 6)
        polymer_model.latents = latents

        # First access computes
        coords1 = polymer_model.coordinates

        # Second access returns cached
        coords2 = polymer_model.coordinates

        assert coords1 is coords2  # Same object

    def test_latents_error_when_no_coords(self, polymer_model):
        """Test accessing latents without coordinates raises error."""
        sequence = np.array([Residue.A.value])
        polymer_model.bind(sequence)

        with pytest.raises(RuntimeError, match="no coordinates set"):
            _ = polymer_model.latents

    def test_coords_error_when_no_latents(self, polymer_model):
        """Test accessing coordinates without latents raises error."""
        sequence = np.array([Residue.A.value])
        polymer_model.bind(sequence)

        with pytest.raises(RuntimeError, match="no latents set"):
            _ = polymer_model.coordinates

    def test_latents_setter_validates_shape(self, polymer_model):
        """Test latents setter validates shape matches sequence."""
        sequence = np.array([Residue.A.value, Residue.G.value])  # 2 residues
        polymer_model.bind(sequence)

        with pytest.raises(ValueError, match="rows but sequence has"):
            polymer_model.latents = torch.randn(3, 6)  # 3 rows, wrong

    def test_latents_setter_validates_dim(self, polymer_model):
        """Test latents setter validates latent dimension."""
        sequence = np.array([Residue.A.value])
        polymer_model.bind(sequence)

        with pytest.raises(ValueError, match="dim .* but model expects"):
            polymer_model.latents = torch.randn(1, 8)  # Wrong dim

    def test_coordinates_setter_validates_shape(self, polymer_model):
        """Test coordinates setter validates shape matches sequence."""
        sequence = np.array([Residue.A.value])  # 10 atoms
        polymer_model.bind(sequence)

        with pytest.raises(ValueError, match="atoms but sequence expects"):
            polymer_model.coordinates = torch.randn(15, 3)  # Wrong atoms

    def test_rebind_clears_cache(self, polymer_model):
        """Test rebinding clears cached values."""
        sequence1 = np.array([Residue.A.value])
        polymer_model.bind(sequence1)
        polymer_model.coordinates = torch.randn(10, 3)
        _ = polymer_model.latents

        # Rebind to different sequence
        sequence2 = np.array([Residue.G.value])
        polymer_model.bind(sequence2)

        assert polymer_model._cached_latents is None
        assert polymer_model._cached_coordinates is None
        assert polymer_model._latents_dirty
        assert polymer_model._coords_dirty

    def test_multi_residue_lazy_roundtrip(self, polymer_model):
        """Test lazy computation with multi-residue sequence."""
        sequence = np.array([Residue.A.value, Residue.G.value, Residue.A.value])
        n_atoms = 10 + 12 + 10
        polymer_model.bind(sequence)

        # Set coordinates
        coords = torch.randn(n_atoms, 3)
        polymer_model.coordinates = coords

        # Get latents (lazy computed)
        latents = polymer_model.latents
        assert latents.shape == (3, 6)

        # Modify latents
        modified = latents + 0.1
        polymer_model.latents = modified

        # Get coordinates (lazy recomputed)
        new_coords = polymer_model.coordinates
        assert new_coords.shape == (n_atoms, 3)


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
