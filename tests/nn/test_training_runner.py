"""Tests for training runner and config parsing."""

from __future__ import annotations

import pytest


# =============================================================================
# Test BaseConfig nested dataclass parsing
# =============================================================================


class TestBaseConfigParsing:
    """Test BaseConfig.from_dict with nested dataclasses."""

    def test_device_override(self):
        """Device override should be applied to training config."""
        pytest.importorskip("torch")
        from ciffy.nn.flow.residue.trainer import ResidueFlowTrainingConfig

        config = ResidueFlowTrainingConfig.from_dict(
            {
                "training": {"epochs": 10},
                "data": {"residue": "A"},
            },
            device="cuda:1",
        )

        assert config.training.device == "cuda:1"
        assert config.training.epochs == 10


# =============================================================================
# Test LatentDiffusion config parsing
# =============================================================================


class TestLatentDiffusionConfig:
    """Test LatentDiffusion config parsing from YAML-like dicts."""

    @pytest.fixture
    def sample_config_dict(self):
        """Sample config dict like what would come from YAML."""
        return {
            "trainer": "latent_diffusion",
            "model": {
                "flow_model_path": "/path/to/flow",
                "num_timesteps": 500,
                "noise_schedule": "linear",
                "denoiser": {
                    "d_model": 128,
                    "num_layers": 4,
                    "num_heads": 4,
                },
            },
            "data": {
                "data_dir": "/path/to/data",
                "batch_size": 16,
            },
            "training": {
                "epochs": 50,
                "lr": 0.0001,
            },
        }

    def test_latent_diffusion_config_parsing(self, sample_config_dict):
        """LatentDiffusionTrainingConfig should parse nested denoiser config."""
        pytest.importorskip("torch")
        from ciffy.nn.diffusion.latent_trainer import LatentDiffusionTrainingConfig
        from ciffy.nn.diffusion.latent_diffusion import LatentDiffusionConfig
        from ciffy.nn.diffusion.latent_denoiser import LatentDenoiserConfig

        config = LatentDiffusionTrainingConfig.from_dict(sample_config_dict)

        # Model config should be proper dataclass
        assert isinstance(config.model, LatentDiffusionConfig)
        assert config.model.num_timesteps == 500
        assert config.model.noise_schedule == "linear"
        assert config.model.flow_model_path == "/path/to/flow"

        # Denoiser config should be proper dataclass, not dict
        assert isinstance(config.model.denoiser, LatentDenoiserConfig)
        assert config.model.denoiser.d_model == 128
        assert config.model.denoiser.num_layers == 4
        assert config.model.denoiser.num_heads == 4


# =============================================================================
# Test Flow trainer result format
# =============================================================================


class TestFlowTrainerResult:
    """Test that flow trainer returns proper result format."""

    def test_train_pca_flow_returns_original_model(self):
        """train_pca_flow should return original model, not compiled wrapper."""
        pytest.importorskip("torch")
        import torch
        import numpy as np
        from ciffy.nn.flow.residue.train import train_pca_flow
        from ciffy.nn.flow.residue.model import PCAFlow

        # Create synthetic data
        n_samples = 50
        n_features = 30
        data = np.random.randn(n_samples, n_features).astype(np.float32)

        # Train with GPU if available (compile only happens on GPU)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        flow, info = train_pca_flow(
            train_data=data,
            test_data=data[:10],
            latent_dim=4,
            n_layers=2,
            hidden_dim=16,
            n_epochs=2,
            device=device,
            verbose=False,
        )

        # Should be PCAFlow, not a compiled wrapper
        # The state_dict should have standard keys
        state_dict = flow.state_dict()
        assert "V" in state_dict, f"state_dict keys: {list(state_dict.keys())}"
        assert "mean" in state_dict


# =============================================================================
# Test ResidueFlowModel save/load roundtrip
# =============================================================================


class TestResidueFlowModelSaveLoad:
    """Test that ResidueFlowModel save/load works correctly."""

    def test_save_load_roundtrip(self):
        """Model should save and load with correct state."""
        pytest.importorskip("torch")
        pytest.importorskip("safetensors")
        import tempfile
        import torch
        from pathlib import Path
        from ciffy.nn.flow.residue.model import ResidueFlowModel, PCAFlow
        from ciffy.biochemistry import Residue

        # Use real adenine atom structure
        residue = Residue.A
        atom_indices = [atom.value for atom in residue]
        n_atoms = len(atom_indices)

        # Create a simple model matching the residue structure
        latent_dim = 4
        V = torch.randn(latent_dim, n_atoms * 3)
        mean = torch.randn(n_atoms * 3)
        flow = PCAFlow(V, mean, n_layers=2, hidden_dim=16)

        model = ResidueFlowModel(
            flow=flow,
            residue=residue,
            atom_indices=atom_indices,
            n_atoms=n_atoms,
        )

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save(tmpdir)

            # Check files exist
            assert (Path(tmpdir) / "config.json").exists()
            assert (Path(tmpdir) / "tensors.safetensors").exists()

            # Load and compare
            loaded = ResidueFlowModel.load(tmpdir)

            assert loaded.residue == model.residue
            assert loaded.n_atoms == model.n_atoms

            # Check parameters match
            orig_state = model.flow.state_dict()
            loaded_state = loaded.flow.state_dict()

            for key in orig_state:
                assert key in loaded_state, f"Missing key: {key}"
                assert torch.allclose(orig_state[key], loaded_state[key]), f"Mismatch in {key}"


# =============================================================================
# Test TrainingResult format
# =============================================================================


class TestTrainingResultFormat:
    """Test TrainingResult has required fields."""

    def test_training_result_has_loss_fields(self):
        """TrainingResult should have final_loss and best_loss fields."""
        from ciffy.nn.runners.training_runner import TrainingResult

        result = TrainingResult(
            name="test",
            config_path="/path/to/config",
            trainer_type="flow",
            status="success",
            final_loss=1.5,
            best_loss=1.2,
            epochs_trained=10,
            total_epochs=10,
        )

        assert result.final_loss == 1.5
        assert result.best_loss == 1.2

    def test_training_result_optional_fields(self):
        """TrainingResult should have sensible defaults for optional fields."""
        from ciffy.nn.runners.training_runner import TrainingResult

        result = TrainingResult(
            name="test",
            config_path="/path/to/config",
            trainer_type="flow",
            status="success",
        )

        assert result.final_loss is None
        assert result.best_loss is None
        assert result.epochs_trained == 0
        assert result.extra_metrics == {}
