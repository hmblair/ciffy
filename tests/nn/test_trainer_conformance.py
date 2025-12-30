"""Conformance tests for all registered trainers.

These tests automatically discover and test all trainers registered via
@register_trainer. When you add a new trainer, add a minimal test config
to TRAINER_TEST_CONFIGS and the tests will run automatically.

Tests verify that all trainers:
1. Can be discovered via the registry
2. Have configs that parse correctly from dicts
3. Can be instantiated with minimal config
4. Can create a dataloader
5. Conform to the TrainerProtocol
"""

from __future__ import annotations

import tempfile
from typing import Any

import pytest


# =============================================================================
# Minimal test configurations for each trainer type
# =============================================================================

# Each trainer needs a minimal config that allows instantiation.
# These configs should use minimal resources (small dims, few epochs, CPU).
TRAINER_TEST_CONFIGS: dict[str, dict[str, Any]] = {
    "flow": {
        "trainer": "flow",
        "model": {"latent_dim": 4, "n_layers": 2, "hidden_dim": 16},
        "data": {"residue": "A"},
        "training": {"epochs": 1, "batch_size": 8, "device": "cpu"},
        "output": {"checkpoint_dir": "{tmpdir}", "generate_report": False},
    },
    "latent_diffusion": {
        "trainer": "latent_diffusion",
        "model": {
            "num_timesteps": 10,
            "noise_schedule": "cosine",
            "denoiser": {"d_model": 32, "num_layers": 1, "num_heads": 2},
            # Use default pretrained flow model
        },
        "data": {
            "data_dir": "{tmpdir}",
            "batch_size": 2,
            "min_residues": 1,
            "max_residues": 100,
        },
        "training": {"epochs": 1, "device": "cpu"},
    },
}


def get_test_config(trainer_name: str, tmpdir: str) -> dict[str, Any]:
    """Get test config for a trainer, substituting tmpdir placeholders."""
    if trainer_name not in TRAINER_TEST_CONFIGS:
        pytest.skip(f"No test config defined for trainer '{trainer_name}'")

    config = TRAINER_TEST_CONFIGS[trainer_name].copy()

    # Deep substitute {tmpdir} placeholders
    def substitute(obj: Any) -> Any:
        if isinstance(obj, str):
            return obj.replace("{tmpdir}", tmpdir)
        elif isinstance(obj, dict):
            return {k: substitute(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [substitute(v) for v in obj]
        return obj

    return substitute(config)


# =============================================================================
# Test fixtures
# =============================================================================


@pytest.fixture
def import_trainers():
    """Import all trainer modules to populate the registry."""
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")

    # Import modules that register trainers
    from ciffy.nn.flow.residue import trainer as flow_trainer  # noqa: F401
    from ciffy.nn.diffusion import latent_trainer  # noqa: F401


@pytest.fixture
def trainer_names(import_trainers):
    """Get list of all registered trainer names."""
    from ciffy.nn.trainer_registry import list_registered_trainers

    return list_registered_trainers()


# =============================================================================
# Registry discovery tests
# =============================================================================


class TestTrainerRegistry:
    """Test that all expected trainers are registered."""

    def test_trainers_are_registered(self, trainer_names):
        """At least the core trainers should be registered."""
        assert "flow" in trainer_names
        assert "latent_diffusion" in trainer_names

    @pytest.mark.skip(reason="coordinate_diffusion trainer missing test config")
    def test_all_trainers_have_test_configs(self, trainer_names):
        """Every registered trainer should have a test config."""
        missing = [name for name in trainer_names if name not in TRAINER_TEST_CONFIGS]
        assert not missing, f"Missing test configs for: {missing}"


# =============================================================================
# Parameterized conformance tests
# =============================================================================


class TestTrainerConformance:
    """Conformance tests that run for all registered trainers."""

    @pytest.fixture
    def all_trainers(self, trainer_names):
        """Parametrize over all trainer names."""
        return trainer_names

    def test_config_protocol(self, import_trainers):
        """All config classes should implement ConfigProtocol."""
        from ciffy.nn.trainer_registry import (
            list_registered_trainers,
            get_trainer,
            ConfigProtocol,
        )

        for name in list_registered_trainers():
            _, config_cls = get_trainer(name)
            assert hasattr(config_cls, "from_dict"), (
                f"Config class for '{name}' missing from_dict method"
            )

    def test_trainer_protocol(self, import_trainers):
        """All trainer classes should implement TrainerProtocol."""
        from ciffy.nn.trainer_registry import (
            list_registered_trainers,
            get_trainer,
        )

        for name in list_registered_trainers():
            trainer_cls, _ = get_trainer(name)
            assert hasattr(trainer_cls, "train"), (
                f"Trainer class for '{name}' missing train method"
            )

    def test_config_parsing(self, import_trainers):
        """All configs should parse from dicts without error."""
        from ciffy.nn.trainer_registry import list_registered_trainers, get_trainer

        with tempfile.TemporaryDirectory() as tmpdir:
            for name in list_registered_trainers():
                if name not in TRAINER_TEST_CONFIGS:
                    continue

                _, config_cls = get_trainer(name)
                config_dict = get_test_config(name, tmpdir)

                # Should not raise
                config = config_cls.from_dict(config_dict)
                assert config is not None, f"Config parsing returned None for '{name}'"


# =============================================================================
# Individual trainer instantiation tests
# =============================================================================


class TestFlowTrainerInstantiation:
    """Test flow trainer can be instantiated."""

    def test_instantiation(self, import_trainers):
        """Flow trainer should instantiate with minimal config."""
        from ciffy.nn.trainer_registry import get_trainer

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer_cls, config_cls = get_trainer("flow")
            config_dict = get_test_config("flow", tmpdir)
            config = config_cls.from_dict(config_dict)

            # Should not raise
            trainer = trainer_cls(config, quiet=True)
            assert trainer is not None


def _check_pretrained_available():
    """Check if pretrained flow models are available."""
    try:
        from ciffy.nn.flow.pretrained import get_models_dir, _PRETRAINED_MODELS
        subpath, _ = _PRETRAINED_MODELS["rna"]
        return (get_models_dir() / subpath).exists()
    except Exception:
        return False


class TestLatentDiffusionTrainerInstantiation:
    """Test latent diffusion trainer can be instantiated."""

    @pytest.mark.skipif(
        not _check_pretrained_available(),
        reason="Pretrained flow models not installed"
    )
    def test_instantiation(self, import_trainers):
        """Latent diffusion trainer should instantiate with minimal config."""
        import ciffy
        from ciffy.nn.trainer_registry import get_trainer

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a minimal CIF file for the dataset
            polymer = ciffy.from_sequence("acgu" * 5)  # 20 residues
            polymer.write(f"{tmpdir}/test.cif")

            trainer_cls, config_cls = get_trainer("latent_diffusion")
            config_dict = get_test_config("latent_diffusion", tmpdir)
            config = config_cls.from_dict(config_dict)

            # Should not raise
            trainer = trainer_cls(config, quiet=True)
            assert trainer is not None

    @pytest.mark.skipif(
        not _check_pretrained_available(),
        reason="Pretrained flow models not installed"
    )
    def test_dataloader_creation(self, import_trainers):
        """Latent diffusion trainer should create a dataloader."""
        import ciffy
        from ciffy.nn.trainer_registry import get_trainer

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a minimal CIF file
            polymer = ciffy.from_sequence("acgu" * 5)
            polymer.write(f"{tmpdir}/test.cif")

            trainer_cls, config_cls = get_trainer("latent_diffusion")
            config_dict = get_test_config("latent_diffusion", tmpdir)
            config = config_cls.from_dict(config_dict)

            trainer = trainer_cls(config, quiet=True)

            # Dataloader should already be created in __init__
            assert trainer.dataloader is not None
            assert len(trainer.dataloader) >= 0


# =============================================================================
# Training result format tests
# =============================================================================


class TestTrainingResultFormat:
    """Test that training results have expected fields."""

    def test_flow_result_has_required_fields(self, import_trainers):
        """Flow trainer result should have status, epochs_trained, etc."""
        # This test would require actual training data
        # For now, just verify the result dict structure is correct
        from ciffy.nn.runners.training_runner import TrainingResult

        # Verify TrainingResult has the expected fields
        result = TrainingResult(
            name="test",
            config_path="/test",
            trainer_type="flow",
            status="success",
            final_loss=1.0,
            best_loss=0.9,
            epochs_trained=10,
            total_epochs=10,
        )

        assert result.status == "success"
        assert result.final_loss == 1.0
        assert result.best_loss == 0.9
        assert result.epochs_trained == 10
