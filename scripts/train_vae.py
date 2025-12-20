#!/usr/bin/env python3
"""
Training script for Polymer VAE.

Trains a variational autoencoder on polymer backbone conformations
using dihedral angle representation.

Usage:
    python scripts/train_vae.py config.yaml

Example config (config.yaml):
    model:
      latent_dim: 64
      hidden_dim: 256
      num_layers: 4
      num_heads: 8
      dropout: 0.1
      beta: 1.0
      beta_schedule: linear  # constant, linear, cosine, cyclical
      beta_warmup_epochs: 50  # null = half of total epochs

    data:
      data_dir: /path/to/cif/files
      scale: chain  # or 'molecule'
      min_atoms: 10
      max_atoms: 5000

    training:
      epochs: 100
      lr: 1e-4
      batch_size: 1  # VAE processes one polymer at a time
      seed: 42
      device: auto  # 'auto', 'cuda', 'mps', or 'cpu'
      num_workers: 0  # DataLoader workers

    output:
      checkpoint_dir: ./checkpoints
      sample_dir: ./samples
      save_every: 10
      n_perturbations: 5
      perturbation_scale: 1.0

    wandb:  # optional
      project: my-project
      group: null
      enabled: true
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

try:
    import torch
except ImportError:
    print("PyTorch is required. Install with: pip install torch")
    sys.exit(1)

from ciffy.nn.vae.trainer import VAEConfig, VAETrainer, VAEModelConfig, DataConfig
from ciffy.nn.base_trainer import TrainingConfig, OutputConfig, WandbConfig
from ciffy.nn.training import get_device


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Legacy Configuration Support
# =============================================================================

# For backward compatibility with experiment_runner.py imports
@dataclass
class ModelConfig:
    """Model architecture configuration (legacy)."""
    latent_dim: int = 64
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1
    beta: float = 1.0
    beta_schedule: str = "constant"
    beta_warmup_epochs: Optional[int] = None
    beta_cycles: int = 4


@dataclass
class LegacyDataConfig:
    """Data loading configuration (legacy)."""
    data_dir: str = "./data"
    scale: str = "chain"
    min_atoms: Optional[int] = None
    max_atoms: Optional[int] = None
    molecule_types: Optional[list[str]] = None
    exclude_ids: Optional[list[str]] = None
    limit: Optional[int] = None


@dataclass
class LegacyTrainingConfig:
    """Training configuration (legacy)."""
    epochs: int = 100
    lr: float = 1e-4
    weight_decay: float = 0.0
    batch_size: int = 1
    seed: Optional[int] = None
    device: str = "auto"
    grad_clip: Optional[float] = 1.0
    num_workers: int = 0


@dataclass
class LegacyOutputConfig:
    """Output configuration (legacy)."""
    checkpoint_dir: str = "./checkpoints"
    sample_dir: str = "./samples"
    save_every: int = 10
    n_perturbations: int = 5
    perturbation_scale: float = 1.0


@dataclass
class Config:
    """Full configuration (legacy format)."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: LegacyDataConfig = field(default_factory=LegacyDataConfig)
    training: LegacyTrainingConfig = field(default_factory=LegacyTrainingConfig)
    output: LegacyOutputConfig = field(default_factory=LegacyOutputConfig)


def load_config(path: str) -> Config:
    """Load configuration from YAML file (legacy format)."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    config = Config()

    if "model" in raw:
        config.model = ModelConfig(**raw["model"])
    if "data" in raw:
        config.data = LegacyDataConfig(**raw["data"])
    if "training" in raw:
        config.training = LegacyTrainingConfig(**raw["training"])
    if "output" in raw:
        config.output = LegacyOutputConfig(**raw["output"])

    return config


def _convert_legacy_config(raw: dict) -> VAEConfig:
    """Convert legacy config dict to new VAEConfig format."""
    model_raw = raw.get("model", {})
    data_raw = raw.get("data", {})
    training_raw = raw.get("training", {})
    output_raw = raw.get("output", {})
    wandb_raw = raw.get("wandb", {})

    return VAEConfig(
        model=VAEModelConfig(
            latent_dim=model_raw.get("latent_dim", 64),
            hidden_dim=model_raw.get("hidden_dim", 256),
            num_layers=model_raw.get("num_layers", 4),
            num_heads=model_raw.get("num_heads", 8),
            dropout=model_raw.get("dropout", 0.1),
            beta=model_raw.get("beta", 1.0),
            beta_schedule=model_raw.get("beta_schedule", "constant"),
            beta_warmup_epochs=model_raw.get("beta_warmup_epochs"),
            beta_cycles=model_raw.get("beta_cycles", 4),
        ),
        data=DataConfig(
            data_dir=data_raw.get("data_dir", "./data"),
            scale=data_raw.get("scale", "chain"),
            min_atoms=data_raw.get("min_atoms"),
            max_atoms=data_raw.get("max_atoms"),
            molecule_types=data_raw.get("molecule_types"),
            exclude_ids=data_raw.get("exclude_ids"),
            limit=data_raw.get("limit"),
        ),
        training=TrainingConfig(
            epochs=training_raw.get("epochs", 100),
            lr=training_raw.get("lr", 1e-4),
            weight_decay=training_raw.get("weight_decay", 0.0),
            grad_clip=training_raw.get("grad_clip", 1.0),
            device=training_raw.get("device", "auto"),
            seed=training_raw.get("seed"),
            num_workers=training_raw.get("num_workers", 0),
        ),
        output=OutputConfig(
            checkpoint_dir=output_raw.get("checkpoint_dir", "./checkpoints"),
            sample_dir=output_raw.get("sample_dir", "./samples"),
            save_every=output_raw.get("save_every", 10),
            n_perturbations=output_raw.get("n_perturbations", 5),
            perturbation_scale=output_raw.get("perturbation_scale", 1.0),
        ),
        wandb=WandbConfig(
            project=wandb_raw.get("project"),
            group=wandb_raw.get("group"),
            name=wandb_raw.get("name"),
            enabled=wandb_raw.get("enabled", True),
        ),
    )


# =============================================================================
# Main Training Function
# =============================================================================


def train_vae(
    config_path: str,
    device_override: str | None = None,
    resume_path: str | None = None,
    experiment_name: str | None = None,
    quiet: bool = False,
    progress_callback=None,
) -> dict[str, Any]:
    """
    Train the Polymer VAE using the new VAETrainer.

    Args:
        config_path: Path to YAML configuration file.
        device_override: Override device from config (e.g., "cuda:1").
        resume_path: Path to checkpoint to resume from.
        experiment_name: Name for this experiment. If provided, output
            directories are suffixed with this name.
        quiet: If True, suppress progress bars and reduce logging.
        progress_callback: Optional callback called after each epoch with
            signature: callback(epoch, total_epochs, metrics) where metrics
            is a dict containing 'loss', 'recon_loss', 'kl_loss', etc.

    Returns:
        Dict containing:
        - 'final_loss': Loss value from final epoch
        - 'best_loss': Best loss achieved during training
        - 'final_recon_loss': Final reconstruction loss
        - 'final_kl_loss': Final KL loss
        - 'epochs_trained': Number of epochs completed
        - 'n_samples': Total samples processed
        - 'device': Device string used
        - 'checkpoint_path': Path to best checkpoint
        - 'error': None if successful, error message otherwise
    """
    try:
        # Load and convert config
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}

        config = _convert_legacy_config(raw)

        # Override device if specified
        if device_override:
            config.training.device = device_override

        if not quiet:
            logger.info(f"Loaded config from {config_path}")

        # Create trainer
        trainer = VAETrainer(
            config=config,
            quiet=quiet,
            experiment_name=experiment_name,
        )

        if not quiet:
            logger.info(f"Using device: {trainer.device}")
            logger.info(f"Dataset size: {len(trainer.dataset)} structures")
            n_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
            logger.info(f"Model parameters: {n_params:,}")

        if len(trainer.dataset) == 0:
            return {
                "error": "No structures found in dataset",
                "device": str(trainer.device),
            }

        # Run training
        result = trainer.train(
            resume_path=resume_path,
            progress_callback=progress_callback,
        )

        return result

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "error": str(e),
            "device": device_override or "unknown",
        }


# =============================================================================
# CLI Entry Point
# =============================================================================


def main():
    """Command-line entry point for training."""
    parser = argparse.ArgumentParser(
        description="Train Polymer VAE on dihedral angles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device from config",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="Enable wandb logging with this project name",
    )
    args = parser.parse_args()

    result = train_vae(
        config_path=args.config,
        device_override=args.device,
        resume_path=args.resume,
        quiet=False,
    )

    if result.get("error"):
        logger.error(f"Training failed: {result['error']}")
        sys.exit(1)

    logger.info(f"Best loss: {result['best_loss']:.4f}")


if __name__ == "__main__":
    main()
