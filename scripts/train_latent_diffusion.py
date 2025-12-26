#!/usr/bin/env python
"""Train a latent diffusion model for polymer structure generation.

This script trains a transformer-based denoiser that operates in the latent
space of a pre-trained PolymerFlowModel. The model learns to generate polymer
structures by denoising random latent vectors conditioned on residue sequence.

Usage:
    # From YAML config
    python scripts/train_latent_diffusion.py config.yaml

    # With command-line overrides
    python scripts/train_latent_diffusion.py --data-dir ./rna_structures --epochs 100

    # Minimal example (uses pretrained RNA flow model)
    python scripts/train_latent_diffusion.py --data-dir ./data/rna

Example config.yaml:
    model:
      denoiser:
        d_model: 256
        num_layers: 6
        num_heads: 8
      num_timesteps: 1000
      noise_schedule: cosine

    data:
      data_dir: ./data/rna_structures
      batch_size: 32
      molecule_types: [RNA]

    training:
      epochs: 100
      lr: 1e-4
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

try:
    from ciffy.nn.diffusion import (
        LatentDiffusionConfig,
        LatentDiffusionModel,
        LatentDiffusionTrainer,
        LatentDiffusionTrainingConfig,
        LatentDenoiserConfig,
    )
    from ciffy.nn.diffusion.latent_trainer import LatentDiffusionDataConfig
    from ciffy.nn.base_trainer import TrainingConfig, OutputConfig, WandbConfig
except ImportError as e:
    print(f"Error importing ciffy modules: {e}")
    print("Make sure ciffy is installed: pip install -e .")
    raise


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train latent diffusion model for polymer structure generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "config",
        nargs="?",
        type=str,
        default=None,
        help="Path to YAML config file",
    )

    # Data options
    parser.add_argument(
        "--data-dir",
        type=str,
        help="Training data directory (CIF files)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size",
    )
    parser.add_argument(
        "--molecule-types",
        nargs="+",
        default=["RNA"],
        help="Molecule types to train on",
    )
    parser.add_argument(
        "--min-residues",
        type=int,
        default=10,
        help="Minimum residues per chain",
    )
    parser.add_argument(
        "--max-residues",
        type=int,
        default=500,
        help="Maximum residues per chain",
    )

    # Model options
    parser.add_argument(
        "--d-model",
        type=int,
        default=256,
        help="Transformer hidden dimension",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=6,
        help="Number of transformer layers",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=8,
        help="Number of attention heads",
    )
    parser.add_argument(
        "--num-timesteps",
        type=int,
        default=1000,
        help="Number of diffusion timesteps",
    )
    parser.add_argument(
        "--noise-schedule",
        type=str,
        default="cosine",
        choices=["cosine", "linear"],
        help="Noise schedule type",
    )
    parser.add_argument(
        "--flow-model-path",
        type=str,
        default=None,
        help="Path to pre-trained PolymerFlowModel (None uses default)",
    )

    # Training options
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay for AdamW",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="Gradient clipping norm",
    )
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.9999,
        help="EMA decay rate",
    )

    # Output options
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints/latent_diffusion",
        help="Directory for saving checkpoints",
    )
    parser.add_argument(
        "--sample-dir",
        type=str,
        default="./samples/latent_diffusion",
        help="Directory for saving generated samples",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Save checkpoint every N epochs",
    )
    parser.add_argument(
        "--val-every",
        type=int,
        default=10,
        help="Run validation every N epochs",
    )
    parser.add_argument(
        "--val-samples",
        type=int,
        default=5,
        help="Number of validation samples",
    )
    parser.add_argument(
        "--val-steps",
        type=int,
        default=50,
        help="DDIM steps for validation sampling",
    )

    # Logging options
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="Weights & Biases project name (enables logging)",
    )
    parser.add_argument(
        "--wandb-name",
        type=str,
        default=None,
        help="Weights & Biases run name",
    )

    # Other options
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to train on (auto, cuda, cpu, mps)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint path",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress bars and reduce logging",
    )

    return parser.parse_args()


def create_config(args: argparse.Namespace) -> LatentDiffusionTrainingConfig:
    """Create training configuration from arguments."""
    # Load from YAML if provided
    if args.config is not None:
        config = LatentDiffusionTrainingConfig.from_yaml(args.config)
    else:
        config = LatentDiffusionTrainingConfig()

    # Apply command-line overrides

    # Data config
    if args.data_dir:
        config.data.data_dir = args.data_dir
    config.data.batch_size = args.batch_size
    config.data.molecule_types = tuple(args.molecule_types)
    config.data.min_residues = args.min_residues
    config.data.max_residues = args.max_residues
    config.data.num_workers = args.num_workers

    # Model config
    config.model.denoiser.d_model = args.d_model
    config.model.denoiser.num_layers = args.num_layers
    config.model.denoiser.num_heads = args.num_heads
    config.model.num_timesteps = args.num_timesteps
    config.model.noise_schedule = args.noise_schedule
    if args.flow_model_path:
        config.model.flow_model_path = args.flow_model_path

    # Training config
    config.training.epochs = args.epochs
    config.training.lr = args.lr
    config.training.weight_decay = args.weight_decay
    config.training.grad_clip = args.grad_clip
    config.training.device = args.device
    config.training.seed = args.seed
    config.training.num_workers = args.num_workers

    # EMA config
    config.ema_decay = args.ema_decay

    # Validation config
    config.val_every = args.val_every
    config.val_samples = args.val_samples
    config.val_steps = args.val_steps

    # Output config
    config.output.checkpoint_dir = args.checkpoint_dir
    config.output.sample_dir = args.sample_dir
    config.output.save_every = args.save_every

    # WandB config
    if args.wandb_project:
        config.wandb.project = args.wandb_project
        config.wandb.enabled = True
    if args.wandb_name:
        config.wandb.name = args.wandb_name

    return config


def create_logger(config: LatentDiffusionTrainingConfig):
    """Create metrics logger if WandB is enabled."""
    if config.wandb.project and config.wandb.enabled:
        try:
            import wandb

            class WandbLogger:
                def __init__(self, config):
                    wandb.init(
                        project=config.wandb.project,
                        name=config.wandb.name,
                        group=config.wandb.group,
                        config=config.to_dict(),
                    )

                def log(self, metrics: dict, step: int) -> None:
                    wandb.log(metrics, step=step)

                def finish(self) -> None:
                    wandb.finish()

            return WandbLogger(config)
        except ImportError:
            logging.warning("wandb not installed, logging disabled")
            return None
    return None


def main() -> None:
    """Main training entry point."""
    args = parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger = logging.getLogger(__name__)

    # Create config
    config = create_config(args)

    # Validate data directory
    if not config.data.data_dir:
        logger.error("No data directory specified. Use --data-dir or config file.")
        return

    data_path = Path(config.data.data_dir)
    if not data_path.exists():
        logger.error(f"Data directory does not exist: {data_path}")
        return

    # Log configuration
    if not args.quiet:
        logger.info("=" * 60)
        logger.info("Latent Diffusion Training")
        logger.info("=" * 60)
        logger.info(f"Data directory: {config.data.data_dir}")
        logger.info(f"Molecule types: {config.data.molecule_types}")
        logger.info(f"Model: d_model={config.model.denoiser.d_model}, "
                   f"layers={config.model.denoiser.num_layers}, "
                   f"heads={config.model.denoiser.num_heads}")
        logger.info(f"Diffusion: {config.model.num_timesteps} steps, "
                   f"{config.model.noise_schedule} schedule")
        logger.info(f"Training: {config.training.epochs} epochs, "
                   f"lr={config.training.lr}, "
                   f"batch_size={config.data.batch_size}")
        logger.info(f"Checkpoints: {config.output.checkpoint_dir}")
        logger.info("=" * 60)

    # Create metrics logger
    metrics_logger = create_logger(config)

    # Create trainer
    trainer = LatentDiffusionTrainer(
        config=config,
        logger=metrics_logger,
        quiet=args.quiet,
    )

    # Log model info
    if not args.quiet:
        n_params = sum(p.numel() for p in trainer.model.denoiser.parameters())
        logger.info(f"Denoiser parameters: {n_params:,}")
        logger.info(f"Latent dimension: {trainer.model.latent_dim}")
        logger.info(f"Cached samples: {len(trainer._latent_cache)}")

    # Train
    try:
        result = trainer.train(resume_path=args.resume)

        if not args.quiet:
            logger.info("\n" + "=" * 60)
            logger.info("Training Complete!")
            logger.info("=" * 60)
            logger.info(f"Final loss: {result['final_loss']:.6f}")
            logger.info(f"Best loss: {result['best_loss']:.6f}")
            logger.info(f"Epochs trained: {result['epochs_trained']}")
            logger.info(f"Checkpoint: {result['checkpoint_path']}")
            logger.info("=" * 60)

    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
