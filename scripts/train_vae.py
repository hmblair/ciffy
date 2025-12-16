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
      device: cuda  # or 'cpu', 'mps'

    output:
      checkpoint_dir: ./checkpoints
      sample_dir: ./samples
      save_every: 10
      n_perturbations: 5
      perturbation_scale: 1.0
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

try:
    import torch
    import torch.optim as optim
    from torch.utils.data import DataLoader
except ImportError:
    print("PyTorch is required. Install with: pip install torch")
    sys.exit(1)

import ciffy
from ciffy import Scale, Molecule
from ciffy.nn import PolymerVAE, PolymerDataset


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    latent_dim: int = 64
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1
    beta: float = 1.0
    use_rope: bool = True
    use_swiglu: bool = True


@dataclass
class DataConfig:
    """Data loading configuration."""
    data_dir: str = "./data"
    scale: str = "chain"  # 'chain' or 'molecule'
    min_atoms: Optional[int] = None
    max_atoms: Optional[int] = None
    molecule_types: Optional[list[str]] = None
    exclude_ids: Optional[list[str]] = None


@dataclass
class TrainingConfig:
    """Training configuration."""
    epochs: int = 100
    lr: float = 1e-4
    weight_decay: float = 0.0
    batch_size: int = 1
    seed: Optional[int] = None
    device: str = "cuda"
    grad_clip: Optional[float] = 1.0


@dataclass
class OutputConfig:
    """Output configuration."""
    checkpoint_dir: str = "./checkpoints"
    sample_dir: str = "./samples"
    save_every: int = 10
    n_perturbations: int = 5
    perturbation_scale: float = 1.0


@dataclass
class Config:
    """Full configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def load_config(path: str) -> Config:
    """Load configuration from YAML file."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    config = Config()

    if "model" in raw:
        config.model = ModelConfig(**raw["model"])
    if "data" in raw:
        config.data = DataConfig(**raw["data"])
    if "training" in raw:
        config.training = TrainingConfig(**raw["training"])
    if "output" in raw:
        config.output = OutputConfig(**raw["output"])

    return config


# =============================================================================
# Training
# =============================================================================


def create_model(config: ModelConfig, device: torch.device) -> PolymerVAE:
    """Create and initialize the VAE model."""
    model = PolymerVAE(
        latent_dim=config.latent_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        dropout=config.dropout,
        beta=config.beta,
    )
    return model.to(device)


def create_dataset(config: DataConfig) -> PolymerDataset:
    """Create the polymer dataset."""
    # Parse scale
    scale = Scale.CHAIN if config.scale.lower() == "chain" else Scale.MOLECULE

    # Parse molecule types if specified
    molecule_types = None
    if config.molecule_types:
        molecule_types = tuple(Molecule[mt.upper()] for mt in config.molecule_types)

    return PolymerDataset(
        directory=config.data_dir,
        scale=scale,
        min_atoms=config.min_atoms,
        max_atoms=config.max_atoms,
        molecule_types=molecule_types,
        exclude_ids=config.exclude_ids,
        backend="torch",
    )


def train_epoch(
    model: PolymerVAE,
    dataset: PolymerDataset,
    optimizer: optim.Optimizer,
    device: torch.device,
    grad_clip: Optional[float] = None,
) -> dict[str, float]:
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    n_samples = 0
    n_skipped = 0

    indices = list(range(len(dataset)))
    random.shuffle(indices)

    for idx in indices:
        try:
            polymer = dataset[idx]

            # Strip non-polymer atoms (ligands, water, modified residues marked as HETATM)
            # This ensures molecule_type is consistent (e.g., RNA without OTHER atoms)
            polymer = polymer.poly()

            # Skip if polymer is too small or wrong type
            if polymer.size(Scale.RESIDUE) < 2:
                n_skipped += 1
                continue

            # Move to device
            polymer = polymer.to(device)

            # Forward pass
            optimizer.zero_grad()
            losses = model.compute_loss(polymer)

            # Check for NaN
            if torch.isnan(losses["loss"]):
                n_skipped += 1
                continue

            # Backward pass
            losses["loss"].backward()

            # Gradient clipping
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            # Accumulate metrics
            total_loss += losses["loss"].item()
            total_recon += losses["recon_loss"].item()
            total_kl += losses["kl_loss"].item()
            n_samples += 1

        except Exception as e:
            logger.warning(f"Skipping sample {idx}: {e}")
            n_skipped += 1
            continue

    if n_samples == 0:
        return {"loss": float("nan"), "recon_loss": float("nan"), "kl_loss": float("nan")}

    return {
        "loss": total_loss / n_samples,
        "recon_loss": total_recon / n_samples,
        "kl_loss": total_kl / n_samples,
        "n_samples": n_samples,
        "n_skipped": n_skipped,
    }


def generate_samples(
    model: PolymerVAE,
    dataset: PolymerDataset,
    device: torch.device,
    output_dir: Path,
    epoch: int,
    n_perturbations: int = 5,
    perturbation_scale: float = 1.0,
) -> None:
    """
    Generate perturbed samples for visualization.

    Selects a random structure, encodes it, perturbs the latent vector
    with noise, and saves the decoded structures.
    """
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select random valid structure
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    template = None
    for idx in indices:
        try:
            polymer = dataset[idx].poly()  # Strip non-polymer atoms
            if polymer.size(Scale.RESIDUE) >= 2:
                template = polymer.to(device)
                break
        except Exception:
            continue

    if template is None:
        logger.warning("Could not find valid structure for sampling")
        return

    with torch.no_grad():
        # Encode the template
        z_mu, z_logvar = model.encode(template)

        # Save original reconstruction
        recon = model.decode(z_mu, template, sample=False)
        recon_cpu = recon.numpy()
        recon_cpu.write(str(output_dir / f"epoch{epoch:04d}_original.cif"))

        # Generate perturbed samples
        for i in range(n_perturbations):
            # Add noise to latent vector
            noise = torch.randn_like(z_mu) * perturbation_scale
            z_perturbed = z_mu + noise

            # Decode perturbed latent
            perturbed = model.decode(z_perturbed, template, sample=False)
            perturbed_cpu = perturbed.numpy()
            perturbed_cpu.write(str(output_dir / f"epoch{epoch:04d}_perturb{i+1}.cif"))

    logger.info(f"Saved {n_perturbations + 1} samples to {output_dir}")


def save_checkpoint(
    model: PolymerVAE,
    optimizer: optim.Optimizer,
    epoch: int,
    metrics: dict,
    config: Config,
    path: Path,
) -> None:
    """Save training checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "config": {
            "model": vars(config.model),
            "data": vars(config.data),
            "training": vars(config.training),
            "output": vars(config.output),
        },
    }, path)

    logger.info(f"Saved checkpoint to {path}")


def load_checkpoint(
    path: Path,
    model: PolymerVAE,
    optimizer: Optional[optim.Optimizer] = None,
) -> int:
    """Load training checkpoint. Returns the epoch number."""
    checkpoint = torch.load(path, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    logger.info(f"Loaded checkpoint from {path} (epoch {checkpoint['epoch']})")
    return checkpoint["epoch"]


# =============================================================================
# Main
# =============================================================================


def main():
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
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded config from {args.config}")

    # Override device if specified
    if args.device:
        config.training.device = args.device

    # Set random seed
    if config.training.seed is not None:
        random.seed(config.training.seed)
        torch.manual_seed(config.training.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.training.seed)
        logger.info(f"Set random seed to {config.training.seed}")

    # Setup device
    if config.training.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        config.training.device = "cpu"
    elif config.training.device == "mps" and not torch.backends.mps.is_available():
        logger.warning("MPS not available, falling back to CPU")
        config.training.device = "cpu"

    device = torch.device(config.training.device)
    logger.info(f"Using device: {device}")

    # Create dataset
    logger.info(f"Loading data from {config.data.data_dir}")
    dataset = create_dataset(config.data)
    logger.info(f"Dataset size: {len(dataset)} structures")

    if len(dataset) == 0:
        logger.error("No structures found in dataset!")
        sys.exit(1)

    # Create model
    model = create_model(config.model, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,}")

    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(Path(args.resume), model, optimizer) + 1

    # Setup output directories
    checkpoint_dir = Path(config.output.checkpoint_dir)
    sample_dir = Path(config.output.sample_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    logger.info("Starting training...")
    best_loss = float("inf")

    for epoch in range(start_epoch, config.training.epochs):
        # Train epoch
        metrics = train_epoch(
            model, dataset, optimizer, device,
            grad_clip=config.training.grad_clip,
        )

        # Log metrics
        logger.info(
            f"Epoch {epoch+1}/{config.training.epochs} | "
            f"Loss: {metrics['loss']:.4f} | "
            f"Recon: {metrics['recon_loss']:.4f} | "
            f"KL: {metrics['kl_loss']:.4f} | "
            f"Samples: {metrics.get('n_samples', 0)} | "
            f"Skipped: {metrics.get('n_skipped', 0)}"
        )

        # Generate samples at end of epoch
        generate_samples(
            model, dataset, device, sample_dir, epoch + 1,
            n_perturbations=config.output.n_perturbations,
            perturbation_scale=config.output.perturbation_scale,
        )

        # Save checkpoint
        if (epoch + 1) % config.output.save_every == 0:
            save_checkpoint(
                model, optimizer, epoch + 1, metrics, config,
                checkpoint_dir / f"checkpoint_epoch{epoch+1:04d}.pt",
            )

        # Save best model
        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            save_checkpoint(
                model, optimizer, epoch + 1, metrics, config,
                checkpoint_dir / "checkpoint_best.pt",
            )

    # Save final checkpoint
    save_checkpoint(
        model, optimizer, config.training.epochs, metrics, config,
        checkpoint_dir / "checkpoint_final.pt",
    )

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
