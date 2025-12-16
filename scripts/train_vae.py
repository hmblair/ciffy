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
from ciffy.nn.training import (
    set_seed,
    get_device,
    save_checkpoint,
    load_checkpoint,
    train_epoch,
    polymer_collate_fn,
    get_worker_init_fn,
    BetaScheduler,
)


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
    # Beta annealing options
    beta_schedule: str = "constant"  # constant, linear, cosine, cyclical
    beta_warmup_epochs: Optional[int] = None  # None = half of total epochs
    beta_cycles: int = 4  # For cyclical schedule


@dataclass
class DataConfig:
    """Data loading configuration."""
    data_dir: str = "./data"
    scale: str = "chain"  # 'chain' or 'molecule'
    min_atoms: Optional[int] = None
    max_atoms: Optional[int] = None
    molecule_types: Optional[list[str]] = None
    exclude_ids: Optional[list[str]] = None
    limit: Optional[int] = None  # Max samples (for overfitting tests)


@dataclass
class TrainingConfig:
    """Training configuration."""
    epochs: int = 100
    lr: float = 1e-4
    weight_decay: float = 0.0
    batch_size: int = 1
    seed: Optional[int] = None
    device: str = "auto"
    grad_clip: Optional[float] = 1.0
    num_workers: int = 0


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
# Model and Dataset Creation
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
        limit=config.limit,
        backend="torch",
    )


def create_dataloader(
    dataset: PolymerDataset,
    config: TrainingConfig,
) -> DataLoader:
    """Create DataLoader with appropriate settings."""
    return DataLoader(
        dataset,
        batch_size=1,  # Always 1 for variable-size molecules
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=polymer_collate_fn,
        worker_init_fn=get_worker_init_fn(config.seed),
        pin_memory=(config.device != "cpu"),
        persistent_workers=(config.num_workers > 0),
    )


# =============================================================================
# VAE-Specific Loss Function
# =============================================================================


def create_vae_loss_fn(device: torch.device):
    """
    Create loss function for VAE training.

    Returns a callable that:
    1. Prepares the polymer (strips non-polymer atoms, validates)
    2. Moves to device
    3. Calls model.compute_loss()
    """
    # Supported molecule types for dihedral VAE
    supported_types = (Molecule.PROTEIN, Molecule.PROTEIN_D, Molecule.RNA, Molecule.DNA)

    def loss_fn(model: PolymerVAE, polymer: ciffy.Polymer) -> dict[str, torch.Tensor]:
        # Strip non-polymer atoms (ligands, water, etc.)
        polymer = polymer.poly()

        # Skip too-small polymers
        if polymer.size(Scale.RESIDUE) < 2:
            return {"loss": torch.tensor(float("nan"))}

        # Skip unsupported molecule types
        mol_type_val = polymer.molecule_type[0]
        if hasattr(mol_type_val, "item"):
            mol_type_val = mol_type_val.item()
        mol_type = Molecule(mol_type_val)

        if mol_type not in supported_types:
            return {"loss": torch.tensor(float("nan"))}

        # Move to device and compute loss
        polymer = polymer.to(device)
        return model.compute_loss(polymer)

    return loss_fn


# =============================================================================
# Sample Generation (VAE-specific)
# =============================================================================


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
    Generate samples for visualization.

    Saves:
    - Original template structure (from dataset)
    - Reconstruction through the VAE
    - Perturbed samples (latent + noise)
    """
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select random valid structure
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    template = None
    for idx in indices:
        try:
            polymer = dataset[idx]
            if polymer is None:
                continue
            polymer = polymer.poly()  # Strip non-polymer atoms
            if polymer.size(Scale.RESIDUE) >= 2:
                template = polymer.to(device)
                break
        except Exception:
            continue

    if template is None:
        logger.warning("Could not find valid structure for sampling")
        return

    with torch.no_grad():
        # Save original template (ground truth)
        template_cpu = template.numpy()
        template_cpu.write(str(output_dir / f"epoch{epoch:04d}_original.cif"))

        # Encode the template
        z_mu, z_logvar = model.encode(template)

        # Save reconstruction (encode then decode with mean latent)
        recon = model.decode(z_mu, template, sample=False)
        recon_cpu = recon.numpy()
        recon_cpu.write(str(output_dir / f"epoch{epoch:04d}_reconstruction.cif"))

        # Generate perturbed samples
        for i in range(n_perturbations):
            # Add noise to latent vector
            noise = torch.randn_like(z_mu) * perturbation_scale
            z_perturbed = z_mu + noise

            # Decode perturbed latent
            perturbed = model.decode(z_perturbed, template, sample=False)
            perturbed_cpu = perturbed.numpy()
            perturbed_cpu.write(str(output_dir / f"epoch{epoch:04d}_perturb{i+1}.cif"))

    logger.info(f"Saved {n_perturbations + 2} samples to {output_dir}")


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
        set_seed(config.training.seed)
        logger.info(f"Set random seed to {config.training.seed}")

    # Setup device
    device = get_device(config.training.device)
    logger.info(f"Using device: {device}")

    # Create dataset and dataloader
    logger.info(f"Loading data from {config.data.data_dir}")
    dataset = create_dataset(config.data)
    logger.info(f"Dataset size: {len(dataset)} structures")

    if len(dataset) == 0:
        logger.error("No structures found in dataset!")
        sys.exit(1)

    dataloader = create_dataloader(dataset, config.training)

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
        ckpt = load_checkpoint(Path(args.resume), model, optimizer)
        start_epoch = ckpt["epoch"] + 1

    # Setup output directories
    checkpoint_dir = Path(config.output.checkpoint_dir)
    sample_dir = Path(config.output.sample_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Create VAE loss function
    loss_fn = create_vae_loss_fn(device)

    # Create beta scheduler for KL annealing
    warmup_epochs = config.model.beta_warmup_epochs
    if warmup_epochs is None:
        warmup_epochs = config.training.epochs // 2  # Default: half of training
    beta_scheduler = BetaScheduler(
        schedule=config.model.beta_schedule,
        target_beta=config.model.beta,
        warmup_epochs=warmup_epochs,
        total_epochs=config.training.epochs,
        n_cycles=config.model.beta_cycles,
    )
    logger.info(f"Beta schedule: {beta_scheduler}")

    # Training loop
    logger.info("Starting training...")
    best_loss = float("inf")

    for epoch in range(start_epoch, config.training.epochs):
        # Update beta for this epoch (KL annealing)
        current_beta = beta_scheduler.get_beta(epoch)
        model.beta = current_beta

        # Train epoch using shared training utility
        metrics = train_epoch(
            model=model,
            dataloader=dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            grad_clip=config.training.grad_clip,
        )

        # Log metrics
        logger.info(
            f"Epoch {epoch+1}/{config.training.epochs} | "
            f"Loss: {metrics.get('loss', float('nan')):.4f} | "
            f"Recon: {metrics.get('recon_loss', float('nan')):.4f} | "
            f"KL: {metrics.get('kl_loss', float('nan')):.4f} | "
            f"Beta: {current_beta:.4f} | "
            f"Samples: {int(metrics.get('n_samples', 0))} | "
            f"Skipped: {int(metrics.get('n_skipped', 0))}"
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
                checkpoint_dir / f"checkpoint_epoch{epoch+1:04d}.pt",
                model, optimizer,
                epoch=epoch + 1,
                metrics=metrics,
                config=config,
            )

        # Save best model
        if metrics.get("loss", float("inf")) < best_loss:
            best_loss = metrics["loss"]
            save_checkpoint(
                checkpoint_dir / "checkpoint_best.pt",
                model, optimizer,
                epoch=epoch + 1,
                metrics=metrics,
                config=config,
            )

    # Save final checkpoint
    save_checkpoint(
        checkpoint_dir / "checkpoint_final.pt",
        model, optimizer,
        epoch=config.training.epochs,
        metrics=metrics,
        config=config,
    )

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
