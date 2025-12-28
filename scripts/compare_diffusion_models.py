#!/usr/bin/env python
"""Compare coordinate diffusion vs latent diffusion for structure prediction.

This script trains both diffusion model types on RNA structures and compares
their performance via RMSD on held-out test data. Outputs predictions as .cif files.

Usage:
    # Quick dry-run on CPU (< 5 minutes)
    python scripts/compare_diffusion_models.py --dry-run

    # Full training (8-12 hours on GPU)
    python scripts/compare_diffusion_models.py --data-dir /path/to/rna

    # Medium training with custom settings
    python scripts/compare_diffusion_models.py --data-dir ./rna --epochs 50

Example Output:
    Model Comparison (RMSD in Angstroms):
    ====================================
    Model                  Mean    Std     Min     Max
    Coordinate Diffusion   3.45    1.23    1.12    8.34
    Latent Diffusion       4.12    1.56    1.45    9.21
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import ciffy
    from ciffy import Polymer, Scale, rmsd
    from ciffy.nn.split import DataSplit, split_by_structure
    from ciffy.nn.diffusion import (
        # Coordinate diffusion
        CoordinateDiffusionConfig,
        CoordinateDiffusionModel,
        CoordinateDiffusionTrainer,
        CoordinateDiffusionTrainingConfig,
        CoordinateDiffusionDataConfig,
        CoordinateDenoiserConfig,
        # Latent diffusion
        LatentDiffusionConfig,
        LatentDiffusionModel,
        LatentDiffusionTrainer,
        LatentDiffusionTrainingConfig,
        LatentDenoiserConfig,
    )
    from ciffy.nn.diffusion.latent_trainer import LatentDiffusionDataConfig
    from ciffy.nn.base_trainer import TrainingConfig, OutputConfig
    from ciffy.nn.flow import PolymerFlowModel
    from ciffy.nn.flow.residue import ResidueFlowTrainer, ResidueFlowTrainingConfig
    from ciffy.biochemistry import Residue
except ImportError as e:
    print(f"Error importing ciffy modules: {e}")
    print("Make sure ciffy is installed: pip install -e .")
    raise


logger = logging.getLogger(__name__)


@dataclass
class ComparisonConfig:
    """Configuration for model comparison experiment."""

    # Data
    data_dir: str = ""
    max_samples: Optional[int] = None  # Limit samples (for testing)
    test_fraction: float = 0.2
    seed: int = 42

    # Model dimensions (small for pipeline verification)
    d_model: int = 128
    num_layers: int = 4
    num_heads: int = 4
    num_timesteps: int = 500

    # Training (reduced for pipeline verification)
    epochs: int = 30
    lr: float = 1e-4
    batch_size: int = 16  # Larger batch with smaller model

    # Flow model settings (for latent diffusion)
    flow_epochs: int = 50  # Epochs to train flow model
    flow_latent_dim: int = 12  # Latent dimensions per residue
    flow_n_layers: int = 4  # Flow layers

    # Evaluation
    num_test_samples: int = 10  # Number of test structures to evaluate
    ddim_steps: int = 50  # Sampling steps

    # Output
    output_dir: str = "./comparison_results"

    # Hardware
    device: str = "auto"
    precision: str = "16-mixed"
    num_devices: int = 1


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare coordinate vs latent diffusion models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="/Users/hmblair/academic/data/structures/rna",
        help="Directory containing CIF files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./comparison_results",
        help="Directory for outputs (checkpoints, predictions)",
    )

    # Dry run mode
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Quick test mode: minimal epochs, small model, CPU only",
    )

    # Training options
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--d-model",
        type=int,
        default=256,
        help="Model hidden dimension",
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
        "--batch-size",
        type=int,
        default=8,
        help="Training batch size",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of training samples (for testing)",
    )

    # Evaluation
    parser.add_argument(
        "--num-test-samples",
        type=int,
        default=10,
        help="Number of test structures for evaluation",
    )
    parser.add_argument(
        "--ddim-steps",
        type=int,
        default=50,
        help="DDIM sampling steps",
    )

    # Other
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device (auto, cuda, cpu, mps)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="16-mixed",
        choices=["32-true", "16-mixed", "bf16-mixed"],
        help="Training precision (16-mixed recommended for speed)",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs for DDP training. NOTE: Current implementation "
             "processes variable-size polymers individually, so multi-GPU "
             "speedup is limited. For faster iteration, use smaller model or fewer epochs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--skip-latent",
        action="store_true",
        help="Skip latent diffusion training (coordinate only)",
    )
    parser.add_argument(
        "--skip-coordinate",
        action="store_true",
        help="Skip coordinate diffusion training (latent only)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce logging output",
    )

    return parser.parse_args()


def get_dry_run_config() -> dict:
    """Get configuration overrides for dry-run mode."""
    return {
        "epochs": 2,
        "d_model": 64,
        "num_layers": 2,
        "num_heads": 4,
        "batch_size": 2,
        "max_samples": 20,
        "num_test_samples": 3,
        "ddim_steps": 5,
        "device": "cpu",
        "precision": "32-true",  # CPU doesn't support mixed precision
        "num_timesteps": 100,
        # Flow model settings
        "flow_epochs": 5,
        "flow_latent_dim": 8,
        "flow_n_layers": 2,
    }


def create_config(args: argparse.Namespace) -> ComparisonConfig:
    """Create comparison config from arguments."""
    config = ComparisonConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        seed=args.seed,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_test_samples=args.num_test_samples,
        ddim_steps=args.ddim_steps,
        device=args.device,
        precision=args.precision,
        num_devices=args.num_gpus,
    )

    # Apply dry-run overrides
    if args.dry_run:
        dry_run = get_dry_run_config()
        config.epochs = dry_run["epochs"]
        config.d_model = dry_run["d_model"]
        config.num_layers = dry_run["num_layers"]
        config.num_heads = dry_run["num_heads"]
        config.batch_size = dry_run["batch_size"]
        config.max_samples = dry_run["max_samples"]
        config.num_test_samples = dry_run["num_test_samples"]
        config.ddim_steps = dry_run["ddim_steps"]
        config.device = dry_run["device"]
        config.precision = dry_run["precision"]
        config.num_timesteps = dry_run["num_timesteps"]
        # Flow model settings
        config.flow_epochs = dry_run["flow_epochs"]
        config.flow_latent_dim = dry_run["flow_latent_dim"]
        config.flow_n_layers = dry_run["flow_n_layers"]

    return config


def load_and_split_data(
    config: ComparisonConfig,
) -> tuple[list[Path], list[Path]]:
    """Load CIF files and split into train/test."""
    data_dir = Path(config.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    cif_files = sorted(data_dir.glob("*.cif"))
    if not cif_files:
        raise ValueError(f"No CIF files found in {data_dir}")

    logger.info(f"Found {len(cif_files)} CIF files")

    # Apply sample limit
    if config.max_samples is not None:
        cif_files = cif_files[: config.max_samples]
        logger.info(f"Limited to {len(cif_files)} samples")

    # Simple random split (for production, use split_by_sequence)
    split = split_by_structure(
        cif_files,
        train=1.0 - config.test_fraction,
        val=0.0,
        test=config.test_fraction,
        seed=config.seed,
    )

    logger.info(f"Split: {len(split.train)} train, {len(split.test)} test")

    return split.train, split.test


def train_flow_model(
    config: ComparisonConfig,
    train_files: list[Path],
    quiet: bool = False,
) -> PolymerFlowModel:
    """Train a PolymerFlowModel for all RNA residue types.

    This trains individual ResidueFlowModels for A, C, G, U and combines
    them into a PolymerFlowModel for use with latent diffusion.
    """
    logger.info("Training PolymerFlowModel for latent diffusion...")

    # Create trainer config
    from ciffy.nn.flow.residue.trainer import (
        ResidueFlowModelConfig,
        ResidueFlowDataConfig,
    )
    from ciffy.nn.base_trainer import TrainingConfig as FlowTrainingConfig

    model_config = ResidueFlowModelConfig(
        latent_dim=config.flow_latent_dim,
        n_layers=config.flow_n_layers,
    )
    data_config = ResidueFlowDataConfig(
        data_dir=str(train_files[0].parent) if train_files else "",
    )
    training_config = FlowTrainingConfig(
        epochs=config.flow_epochs,
        device=config.device,
    )

    flow_config = ResidueFlowTrainingConfig(
        model=model_config,
        data=data_config,
        training=training_config,
    )

    trainer = ResidueFlowTrainer(flow_config, quiet=quiet)

    # Train for each RNA residue type
    residue_models = {}
    rna_residues = [Residue.A, Residue.C, Residue.G, Residue.U]

    for residue in rna_residues:
        try:
            result = trainer.train_single(train_files, residue, verbose=not quiet)
            residue_models[residue.value] = result.model
            logger.info(f"  {residue.name}: train RMSD={result.train_rmsd:.3f}Å")
        except ValueError as e:
            logger.warning(f"  {residue.name}: skipped ({e})")
            continue

    if not residue_models:
        raise ValueError("No residue models trained successfully")

    # Combine into PolymerFlowModel
    flow_model = PolymerFlowModel(residue_models)

    # Save the flow model
    flow_dir = Path(config.output_dir) / "flow_model"
    flow_model.save(flow_dir)
    logger.info(f"Saved flow model to {flow_dir}")

    return flow_model


def train_coordinate_diffusion(
    config: ComparisonConfig,
    train_files: list[Path],
    quiet: bool = False,
) -> tuple[CoordinateDiffusionModel, Path]:
    """Train coordinate diffusion model."""
    logger.info("Training Coordinate Diffusion Model...")

    # Create training config
    denoiser_config = CoordinateDenoiserConfig(
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        num_timesteps=config.num_timesteps,
    )
    model_config = CoordinateDiffusionConfig(
        denoiser=denoiser_config,
        num_timesteps=config.num_timesteps,
    )
    data_config = CoordinateDiffusionDataConfig(
        batch_size=config.batch_size,
        molecule_types=("RNA",),
        min_atoms=10,
        max_atoms=5000,
    )
    training_config = TrainingConfig(
        epochs=config.epochs,
        lr=config.lr,
        device=config.device,
        precision=config.precision,
        num_devices=config.num_devices,
    )
    output_config = OutputConfig(
        checkpoint_dir=str(Path(config.output_dir) / "coord_checkpoints"),
        sample_dir=str(Path(config.output_dir) / "coord_samples"),
        save_every=max(1, config.epochs // 5),
    )

    full_config = CoordinateDiffusionTrainingConfig(
        model=model_config,
        data=data_config,
        training=training_config,
        output=output_config,
        val_every=10,  # Validate every 10 epochs
        val_samples=3,
        val_steps=config.ddim_steps,
    )

    # Create trainer
    trainer = CoordinateDiffusionTrainer(full_config, quiet=quiet)

    # Train
    start_time = time.time()
    result = trainer.train(cif_files=train_files)
    elapsed = time.time() - start_time

    logger.info(f"Coordinate diffusion training complete in {elapsed:.1f}s")
    logger.info(f"Final loss: {result['final_loss']:.6f}")

    return trainer.model, Path(result["checkpoint_path"])


def train_latent_diffusion(
    config: ComparisonConfig,
    train_files: list[Path],
    flow_model: PolymerFlowModel,
    quiet: bool = False,
) -> tuple[LatentDiffusionModel, Path]:
    """Train latent diffusion model using a pre-trained flow model."""
    logger.info("Training Latent Diffusion Model...")

    # Create training config
    denoiser_config = LatentDenoiserConfig(
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        latent_dim=flow_model.latent_dim,  # Match flow model
    )
    model_config = LatentDiffusionConfig(
        denoiser=denoiser_config,
        num_timesteps=config.num_timesteps,
    )
    data_config = LatentDiffusionDataConfig(
        data_dir=str(train_files[0].parent) if train_files else "",
        batch_size=config.batch_size * 2,  # Latent is more memory efficient
        molecule_types=("RNA",),
        min_residues=5,
        max_residues=5000,  # Allow larger structures
    )
    training_config = TrainingConfig(
        epochs=config.epochs,
        lr=config.lr,
        device=config.device,
        precision=config.precision,
        num_devices=config.num_devices,
    )
    output_config = OutputConfig(
        checkpoint_dir=str(Path(config.output_dir) / "latent_checkpoints"),
        sample_dir=str(Path(config.output_dir) / "latent_samples"),
        save_every=max(1, config.epochs // 5),
    )

    full_config = LatentDiffusionTrainingConfig(
        model=model_config,
        data=data_config,
        training=training_config,
        output=output_config,
        val_every=10,  # Validate every 10 epochs
        val_samples=3,
        val_steps=config.ddim_steps,
    )

    # Create model with our trained flow model
    latent_model = LatentDiffusionModel(model_config, flow_model=flow_model)

    # Create trainer with pre-built model
    # Note: trainer handles data loading from data_config.data_dir internally
    trainer = LatentDiffusionTrainer(full_config, model=latent_model, quiet=quiet)

    # Train
    start_time = time.time()
    result = trainer.train()
    elapsed = time.time() - start_time

    logger.info(f"Latent diffusion training complete in {elapsed:.1f}s")
    logger.info(f"Final loss: {result['final_loss']:.6f}")

    return trainer.model, Path(result["checkpoint_path"])


def evaluate_model(
    model,
    test_files: list[Path],
    num_samples: int,
    ddim_steps: int,
    output_dir: Path,
    model_name: str,
) -> dict:
    """Evaluate model on test structures."""
    import torch

    logger.info(f"Evaluating {model_name} on {min(num_samples, len(test_files))} test structures...")

    model.eval()
    results = []
    output_dir.mkdir(parents=True, exist_ok=True)

    from ciffy import Molecule

    for i, cif_path in enumerate(test_files[:num_samples]):
        try:
            # Load structure - filter to RNA only and remove modified residues
            polymer = ciffy.load(str(cif_path)).by_type(Molecule.RNA).poly().canonical()

            # Get original coordinates
            original_coords = polymer.coordinates.copy()

            # Generate sample
            with torch.no_grad():
                samples = model.sample(
                    polymer,
                    n_samples=1,
                    num_steps=ddim_steps,
                )

            if not samples:
                logger.warning(f"No samples generated for {cif_path.name}")
                continue

            generated = samples[0]

            # Intersect to ensure matching atoms before RMSD
            original_matched, generated_matched = ciffy.intersect(polymer, generated)

            # Compute RMSD on matched atoms
            rmsd_val = float(rmsd(generated_matched.coordinates, original_matched.coordinates))
            results.append({
                "file": cif_path.name,
                "rmsd": rmsd_val,
                "n_atoms": original_matched.size(),
            })

            # Save prediction
            pred_path = output_dir / f"{cif_path.stem}_{model_name}_pred.cif"
            generated.write(str(pred_path))

            logger.info(f"  {cif_path.name}: RMSD = {rmsd_val:.2f} A ({original_matched.size()} matched atoms)")

        except Exception as e:
            logger.warning(f"Failed to evaluate {cif_path.name}: {e}")
            continue

    if not results:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}

    rmsds = [r["rmsd"] for r in results]
    return {
        "mean": float(np.mean(rmsds)),
        "std": float(np.std(rmsds)),
        "min": float(np.min(rmsds)),
        "max": float(np.max(rmsds)),
        "n_evaluated": len(results),
        "results": results,
    }


def print_comparison_table(
    coord_metrics: Optional[dict],
    latent_metrics: Optional[dict],
) -> None:
    """Print formatted comparison table."""
    print("\n" + "=" * 60)
    print("Model Comparison (RMSD in Angstroms)")
    print("=" * 60)
    print(f"{'Model':<25} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("-" * 60)

    if coord_metrics:
        print(
            f"{'Coordinate Diffusion':<25} "
            f"{coord_metrics['mean']:>8.2f} "
            f"{coord_metrics['std']:>8.2f} "
            f"{coord_metrics['min']:>8.2f} "
            f"{coord_metrics['max']:>8.2f}"
        )

    if latent_metrics:
        print(
            f"{'Latent Diffusion':<25} "
            f"{latent_metrics['mean']:>8.2f} "
            f"{latent_metrics['std']:>8.2f} "
            f"{latent_metrics['min']:>8.2f} "
            f"{latent_metrics['max']:>8.2f}"
        )

    print("=" * 60)


def main() -> None:
    """Main comparison entry point."""
    args = parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Enable Tensor Core acceleration
    try:
        import torch
        torch.set_float32_matmul_precision('high')
    except Exception:
        pass

    # Create config
    config = create_config(args)

    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Log configuration
    if args.dry_run:
        logger.info("=" * 60)
        logger.info("DRY RUN MODE - Quick test with minimal settings")
        logger.info("=" * 60)
    else:
        logger.info("=" * 60)
        logger.info("Diffusion Model Comparison")
        logger.info("=" * 60)

    logger.info(f"Data directory: {config.data_dir}")
    logger.info(f"Output directory: {config.output_dir}")
    logger.info(f"Model: d_model={config.d_model}, layers={config.num_layers}")
    logger.info(f"Training: {config.epochs} epochs, batch_size={config.batch_size}")
    logger.info(f"Device: {config.device}")
    logger.info("=" * 60)

    # Load and split data
    try:
        train_files, test_files = load_and_split_data(config)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    coord_model = None
    latent_model = None
    coord_metrics = None
    latent_metrics = None

    # Train coordinate diffusion
    if not args.skip_coordinate:
        try:
            coord_model, coord_ckpt = train_coordinate_diffusion(
                config, train_files, quiet=args.quiet
            )
            logger.info(f"Saved checkpoint: {coord_ckpt}")
        except Exception as e:
            logger.error(f"Coordinate diffusion training failed: {e}")
            if args.dry_run:
                raise  # Re-raise in dry-run to catch bugs
    else:
        logger.info("Skipping coordinate diffusion training")

    # Train latent diffusion (requires training flow model first)
    if not args.skip_latent:
        try:
            # First train the flow model for encoding coordinates to latents
            flow_model = train_flow_model(config, train_files, quiet=args.quiet)

            # Then train the latent diffusion model
            latent_model, latent_ckpt = train_latent_diffusion(
                config, train_files, flow_model=flow_model, quiet=args.quiet
            )
            logger.info(f"Saved checkpoint: {latent_ckpt}")
        except ValueError as e:
            # Data compatibility issues (modified nucleotides, etc.)
            if "incompatible" in str(e).lower():
                logger.warning(f"Latent diffusion skipped: {e}")
                logger.warning("The data contains modified nucleotides or non-standard residues.")
                logger.warning("Use --skip-latent or provide pure RNA data with standard A/C/G/U residues.")
            else:
                logger.error(f"Latent diffusion training failed: {e}")
                if args.dry_run:
                    raise
        except Exception as e:
            logger.error(f"Latent diffusion training failed: {e}")
            if args.dry_run:
                raise
    else:
        logger.info("Skipping latent diffusion training")

    # Evaluate models
    if coord_model is not None:
        try:
            coord_metrics = evaluate_model(
                coord_model,
                test_files,
                config.num_test_samples,
                config.ddim_steps,
                output_dir / "predictions_coordinate",
                "coord",
            )
        except Exception as e:
            logger.error(f"Coordinate diffusion evaluation failed: {e}")
            if args.dry_run:
                raise

    if latent_model is not None:
        try:
            latent_metrics = evaluate_model(
                latent_model,
                test_files,
                config.num_test_samples,
                config.ddim_steps,
                output_dir / "predictions_latent",
                "latent",
            )
        except Exception as e:
            logger.error(f"Latent diffusion evaluation failed: {e}")
            if args.dry_run:
                raise

    # Print comparison
    print_comparison_table(coord_metrics, latent_metrics)

    # Save results
    results_path = output_dir / "comparison_results.txt"
    with open(results_path, "w") as f:
        f.write("Diffusion Model Comparison Results\n")
        f.write("=" * 40 + "\n\n")

        if coord_metrics:
            f.write("Coordinate Diffusion:\n")
            f.write(f"  Mean RMSD: {coord_metrics['mean']:.3f} A\n")
            f.write(f"  Std RMSD:  {coord_metrics['std']:.3f} A\n")
            f.write(f"  Samples:   {coord_metrics.get('n_evaluated', 'N/A')}\n\n")

        if latent_metrics:
            f.write("Latent Diffusion:\n")
            f.write(f"  Mean RMSD: {latent_metrics['mean']:.3f} A\n")
            f.write(f"  Std RMSD:  {latent_metrics['std']:.3f} A\n")
            f.write(f"  Samples:   {latent_metrics.get('n_evaluated', 'N/A')}\n")

    logger.info(f"\nResults saved to: {results_path}")
    logger.info(f"Predictions saved to: {output_dir}")

    if args.dry_run:
        logger.info("\nDry run completed successfully!")
        logger.info("To run full training, use: python scripts/compare_diffusion_models.py --data-dir <path>")


if __name__ == "__main__":
    main()
