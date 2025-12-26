#!/usr/bin/env python3
"""
Compare flow model performance across different PCA dimensions.

This script trains flow models with different latent dimensions and compares
their reconstruction quality and variance explained.

Example:
    python scripts/compare_pca_dimensions.py --data_dir data/rna_training/ --output_dir experiments/pca_sweep

For quick testing:
    python scripts/compare_pca_dimensions.py --data_dir tests/data/ --dims 4,8,12 --epochs 50
"""

import argparse
from pathlib import Path
from glob import glob

from ciffy.biochemistry import Residue
from ciffy.nn.runners import (
    run_flow_experiments,
    format_flow_results_table,
    create_latent_dim_sweep,
    FlowExperimentConfig,
)


# Standard RNA residue types
RNA_RESIDUES = [Residue.A, Residue.C, Residue.G, Residue.U]


def main():
    parser = argparse.ArgumentParser(
        description="Compare flow model performance across PCA dimensions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing CIF files for training",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save trained models (optional)",
    )
    parser.add_argument(
        "--max_structures",
        type=int,
        default=None,
        help="Maximum number of CIF files to use for training (default: all)",
    )
    parser.add_argument(
        "--dims",
        type=str,
        default="4,8,12,16,20",
        help="Comma-separated latent dimensions to test (default: 4,8,12,16,20)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Number of training epochs (default: 200)",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=8,
        help="Number of flow layers (default: 8)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to train on (default: cpu)",
    )
    parser.add_argument(
        "--residues",
        type=str,
        default=None,
        help="Comma-separated residue names to train (default: A,C,G,U)",
    )
    parser.add_argument(
        "--show_residues",
        action="store_true",
        help="Show per-residue breakdown in results",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        default=True,
        help="Run experiments in parallel (default: True)",
    )
    parser.add_argument(
        "--no_parallel",
        action="store_true",
        help="Run experiments sequentially",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=None,
        help="Maximum parallel workers (default: auto based on CPU count)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-experiment output",
    )
    parser.add_argument(
        "--train_split",
        type=float,
        default=0.8,
        help="Fraction of structures for training (default: 0.8)",
    )
    parser.add_argument(
        "--test_split",
        type=float,
        default=0.2,
        help="Fraction of structures for testing (default: 0.2)",
    )
    parser.add_argument(
        "--split_seed",
        type=int,
        default=42,
        help="Random seed for train/test split (default: 42)",
    )

    args = parser.parse_args()

    # Parse dimensions
    latent_dims = [int(d.strip()) for d in args.dims.split(",")]

    # Parse residues
    if args.residues:
        residue_names = [r.strip().upper() for r in args.residues.split(",")]
        residues = []
        for name in residue_names:
            # Handle both short (A) and long (ADE) names
            if hasattr(Residue, name):
                residues.append(getattr(Residue, name))
            else:
                # Try mapping long names to short
                name_map = {"ADE": "A", "CYT": "C", "GUA": "G", "URA": "U"}
                short_name = name_map.get(name, name)
                if hasattr(Residue, short_name):
                    residues.append(getattr(Residue, short_name))
                else:
                    print(f"Warning: Unknown residue '{name}', skipping")
    else:
        residues = RNA_RESIDUES

    # Find CIF files
    data_dir = Path(args.data_dir)
    cif_patterns = [str(data_dir / "*.cif"), str(data_dir / "**/*.cif")]
    cif_files = []
    for pattern in cif_patterns:
        cif_files.extend(glob(pattern, recursive=True))
    cif_files = list(set(cif_files))  # Remove duplicates

    if not cif_files:
        print(f"Error: No CIF files found in {data_dir}")
        return 1

    # Limit number of structures if requested
    total_found = len(cif_files)
    if args.max_structures is not None and total_found > args.max_structures:
        cif_files = cif_files[:args.max_structures]
        print(f"Using {len(cif_files)} of {total_found} CIF files")
    else:
        print(f"Found {total_found} CIF files")
    print(f"Training on residues: {[r.name for r in residues]}")
    print(f"Testing dimensions: {latent_dims}")
    print(f"Train/test split: {args.train_split:.0%}/{args.test_split:.0%} (seed={args.split_seed})")
    print()

    # Create experiment configurations
    configs = create_latent_dim_sweep(
        base_name="pca",
        latent_dims=latent_dims,
        n_layers=args.layers,
        n_epochs=args.epochs,
        train_split=args.train_split,
        test_split=args.test_split,
        split_seed=args.split_seed,
    )

    # Run experiments
    parallel = args.parallel and not args.no_parallel
    results = run_flow_experiments(
        configs,
        cif_paths=cif_files,
        residues=residues,
        device=args.device,
        output_dir=args.output_dir,
        parallel=parallel,
        max_workers=args.max_workers,
        verbose=not args.quiet,
    )

    # Print final comparison
    print("\n" + "=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)
    print(format_flow_results_table(results, show_residues=args.show_residues))

    # Find best configuration
    successful = [r for r in results if r.status == "success"]
    if successful:
        # Best by test RMSD (held-out data)
        best_rmsd = min(successful, key=lambda r: r.mean_test_rmsd)
        # Best by variance explained
        best_var = max(successful, key=lambda r: r.mean_var_explained)

        print("\n" + "-" * 40)
        print("RECOMMENDATIONS")
        print("-" * 40)
        print(f"Best reconstruction (Test RMSD): {best_rmsd.name}")
        print(f"  Train RMSD: {best_rmsd.mean_train_rmsd:.4f}")
        print(f"  Test RMSD:  {best_rmsd.mean_test_rmsd:.4f}")
        print(f"  Variance: {best_rmsd.mean_var_explained*100:.1f}%")
        print(f"  Params: {best_rmsd.total_params:,}")
        print()
        print(f"Highest variance explained: {best_var.name}")
        print(f"  Train RMSD: {best_var.mean_train_rmsd:.4f}")
        print(f"  Test RMSD:  {best_var.mean_test_rmsd:.4f}")
        print(f"  Variance: {best_var.mean_var_explained*100:.1f}%")

    return 0


if __name__ == "__main__":
    exit(main())
