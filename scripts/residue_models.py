#!/usr/bin/env python
"""
Unified script for training and sampling residue-level generative models.

Supports three model types via ciffy.nn.residue:
- flow: PCA + normalizing flow (exact density, fast)
- vae: MLP encoder/decoder VAE
- consolidated: Shared encoder VAE (4x more training data)

Examples:
    # Train all models
    python scripts/residue_models.py train --data-dir /path/to/cifs

    # Train specific model
    python scripts/residue_models.py train --model-type flow

    # Sample chains from trained model
    python scripts/residue_models.py sample --model-dir outputs/models/flow

    # Train and sample in one go
    python scripts/residue_models.py train --sample
"""

import argparse
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")


def train(args):
    """Train residue models."""
    import os
    os.environ["CIFFY_LOG_LEVEL"] = "WARNING"

    from ciffy.nn import residue

    cif_paths = sorted(Path(args.data_dir).glob("*.cif"))[:args.max_files]
    if not cif_paths:
        print(f"No CIF files found in {args.data_dir}")
        return

    print("=" * 60)
    print("Residue Model Training")
    print("=" * 60)
    print(f"  Data: {len(cif_paths)} CIF files")
    print(f"  Residues: {args.residues}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Accelerator: {args.accelerator}")
    print()

    model_types = args.model_type.split(",") if args.model_type != "all" else ["flow", "vae", "consolidated"]
    models = {}

    for model_type in model_types:
        print(f"\n{'=' * 60}")
        print(f"Training {model_type.upper()} model...")
        print("=" * 60)

        output_dir = Path(args.output_dir) / model_type

        model = residue.train(
            cif_paths=cif_paths,
            residues=args.residues,
            model_type=model_type,
            n_epochs=args.epochs,
            latent_dim=args.latent_dim,
            batch_size=args.batch_size,
            accelerator=args.accelerator,
            output_dir=output_dir,
            verbose=True,
        )
        models[model_type] = model

        print(f"\nSaved to {output_dir}/")

    # Optionally sample chains
    if args.sample:
        print("\n" + "=" * 60)
        print("Sampling chains...")
        print("=" * 60)

        np.random.seed(args.seed)
        sequence = args.sequence or "".join(np.random.choice(list("acgu"), args.chain_length))
        print(f"  Sequence: {sequence}")

        chains_dir = Path(args.output_dir) / "chains"
        chains_dir.mkdir(parents=True, exist_ok=True)

        for model_type, model in models.items():
            for i in range(args.n_samples):
                polymer = model.sample_from_sequence(sequence)
                output_file = chains_dir / f"{model_type}_{i}.cif"
                polymer.write(str(output_file))
                print(f"  Saved: {output_file.name} ({polymer.size()} atoms)")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


def sample(args):
    """Sample chains from a trained model."""
    from ciffy.nn import residue

    print("=" * 60)
    print("Chain Sampling")
    print("=" * 60)

    model = residue.load(args.model_dir)
    print(f"  Loaded model from {args.model_dir}")

    np.random.seed(args.seed)
    sequence = args.sequence or "".join(np.random.choice(list("acgu"), args.chain_length))
    print(f"  Sequence: {sequence}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(args.n_samples):
        polymer = model.sample_from_sequence(sequence)
        output_file = output_dir / f"chain_{i}.cif"
        polymer.write(str(output_file))
        print(f"  Saved: {output_file.name} ({polymer.size()} atoms)")

    # Verify geometry
    if args.verify:
        print("\nVerifying chain geometry...")
        import torch
        from ciffy.biochemistry import Sugar, PhosphateGroup

        polymer = model.sample_from_sequence(sequence)
        coords = torch.tensor(polymer.coordinates, dtype=torch.float32)
        atoms = torch.tensor(polymer.atoms, dtype=torch.long)

        o3p_vals = Sugar.O3p.index()
        p_vals = PhosphateGroup.P.index()

        o3p_mask = torch.isin(atoms, torch.tensor(o3p_vals))
        p_mask = torch.isin(atoms, torch.tensor(p_vals))

        o3p_coords = coords[o3p_mask]
        p_coords = coords[p_mask]

        if len(o3p_coords) > 1 and len(p_coords) > 1:
            link_distances = torch.norm(o3p_coords[:-1] - p_coords[1:], dim=1)
            mean_dist = link_distances.mean().item()
            std_dist = link_distances.std().item()
            print(f"  O3'-P distances: {mean_dist:.3f} +/- {std_dist:.3f} A (expected ~1.60 A)")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


def evaluate(args):
    """Evaluate trained models on reconstruction quality."""
    import torch
    import torch.nn.functional as F
    from ciffy.nn import residue
    from ciffy.nn.flow.residue.data import extract_residues_with_links
    from ciffy.biochemistry import Residue

    print("=" * 60)
    print("Model Evaluation")
    print("=" * 60)

    cif_paths = sorted(Path(args.data_dir).glob("*.cif"))[:args.max_files]

    model_dirs = list(Path(args.model_dir).iterdir()) if Path(args.model_dir).is_dir() else [Path(args.model_dir)]
    model_dirs = [d for d in model_dirs if d.is_dir() and (d / "polymer_model").exists()]

    results = {}

    for model_dir in model_dirs:
        model_name = model_dir.name
        print(f"\nEvaluating {model_name}...")

        model = residue.load(str(model_dir))
        coord_rmsds = []

        for res_char in args.residues:
            res = getattr(Residue, res_char.upper())
            coords, transforms, atoms = extract_residues_with_links(
                cif_paths[:50], res, min_coverage=0.9, verbose=False
            )

            if len(coords) == 0:
                continue

            # Get the residue model
            res_model = None
            for key, m in model.residue_models.items():
                if hasattr(m, 'residue') and m.residue == res:
                    res_model = m
                    break

            if res_model is None:
                continue

            res_model.eval()
            coords_t = torch.tensor(coords[:100], dtype=torch.float32)

            with torch.no_grad():
                z = res_model.encode(coords_t)
                recon_coords, _ = res_model.decode(z)

            from ciffy.operations.metrics import rmsd
            coord_rmsd = rmsd(recon_coords, coords_t).mean().item()
            coord_rmsds.append(coord_rmsd)

        results[model_name] = np.mean(coord_rmsds) if coord_rmsds else float('nan')

    print(f"\n{'Model':<20} {'Coord RMSD (A)':>15}")
    print("-" * 37)
    for name, rmsd_val in sorted(results.items(), key=lambda x: x[1]):
        print(f"{name:<20} {rmsd_val:>15.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Train and sample residue-level generative models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train command
    train_parser = subparsers.add_parser("train", help="Train residue models")
    train_parser.add_argument("--data-dir", default="/Users/hmblair/academic/data/structures/rna")
    train_parser.add_argument("--max-files", type=int, default=500)
    train_parser.add_argument("--model-type", default="all", help="flow, vae, consolidated, or all")
    train_parser.add_argument("--residues", default="ACGU")
    train_parser.add_argument("--epochs", type=int, default=100)
    train_parser.add_argument("--latent-dim", type=int, default=12)
    train_parser.add_argument("--batch-size", type=int, default=256)
    train_parser.add_argument("--accelerator", default="cpu")
    train_parser.add_argument("--output-dir", default="outputs/models")
    train_parser.add_argument("--sample", action="store_true", help="Sample chains after training")
    train_parser.add_argument("--sequence", help="Sequence to sample (random if not specified)")
    train_parser.add_argument("--chain-length", type=int, default=20)
    train_parser.add_argument("--n-samples", type=int, default=3)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.set_defaults(func=train)

    # Sample command
    sample_parser = subparsers.add_parser("sample", help="Sample chains from trained model")
    sample_parser.add_argument("--model-dir", required=True, help="Path to trained model")
    sample_parser.add_argument("--sequence", help="Sequence to sample (random if not specified)")
    sample_parser.add_argument("--chain-length", type=int, default=20)
    sample_parser.add_argument("--n-samples", type=int, default=10)
    sample_parser.add_argument("--output-dir", default="outputs/chains")
    sample_parser.add_argument("--seed", type=int, default=42)
    sample_parser.add_argument("--verify", action="store_true", help="Verify chain geometry")
    sample_parser.set_defaults(func=sample)

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate model reconstruction")
    eval_parser.add_argument("--model-dir", required=True, help="Path to model(s)")
    eval_parser.add_argument("--data-dir", default="/Users/hmblair/academic/data/structures/rna")
    eval_parser.add_argument("--max-files", type=int, default=100)
    eval_parser.add_argument("--residues", default="ACGU")
    eval_parser.set_defaults(func=evaluate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
