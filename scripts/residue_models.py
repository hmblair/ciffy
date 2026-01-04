#!/usr/bin/env python
"""
Unified script for training and sampling residue-level generative models.

Supports two model types via ciffy.nn.residue:
- flow: PCA + normalizing flow (exact density, fast)
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

    model_types = args.model_type.split(",") if args.model_type != "all" else ["flow", "consolidated"]
    models = {}

    for model_type in model_types:
        print(f"\n{'=' * 60}")
        print(f"Training {model_type.upper()} model...")
        print("=" * 60)

        output_dir = Path(args.output_dir) / model_type

        # Only pass transform_scale for flow models
        extra_kwargs = {}
        if model_type == "flow":
            extra_kwargs["transform_scale"] = args.transform_scale

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
            **extra_kwargs,
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
                polymer = model.sample_from_sequence(sequence, id=model_type)
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

    model_path = Path(args.model_dir)
    model = residue.load(model_path)
    model_type = model_path.name  # e.g., "flow" or "consolidated"
    print(f"  Loaded model from {args.model_dir}")

    np.random.seed(args.seed)
    sequence = args.sequence or "".join(np.random.choice(list("acgu"), args.chain_length))
    print(f"  Sequence: {sequence}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(args.n_samples):
        polymer = model.sample_from_sequence(sequence, id=model_type)
        output_file = output_dir / f"chain_{i}.cif"
        polymer.write(str(output_file))
        print(f"  Saved: {output_file.name} ({polymer.size()} atoms, id={model_type})")

    # Verify geometry
    if args.verify:
        print("\nVerifying chain geometry...")
        import torch
        from ciffy.biochemistry import Sugar, PhosphateGroup

        polymer = model.sample_from_sequence(sequence, id=model_type)
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


def analyze(args):
    """Analyze trained models with visualizations."""
    import os
    os.environ["CIFFY_LOG_LEVEL"] = "WARNING"

    import torch
    import hmbp
    from scipy import stats
    from ciffy.nn import residue
    from ciffy.nn.flow.residue.data import extract_residues_with_links
    from ciffy.biochemistry import Residue, Sugar, PhosphateGroup
    from ciffy.operations.metrics import rmsd

    print("=" * 60)
    print("Model Analysis")
    print("=" * 60)

    # Setup output directory
    figures_dir = Path(args.output_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    cif_paths = sorted(Path(args.data_dir).glob("*.cif"))[:args.max_files]
    print(f"  Data: {len(cif_paths)} CIF files")

    # Find model directories - detect which have loadable models
    model_base = Path(args.model_dir)
    model_dirs = []
    model_names = []

    def is_loadable_model(path: Path) -> bool:
        """Check if a directory contains a loadable PolymerModel."""
        if not (path / "config.json").exists():
            return False
        # Check if it has per-residue dirs (flow/vae) or consolidated subdir
        has_residue_dirs = any((path / r).is_dir() and (path / r / "config.json").exists()
                               for r in ["A", "C", "G", "U"])
        has_consolidated = (path / "consolidated" / "config.json").exists()
        return has_residue_dirs or has_consolidated

    if is_loadable_model(model_base):
        model_dirs = [model_base]
        model_names = [model_base.name]
    else:
        for d in model_base.iterdir():
            if d.is_dir() and is_loadable_model(d):
                model_dirs.append(d)
                model_names.append(d.name)

    if not model_dirs:
        print(f"  No loadable models found in {model_base}")
        return

    print(f"  Models: {', '.join(model_names)}")

    # ==========================================================================
    # 1. Reconstruction Error Analysis
    # ==========================================================================
    print("\n[1/4] Computing reconstruction errors...")

    all_rmsds = {name: [] for name in model_names}
    residue_rmsds = {name: {r: [] for r in args.residues} for name in model_names}

    for model_dir in model_dirs:
        model_name = model_dir.name
        model = residue.load(str(model_dir))

        for res_char in args.residues:
            res = getattr(Residue, res_char.upper())
            coords, transforms, atoms = extract_residues_with_links(
                cif_paths, res, min_coverage=0.9, verbose=False
            )

            if len(coords) == 0:
                continue

            # Get residue model
            res_model = None
            for key, m in model.residue_models.items():
                if hasattr(m, 'residue') and m.residue == res:
                    res_model = m
                    break

            if res_model is None:
                continue

            res_model.eval()
            coords_t = torch.tensor(coords[:500], dtype=torch.float32)

            with torch.no_grad():
                z = res_model.encode(coords_t)
                recon_coords, _ = res_model.decode(z)

            rmsds = rmsd(recon_coords, coords_t).numpy()
            all_rmsds[model_name].extend(rmsds.tolist())
            residue_rmsds[model_name][res_char].extend(rmsds.tolist())

    # Plot reconstruction RMSD histogram overlay
    rmsd_data = [np.array(all_rmsds[name]) for name in model_names]
    hmbp.quick_histogram_overlay(
        rmsd_data,
        labels=model_names,
        title="Reconstruction RMSD by Model",
        xlabel="RMSD (Å)",
        path=str(figures_dir / "reconstruction_rmsd.png"),
    )
    print(f"    Saved: reconstruction_rmsd.png")

    # Plot per-residue RMSD comparison
    for res_char in args.residues:
        res_data = [np.array(residue_rmsds[name][res_char]) for name in model_names
                    if len(residue_rmsds[name][res_char]) > 0]
        res_labels = [name for name in model_names
                      if len(residue_rmsds[name][res_char]) > 0]
        if res_data:
            hmbp.quick_histogram_overlay(
                res_data,
                labels=res_labels,
                title=f"Reconstruction RMSD - Residue {res_char}",
                xlabel="RMSD (Å)",
                path=str(figures_dir / f"reconstruction_rmsd_{res_char}.png"),
            )
            print(f"    Saved: reconstruction_rmsd_{res_char}.png")

    # ==========================================================================
    # 2. Latent Space Gaussianity
    # ==========================================================================
    print("\n[2/4] Analyzing latent space gaussianity...")

    latent_stats = {name: {} for name in model_names}

    for model_dir in model_dirs:
        model_name = model_dir.name
        model = residue.load(str(model_dir))

        all_latents = []
        for res_char in args.residues:
            res = getattr(Residue, res_char.upper())
            coords, transforms, atoms = extract_residues_with_links(
                cif_paths, res, min_coverage=0.9, verbose=False
            )

            if len(coords) == 0:
                continue

            res_model = None
            for key, m in model.residue_models.items():
                if hasattr(m, 'residue') and m.residue == res:
                    res_model = m
                    break

            if res_model is None:
                continue

            res_model.eval()
            coords_t = torch.tensor(coords[:500], dtype=torch.float32)

            with torch.no_grad():
                z = res_model.encode(coords_t)

            all_latents.append(z.numpy())

        if all_latents:
            latents = np.concatenate(all_latents, axis=0)

            # Compute per-dimension statistics
            means = latents.mean(axis=0)
            stds = latents.std(axis=0)

            # Shapiro-Wilk test for normality (on subset for speed)
            n_test = min(500, len(latents))
            p_values = []
            for dim in range(latents.shape[1]):
                _, p = stats.shapiro(latents[:n_test, dim])
                p_values.append(p)

            latent_stats[model_name] = {
                'means': means,
                'stds': stds,
                'p_values': np.array(p_values),
                'latents': latents,
            }

    # Plot latent dimension statistics
    for model_name, lstats in latent_stats.items():
        if not lstats:
            continue

        latents = lstats['latents']
        n_dims = latents.shape[1]

        # Histogram of first few latent dimensions
        latent_dims = [latents[:, i] for i in range(min(4, n_dims))]
        hmbp.quick_histogram_overlay(
            latent_dims,
            labels=[f"z{i}" for i in range(len(latent_dims))],
            title=f"Latent Distributions - {model_name}",
            xlabel="Value",
            path=str(figures_dir / f"latent_hist_{model_name}.png"),
        )
        print(f"    Saved: latent_hist_{model_name}.png")

        # Q-Q plot data - compare to standard normal
        for dim in range(min(2, n_dims)):
            sorted_data = np.sort(latents[:, dim])
            n = len(sorted_data)
            theoretical = stats.norm.ppf(np.linspace(0.01, 0.99, n))

            hmbp.quick_scatter(
                theoretical, sorted_data[:len(theoretical)],
                title=f"Q-Q Plot z{dim} - {model_name}",
                xlabel="Theoretical Quantiles",
                ylabel="Sample Quantiles",
                path=str(figures_dir / f"qq_z{dim}_{model_name}.png"),
            )
            print(f"    Saved: qq_z{dim}_{model_name}.png")

    # ==========================================================================
    # 3. Bond Length Distributions
    # ==========================================================================
    print("\n[3/4] Analyzing bond length distributions...")

    # Expected bond lengths (from crystallography)
    EXPECTED_BONDS = {
        "O3'-P": 1.607,
        "P-O5'": 1.593,
        "C1'-N": 1.47,  # Glycosidic bond
    }

    bond_lengths = {name: {"O3'-P": [], "P-O5'": []} for name in model_names}

    for model_dir in model_dirs:
        model_name = model_dir.name
        model = residue.load(str(model_dir))

        # Sample chains
        np.random.seed(42)
        for _ in range(args.n_chains):
            sequence = "".join(np.random.choice(list("acgu"), 20))
            polymer = model.sample_from_sequence(sequence)

            coords = torch.tensor(polymer.coordinates, dtype=torch.float32)
            atoms = torch.tensor(polymer.atoms, dtype=torch.long)

            # O3'-P bonds (phosphodiester)
            o3p_vals = torch.tensor(Sugar.O3p.index())
            p_vals = torch.tensor(PhosphateGroup.P.index())

            o3p_mask = torch.isin(atoms, o3p_vals)
            p_mask = torch.isin(atoms, p_vals)

            o3p_coords = coords[o3p_mask]
            p_coords = coords[p_mask]

            if len(o3p_coords) > 1 and len(p_coords) > 1:
                # O3' of residue i bonds to P of residue i+1
                dists = torch.norm(o3p_coords[:-1] - p_coords[1:], dim=1)
                bond_lengths[model_name]["O3'-P"].extend(dists.tolist())

            # P-O5' bonds
            o5p_vals = torch.tensor(Sugar.O5p.index())
            o5p_mask = torch.isin(atoms, o5p_vals)
            o5p_coords = coords[o5p_mask]

            if len(p_coords) > 0 and len(o5p_coords) > 0:
                # P and O5' within same residue
                min_len = min(len(p_coords), len(o5p_coords))
                dists = torch.norm(p_coords[:min_len] - o5p_coords[:min_len], dim=1)
                bond_lengths[model_name]["P-O5'"].extend(dists.tolist())

    # Plot bond length distributions
    for bond_name, expected in EXPECTED_BONDS.items():
        if bond_name not in list(bond_lengths.values())[0]:
            continue

        bond_data = [np.array(bond_lengths[name][bond_name]) for name in model_names
                     if len(bond_lengths[name][bond_name]) > 0]
        bond_labels = [name for name in model_names
                       if len(bond_lengths[name][bond_name]) > 0]

        if bond_data:
            safe_name = bond_name.replace("'", "p").replace("-", "_")
            hmbp.quick_histogram_overlay(
                bond_data,
                labels=bond_labels,
                title=f"{bond_name} Bond Lengths (expected: {expected:.3f} Å)",
                xlabel="Distance (Å)",
                path=str(figures_dir / f"bond_{safe_name}.png"),
            )
            print(f"    Saved: bond_{safe_name}.png")

    # ==========================================================================
    # 4. Summary Statistics
    # ==========================================================================
    print("\n[4/4] Computing summary statistics...")

    print(f"\n{'Model':<15} {'RMSD (Å)':<12} {'Latent μ':<12} {'Latent σ':<12} {'Gauss p':<10}")
    print("-" * 61)

    for model_name in model_names:
        rmsd_mean = np.mean(all_rmsds[model_name]) if all_rmsds[model_name] else float('nan')

        if model_name in latent_stats and latent_stats[model_name]:
            lat_mean = np.mean(np.abs(latent_stats[model_name]['means']))
            lat_std = np.mean(latent_stats[model_name]['stds'])
            gauss_p = np.mean(latent_stats[model_name]['p_values'])
        else:
            lat_mean = lat_std = gauss_p = float('nan')

        print(f"{model_name:<15} {rmsd_mean:<12.4f} {lat_mean:<12.4f} {lat_std:<12.4f} {gauss_p:<10.4f}")

    # Bond length summary
    print(f"\n{'Model':<15} {'O3p-P mean':<12} {'O3p-P std':<12}")
    print("-" * 39)
    for model_name in model_names:
        if bond_lengths[model_name]["O3'-P"]:
            bl = np.array(bond_lengths[model_name]["O3'-P"])
            print(f"{model_name:<15} {bl.mean():<12.3f} {bl.std():<12.3f}")

    expected_o3p_p = EXPECTED_BONDS["O3'-P"]
    print(f"\nExpected O3'-P: {expected_o3p_p:.3f} Å")

    print("\n" + "=" * 60)
    print(f"Figures saved to: {figures_dir}/")
    print("=" * 60)


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
    train_parser.add_argument("--model-type", default="all", help="flow, consolidated, or all")
    train_parser.add_argument("--residues", default="ACGU")
    train_parser.add_argument("--epochs", type=int, default=100)
    train_parser.add_argument("--latent-dim", type=int, default=12)
    train_parser.add_argument("--batch-size", type=int, default=256)
    train_parser.add_argument("--transform-scale", type=float, default=1.0, help="Scale transforms for flow models (4.0 for better helix geometry)")
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

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze models with visualizations")
    analyze_parser.add_argument("--model-dir", default="outputs/models", help="Path to model(s)")
    analyze_parser.add_argument("--data-dir", default="/Users/hmblair/academic/data/structures/rna")
    analyze_parser.add_argument("--max-files", type=int, default=100)
    analyze_parser.add_argument("--residues", default="ACGU")
    analyze_parser.add_argument("--n-chains", type=int, default=50, help="Chains to sample for bond analysis")
    analyze_parser.add_argument("--output-dir", default="outputs/figures", help="Where to save figures")
    analyze_parser.set_defaults(func=analyze)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
