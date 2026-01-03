"""Compare residue model architectures using the unified training API.

Trains Flow, VAE, and Consolidated VAE models via ciffy.nn.residue.train(),
evaluates reconstruction quality, samples 20-mer chains, and generates comparison plots.
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

import hmbp


def evaluate_reconstruction(models: dict, cif_paths: list, residues: str = "ACGU"):
    """Evaluate reconstruction quality for trained models.

    Args:
        models: Dict mapping model name to PolymerModel.
        cif_paths: CIF files to evaluate on.
        residues: Residue types to evaluate.

    Returns:
        Dict of {model_name: {coord_rmsd: float, transform_mse: float}}
    """
    from ciffy.nn.flow.residue.data import extract_residues_with_links
    from ciffy.operations.metrics import rmsd as kabsch_rmsd
    from ciffy.biochemistry import Residue

    results = {name: {"coord_rmsd": [], "transform_mse": []} for name in models}

    for res_char in residues:
        res = getattr(Residue, res_char.upper())

        # Extract test data
        coords, transforms, atoms = extract_residues_with_links(
            cif_paths[:50],  # Use subset for eval
            res,
            min_coverage=0.9,
            verbose=False,
        )

        if len(coords) == 0:
            continue

        n_atoms = len(atoms)
        n_coord_dims = n_atoms * 3
        coords_flat = coords.reshape(len(coords), -1)
        data = np.concatenate([coords_flat, transforms], axis=1)
        data = torch.tensor(data[:200], dtype=torch.float32)  # Limit eval samples
        coords_t = data[:, :n_coord_dims].reshape(-1, n_atoms, 3)
        transforms_t = data[:, n_coord_dims:]

        for name, polymer_model in models.items():
            # PolymerModel stores models by residue value (int) or string key
            model = None
            for key, m in polymer_model.residue_models.items():
                if hasattr(m, 'residue') and m.residue == res:
                    model = m
                    break

            if model is None:
                continue
            model.eval()

            with torch.no_grad():
                # All models implement encode/decode via ResidueGenerativeCore
                if hasattr(model, 'flow'):
                    # Flow model: uses PCAFlow with flat data
                    z = model.flow.encode(data)
                    recon_flat = model.flow.decode(z)
                    recon_coords = recon_flat[:, :n_coord_dims].reshape(-1, n_atoms, 3)
                    recon_transforms = recon_flat[:, n_coord_dims:]
                elif hasattr(model, '_model'):
                    # ConsolidatedResidueView: uses separate coords/transforms
                    z = model.encode(coords_t)
                    recon_coords, recon_transforms = model.decode(z.unsqueeze(0) if z.dim() == 1 else z)
                else:
                    # ResidueVAE: uses flat data format
                    recon_flat, _, _ = model(data)
                    recon_coords = recon_flat[:, :n_coord_dims].reshape(-1, n_atoms, 3)
                    recon_transforms = recon_flat[:, n_coord_dims:]

                coord_rmsd = kabsch_rmsd(recon_coords, coords_t).mean().item()
                transform_mse = F.mse_loss(recon_transforms, transforms_t).item()

                results[name]["coord_rmsd"].append(coord_rmsd)
                results[name]["transform_mse"].append(transform_mse)

    # Average across residue types
    for name in results:
        if results[name]["coord_rmsd"]:
            results[name]["coord_rmsd"] = np.mean(results[name]["coord_rmsd"])
            results[name]["transform_mse"] = np.mean(results[name]["transform_mse"])
        else:
            results[name]["coord_rmsd"] = float('nan')
            results[name]["transform_mse"] = float('nan')

    return results


def generate_plots(eval_results, output_path):
    """Generate comparison plots."""
    figures_path = output_path / "figures"
    figures_path.mkdir(exist_ok=True)

    model_names = list(eval_results.keys())

    # Reconstruction RMSD comparison
    rmsd_values = [eval_results[name]["coord_rmsd"] for name in model_names]
    hmbp.quick_bar(
        rmsd_values,
        model_names,
        title="Reconstruction Quality",
        ylabel="Coordinate RMSD (A)",
        path=str(figures_path / "reconstruction_rmsd.png"),
    )

    # Transform MSE comparison
    transform_values = [eval_results[name]["transform_mse"] for name in model_names]
    hmbp.quick_bar(
        transform_values,
        model_names,
        title="Transform Prediction",
        ylabel="Transform MSE",
        path=str(figures_path / "transform_mse.png"),
    )

    print(f"\n  Plots saved to {figures_path}/")


def main():
    import argparse
    from ciffy.nn import residue

    parser = argparse.ArgumentParser(description="Compare residue model architectures")
    parser.add_argument("--data-dir", default="/Users/hmblair/academic/data/structures/rna")
    parser.add_argument("--max-files", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--chain-length", type=int, default=20)
    parser.add_argument("--output-dir", default="outputs/model_comparison")
    parser.add_argument("--accelerator", default="cpu", help="cpu, gpu, or mps")
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Resolve CIF paths
    cif_paths = sorted(Path(args.data_dir).glob("*.cif"))[:args.max_files]

    print("=" * 70)
    print("Residue Model Comparison (Unified API)")
    print("=" * 70)
    print(f"\n  Data: {len(cif_paths)} CIF files")
    print(f"  Models: {list(residue.available_models().keys())}")

    models = {}

    # Train Flow model
    print("\n[1/4] Training Flow model...")
    flow_model = residue.train(
        cif_paths=cif_paths,
        residues="ACGU",
        model_type="flow",
        n_epochs=args.epochs,
        accelerator=args.accelerator,
        output_dir=output_path / "flow",
        verbose=True,
    )
    models["Flow"] = flow_model

    # Train VAE model
    print("\n[2/4] Training VAE model...")
    vae_model = residue.train(
        cif_paths=cif_paths,
        residues="ACGU",
        model_type="vae",
        n_epochs=args.epochs,
        accelerator=args.accelerator,
        output_dir=output_path / "vae",
        verbose=True,
    )
    models["VAE"] = vae_model

    # Train Consolidated VAE model
    print("\n[3/4] Training Consolidated VAE model...")
    consolidated_model = residue.train(
        cif_paths=cif_paths,
        residues="ACGU",
        model_type="consolidated",
        n_epochs=args.epochs,
        accelerator=args.accelerator,
        output_dir=output_path / "consolidated",
        verbose=True,
    )
    models["Consolidated"] = consolidated_model

    # Evaluate reconstruction
    print("\n[4/4] Evaluating reconstruction quality...")
    eval_results = evaluate_reconstruction(models, cif_paths)

    print(f"\n  {'Model':<15} {'Coord RMSD':>12} {'Transform MSE':>14}")
    print("  " + "-" * 43)
    for name, metrics in eval_results.items():
        print(f"  {name:<15} {metrics['coord_rmsd']:>12.4f} {metrics['transform_mse']:>14.6f}")

    # Sample chains
    print(f"\n  Sampling {args.chain_length}-mer chains...")
    np.random.seed(42)
    sequence = "".join(np.random.choice(list("acgu"), args.chain_length))
    print(f"  Sequence: {sequence}")

    chains_path = output_path / "chains"
    chains_path.mkdir(exist_ok=True)

    for name, model in models.items():
        polymer = model.sample_from_sequence(sequence)
        output_file = chains_path / f"{name}_chain.cif"
        polymer.write(str(output_file))
        print(f"    Saved: {output_file.name}")

    # Generate plots
    print("\n  Generating plots...")
    generate_plots(eval_results, output_path)

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"\n  Output directory: {output_path}/")
    print(f"  Chains saved to: {chains_path}/")

    best = min(eval_results, key=lambda x: eval_results[x]["coord_rmsd"])
    print(f"\n  Best reconstruction (lowest RMSD): {best} ({eval_results[best]['coord_rmsd']:.4f} A)")

    # Show how to load saved models
    print("\n  To load trained models:")
    print(f"    flow_model = residue.load('{output_path}/flow')")
    print(f"    vae_model = residue.load('{output_path}/vae')")
    print(f"    consolidated_model = residue.load('{output_path}/consolidated')")

    print("\n" + "=" * 70)
    print("Done!")


if __name__ == "__main__":
    main()
