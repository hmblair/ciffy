"""
Train a PolymerFlowModel for RNA (A, U, G, C).

Usage:
    python scripts/train_rna_flow.py --data /home/hmblair/data/rna --output /home/hmblair/models/rna_flow
"""

import argparse
from pathlib import Path

import torch

from ciffy.biochemistry import Residue
from ciffy.nn.flow import PolymerFlowModel
from ciffy.nn.flow.residue import ResidueFlowModel
from ciffy.nn.flow.residue.model import ResidueFlowConfig


def train_rna_flow(
    data_dir: str,
    output_dir: str,
    n_epochs: int = 200,
    latent_dim: int = 12,
    n_layers: int = 6,
    use_rotation: bool = True,
    noise_std: float = 0.05,
    device: str = "cuda",
):
    """Train flow models for all RNA bases and combine into PolymerFlowModel."""
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get all CIF files
    cif_files = sorted(data_path.glob("*.cif"))
    print(f"Found {len(cif_files)} CIF files in {data_path}")

    if len(cif_files) == 0:
        raise ValueError(f"No CIF files found in {data_path}")

    # RNA residue types
    rna_residues = [Residue.A, Residue.U, Residue.G, Residue.C]

    # Create config
    config = ResidueFlowConfig(
        latent_dim=latent_dim,
        n_layers=n_layers,
        use_rotation=use_rotation,
        noise_std=noise_std,
    )
    print(f"Config: {config}")

    # Train a model for each residue type
    residue_models = {}

    for residue in rna_residues:
        print(f"\n{'='*60}")
        print(f"Training model for {residue.name}")
        print(f"{'='*60}")

        try:
            model, info = ResidueFlowModel.from_structures(
                cif_files,
                residue,
                config=config,
                n_epochs=n_epochs,
                device=device,
                verbose=True,
            )

            # Save individual model
            model_path = output_path / residue.name
            model.save(model_path)
            print(f"Saved {residue.name} model to {model_path}")

            # Store for PolymerFlowModel
            residue_models[residue] = model

            # Print metrics
            print(f"  Samples: {info.get('n_samples', 'N/A')}")
            print(f"  Test RMSD: {info.get('test_rmsd', 'N/A'):.4f} Å")
            print(f"  Test NLL: {info.get('test_nll', 'N/A'):.2f}")

        except Exception as e:
            print(f"Failed to train {residue.name}: {e}")
            continue

    if not residue_models:
        raise ValueError("No models were successfully trained")

    # Create and save PolymerFlowModel
    print(f"\n{'='*60}")
    print("Creating PolymerFlowModel")
    print(f"{'='*60}")

    polymer_model = PolymerFlowModel(residue_models)
    polymer_model.save(output_path)
    print(f"Saved PolymerFlowModel to {output_path}")
    print(f"Supported residues: {[r.name for r in polymer_model.supported_residues]}")
    print(f"Latent dim: {polymer_model.latent_dim}")

    return polymer_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RNA PolymerFlowModel")
    parser.add_argument("--data", type=str, required=True, help="Path to CIF files")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--latent-dim", type=int, default=12, help="Latent dimension")
    parser.add_argument("--n-layers", type=int, default=6, help="Number of flow layers")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--no-rotation", action="store_true", help="Disable rotation layers")
    parser.add_argument("--noise", type=float, default=0.05, help="Noise std for regularization")

    args = parser.parse_args()

    train_rna_flow(
        data_dir=args.data,
        output_dir=args.output,
        n_epochs=args.epochs,
        latent_dim=args.latent_dim,
        n_layers=args.n_layers,
        device=args.device,
        use_rotation=not args.no_rotation,
        noise_std=args.noise,
    )
