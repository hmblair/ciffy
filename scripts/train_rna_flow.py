"""
Train a PolymerFlowModel for RNA (A, U, G, C) using Lightning.

Usage:
    python scripts/train_rna_flow.py --data /home/hmblair/data/rna --output /home/hmblair/models/rna_flow
"""

import argparse
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger

from ciffy.biochemistry import Residue
from ciffy.nn.flow import PolymerFlowModel
from ciffy.nn.lightning import FlowDataModule, ResidueFlowModule
from ciffy.nn.lightning.modules.residue_flow import (
    ResidueFlowFullConfig,
    ResidueFlowModelConfig,
    ResidueFlowDataConfig,
)
from ciffy.nn.config import TrainingConfig


def train_rna_flow(
    data_dir: str,
    output_dir: str,
    n_epochs: int = 200,
    latent_dim: int = 12,
    n_layers: int = 6,
    hidden_dim: int = 64,
    use_rotation: bool = True,
    noise_std: float = 0.05,
    batch_size: int = 256,
    lr: float = 1e-3,
    accelerator: str = "auto",
    use_wandb: bool = False,
    project: str = "ciffy-flow",
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
    config = ResidueFlowFullConfig(
        model=ResidueFlowModelConfig(
            latent_dim=latent_dim,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            use_rotation=use_rotation,
            noise_std=noise_std,
        ),
        data=ResidueFlowDataConfig(
            batch_size=batch_size,
        ),
        training=TrainingConfig(
            lr=lr,
            epochs=n_epochs,
        ),
    )
    print(f"Config: latent_dim={latent_dim}, n_layers={n_layers}, hidden_dim={hidden_dim}")

    # Train a model for each residue type
    residue_models = {}

    for residue in rna_residues:
        print(f"\n{'='*60}")
        print(f"Training model for {residue.name}")
        print(f"{'='*60}")

        try:
            # Create data module
            dm = FlowDataModule(
                cif_paths=list(cif_files),
                residue=residue,
                batch_size=batch_size,
            )

            # Create Lightning module
            module = ResidueFlowModule(config, residue)

            # Setup logger
            logger = None
            if use_wandb:
                logger = WandbLogger(
                    project=project,
                    name=f"flow-{residue.name}",
                    tags=["flow", residue.name],
                )

            # Callbacks
            callbacks = [
                ModelCheckpoint(
                    dirpath=output_path / "checkpoints" / residue.name,
                    filename="best",
                    monitor="val/nll",
                    mode="min",
                    save_top_k=1,
                ),
                EarlyStopping(
                    monitor="val/nll",
                    patience=20,
                    mode="min",
                ),
            ]

            # Create trainer
            trainer = L.Trainer(
                max_epochs=n_epochs,
                accelerator=accelerator,
                logger=logger,
                callbacks=callbacks,
                enable_progress_bar=True,
                log_every_n_steps=10,
            )

            # Train
            trainer.fit(module, dm)

            # Get trained model
            model = module.get_model()

            # Save individual model
            model_path = output_path / residue.name
            model.save(model_path)
            print(f"Saved {residue.name} model to {model_path}")

            # Store for PolymerFlowModel
            residue_models[residue] = model

            # Print info
            print(f"  Atoms: {model.n_atoms}")
            print(f"  Latent dim: {model.latent_dim}")

        except Exception as e:
            print(f"Failed to train {residue.name}: {e}")
            import traceback
            traceback.print_exc()
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
    print(f"Supported residues: {list(residue_models.keys())}")
    print(f"Latent dim: {polymer_model.latent_dim}")

    return polymer_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RNA PolymerFlowModel")
    parser.add_argument("--data", type=str, required=True, help="Path to CIF files")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--latent-dim", type=int, default=12, help="Latent dimension")
    parser.add_argument("--n-layers", type=int, default=6, help="Number of flow layers")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--accelerator", type=str, default="auto", help="Accelerator (auto, cpu, gpu)")
    parser.add_argument("--no-rotation", action="store_true", help="Disable rotation layers")
    parser.add_argument("--noise", type=float, default=0.05, help="Noise std for regularization")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--project", type=str, default="ciffy-flow", help="W&B project name")

    args = parser.parse_args()

    train_rna_flow(
        data_dir=args.data,
        output_dir=args.output,
        n_epochs=args.epochs,
        latent_dim=args.latent_dim,
        n_layers=args.n_layers,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        lr=args.lr,
        accelerator=args.accelerator,
        use_rotation=not args.no_rotation,
        noise_std=args.noise,
        use_wandb=args.wandb,
        project=args.project,
    )
