#!/usr/bin/env python3
"""
Example: Training ResidueFlowModels and sampling RNA conformations.

This example demonstrates the complete workflow for:
1. Training flow models on RNA residues from CIF structures using Lightning
2. Creating a PolymerFlowModel for multi-residue sampling
3. Sampling multiple conformations for a given sequence
4. Saving sampled structures to CIF files

The flow model learns a low-dimensional latent space that captures
the conformational variability of RNA residues, enabling realistic
sampling of new structures.
"""

import numpy as np
import torch
from pathlib import Path

import ciffy
from ciffy import Scale
from ciffy import Residue
from ciffy.nn.flow import (
    ResidueFlowModel,
    ResidueFlowConfig,
    PolymerFlowModel,
)


def train_residue_models(
    cif_paths: list[Path],
    residues: list,
    latent_dim: int = 8,
    n_layers: int = 4,
    hidden_dim: int = 32,
    n_epochs: int = 50,
    verbose: bool = True,
) -> dict:
    """
    Train flow models for each residue type using Lightning.

    Args:
        cif_paths: List of CIF file paths for training data.
        residues: List of residue types to train (e.g., [Residue.A, Residue.C]).
        latent_dim: Latent space dimension.
        n_layers: Number of flow layers.
        hidden_dim: Hidden layer size.
        n_epochs: Number of training epochs per residue.
        verbose: Whether to print training progress.

    Returns:
        Dictionary mapping residue type to trained ResidueFlowModel.
    """
    import lightning as L
    from ciffy.nn.lightning import FlowDataModule, ResidueFlowModule
    from ciffy.nn.lightning.modules.residue_flow import (
        ResidueFlowFullConfig,
        ResidueFlowModelConfig,
        ResidueFlowDataConfig,
    )
    from ciffy.nn.config import TrainingConfig

    # Create config
    config = ResidueFlowFullConfig(
        model=ResidueFlowModelConfig(
            latent_dim=latent_dim,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
        ),
        data=ResidueFlowDataConfig(
            batch_size=256,
        ),
        training=TrainingConfig(
            lr=1e-3,
            epochs=n_epochs,
        ),
    )

    models = {}

    for residue in residues:
        if verbose:
            print(f"\nTraining model for {residue.name}...")

        try:
            # Create data module
            dm = FlowDataModule(
                cif_paths=list(cif_paths),
                residue=residue,
                batch_size=256,
                min_coverage=0.5,  # Lower for small datasets
            )

            # Create Lightning module
            module = ResidueFlowModule(config, residue)

            # Create trainer
            trainer = L.Trainer(
                max_epochs=n_epochs,
                accelerator="auto",
                enable_progress_bar=verbose,
                enable_model_summary=False,
                logger=False,
            )

            # Train
            trainer.fit(module, dm)

            # Get trained model
            model = module.get_model()
            models[residue] = model

            if verbose:
                print(f"  {residue.name}: {model.n_atoms} atoms, latent_dim={model.latent_dim}")

        except Exception as e:
            if verbose:
                print(f"  {residue.name}: Failed - {e}")

    return models


def sample_conformations(
    polymer_model: PolymerFlowModel,
    sequence: list,
    n_samples: int = 10,
) -> list[np.ndarray]:
    """
    Sample multiple conformations for a sequence.

    Args:
        polymer_model: Trained PolymerFlowModel.
        sequence: List of residue types (e.g., [Residue.A, Residue.G]).
        n_samples: Number of conformations to sample.

    Returns:
        List of coordinate arrays, each with shape (n_atoms, 3).
    """
    # Convert residue types to int array (PolymerFlowModel expects int sequence)
    sequence_int = np.array([r.value for r in sequence], dtype=np.int64)

    # Use the built-in sample method
    with torch.no_grad():
        samples_tensor = polymer_model.sample(sequence_int, n_samples=n_samples)

    # Convert list of tensors to list of numpy arrays
    return [s.numpy() for s in samples_tensor]


def main():
    # =========================================================================
    # Configuration
    # =========================================================================

    # Use test data files
    data_dir = Path(__file__).parent.parent / "tests" / "data"
    cif_paths = sorted(data_dir.glob("*.cif"))

    print("=" * 60)
    print("RNA Flow Model Training and Sampling Example")
    print("=" * 60)
    print(f"\nFound {len(cif_paths)} CIF files for training:")
    for p in cif_paths:
        print(f"  - {p.name}")

    # RNA nucleotides to train (all 4 standard bases)
    rna_residues = [Residue.A, Residue.C, Residue.G, Residue.U]

    # Flow model configuration
    # Note: Small values for quick demo; increase for production
    n_epochs = 30  # Quick training for demo

    # =========================================================================
    # Train Residue Models
    # =========================================================================

    print("\n" + "=" * 60)
    print("Training ResidueFlowModels (using Lightning)")
    print("=" * 60)

    models = train_residue_models(
        cif_paths,
        rna_residues,
        latent_dim=8,
        n_layers=4,
        hidden_dim=32,
        n_epochs=n_epochs,
        verbose=True,
    )

    if not models:
        print("\nNo models were trained successfully. Try with more/different data.")
        return

    print(f"\nSuccessfully trained {len(models)} models:")
    for res, model in models.items():
        print(f"  {res.name}: latent_dim={model.latent_dim}, n_atoms={model.n_atoms}")

    # =========================================================================
    # Create PolymerFlowModel
    # =========================================================================

    print("\n" + "=" * 60)
    print("Creating PolymerFlowModel")
    print("=" * 60)

    # Create polymer model from residue models
    polymer_model = PolymerFlowModel(models)
    print(f"\nPolymerFlowModel created:")
    print(f"  Supported residues: {list(models.keys())}")
    print(f"  Latent dimension: {polymer_model.latent_dim}")

    # =========================================================================
    # Sample Conformations
    # =========================================================================

    print("\n" + "=" * 60)
    print("Sampling Conformations")
    print("=" * 60)

    # Define a target sequence (use residues we have models for)
    available_residues = list(models.keys())
    sequence = available_residues * 2  # Repeat to get longer sequence
    sequence_str = "".join(r.abbrev for r in sequence)

    print(f"\nTarget sequence: {sequence_str} ({len(sequence)} residues)")

    # Sample multiple conformations
    n_samples = 5
    print(f"Sampling {n_samples} conformations...")

    samples = sample_conformations(
        polymer_model,
        sequence,
        n_samples=n_samples,
    )

    print(f"\nSampled {len(samples)} conformations:")
    for i, coords in enumerate(samples):
        # Compute radius of gyration as a measure of compactness
        centroid = coords.mean(axis=0)
        rg = np.sqrt(((coords - centroid) ** 2).sum(axis=1).mean())
        print(f"  Sample {i+1}: {coords.shape[0]} atoms, Rg = {rg:.2f} Å")

    # =========================================================================
    # Save Samples as CIF files
    # =========================================================================

    print("\n" + "=" * 60)
    print("Saving Sampled Structures")
    print("=" * 60)

    output_dir = Path("/tmp/rna_flow_samples")
    output_dir.mkdir(exist_ok=True)

    # Create a template with ONLY the atoms the flow model uses
    # This avoids hydrogens and other atoms that would have incorrect positions
    template = ciffy.from_sequence(sequence_str, atoms=polymer_model.atom_filter)
    print(f"\nTemplate polymer: {template.size()} atoms, {template.size(Scale.RESIDUE)} residues")
    print(f"  (Using only atoms from flow model, excluding hydrogens)")

    # Convert each sample to a polymer and save as CIF
    for i, coords in enumerate(samples):
        # Template already has the right atoms - just update coordinates
        sampled_polymer = template.with_coordinates(coords)
        output_path = output_dir / f"sample_{i+1:03d}.cif"
        sampled_polymer.write(str(output_path))
        print(f"  Saved: {output_path}")

    # =========================================================================
    # Summary
    # =========================================================================

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"""
Trained {len(models)} ResidueFlowModels on {len(cif_paths)} structures.
Created PolymerFlowModel for sequences containing: {list(models.keys())}
Sampled {n_samples} conformations for sequence '{sequence_str}'.
Saved CIF files to: {output_dir}

The CIF files contain only atoms the flow model was trained on (heavy atoms
with sufficient coverage). Hydrogens and rare atoms are excluded.

Next steps:
- Train with more data for better models (hundreds of structures)
- Increase latent_dim (12-16) and n_layers (8) for more expressive models
- Use ciffy.flow.train() for a simpler high-level API
- Add hydrogens with a molecular dynamics package if needed
""")


if __name__ == "__main__":
    main()
