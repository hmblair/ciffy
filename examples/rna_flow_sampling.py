#!/usr/bin/env python3
"""
Example: Training ResidueFlowModels and sampling RNA conformations.

This example demonstrates the complete workflow for:
1. Training flow models on RNA residues from CIF structures
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
    config: ResidueFlowConfig,
    n_epochs: int = 50,
    verbose: bool = True,
) -> dict:
    """
    Train flow models for each residue type.

    Args:
        cif_paths: List of CIF file paths for training data.
        residues: List of residue types to train (e.g., [Residue.A, Residue.C]).
        config: Configuration for the flow model.
        n_epochs: Number of training epochs per residue.
        verbose: Whether to print training progress.

    Returns:
        Dictionary mapping residue type to trained ResidueFlowModel.
    """
    models = {}

    for residue in residues:
        if verbose:
            print(f"\nTraining model for {residue.name}...")

        try:
            model = ResidueFlowModel.from_structures(
                cif_paths,
                residue,
                config=config,
                n_epochs=n_epochs,
                verbose=verbose,
            )
            models[residue] = model

            if verbose:
                print(f"  {residue.name}: {model.n_atoms} atoms, "
                      f"{model.var_explained*100:.1f}% variance explained, "
                      f"PCA RMSD = {model.pca_rmsd:.3f} Å")

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
    config = ResidueFlowConfig(
        latent_dim=8,       # Dimensionality of latent space
        n_layers=4,         # Number of flow layers
        hidden_dim=32,      # Hidden layer size
        min_coverage=0.5,   # Minimum atom coverage (lower for small datasets)
    )

    n_epochs = 30  # Quick training for demo

    # =========================================================================
    # Train Residue Models
    # =========================================================================

    print("\n" + "=" * 60)
    print("Training ResidueFlowModels")
    print("=" * 60)

    models = train_residue_models(
        cif_paths,
        rna_residues,
        config,
        n_epochs=n_epochs,
        verbose=True,
    )

    if not models:
        print("\nNo models were trained successfully. Try with more/different data.")
        return

    print(f"\nSuccessfully trained {len(models)} models:")
    for res, model in models.items():
        print(f"  {res.name}: latent_dim={config.latent_dim}, n_atoms={model.n_atoms}")

    # =========================================================================
    # Create PolymerFlowModel
    # =========================================================================

    print("\n" + "=" * 60)
    print("Creating PolymerFlowModel")
    print("=" * 60)

    # Create polymer model from residue models
    polymer_model = PolymerFlowModel(models)
    print(f"\nPolymerFlowModel created:")
    print(f"  Supported residues: {[r.name for r in polymer_model.supported_residues]}")
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
    # Encode and Reconstruct (using a real structure)
    # =========================================================================

    print("\n" + "=" * 60)
    print("Encode-Decode Roundtrip")
    print("=" * 60)

    # Load a structure with RNA residues (9MDS has RNA)
    # Find a structure that has A or G residues
    for cif_path in cif_paths:
        polymer = ciffy.load(str(cif_path)).poly()
        seq_indices = set(polymer.sequence.tolist())
        if Residue.A.value in seq_indices or Residue.G.value in seq_indices:
            print(f"\nUsing structure: {polymer.pdb_id}")
            print(f"  Chains: {polymer.names}")
            print(f"  Residues: {polymer.size(Scale.RESIDUE)}")
            break
    else:
        print("No structures with trainable residues found.")
        polymer = None

    if polymer is not None:
        # Find residues that we can encode
        encodable = []
        for idx in polymer.sequence[:20]:
            try:
                res = Residue.from_index(int(idx))
                if res in models:
                    encodable.append(res)
            except (ValueError, KeyError):
                pass

        if encodable:
            print(f"\nEncodable residues in first 20: {[r.name for r in encodable]}")
            print(f"  (Total: {len(encodable)} residues can be encoded)")

            # Note: Full encode/decode requires matching atoms to model's atom order
            # The ResidueFlowModel expects coordinates in a specific order
            print("\n  For full encode/decode workflow, use:")
            print("    - extract_residues_with_links() to get aligned coordinates")
            print("    - polymer_model.encode_polymer() for direct polymer encoding")
            print("    - polymer_model.decode_to_polymer() for polymer output")
        else:
            print("No encodable residues found in first 20 residues.")

    # =========================================================================
    # Summary
    # =========================================================================

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"""
Trained {len(models)} ResidueFlowModels on {len(cif_paths)} structures.
Created PolymerFlowModel for sequences containing: {[r.name for r in models.keys()]}
Sampled {n_samples} conformations for sequence '{sequence_str}'.
Saved CIF files to: {output_dir}

The CIF files contain only atoms the flow model was trained on (heavy atoms
with sufficient coverage). Hydrogens and rare atoms are excluded.

Next steps:
- Train with more data for better models (hundreds of structures)
- Increase latent_dim (12-16) and n_layers (8) for more expressive models
- Use ResidueFlowTrainer for batch training multiple residue types
- Add hydrogens with a molecular dynamics package if needed
""")


if __name__ == "__main__":
    main()
