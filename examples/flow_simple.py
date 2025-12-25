#!/usr/bin/env python3
"""
Example: Simple flow model usage with the high-level ciffy.flow API.

This example demonstrates the simplified workflow for:
1. Sampling polymer conformations from sequences
2. Training custom flow models on your data
3. Encoding existing structures to latent space
4. Decoding latents back to polymers

For the lower-level API with more control, see rna_flow_sampling.py.
"""

from pathlib import Path

# High-level flow API - all common operations in one import
from ciffy import flow


def sample_from_sequence():
    """Sample conformations from a sequence - the simplest workflow."""
    print("=" * 60)
    print("Sample from Sequence")
    print("=" * 60)

    # Sample a single conformation (2 lines!)
    polymer = flow.sample("acgu")
    print(f"\nSampled polymer: {polymer.size()} atoms")

    # Save to file
    output_path = Path("/tmp/flow_sample.cif")
    polymer.write(str(output_path))
    print(f"Saved to: {output_path}")

    # Sample multiple conformations
    samples = flow.sample("acguacgu", n_samples=5)
    print(f"\nSampled {len(samples)} conformations:")
    for i, p in enumerate(samples):
        print(f"  Sample {i+1}: {p.size()} atoms")


def train_custom_model():
    """Train a custom flow model on your data."""
    print("\n" + "=" * 60)
    print("Train Custom Model")
    print("=" * 60)

    # Find training data
    data_dir = Path(__file__).parent.parent / "tests" / "data"
    cif_files = list(data_dir.glob("*.cif"))

    if not cif_files:
        print("No CIF files found in tests/data/")
        return None

    print(f"\nTraining on {len(cif_files)} structures...")

    # Train with minimal config for quick demo
    model = flow.train(
        cif_files,
        residues="ACGU",  # RNA nucleotides
        n_epochs=30,      # Quick training for demo
        latent_dim=8,
        n_layers=4,
        hidden_dim=32,
    )

    print(f"Trained model with {len(model.residue_types)} residue types")
    return model


def encode_decode_workflow():
    """Encode existing structures and decode modified latents."""
    print("\n" + "=" * 60)
    print("Encode/Decode Workflow")
    print("=" * 60)

    import ciffy
    import torch

    # Load an existing structure
    data_dir = Path(__file__).parent.parent / "tests" / "data"
    cif_path = data_dir / "9MDS.cif"

    if not cif_path.exists():
        print("Test file not found, skipping encode/decode demo")
        return

    polymer = ciffy.load(str(cif_path)).by_type(ciffy.RNA).poly()
    if polymer.size() == 0:
        print("No RNA found in structure, skipping encode/decode demo")
        return

    # Take first chain for demo
    chain = list(polymer.chains())[0]
    print(f"\nUsing chain {chain.names[0]}: {chain.size(ciffy.RESIDUE)} residues")

    # Encode to latent space
    latents = flow.encode(chain)
    print(f"Latent shape: {latents.shape}")

    # Modify latents (add small noise)
    modified = latents + torch.randn_like(latents) * 0.1

    # Decode back to polymer
    new_polymer = flow.decode(modified, chain)
    print(f"Decoded polymer: {new_polymer.size()} atoms")

    # Compute RMSD between original and modified
    rmsd = ciffy.rmsd(chain, new_polymer)
    print(f"RMSD after latent perturbation: {rmsd:.2f} Angstroms")


def sample_with_custom_model(model):
    """Use a custom trained model for sampling."""
    print("\n" + "=" * 60)
    print("Sample with Custom Model")
    print("=" * 60)

    if model is None:
        print("No model provided, skipping")
        return

    # Sample using the trained model
    samples = flow.sample("acgu", n_samples=3, model=model)

    print(f"\nSampled {len(samples)} conformations with custom model:")
    for i, p in enumerate(samples):
        print(f"  Sample {i+1}: {p.size()} atoms")


def main():
    print("ciffy.flow - High-Level Flow Model API")
    print()

    # Simplest workflow: sample from sequence
    sample_from_sequence()

    # Train a custom model
    model = train_custom_model()

    # Use custom model for sampling
    sample_with_custom_model(model)

    # Encode/decode workflow
    encode_decode_workflow()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
The ciffy.flow API provides simple functions for common workflows:

    from ciffy import flow

    # Sample conformations
    polymer = flow.sample("acgu")

    # Train on your data
    model = flow.train(["data/*.cif"], residues="ACGU")

    # Encode to latent space
    latents = flow.encode(polymer)

    # Decode latents
    new_polymer = flow.decode(latents, "acgu")

For more control, use the lower-level API in ciffy.nn.flow.
See rna_flow_sampling.py for a detailed example.
""")


if __name__ == "__main__":
    main()
