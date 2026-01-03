#!/usr/bin/env python
"""Test chain sampling with the new frame system.

Quick test that trains a model and samples chains to verify:
1. Residue model training works
2. PolymerModel chain assembly works
3. Output CIF files are valid
"""

import warnings
warnings.filterwarnings("ignore")

import os
os.environ["CIFFY_LOG_LEVEL"] = "WARNING"

from pathlib import Path
import numpy as np


def main():
    import argparse
    from ciffy.nn import residue

    parser = argparse.ArgumentParser(description="Test chain sampling")
    parser.add_argument("--data-dir", default="/Users/hmblair/academic/data/structures/rna")
    parser.add_argument("--max-files", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--chain-length", type=int, default=10)
    parser.add_argument("--output-dir", default="outputs/chain_test")
    parser.add_argument("--model-type", default="flow", choices=["flow", "vae", "consolidated"])
    parser.add_argument("--accelerator", default="cpu", help="cpu, gpu, or mps")
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cif_paths = sorted(Path(args.data_dir).glob("*.cif"))[:args.max_files]

    print("=" * 60)
    print("Chain Sampling Test")
    print("=" * 60)
    print(f"  Data: {len(cif_paths)} CIF files")
    print(f"  Model: {args.model_type}")
    print(f"  Epochs: {args.epochs}")
    print()

    # Train model
    print("Training model...")
    model = residue.train(
        cif_paths=cif_paths,
        residues="ACGU",
        model_type=args.model_type,
        n_epochs=args.epochs,
        output_dir=output_path / "model",
        accelerator=args.accelerator,
        verbose=True,
    )

    # Sample chains
    print(f"\nSampling {args.chain_length}-mer chains...")
    np.random.seed(42)
    sequence = "".join(np.random.choice(list("acgu"), args.chain_length))
    print(f"  Sequence: {sequence}")

    chains_path = output_path / "chains"
    chains_path.mkdir(exist_ok=True)

    # Sample multiple chains
    for i in range(3):
        polymer = model.sample_from_sequence(sequence)
        output_file = chains_path / f"chain_{i}.cif"
        polymer.write(str(output_file))
        print(f"  Saved: {output_file.name} ({polymer.size()} atoms)")

    # Verify chain geometry
    print("\nVerifying chain geometry...")
    import torch

    polymer = model.sample_from_sequence(sequence)

    # Check O3'-P bond distances between residues
    coords = torch.tensor(polymer.coordinates, dtype=torch.float32)
    atoms = torch.tensor(polymer.atoms, dtype=torch.long)

    # Find O3' and P atoms
    from ciffy.biochemistry import Sugar, PhosphateGroup
    o3p_val = Sugar.O3p.index()[0]  # Get first value
    p_val = PhosphateGroup.P.index()[0]

    o3p_mask = atoms == o3p_val
    p_mask = atoms == p_val

    o3p_coords = coords[o3p_mask]
    p_coords = coords[p_mask]

    # O3' of residue i should be ~1.6A from P of residue i+1
    if len(o3p_coords) > 1 and len(p_coords) > 1:
        # Skip first P (no preceding O3') and last O3' (no following P)
        link_distances = torch.norm(o3p_coords[:-1] - p_coords[1:], dim=1)
        mean_dist = link_distances.mean().item()
        std_dist = link_distances.std().item()
        print(f"  O3'-P link distances: {mean_dist:.3f} ± {std_dist:.3f} A")
        print(f"  Expected: ~1.60 A (phosphodiester bond)")

        if abs(mean_dist - 1.6) < 0.3:
            print("  ✓ Chain geometry looks correct!")
        else:
            print("  ⚠ Chain geometry may have issues")

    print("\n" + "=" * 60)
    print(f"Done! Output in {output_path}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
