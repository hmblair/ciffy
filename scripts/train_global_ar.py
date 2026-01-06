#!/usr/bin/env python3
"""
Train CoordinateARModel for coordinate prediction with global conditioning.

Uses the simplified Polymer-based interface - no complex batching or collation.
Each polymer is processed one at a time with model.compute_loss(polymer).
"""

import sys
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import ciffy
from ciffy.nn import PolymerDataset
from ciffy.nn.autoregressive import CoordinateARModel, CoordinateARModelConfig
from ciffy.biochemistry import Residue, Scale, Molecule


def get_residue_atom_counts(dataset: PolymerDataset, n_samples: int = 100) -> dict:
    """Get atom counts for each residue type from dataset."""
    from collections import Counter

    atom_counts = {res: Counter() for res in [Residue.A, Residue.C, Residue.G, Residue.U]}

    for i in range(min(n_samples, len(dataset))):
        try:
            poly = dataset[i]
            if poly is None:
                continue
            poly = poly.strip()
            if poly.size() == 0:
                continue

            seq = poly.sequence.numpy() if hasattr(poly.sequence, 'numpy') else poly.sequence
            counts = poly.counts(Scale.RESIDUE)
            counts = counts.numpy() if hasattr(counts, 'numpy') else counts

            for j, (res_val, n_atoms) in enumerate(zip(seq, counts)):
                for res in atom_counts:
                    if res.value == res_val:
                        atom_counts[res][int(n_atoms)] += 1
                        break
        except Exception:
            continue

    # Get most common atom count for each residue
    result = {}
    for res, counter in atom_counts.items():
        if counter:
            most_common = counter.most_common(1)[0][0]
            result[res] = most_common
            print(f"  {res.name}: {most_common} atoms")

    return result


def train(
    data_dir: str,
    output_dir: Path,
    d_model: int = 256,
    num_layers: int = 6,
    num_epochs: int = 100,
    lr: float = 1e-4,
    max_chains: int = None,
    max_residues: int = 128,
    device: str = "cuda",
):
    """Train CoordinateARModel with simplified Polymer interface."""
    print(f"\n=== Training CoordinateARModel ===")
    print(f"  d_model={d_model}, num_layers={num_layers}")
    print(f"  num_epochs={num_epochs}, lr={lr}")

    # Load dataset
    dataset = PolymerDataset(
        data_dir,
        scale=Scale.CHAIN,
        molecule_types=Molecule.RNA,
        min_residues=6,
        max_residues=max_residues,
        backend="numpy",
        limit=max_chains,
    )
    print(f"  Chains: {len(dataset)}")

    if len(dataset) == 0:
        print("No chains found, exiting")
        return None

    # Get residue atom counts
    print("\n=== Getting Residue Atom Info ===")
    residue_atoms = get_residue_atom_counts(dataset)

    if len(residue_atoms) == 0:
        print("No residue info found, exiting")
        return None

    # Create model
    config = CoordinateARModelConfig(
        d_model=d_model,
        num_layers=num_layers,
        num_heads=8,
        dropout=0.1,
    )
    model = CoordinateARModel(residue_atoms, config).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Split train/val
    n_val = max(1, len(dataset) // 10)
    indices = torch.randperm(len(dataset)).tolist()
    train_indices = indices[n_val:]
    val_indices = indices[:n_val]

    print(f"  Train: {len(train_indices)}, Val: {len(val_indices)}")

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_loss = 0.0
        n_chains = 0

        np.random.shuffle(train_indices)
        for idx in train_indices:
            try:
                poly = dataset[idx]
                if poly is None:
                    continue

                poly = poly.strip()
                if poly.size() == 0 or poly.size(Scale.RESIDUE) < 4:
                    continue

                optimizer.zero_grad()
                loss = model.compute_loss(poly)

                if torch.isnan(loss):
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_chains += 1

            except Exception as e:
                if 'cuda' in device:
                    torch.cuda.empty_cache()
                continue

        train_loss = epoch_loss / max(1, n_chains)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        n_val_chains = 0

        with torch.no_grad():
            for idx in val_indices:
                try:
                    poly = dataset[idx]
                    if poly is None:
                        continue

                    poly = poly.strip()
                    if poly.size() == 0 or poly.size(Scale.RESIDUE) < 4:
                        continue

                    loss = model.compute_loss(poly)
                    if not torch.isnan(loss):
                        val_loss += loss.item()
                        n_val_chains += 1
                except Exception:
                    continue

        val_loss = val_loss / max(1, n_val_chains)
        val_losses.append(val_loss)

        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save(str(output_dir / "model"))

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}: train={train_loss:.4f}, val={val_loss:.4f} ({n_chains} chains)")

    print(f"\nBest validation loss: {best_val_loss:.4f}")

    np.savez(
        output_dir / "training_curves.npz",
        train_losses=train_losses,
        val_losses=val_losses,
    )

    return model


def sample_structures(model: CoordinateARModel, output_dir: Path, device: str = "cuda"):
    """Sample structures from the trained model."""
    print("\n=== Sampling Structures ===")

    test_sequences = ["ACGU", "GGCGCG", "ACGUACGU"]

    for seq_str in test_sequences:
        print(f"\nSequence: {seq_str}")

        # Create template from sequence
        template = ciffy.from_sequence(seq_str.lower())

        try:
            # Sample using the model
            polymers = model.sample(template, n_samples=1, temperature=0.0)
            poly = polymers[0]

            print(f"  Generated: {poly.size()} atoms, {poly.size(Scale.RESIDUE)} residues")

            # Save
            poly.write(str(output_dir / f"sampled_{seq_str}.cif"))
            print(f"  Saved: {output_dir / f'sampled_{seq_str}.cif'}")

        except Exception as e:
            print(f"  Error: {e}")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str,
                        default="/home/hmblair/data/rna",
                        help="Directory with CIF files")
    parser.add_argument("--output-dir", type=str, default="outputs/global_ar",
                        help="Output directory")
    parser.add_argument("--max-chains", type=int, default=None)
    parser.add_argument("--max-residues", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = train(
        args.data_dir,
        output_dir,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_epochs=args.num_epochs,
        lr=args.lr,
        max_chains=args.max_chains,
        max_residues=args.max_residues,
        device=device,
    )

    if model is not None:
        sample_structures(model, output_dir, device)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
