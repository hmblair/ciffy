#!/usr/bin/env python3
"""
Train autoregressive model for residue latent prediction.

Steps:
1. Fit PCA+Quantile encoders for each residue type (using common atoms)
2. Train AR model to predict latents autoregressively
3. Sample and evaluate on test sequences
"""

import os
import sys
from pathlib import Path
from glob import glob
import json

import torch
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import ciffy
from ciffy.nn import (
    PCAQuantileResidueModel,
    fit_all_residues,
    PolymerModel,
    ResidueLatentARModel,
    ResidueLatentARModelConfig,
)
from ciffy.biochemistry import Residue, Scale
from ciffy.nn.flow.residue.data import extract_residues_with_links


def fit_residue_encoders(
    cif_paths: list,
    output_dir: Path,
    latent_dim: int = 12,
    min_coverage: float = 0.9,
):
    """Fit PCA+Quantile encoders for each residue type using common atoms."""
    print("\n=== Fitting PCA+Quantile Encoders ===")

    encoder_dir = output_dir / "encoders"
    encoder_dir.mkdir(parents=True, exist_ok=True)

    encoders = {}

    for res in [Residue.A, Residue.C, Residue.G, Residue.U]:
        print(f"\n  {res.name}:")
        try:
            # Extract residues with consistent atom ordering
            coords, transforms, atoms = extract_residues_with_links(
                cif_paths, res, min_coverage=min_coverage, verbose=False
            )
        except ValueError as e:
            print(f"    Skipping: {e}")
            continue

        if len(coords) < 100:
            print(f"    Skipping: only {len(coords)} samples")
            continue

        print(f"    Samples: {len(coords)}, Atoms: {len(atoms)}")

        # Convert to tensors
        coords_tensor = torch.tensor(coords, dtype=torch.float32)
        transforms_tensor = torch.tensor(transforms, dtype=torch.float32)

        # Flatten coords and concatenate with transforms
        N = len(coords)
        flat_coords = coords_tensor.reshape(N, -1)  # (N, n_atoms*3)
        data = torch.cat([flat_coords, transforms_tensor], dim=-1)

        # Fit PCA
        data_centered = data - data.mean(dim=0, keepdim=True)
        U, S, Vt = torch.linalg.svd(data_centered, full_matrices=False)
        V = Vt[:latent_dim]  # (latent_dim, D)
        mean = data.mean(dim=0)

        # Project to latent space
        latents = (data - mean) @ V.T  # (N, latent_dim)

        # Fit quantile splines
        from ciffy.nn.pca_quantile import QuantileSpline, PCAQuantileConfig
        splines = []
        for d in range(latent_dim):
            spline = QuantileSpline(n_knots=128)
            spline.fit(latents[:, d])
            splines.append(spline)

        # Create model
        config = PCAQuantileConfig(latent_dim=latent_dim)
        model = PCAQuantileResidueModel(
            V=V,
            mean=mean,
            n_atoms=len(atoms),
            atom_indices=atoms.tolist(),
            residue=res,
            config=config,
        )
        model.splines = torch.nn.ModuleList(splines)

        # Save
        model.save(str(encoder_dir / res.name))
        encoders[res] = model

        # Print stats
        z = model.encode(coords_tensor, transforms_tensor)
        print(f"    Latent stats: μ={z.mean():.3f}, σ={z.std():.3f}")

    return encoders


def encode_polymers(cif_paths: list, encoders: dict, min_coverage: float = 0.9, max_seq_len: int = 256) -> list:
    """Encode polymers to latent sequences using extract_residues_with_links."""
    print("\n=== Encoding Polymers to Latents ===")

    from ciffy.nn.flow.residue.data import _remap_to_common

    encoded = []
    skipped_polymers = 0
    skipped_residues = 0

    for path in cif_paths:
        try:
            polymer = ciffy.load(path)
            polymer = polymer.molecule_type(ciffy.Molecule.RNA).poly().strip()

            if polymer.size() == 0 or polymer.size(Scale.RESIDUE) < 4:
                skipped_polymers += 1
                continue

            # Encode each residue, skipping those with missing atoms
            sequence = []
            latents = []

            for i in range(polymer.size(Scale.RESIDUE)):
                res_polymer = polymer.residue(i)
                res_type_idx = res_polymer.sequence[0]

                # Find encoder for this residue type
                encoder = None
                for res, enc in encoders.items():
                    if res.value == res_type_idx:
                        encoder = enc
                        break

                if encoder is None:
                    # Unknown residue type - skip this residue
                    skipped_residues += 1
                    continue

                # Get coordinates and reorder to match encoder's atom order
                coords = res_polymer.coordinates
                atoms = res_polymer.atoms

                # Check if all required atoms are present
                common_atoms = encoder._atom_indices
                present = set(atoms.tolist())
                needed = set(common_atoms)
                if not needed.issubset(present):
                    # Missing atoms - skip this residue (common at 5'/3' termini)
                    skipped_residues += 1
                    continue

                # Reorder to common atoms
                try:
                    reordered = _remap_to_common(coords, atoms.tolist(), common_atoms)
                except Exception:
                    skipped_residues += 1
                    continue

                coords_tensor = torch.tensor(reordered, dtype=torch.float32)
                # Use zero transforms for now (we don't have link info per-residue)
                z = encoder.encode(coords_tensor.unsqueeze(0))  # (1, latent_dim)

                sequence.append(res_type_idx)
                latents.append(z.squeeze(0))

            # Accept polymers with at least 4 successfully encoded residues
            if len(sequence) >= 4:
                # Truncate long sequences
                if len(sequence) > max_seq_len:
                    sequence = sequence[:max_seq_len]
                    latents = latents[:max_seq_len]
                encoded.append({
                    'sequence': torch.tensor(sequence, dtype=torch.long),
                    'latents': torch.stack(latents),
                })
            else:
                skipped_polymers += 1

        except Exception as e:
            skipped_polymers += 1
            continue

    print(f"Encoded {len(encoded)} polymers ({skipped_polymers} skipped, {skipped_residues} residues skipped)")
    return encoded


def train_ar_model(
    encoded_data: list,
    output_dir: Path,
    latent_dim: int = 12,
    d_model: int = 256,
    num_layers: int = 6,
    num_heads: int = 8,
    batch_size: int = 32,
    num_epochs: int = 100,
    lr: float = 1e-4,
    device: str = "cuda",
):
    """Train the autoregressive model."""
    print("\n=== Training AR Model ===")
    print(f"  latent_dim={latent_dim}, d_model={d_model}")
    print(f"  num_layers={num_layers}, num_heads={num_heads}")
    print(f"  batch_size={batch_size}, num_epochs={num_epochs}, lr={lr}")

    # Split train/val
    n_val = max(1, len(encoded_data) // 10)
    indices = torch.randperm(len(encoded_data)).tolist()
    train_data = [encoded_data[i] for i in indices[n_val:]]
    val_data = [encoded_data[i] for i in indices[:n_val]]

    print(f"  Train: {len(train_data)}, Val: {len(val_data)}")

    # Create model
    config = ResidueLatentARModelConfig(
        latent_dim=latent_dim,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=0.1,
        predict_std=True,  # Predict uncertainty
    )
    model = ResidueLatentARModel(config).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Collate function
    def collate(batch):
        sequences = [item['sequence'] for item in batch]
        latents = [item['latents'] for item in batch]

        max_len = max(len(s) for s in sequences)
        B = len(batch)

        padded_seq = torch.zeros(B, max_len, dtype=torch.long)
        padded_lat = torch.zeros(B, max_len, latent_dim)
        mask = torch.ones(B, max_len, dtype=torch.bool)

        for i, (seq, lat) in enumerate(zip(sequences, latents)):
            L = len(seq)
            padded_seq[i, :L] = seq
            padded_lat[i, :L] = lat
            mask[i, :L] = False

        return {
            'sequence': padded_seq.to(device),
            'latents': padded_lat.to(device),
            'padding_mask': mask.to(device),
        }

    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_loss = 0
        n_batches = 0

        # Shuffle and batch
        indices = torch.randperm(len(train_data)).tolist()
        for start in range(0, len(train_data), batch_size):
            batch_indices = indices[start:start + batch_size]
            batch = [train_data[i] for i in batch_indices]
            batch = collate(batch)

            optimizer.zero_grad()
            loss = model.compute_loss(
                batch['sequence'],
                batch['latents'],
                batch['padding_mask'],
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        train_loss = epoch_loss / n_batches
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0
        n_val_batches = 0

        with torch.no_grad():
            for start in range(0, len(val_data), batch_size):
                batch = val_data[start:start + batch_size]
                batch = collate(batch)

                loss = model.compute_loss(
                    batch['sequence'],
                    batch['latents'],
                    batch['padding_mask'],
                )
                val_loss += loss.item()
                n_val_batches += 1

        val_loss = val_loss / max(1, n_val_batches)
        val_losses.append(val_loss)

        scheduler.step()

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save(str(output_dir / "ar_model"))

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

    print(f"\nBest validation loss: {best_val_loss:.4f}")

    # Save training curves
    np.savez(
        output_dir / "training_curves.npz",
        train_losses=train_losses,
        val_losses=val_losses,
    )

    return model


def sample_and_evaluate(
    model: ResidueLatentARModel,
    encoders: dict,
    test_sequences: list,
    device: str = "cuda",
):
    """Sample latents and decode to structures."""
    print("\n=== Sampling Predictions ===")

    model.eval()

    # Map char to Residue value (RNA nucleotides)
    char_to_value = {
        'A': Residue.A.value,
        'C': Residue.C.value,
        'G': Residue.G.value,
        'U': Residue.U.value,
    }

    for seq_str in test_sequences:
        print(f"\nSequence: {seq_str}")

        # Convert to indices
        seq_indices = []
        for char in seq_str.upper():
            val = char_to_value.get(char)
            if val is None:
                print(f"  Unknown residue: {char}, skipping")
                continue
            seq_indices.append(val)

        sequence = torch.tensor(seq_indices, dtype=torch.long, device=device).unsqueeze(0)

        # Generate latents at different temperatures
        for temp in [0.0, 0.5, 1.0]:
            with torch.no_grad():
                latents = model.generate(sequence, temperature=temp)

            # Compute latent statistics
            z = latents[0]  # (seq_len, latent_dim)
            print(f"  T={temp}: latent μ={z.mean():.3f}, σ={z.std():.3f}, "
                  f"range=[{z.min():.2f}, {z.max():.2f}]")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str,
                        default="/home/hblair/data/structures/rna",
                        help="Directory with CIF files")
    parser.add_argument("--output-dir", type=str, default="outputs/ar_model",
                        help="Output directory")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Max CIF files to load")
    parser.add_argument("--latent-dim", type=int, default=12)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Get CIF paths
    cif_paths = sorted(glob(os.path.join(args.data_dir, "*.cif")))
    if args.max_files:
        cif_paths = cif_paths[:args.max_files]
    cif_paths = [Path(p) for p in cif_paths]

    print(f"Found {len(cif_paths)} CIF files")

    if len(cif_paths) == 0:
        print("No CIF files found, exiting")
        return

    # Fit encoders using extract_residues_with_links (handles atom normalization)
    encoders = fit_residue_encoders(cif_paths, output_dir, args.latent_dim)

    if len(encoders) == 0:
        print("No encoders fitted, exiting")
        return

    # Encode polymers
    encoded_data = encode_polymers(cif_paths, encoders)

    if len(encoded_data) == 0:
        print("No encoded data, exiting")
        return

    # Train AR model
    model = train_ar_model(
        encoded_data,
        output_dir,
        latent_dim=args.latent_dim,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        device=device,
    )

    # Sample predictions
    test_sequences = [
        "ACGU",
        "GGCGCG",
        "ACGUACGU",
        "GCGCGCGCGC",
        "AAAAAAUUUUUU",
    ]
    sample_and_evaluate(model, encoders, test_sequences, device)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
