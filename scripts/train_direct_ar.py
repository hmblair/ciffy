#!/usr/bin/env python3
"""
Train DirectCoordAR model for direct coordinate prediction.

This model directly predicts coordinates and transforms autoregressively,
without intermediate latent representations.

Steps:
1. Extract training data from CIF files (coords + transforms per residue)
2. Build polymer-level sequences
3. Train DirectCoordAR to predict coords autoregressively
4. Sample and generate actual structures
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
from ciffy.nn import DirectCoordAR, DirectCoordARConfig
from ciffy.biochemistry import Residue, Scale, Molecule
from ciffy.nn.flow.residue.data import extract_residues_with_links, _remap_to_common
from ciffy.biochemistry.linking import LINKING_BY_TYPE, GLYCOSIDIC_FRAME
from ciffy.backend import to_numpy


def get_residue_atom_counts(cif_paths: list, min_coverage: float = 0.9) -> dict:
    """Get atom counts for each residue type from training data."""
    residue_atoms = {}
    atom_indices = {}

    for res in [Residue.A, Residue.C, Residue.G, Residue.U]:
        try:
            coords, transforms, atoms = extract_residues_with_links(
                cif_paths[:50], res, min_coverage=min_coverage, verbose=False
            )
            residue_atoms[res.value] = len(atoms)
            atom_indices[res.value] = atoms.tolist()
            print(f"  {res.name}: {len(atoms)} atoms, {len(coords)} samples")
        except ValueError as e:
            print(f"  {res.name}: skipped ({e})")

    return residue_atoms, atom_indices


def build_polymer_sequences(
    cif_paths: list,
    atom_indices: dict,
    max_seq_len: int = 256,
    max_bond_length: float = 2.0,
    verbose: bool = True,
) -> list:
    """
    Build training sequences from polymers.

    Each sequence is a list of (residue_value, coords, transform) tuples,
    where transform positions THIS residue relative to predecessor.
    """
    from ciffy.geometry.transforms import (
        extract_frame_positions,
        frame_from_positions,
        compute_relative_transform,
    )

    sequences = []
    skipped = 0

    for path_idx, path in enumerate(cif_paths):
        if verbose and path_idx % 100 == 0:
            print(f"  Processing {path_idx}/{len(cif_paths)}...")

        try:
            poly = ciffy.load(str(path)).molecule_type(Molecule.RNA).poly().strip()
            if poly.size() == 0 or poly.size(Scale.RESIDUE) < 4:
                skipped += 1
                continue

            # Align to glycosidic frame
            aligned, Rs = poly.align(GLYCOSIDIC_FRAME)
            Rs = to_numpy(Rs)
            origins = to_numpy(poly.gather([GLYCOSIDIC_FRAME.origin])[:, 0])

            # Get per-residue data
            seq = to_numpy(poly.sequence)
            counts = to_numpy(aligned.counts(Scale.RESIDUE))
            offsets = np.concatenate([[0], np.cumsum(counts)])
            coords = to_numpy(aligned.coordinates)
            atoms = to_numpy(aligned.atoms)
            orig_coords = to_numpy(poly.coordinates)

            n_res = poly.size(Scale.RESIDUE)

            # Build sequence
            chain_seq = []
            link_def = LINKING_BY_TYPE[Molecule.RNA]

            for j in range(n_res):
                res_val = int(seq[j])

                # Skip unknown residue types
                if res_val not in atom_indices:
                    continue

                sj, ej = offsets[j], offsets[j + 1]
                coords_j = coords[sj:ej]
                atoms_j = atoms[sj:ej].tolist()

                # Remap to common atoms
                common_atoms = atom_indices[res_val]
                if not set(common_atoms).issubset(set(atoms_j)):
                    # Missing atoms - skip
                    continue

                remapped = _remap_to_common(coords_j, atoms_j, common_atoms)

                # Compute transform
                if j == 0:
                    # First residue: identity transform
                    transform = np.zeros(6, dtype=np.float32)
                else:
                    # Check bond to previous
                    prev_val = int(seq[j - 1])
                    if prev_val not in atom_indices:
                        # Previous was skipped, treat as chain break
                        transform = np.zeros(6, dtype=np.float32)
                    else:
                        # Compute relative transform from prev to this
                        si, ei = offsets[j - 1], offsets[j]
                        coords_i = coords[si:ei]
                        atoms_i = atoms[si:ei].tolist()
                        common_i = atom_indices[prev_val]

                        if not set(common_i).issubset(set(atoms_i)):
                            transform = np.zeros(6, dtype=np.float32)
                        else:
                            remapped_i = _remap_to_common(coords_i, atoms_i, common_i)
                            atoms_i_arr = np.array(common_i, dtype=np.int64)
                            atoms_j_arr = np.array(common_atoms, dtype=np.int64)

                            # Transform j's coords to i's frame
                            R_j_to_i = Rs[j].T @ Rs[j - 1]
                            t_j_to_i = (origins[j] - origins[j - 1]) @ Rs[j - 1]
                            coords_j_in_i = remapped @ R_j_to_i + t_j_to_i

                            # Extract frames and compute relative transform
                            prev_pos = extract_frame_positions(
                                remapped_i, atoms_i_arr, link_def.prev_frame
                            )
                            o3p_origin, o3p_R = frame_from_positions(prev_pos)

                            next_pos = extract_frame_positions(
                                coords_j_in_i, atoms_j_arr, link_def.next_frame
                            )
                            p_origin, p_R = frame_from_positions(next_pos)

                            transform = compute_relative_transform(
                                o3p_origin, o3p_R, p_origin, p_R
                            )

                chain_seq.append((res_val, remapped.astype(np.float32), transform))

            # Accept sequences with at least 4 residues
            if len(chain_seq) >= 4:
                # Truncate if needed
                if len(chain_seq) > max_seq_len:
                    chain_seq = chain_seq[:max_seq_len]
                sequences.append(chain_seq)

        except Exception as e:
            skipped += 1
            continue

    if verbose:
        print(f"Built {len(sequences)} sequences ({skipped} skipped)")

    return sequences


def collate_sequences(batch: list, max_atoms: int, device: str = "cpu") -> dict:
    """Collate variable-length sequences into tensors."""
    B = len(batch)
    max_len = max(len(seq) for seq in batch)

    # Initialize tensors
    sequences = torch.zeros(B, max_len, dtype=torch.long, device=device)
    coords = torch.zeros(B, max_len, max_atoms, 3, dtype=torch.float32, device=device)
    transforms = torch.zeros(B, max_len, 6, dtype=torch.float32, device=device)
    n_atoms = torch.zeros(B, max_len, dtype=torch.long, device=device)
    padding_mask = torch.ones(B, max_len, dtype=torch.bool, device=device)

    for i, seq in enumerate(batch):
        L = len(seq)
        for j, (res_val, res_coords, res_transform) in enumerate(seq):
            sequences[i, j] = res_val
            n_res_atoms = res_coords.shape[0]
            coords[i, j, :n_res_atoms] = torch.tensor(res_coords)
            transforms[i, j] = torch.tensor(res_transform)
            n_atoms[i, j] = n_res_atoms
        padding_mask[i, :L] = False

    return {
        "sequence": sequences,
        "coords": coords,
        "transforms": transforms,
        "n_atoms": n_atoms,
        "padding_mask": padding_mask,
    }


def train_model(
    sequences: list,
    residue_atoms: dict,
    output_dir: Path,
    d_model: int = 256,
    num_layers: int = 6,
    num_heads: int = 8,
    batch_size: int = 32,
    num_epochs: int = 100,
    lr: float = 1e-4,
    device: str = "cuda",
):
    """Train the DirectCoordAR model."""
    print(f"\n=== Training DirectCoordAR ===")
    print(f"  d_model={d_model}, num_layers={num_layers}, num_heads={num_heads}")
    print(f"  batch_size={batch_size}, num_epochs={num_epochs}, lr={lr}")

    # Split train/val
    n_val = max(1, len(sequences) // 10)
    indices = torch.randperm(len(sequences)).tolist()
    train_data = [sequences[i] for i in indices[n_val:]]
    val_data = [sequences[i] for i in indices[:n_val]]

    print(f"  Train: {len(train_data)}, Val: {len(val_data)}")

    # Create model
    config = DirectCoordARConfig(
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=0.1,
    )
    model = DirectCoordAR(residue_atoms, config).to(device)

    max_atoms = model.max_atoms
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_loss = 0
        n_batches = 0

        indices = torch.randperm(len(train_data)).tolist()
        for start in range(0, len(train_data), batch_size):
            batch_indices = indices[start:start + batch_size]
            batch = [train_data[i] for i in batch_indices]
            batch = collate_sequences(batch, max_atoms, device)

            optimizer.zero_grad()
            loss = model.compute_loss(
                batch["sequence"],
                batch["coords"],
                batch["transforms"],
                batch["n_atoms"],
                batch["padding_mask"],
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
                batch = collate_sequences(batch, max_atoms, device)

                loss = model.compute_loss(
                    batch["sequence"],
                    batch["coords"],
                    batch["transforms"],
                    batch["n_atoms"],
                    batch["padding_mask"],
                )
                val_loss += loss.item()
                n_val_batches += 1

        val_loss = val_loss / max(1, n_val_batches)
        val_losses.append(val_loss)

        scheduler.step()

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save(str(output_dir / "model"))

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}: train={train_loss:.6f}, val={val_loss:.6f}")

    print(f"\nBest validation loss: {best_val_loss:.6f}")

    # Save training curves
    np.savez(
        output_dir / "training_curves.npz",
        train_losses=train_losses,
        val_losses=val_losses,
    )

    return model


def sample_and_build(
    model: DirectCoordAR,
    atom_indices: dict,
    test_sequences: list,
    output_dir: Path,
    device: str = "cuda",
):
    """Sample from model and build actual polymer structures."""
    print("\n=== Sampling Structures ===")

    model.eval()
    char_to_res = {'A': Residue.A, 'C': Residue.C, 'G': Residue.G, 'U': Residue.U}

    for seq_str in test_sequences:
        print(f"\nSequence: {seq_str}")

        # Convert to tensor
        seq_values = [char_to_res[c].value for c in seq_str.upper()]
        sequence = torch.tensor(seq_values, dtype=torch.long, device=device)

        # Generate
        with torch.no_grad():
            coords, transforms = model.generate(sequence, temperature=0.0)

        coords = coords[0].cpu().numpy()  # (L, max_atoms, 3)
        transforms = transforms[0].cpu().numpy()  # (L, 6)

        print(f"  Generated: coords shape={coords.shape}, transforms shape={transforms.shape}")

        # Build polymer using extend()
        try:
            poly = ciffy.Polymer()
            for i, char in enumerate(seq_str.upper()):
                res = char_to_res[char]
                n_atoms = model.residue_atoms[res.value]
                res_coords = coords[i, :n_atoms]
                res_transform = transforms[i]
                # atom_indices keys might be strings after JSON, convert to int
                res_atoms = atom_indices.get(res.value) or atom_indices.get(str(res.value))

                if res_atoms is None:
                    print(f"  Warning: No atoms for {res.name}")
                    continue

                # Get elements from atom indices by creating subset AtomGroup
                atom_group = res.subset(set(res_atoms))
                res_elements = atom_group.elements().tolist()

                if i == 0:
                    # First residue: no transform
                    poly = poly.extend(res, res_coords, atoms=res_atoms, elements=res_elements)
                else:
                    # Use transform to position
                    poly = poly.extend(res, res_coords, res_transform, atoms=res_atoms, elements=res_elements)

            # Save structure
            out_path = output_dir / f"sampled_{seq_str}.cif"
            poly.write(str(out_path))
            print(f"  Saved: {out_path}")

            # Check bond lengths
            if poly.size(Scale.RESIDUE) >= 2:
                from ciffy.biochemistry.linking import LINKING_BY_TYPE
                link_def = LINKING_BY_TYPE[Molecule.RNA]
                orig_coords = to_numpy(poly.coordinates)
                atoms = to_numpy(poly.atoms)
                counts = to_numpy(poly.counts(Scale.RESIDUE))
                offsets = np.concatenate([[0], np.cumsum(counts)])

                o3p_values = to_numpy(link_def.prev_atom.index())
                p_values = to_numpy(link_def.next_atom.index())

                bond_lengths = []
                for j in range(1, poly.size(Scale.RESIDUE)):
                    si, ei = offsets[j - 1], offsets[j]
                    sj, ej = offsets[j], offsets[j + 1]

                    atoms_i = atoms[si:ei]
                    atoms_j = atoms[sj:ej]

                    o3p_mask = np.isin(atoms_i, o3p_values)
                    p_mask = np.isin(atoms_j, p_values)

                    if o3p_mask.any() and p_mask.any():
                        o3p_pos = orig_coords[si:ei][o3p_mask.argmax()]
                        p_pos = orig_coords[sj:ej][p_mask.argmax()]
                        dist = np.linalg.norm(p_pos - o3p_pos)
                        bond_lengths.append(dist)

                if bond_lengths:
                    print(f"  O3'-P bond lengths: mean={np.mean(bond_lengths):.2f} Å, "
                          f"range=[{np.min(bond_lengths):.2f}, {np.max(bond_lengths):.2f}]")

        except Exception as e:
            print(f"  Error building polymer: {e}")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str,
                        default="/home/hblair/data/structures/rna",
                        help="Directory with CIF files")
    parser.add_argument("--output-dir", type=str, default="outputs/direct_ar",
                        help="Output directory")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Max CIF files to load")
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

    # Get residue atom counts
    print("\n=== Getting Residue Atom Counts ===")
    residue_atoms, atom_indices = get_residue_atom_counts(cif_paths)

    if len(residue_atoms) == 0:
        print("No residue atom counts found, exiting")
        return

    # Save atom indices for later use
    with open(output_dir / "atom_indices.json", "w") as f:
        json.dump(atom_indices, f, indent=2)

    # Build training sequences
    print("\n=== Building Training Sequences ===")
    sequences = build_polymer_sequences(cif_paths, atom_indices)

    if len(sequences) == 0:
        print("No sequences built, exiting")
        return

    # Print sequence stats
    lengths = [len(s) for s in sequences]
    print(f"  Sequence lengths: mean={np.mean(lengths):.1f}, "
          f"min={np.min(lengths)}, max={np.max(lengths)}")

    # Train model
    model = train_model(
        sequences,
        residue_atoms,
        output_dir,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        device=device,
    )

    # Sample structures
    test_sequences = ["ACGU", "GGCGCG", "ACGUACGU"]
    sample_and_build(model, atom_indices, test_sequences, output_dir, device)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
