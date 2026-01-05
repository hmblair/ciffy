#!/usr/bin/env python3
"""
Train CoordinateAR model for coordinate prediction with global conditioning.

Unlike DirectCoordAR which only sees local frames, CoordinateAR conditions
on the assembled global structure, enabling long-range awareness.

Steps:
1. Use PolymerDataset for proper chain-level iteration
2. Extract training data from chains (local coords + transforms)
3. Compute global positions by assembling chains
4. Train CoordinateAR to predict coords conditioned on global structure
5. Sample and generate structures
"""

import os
import sys
from pathlib import Path
from glob import glob
import json

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import ciffy
from ciffy.nn import CoordinateAR, CoordinateARConfig, PolymerDataset
from ciffy.biochemistry import Residue, Scale, Molecule
from ciffy.nn.flow.residue.data import extract_residues_with_links, _remap_to_common
from ciffy.biochemistry.linking import LINKING_BY_TYPE, GLYCOSIDIC_FRAME
from ciffy.backend import to_numpy


def axis_angle_to_rotation(axis_angle: np.ndarray) -> np.ndarray:
    """Convert axis-angle to rotation matrix."""
    angle = np.linalg.norm(axis_angle)
    if angle < 1e-8:
        return np.eye(3)
    axis = axis_angle / angle
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def rotation_to_axis_angle(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to axis-angle."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
    if angle < 1e-8:
        return np.zeros(3)
    axis = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1]
    ])
    axis = axis / (np.linalg.norm(axis) + 1e-8) * angle
    return axis


def apply_transform(prev_centroid, prev_R, transform):
    """Apply SE(3) transform to get new global position."""
    rot_aa = transform[:3]
    trans = transform[3:]
    rel_R = axis_angle_to_rotation(rot_aa)
    new_R = prev_R @ rel_R
    new_centroid = prev_centroid + prev_R @ trans
    return new_centroid, new_R


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


def process_single_chain(
    poly,
    atom_indices: dict,
    max_seq_len: int = 256,
) -> list | None:
    """
    Process a single chain into training data with global position information.

    Returns list of dicts for each residue, or None if chain is invalid.
    """
    from ciffy.geometry.transforms import (
        extract_frame_positions,
        frame_from_positions,
        compute_relative_transform,
    )

    try:
        # Filter to polymer atoms and strip unresolved residues
        poly = poly.poly().strip()
        if poly.size() == 0 or poly.size(Scale.RESIDUE) < 4:
            return None

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

        n_res = poly.size(Scale.RESIDUE)
        link_def = LINKING_BY_TYPE[Molecule.RNA]

        # Build sequence with global positions
        chain_data = []
        global_centroid = np.zeros(3)
        global_R = np.eye(3)

        for j in range(n_res):
            res_val = int(seq[j])

            if res_val not in atom_indices:
                continue

            sj, ej = offsets[j], offsets[j + 1]
            coords_j = coords[sj:ej]
            atoms_j = atoms[sj:ej].tolist()

            common_atoms = atom_indices[res_val]
            if not set(common_atoms).issubset(set(atoms_j)):
                continue

            remapped = _remap_to_common(coords_j, atoms_j, common_atoms)

            # Compute transform
            if j == 0 or len(chain_data) == 0:
                transform = np.zeros(6, dtype=np.float32)
                # First residue: global position is just the centroid
                global_centroid = remapped.mean(axis=0)
                global_R = np.eye(3)
            else:
                # Get previous residue info
                prev_data = chain_data[-1]
                prev_val = prev_data['res_val']
                prev_coords = prev_data['local_coords']
                prev_atoms = np.array(atom_indices[prev_val], dtype=np.int64)

                si, ei = offsets[j - 1], offsets[j]
                coords_i_aligned = coords[si:ei]
                atoms_i = atoms[si:ei].tolist()
                common_i = atom_indices[prev_val]

                if not set(common_i).issubset(set(atoms_i)):
                    transform = np.zeros(6, dtype=np.float32)
                else:
                    remapped_i = _remap_to_common(coords_i_aligned, atoms_i, common_i)
                    atoms_j_arr = np.array(common_atoms, dtype=np.int64)

                    # Transform j's coords to i's frame
                    R_j_to_i = Rs[j].T @ Rs[j - 1]
                    t_j_to_i = (origins[j] - origins[j - 1]) @ Rs[j - 1]
                    coords_j_in_i = remapped @ R_j_to_i + t_j_to_i

                    prev_pos = extract_frame_positions(remapped_i, prev_atoms, link_def.prev_frame)
                    o3p_origin, o3p_R = frame_from_positions(prev_pos)

                    next_pos = extract_frame_positions(coords_j_in_i, atoms_j_arr, link_def.next_frame)
                    p_origin, p_R = frame_from_positions(next_pos)

                    transform = compute_relative_transform(o3p_origin, o3p_R, p_origin, p_R)

                # Update global position
                global_centroid, global_R = apply_transform(
                    prev_data['global_centroid'],
                    prev_data['global_R'],
                    transform
                )

            chain_data.append({
                'res_val': res_val,
                'local_coords': remapped.astype(np.float32),
                'transform': transform.astype(np.float32),
                'global_centroid': global_centroid.astype(np.float32),
                'global_R': global_R.astype(np.float32),
                'global_orientation': rotation_to_axis_angle(global_R).astype(np.float32),
            })

        if len(chain_data) >= 4:
            if len(chain_data) > max_seq_len:
                chain_data = chain_data[:max_seq_len]
            return chain_data
        return None

    except Exception:
        return None


def build_polymer_sequences_with_global(
    dataset: PolymerDataset,
    atom_indices: dict,
    max_seq_len: int = 256,
    verbose: bool = True,
) -> list:
    """
    Build training sequences from PolymerDataset with global position information.

    Each sequence contains:
    - local_coords: coordinates in glycosidic frame
    - transforms: SE(3) transform to position each residue
    - global_centroids: assembled global centroid positions
    - global_orientations: assembled global orientations (axis-angle)
    """
    sequences = []
    skipped = 0

    for idx in range(len(dataset)):
        if verbose and idx % 200 == 0:
            print(f"  Processing chain {idx}/{len(dataset)}...")

        poly = dataset[idx]
        if poly is None:
            skipped += 1
            continue

        chain_data = process_single_chain(poly, atom_indices, max_seq_len)
        if chain_data is not None:
            sequences.append(chain_data)
        else:
            skipped += 1

    if verbose:
        print(f"Built {len(sequences)} sequences from {len(dataset)} chains ({skipped} skipped)")

    return sequences


def collate_sequences(batch: list, max_atoms: int, device: str = "cpu") -> dict:
    """Collate variable-length sequences into tensors."""
    B = len(batch)
    max_len = max(len(seq) for seq in batch)

    sequences = torch.zeros(B, max_len, dtype=torch.long, device=device)
    local_coords = torch.zeros(B, max_len, max_atoms, 3, dtype=torch.float32, device=device)
    transforms = torch.zeros(B, max_len, 6, dtype=torch.float32, device=device)
    global_centroids = torch.zeros(B, max_len, 3, dtype=torch.float32, device=device)
    global_orientations = torch.zeros(B, max_len, 3, dtype=torch.float32, device=device)
    n_atoms = torch.zeros(B, max_len, dtype=torch.long, device=device)
    padding_mask = torch.ones(B, max_len, dtype=torch.bool, device=device)

    for i, seq in enumerate(batch):
        L = len(seq)
        for j, data in enumerate(seq):
            sequences[i, j] = data['res_val']
            n_res_atoms = data['local_coords'].shape[0]
            local_coords[i, j, :n_res_atoms] = torch.tensor(data['local_coords'])
            transforms[i, j] = torch.tensor(data['transform'])
            global_centroids[i, j] = torch.tensor(data['global_centroid'])
            global_orientations[i, j] = torch.tensor(data['global_orientation'])
            n_atoms[i, j] = n_res_atoms
        padding_mask[i, :L] = False

    return {
        "sequence": sequences,
        "local_coords": local_coords,
        "transforms": transforms,
        "global_centroids": global_centroids,
        "global_orientations": global_orientations,
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
    """Train the CoordinateAR model."""
    print(f"\n=== Training CoordinateAR ===")
    print(f"  d_model={d_model}, num_layers={num_layers}, num_heads={num_heads}")
    print(f"  batch_size={batch_size}, num_epochs={num_epochs}, lr={lr}")

    n_val = max(1, len(sequences) // 10)
    indices = torch.randperm(len(sequences)).tolist()
    train_data = [sequences[i] for i in indices[n_val:]]
    val_data = [sequences[i] for i in indices[:n_val]]

    print(f"  Train: {len(train_data)}, Val: {len(val_data)}")

    config = CoordinateARConfig(
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=0.1,
    )
    model = CoordinateAR(residue_atoms, config).to(device)

    max_atoms = model.max_atoms
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
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
                batch["local_coords"],
                batch["transforms"],
                batch["global_centroids"],
                batch["global_orientations"],
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

        model.eval()
        val_loss = 0
        n_val_batches = 0

        with torch.no_grad():
            for start in range(0, len(val_data), batch_size):
                batch = val_data[start:start + batch_size]
                batch = collate_sequences(batch, max_atoms, device)

                loss = model.compute_loss(
                    batch["sequence"],
                    batch["local_coords"],
                    batch["transforms"],
                    batch["global_centroids"],
                    batch["global_orientations"],
                    batch["n_atoms"],
                    batch["padding_mask"],
                )
                val_loss += loss.item()
                n_val_batches += 1

        val_loss = val_loss / max(1, n_val_batches)
        val_losses.append(val_loss)

        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save(str(output_dir / "model"))

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}: train={train_loss:.6f}, val={val_loss:.6f}")

    print(f"\nBest validation loss: {best_val_loss:.6f}")

    np.savez(
        output_dir / "training_curves.npz",
        train_losses=train_losses,
        val_losses=val_losses,
    )

    return model


def sample_and_build(
    model: CoordinateAR,
    atom_indices: dict,
    test_sequences: list,
    output_dir: Path,
    device: str = "cuda",
):
    """Sample from model and build polymer structures."""
    print("\n=== Sampling Structures ===")

    model.eval()
    char_to_res = {'A': Residue.A, 'C': Residue.C, 'G': Residue.G, 'U': Residue.U}

    for seq_str in test_sequences:
        print(f"\nSequence: {seq_str}")

        seq_values = [char_to_res[c].value for c in seq_str.upper()]
        sequence = torch.tensor(seq_values, dtype=torch.long, device=device)

        with torch.no_grad():
            coords, transforms = model.generate(sequence, temperature=0.0)

        coords = coords[0].cpu().numpy()
        transforms = transforms[0].cpu().numpy()

        print(f"  Generated: coords shape={coords.shape}")

        try:
            poly = ciffy.Polymer()
            for i, char in enumerate(seq_str.upper()):
                res = char_to_res[char]
                n_atoms = model.residue_atoms[res.value]
                res_coords = coords[i, :n_atoms]
                res_transform = transforms[i]
                res_atoms = atom_indices.get(res.value) or atom_indices.get(str(res.value))

                if res_atoms is None:
                    continue

                atom_group = res.subset(set(res_atoms))
                res_elements = atom_group.elements().tolist()

                if i == 0:
                    poly = poly.extend(res, res_coords, atoms=res_atoms, elements=res_elements)
                else:
                    poly = poly.extend(res, res_coords, res_transform, atoms=res_atoms, elements=res_elements)

            print(f"  Built: {poly.size()} atoms, {poly.size(Scale.RESIDUE)} residues")

            # Check bond lengths
            from ciffy.biochemistry.linking import LINKING_BY_TYPE
            link_def = LINKING_BY_TYPE[Molecule.RNA]
            orig_coords = to_numpy(poly.coordinates)
            atoms_arr = to_numpy(poly.atoms)
            counts = to_numpy(poly.counts(Scale.RESIDUE))
            offsets = np.concatenate([[0], np.cumsum(counts)])

            o3p_values = to_numpy(link_def.prev_atom.index())
            p_values = to_numpy(link_def.next_atom.index())

            bond_lengths = []
            for j in range(1, poly.size(Scale.RESIDUE)):
                si, ei = offsets[j - 1], offsets[j]
                sj, ej = offsets[j], offsets[j + 1]

                atoms_i = atoms_arr[si:ei]
                atoms_j = atoms_arr[sj:ej]

                o3p_mask = np.isin(atoms_i, o3p_values)
                p_mask = np.isin(atoms_j, p_values)

                if o3p_mask.any() and p_mask.any():
                    o3p_pos = orig_coords[si:ei][o3p_mask.argmax()]
                    p_pos = orig_coords[sj:ej][p_mask.argmax()]
                    dist = np.linalg.norm(p_pos - o3p_pos)
                    bond_lengths.append(dist)

            if bond_lengths:
                print(f"  O3'-P bonds: mean={np.mean(bond_lengths):.2f} Å, "
                      f"range=[{np.min(bond_lengths):.2f}, {np.max(bond_lengths):.2f}]")

            poly = poly._clone(pdb_id=f'GAR_{seq_str[:4]}')
            poly.write(str(output_dir / f"sampled_{seq_str}.cif"))
            print(f"  Saved: {output_dir / f'sampled_{seq_str}.cif'}")

        except Exception as e:
            print(f"  Error: {e}")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str,
                        default="/home/hblair/data/structures/rna",
                        help="Directory with CIF files")
    parser.add_argument("--output-dir", type=str, default="outputs/global_ar",
                        help="Output directory")
    parser.add_argument("--max-chains", type=int, default=None,
                        help="Maximum number of chains to use")
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

    # Use PolymerDataset for proper chain-level iteration
    print(f"\n=== Loading Dataset ===")
    dataset = PolymerDataset(
        args.data_dir,
        scale=Scale.CHAIN,           # Iterate over individual chains
        molecule_types=Molecule.RNA,  # Only RNA chains
        min_residues=4,              # At least 4 residues
        backend="numpy",
        limit=args.max_chains,
    )

    print(f"Found {len(dataset)} RNA chains")

    if len(dataset) == 0:
        print("No chains found, exiting")
        return

    # Get residue atom counts from a subset of chains
    cif_paths = sorted(glob(os.path.join(args.data_dir, "*.cif")))[:50]
    cif_paths = [Path(p) for p in cif_paths]

    print("\n=== Getting Residue Atom Counts ===")
    residue_atoms, atom_indices = get_residue_atom_counts(cif_paths)

    if len(residue_atoms) == 0:
        print("No residue atom counts found, exiting")
        return

    with open(output_dir / "atom_indices.json", "w") as f:
        json.dump(atom_indices, f, indent=2)

    print("\n=== Building Training Sequences ===")
    sequences = build_polymer_sequences_with_global(dataset, atom_indices)

    if len(sequences) == 0:
        print("No sequences built, exiting")
        return

    lengths = [len(s) for s in sequences]
    print(f"  Sequence lengths: mean={np.mean(lengths):.1f}, "
          f"min={np.min(lengths)}, max={np.max(lengths)}")

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

    test_sequences = ["ACGU", "GGCGCG", "ACGUACGU"]
    sample_and_build(model, atom_indices, test_sequences, output_dir, device)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
