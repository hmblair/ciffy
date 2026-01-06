#!/usr/bin/env python3
"""
Train AtomAR model for all-atom conditioned structure generation.

AtomAR predicts residue coordinates one residue at a time, conditioned on
all atoms from previously placed residues. This provides rich spatial
context while avoiding arbitrary atom ordering within residues.

Steps:
1. Use PolymerDataset for chain-level iteration
2. Extract atom-level data (types, elements, coordinates)
3. Track residue boundaries for per-residue loss computation
4. Train AtomAR with teacher forcing
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
from ciffy.nn import PolymerDataset
from ciffy.nn.autoregressive import AtomAR, AtomARConfig
from ciffy.biochemistry import Residue, Scale, Molecule, Atom
from ciffy.backend import to_numpy


def get_residue_atom_info(cif_paths: list, min_samples: int = 100) -> dict:
    """
    Get common atom sets for each residue type.

    Returns dict mapping residue value to list of atom indices.
    """
    from collections import defaultdict

    atom_counts = defaultdict(lambda: defaultdict(int))

    for path in cif_paths[:100]:
        try:
            poly = ciffy.load(str(path)).molecule_type(Molecule.RNA).poly().strip()
            if poly.size() == 0:
                continue

            seq = to_numpy(poly.sequence)
            atoms = to_numpy(poly.atoms)
            counts = to_numpy(poly.counts(Scale.RESIDUE))
            offsets = np.concatenate([[0], np.cumsum(counts)])

            for i in range(poly.size(Scale.RESIDUE)):
                res_val = int(seq[i])
                s, e = offsets[i], offsets[i + 1]
                res_atoms = frozenset(atoms[s:e].tolist())
                atom_counts[res_val][res_atoms] += 1

        except Exception:
            continue

    # Get most common atom set for each residue
    atom_indices = {}
    for res_val, counts in atom_counts.items():
        if not counts or res_val < 0:
            continue
        # Sort by count, take most common
        most_common = max(counts.items(), key=lambda x: x[1])
        if most_common[1] >= min_samples:
            try:
                res_name = Residue.from_index(res_val).name
            except KeyError:
                continue  # Skip unknown residue types
            atom_indices[res_val] = sorted(most_common[0])
            print(f"  {res_name}: {len(atom_indices[res_val])} atoms, {most_common[1]} samples")

    return atom_indices


def process_chain_for_atom_ar(
    poly,
    atom_indices: dict,
    max_residues: int = 256,
) -> dict | None:
    """
    Process a single chain into training data for AtomAR.

    Returns dict with:
    - atoms: (n_atoms,) atom type indices
    - residues_per_atom: (n_atoms,) residue type for each atom
    - elements: (n_atoms,) element type indices
    - coords: (n_atoms, 3) coordinates
    - residue_types: (n_residues,) residue type sequence
    - residue_boundaries: (n_residues+1,) cumulative atom counts
    """
    try:
        poly = poly.poly().strip()
        if poly.size() == 0 or poly.size(Scale.RESIDUE) < 6:  # Need 6 to have 4 internal
            return None

        # Center the structure
        poly, _ = poly.center()

        seq = to_numpy(poly.sequence)
        atoms_arr = to_numpy(poly.atoms)
        elements_arr = to_numpy(poly.elements)
        coords_arr = to_numpy(poly.coordinates)
        counts = to_numpy(poly.counts(Scale.RESIDUE))
        offsets = np.concatenate([[0], np.cumsum(counts)])

        n_res = poly.size(Scale.RESIDUE)

        # Filter residues to those with complete atom sets
        # Skip first and last residue (terminal atoms differ)
        valid_atoms = []
        valid_residues = []
        valid_elements = []
        valid_coords = []
        valid_res_types = []
        boundaries = [0]

        for i in range(1, n_res - 1):  # Skip terminals
            res_val = int(seq[i])
            if res_val not in atom_indices:
                continue

            s, e = offsets[i], offsets[i + 1]
            res_atoms = atoms_arr[s:e].tolist()
            res_elements = elements_arr[s:e]
            res_coords = coords_arr[s:e]

            # Check if all common atoms are present
            common_atoms = atom_indices[res_val]
            if not set(common_atoms).issubset(set(res_atoms)):
                continue

            # Remap to common atom order
            atom_to_idx = {a: i for i, a in enumerate(res_atoms)}
            indices = [atom_to_idx[a] for a in common_atoms]

            remapped_atoms = np.array(common_atoms, dtype=np.int64)
            remapped_elements = res_elements[indices]
            remapped_coords = res_coords[indices]

            valid_atoms.append(remapped_atoms)
            valid_elements.append(remapped_elements)
            valid_coords.append(remapped_coords)
            valid_residues.append(np.full(len(common_atoms), res_val, dtype=np.int64))
            valid_res_types.append(res_val)
            boundaries.append(boundaries[-1] + len(common_atoms))

        if len(valid_res_types) < 4:
            return None

        # Truncate if too long
        if len(valid_res_types) > max_residues:
            valid_res_types = valid_res_types[:max_residues]
            valid_atoms = valid_atoms[:max_residues]
            valid_residues = valid_residues[:max_residues]
            valid_elements = valid_elements[:max_residues]
            valid_coords = valid_coords[:max_residues]
            boundaries = [0] + [sum(len(a) for a in valid_atoms[:i+1]) for i in range(len(valid_atoms))]

        return {
            'atoms': np.concatenate(valid_atoms).astype(np.int64),
            'residues_per_atom': np.concatenate(valid_residues).astype(np.int64),
            'elements': np.concatenate(valid_elements).astype(np.int64),
            'coords': np.concatenate(valid_coords).astype(np.float32),
            'residue_types': np.array(valid_res_types, dtype=np.int64),
            'residue_boundaries': np.array(boundaries, dtype=np.int64),
        }

    except Exception:
        return None


def build_training_data(
    dataset: PolymerDataset,
    atom_indices: dict,
    max_residues: int = 256,
    verbose: bool = True,
) -> list:
    """Build training data from PolymerDataset."""
    data = []
    skipped = 0

    for idx in range(len(dataset)):
        if verbose and idx % 500 == 0:
            print(f"  Processing chain {idx}/{len(dataset)}...")

        poly = dataset[idx]
        if poly is None:
            skipped += 1
            continue

        chain_data = process_chain_for_atom_ar(poly, atom_indices, max_residues)
        if chain_data is not None:
            data.append(chain_data)
        else:
            skipped += 1

    if verbose:
        print(f"Built {len(data)} chains from {len(dataset)} ({skipped} skipped)")

    return data


def collate_batch(batch: list, device: str = "cpu") -> dict:
    """Collate variable-length chains into padded tensors."""
    B = len(batch)

    # Find max sizes
    max_atoms = max(len(d['atoms']) for d in batch)
    max_residues = max(len(d['residue_types']) for d in batch)

    # Initialize tensors
    atoms = torch.zeros(B, max_atoms, dtype=torch.long, device=device)
    residues_per_atom = torch.zeros(B, max_atoms, dtype=torch.long, device=device)
    elements = torch.zeros(B, max_atoms, dtype=torch.long, device=device)
    coords = torch.zeros(B, max_atoms, 3, dtype=torch.float32, device=device)
    atom_mask = torch.zeros(B, max_atoms, dtype=torch.bool, device=device)

    residue_types = torch.zeros(B, max_residues, dtype=torch.long, device=device)
    residue_boundaries = torch.zeros(B, max_residues + 1, dtype=torch.long, device=device)
    residue_mask = torch.zeros(B, max_residues, dtype=torch.bool, device=device)

    for i, d in enumerate(batch):
        n_atoms = len(d['atoms'])
        n_res = len(d['residue_types'])

        atoms[i, :n_atoms] = torch.from_numpy(d['atoms'])
        residues_per_atom[i, :n_atoms] = torch.from_numpy(d['residues_per_atom'])
        elements[i, :n_atoms] = torch.from_numpy(d['elements'])
        coords[i, :n_atoms] = torch.from_numpy(d['coords'])
        atom_mask[i, :n_atoms] = True

        residue_types[i, :n_res] = torch.from_numpy(d['residue_types'])
        residue_boundaries[i, :n_res + 1] = torch.from_numpy(d['residue_boundaries'])
        residue_mask[i, :n_res] = True

    return {
        'atoms': atoms,
        'residues_per_atom': residues_per_atom,
        'elements': elements,
        'coords': coords,
        'atom_mask': atom_mask,
        'residue_types': residue_types,
        'residue_boundaries': residue_boundaries,
        'residue_mask': residue_mask,
    }


def train_model(
    train_data: list,
    val_data: list,
    output_dir: Path,
    config: AtomARConfig,
    batch_size: int = 16,
    num_epochs: int = 100,
    lr: float = 1e-4,
    device: str = "cuda",
):
    """Train the AtomAR model."""
    print(f"\n=== Training AtomAR ===")
    print(f"  d_model={config.d_model}, encoder_layers={config.num_encoder_layers}")
    print(f"  decoder_layers={config.num_decoder_layers}, heads={config.num_heads}")
    print(f"  batch_size={batch_size}, num_epochs={num_epochs}, lr={lr}")
    print(f"  Train: {len(train_data)}, Val: {len(val_data)}")

    model = AtomAR(config).to(device)

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
            batch = collate_batch(batch, device)

            optimizer.zero_grad()
            loss = model.compute_loss(
                batch['atoms'],
                batch['residues_per_atom'],
                batch['elements'],
                batch['coords'],
                batch['atom_mask'],
                batch['residue_types'],
                batch['residue_boundaries'],
                batch['residue_mask'],
            )

            if torch.isnan(loss):
                print(f"  WARNING: NaN loss at epoch {epoch+1}, batch {n_batches}")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        train_loss = epoch_loss / max(1, n_batches)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0
        n_val_batches = 0

        with torch.no_grad():
            for start in range(0, len(val_data), batch_size):
                batch = val_data[start:start + batch_size]
                batch = collate_batch(batch, device)

                loss = model.compute_loss(
                    batch['atoms'],
                    batch['residues_per_atom'],
                    batch['elements'],
                    batch['coords'],
                    batch['atom_mask'],
                    batch['residue_types'],
                    batch['residue_boundaries'],
                    batch['residue_mask'],
                )

                if not torch.isnan(loss):
                    val_loss += loss.item()
                    n_val_batches += 1

        val_loss = val_loss / max(1, n_val_batches)
        val_losses.append(val_loss)

        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save(str(output_dir / "model"))

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}: train={train_loss:.4f}, val={val_loss:.4f}")

    print(f"\nBest validation loss: {best_val_loss:.4f}")

    np.savez(
        output_dir / "training_curves.npz",
        train_losses=train_losses,
        val_losses=val_losses,
    )

    return model


def sample_structure(
    model: AtomAR,
    atom_indices: dict,
    seq_str: str,
    device: str = "cuda",
    temperature: float = 0.0,
):
    """Generate a structure from sequence."""
    char_to_res = {'A': Residue.A, 'C': Residue.C, 'G': Residue.G, 'U': Residue.U}

    # Build input tensors
    residue_types = []
    atoms_per_residue = []
    elements_per_residue = []

    for char in seq_str.upper():
        res = char_to_res[char]
        res_val = res.value

        if res_val not in atom_indices:
            print(f"Warning: residue {char} not in atom_indices, skipping")
            continue

        atom_list = atom_indices[res_val]
        atom_group = res.subset(set(atom_list))
        elem_list = atom_group.elements().tolist()

        residue_types.append(res_val)
        atoms_per_residue.append(torch.tensor(atom_list, dtype=torch.long, device=device))
        elements_per_residue.append(torch.tensor(elem_list, dtype=torch.long, device=device))

    if len(residue_types) == 0:
        return None

    residue_types = torch.tensor(residue_types, dtype=torch.long, device=device)

    # Generate
    model.eval()
    with torch.no_grad():
        coords_list = model.generate(
            residue_types,
            atoms_per_residue,
            elements_per_residue,
            temperature=temperature,
        )

    return coords_list, atoms_per_residue, elements_per_residue, residue_types


def build_polymer_from_coords(
    coords_list: list,
    atoms_per_residue: list,
    elements_per_residue: list,
    residue_types: torch.Tensor,
    atom_indices: dict,
) -> ciffy.Polymer:
    """Build a Polymer from generated coordinates."""
    poly = ciffy.Polymer()

    for i in range(len(coords_list)):
        res_val = residue_types[0, i].item()
        res = Residue.from_index(res_val)

        coords = coords_list[i][0].cpu().numpy()  # Remove batch dim
        atom_list = atoms_per_residue[i][0].cpu().numpy().tolist()
        elem_list = elements_per_residue[i][0].cpu().numpy().tolist()

        if i == 0:
            poly = poly.extend(res, coords, atoms=atom_list, elements=elem_list)
        else:
            # For subsequent residues, use identity transform (coords are in global frame)
            transform = np.zeros(6, dtype=np.float32)
            poly = poly.extend(res, coords, transform, atoms=atom_list, elements=elem_list)

    return poly


def evaluate_bond_lengths(poly) -> dict:
    """Evaluate O3'-P bond lengths in a polymer."""
    from ciffy.biochemistry.linking import LINKING_BY_TYPE

    link_def = LINKING_BY_TYPE[Molecule.RNA]
    coords = to_numpy(poly.coordinates)
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
            o3p_pos = coords[si:ei][o3p_mask.argmax()]
            p_pos = coords[sj:ej][p_mask.argmax()]
            dist = np.linalg.norm(p_pos - o3p_pos)
            bond_lengths.append(dist)

    if bond_lengths:
        return {
            'mean': np.mean(bond_lengths),
            'std': np.std(bond_lengths),
            'min': np.min(bond_lengths),
            'max': np.max(bond_lengths),
        }
    return {}


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str,
                        default="/home/hmblair/data/rna",
                        help="Directory with CIF files")
    parser.add_argument("--output-dir", type=str, default="outputs/atom_ar",
                        help="Output directory")
    parser.add_argument("--max-chains", type=int, default=None,
                        help="Maximum number of chains to use")
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-encoder-layers", type=int, default=4)
    parser.add_argument("--num-decoder-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load dataset
    print(f"\n=== Loading Dataset ===")
    dataset = PolymerDataset(
        args.data_dir,
        scale=Scale.CHAIN,
        molecule_types=Molecule.RNA,
        min_residues=4,
        backend="numpy",
        limit=args.max_chains,
    )
    print(f"Found {len(dataset)} RNA chains")

    if len(dataset) == 0:
        print("No chains found, exiting")
        return

    # Get residue atom info
    cif_paths = sorted(glob(os.path.join(args.data_dir, "*.cif")))[:100]
    print("\n=== Getting Residue Atom Info ===")
    atom_indices = get_residue_atom_info(cif_paths)

    if len(atom_indices) == 0:
        print("No residue atom info found, exiting")
        return

    # Save atom indices
    with open(output_dir / "atom_indices.json", "w") as f:
        json.dump({str(k): v for k, v in atom_indices.items()}, f, indent=2)

    # Build training data
    print("\n=== Building Training Data ===")
    all_data = build_training_data(dataset, atom_indices, max_residues=64)  # Limit to avoid OOM

    if len(all_data) < 10:
        print(f"Only {len(all_data)} chains, need at least 10")
        return

    # Print statistics
    n_residues = [len(d['residue_types']) for d in all_data]
    n_atoms = [len(d['atoms']) for d in all_data]
    print(f"  Residues per chain: mean={np.mean(n_residues):.1f}, "
          f"min={np.min(n_residues)}, max={np.max(n_residues)}")
    print(f"  Atoms per chain: mean={np.mean(n_atoms):.1f}, "
          f"min={np.min(n_atoms)}, max={np.max(n_atoms)}")

    # Split data
    n_val = max(1, len(all_data) // 10)
    indices = torch.randperm(len(all_data)).tolist()
    train_data = [all_data[i] for i in indices[n_val:]]
    val_data = [all_data[i] for i in indices[:n_val]]

    # Configure model
    config = AtomARConfig(
        d_model=args.d_model,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        num_heads=args.num_heads,
        dropout=0.1,
        max_atoms=min(max(n_atoms) + 100, 2048),  # Cap to avoid OOM
        max_residues=min(max(n_residues) + 10, 128),
    )

    # Train
    model = train_model(
        train_data,
        val_data,
        output_dir,
        config,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        device=device,
    )

    # Sample test structures
    print("\n=== Sampling Test Structures ===")
    test_sequences = ["ACGU", "GGCGCG", "ACGUACGU"]

    for seq_str in test_sequences:
        print(f"\nSequence: {seq_str}")

        result = sample_structure(model, atom_indices, seq_str, device, temperature=0.0)
        if result is None:
            print("  Failed to generate")
            continue

        coords_list, atoms_per_res, elems_per_res, res_types = result

        # Build polymer (note: coords are in global frame, no transforms needed)
        poly = ciffy.Polymer()
        for i in range(len(coords_list)):
            res_val = res_types[i].item()
            res = Residue.from_index(res_val)

            coords = coords_list[i][0].cpu().numpy()
            atom_list = atoms_per_res[i][0].cpu().numpy().tolist()
            elem_list = elems_per_res[i][0].cpu().numpy().tolist()

            # First residue: no transform
            # Subsequent: we need to handle positioning differently
            # For now, just place all coordinates as given (they're predicted in global frame)
            if i == 0:
                poly = poly.extend(res, coords, atoms=atom_list, elements=elem_list)
            else:
                # Use identity transform since coords are already global
                transform = np.zeros(6, dtype=np.float32)
                poly = poly.extend(res, coords, transform, atoms=atom_list, elements=elem_list)

        print(f"  Built: {poly.size()} atoms, {poly.size(Scale.RESIDUE)} residues")

        # Evaluate
        stats = evaluate_bond_lengths(poly)
        if stats:
            print(f"  O3'-P bonds: mean={stats['mean']:.2f} A, std={stats['std']:.2f} A")
            print(f"    range=[{stats['min']:.2f}, {stats['max']:.2f}]")

        # Save
        poly = poly._clone(pdb_id=f'AAR_{seq_str[:6]}')
        poly.write(str(output_dir / f"sampled_{seq_str}.cif"))
        print(f"  Saved: {output_dir / f'sampled_{seq_str}.cif'}")

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
