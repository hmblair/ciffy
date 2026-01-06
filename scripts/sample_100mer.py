#!/usr/bin/env python3
"""Sample a 100-mer from the trained CoordinateARModel model."""

import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ciffy.nn import CoordinateARModel
from ciffy.biochemistry import Residue, Scale, Molecule
from ciffy.backend import to_numpy
import ciffy
import json


def main():
    model_dir = Path("outputs/global_ar_full")

    # Load the trained model
    print("Loading model...")
    model = CoordinateARModel.load(str(model_dir / "model"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    # Load atom indices
    with open(model_dir / "atom_indices.json") as f:
        atom_indices = json.load(f)
    atom_indices = {int(k): v for k, v in atom_indices.items()}

    # Generate a random 100-mer sequence
    np.random.seed(42)
    seq_chars = [np.random.choice(['A', 'C', 'G', 'U']) for _ in range(100)]
    seq_str = ''.join(seq_chars)
    print(f"Sequence: {seq_str[:20]}...{seq_str[-20:]} (100 nt)")

    char_to_res = {'A': Residue.A, 'C': Residue.C, 'G': Residue.G, 'U': Residue.U}
    seq_values = [char_to_res[c].value for c in seq_str]
    sequence = torch.tensor(seq_values, dtype=torch.long, device=device)

    # Generate
    print("Generating structure...")
    with torch.no_grad():
        coords, transforms = model.generate(sequence, temperature=0.8)

    coords = coords[0].cpu().numpy()
    transforms = transforms[0].cpu().numpy()
    print(f"Generated: coords shape={coords.shape}")

    # Build polymer
    poly = ciffy.Polymer()
    for i, char in enumerate(seq_str):
        res = char_to_res[char]
        n_atoms = model.residue_atoms[res.value]
        res_coords = coords[i, :n_atoms]
        res_transform = transforms[i]
        res_atoms = atom_indices.get(res.value)

        if res_atoms is None:
            continue

        atom_group = res.subset(set(res_atoms))
        res_elements = atom_group.elements().tolist()

        if i == 0:
            poly = poly.extend(res, res_coords, atoms=res_atoms, elements=res_elements)
        else:
            poly = poly.extend(res, res_coords, res_transform, atoms=res_atoms, elements=res_elements)

    print(f"Built: {poly.size()} atoms, {poly.size(Scale.RESIDUE)} residues")

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
        print(f"O3'-P bonds: mean={np.mean(bond_lengths):.2f} A, std={np.std(bond_lengths):.2f} A")
        print(f"  range=[{np.min(bond_lengths):.2f}, {np.max(bond_lengths):.2f}]")

    # Save
    poly = poly._clone(pdb_id='GAR_100mer')
    output_path = model_dir / "sampled_100mer.cif"
    poly.write(str(output_path))
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
