"""Generate prediction plots for equivariant reactivity model.

Loads trained model from equi_split checkpoint and generates:
- Scatter plots: predicted vs target for SHAPE and DMS (all samples)
- Line plots: per-residue profiles for individual samples

Output saved to: /scratch/users/hmblair/ciffy/equi_react/equi_split/figures/

Usage:
    rex sherlock -d --gpu scripts/plot_predictions.py
"""

import torch
import numpy as np
import hmbp
import h5py
import os

from ciffy import Scale, Molecule, Reduction
from ciffy.nn import PolymerDataset, PolymerEmbedding
from ciffy.nn.geometric import EquivariantTransformer, Repr
from ciffy.rna import ReactivityIndex


def normalize(x, dim=0):
    """Normalize tensor to [0, 1] range."""
    x_min = x.min(dim=dim, keepdim=True).values
    x_max = x.max(dim=dim, keepdim=True).values
    return (x - x_min) / (x_max - x_min + 1e-8)


class EquivariantReactivityModel(torch.nn.Module):
    """Small equivariant model for reactivity prediction."""

    def __init__(self):
        super().__init__()
        lvals = [0, 1]
        embed_dim = 32

        in_repr = Repr(lvals=lvals, mult=embed_dim)
        hidden_repr = Repr(lvals=lvals, mult=8)
        out_repr = Repr(lvals=lvals, mult=8)

        self.embedding = PolymerEmbedding(
            scale=Scale.ATOM,
            atom_dim=embed_dim // 2,
            residue_dim=embed_dim // 4,
            element_dim=embed_dim // 4,
            dropout=0.0,
        )
        self.transformer = EquivariantTransformer(
            in_repr=in_repr,
            out_repr=out_repr,
            hidden_repr=hidden_repr,
            hidden_layers=2,
            edge_dim=16,
            edge_hidden_dim=32,
            k_neighbors=64,
            nheads=4,
            dropout=0.0,
            attn_dropout=0.0,
            transition=True,
            residual_scale=0.5,
            qk_norm=True,
            seq_pos_dim=8,
            rank=0,
        )
        self.out_proj = torch.nn.Linear(8, 2)
        self.in_repr = in_repr

    def forward(self, polymer):
        n_atoms = polymer.size(Scale.ATOM)
        if n_atoms == 0:
            return polymer.coordinates.new_zeros(0, 2)
        embed = self.embedding(polymer)
        repr_dim = self.in_repr.dim()
        features = embed.new_zeros(n_atoms, embed.size(1), repr_dim)
        features[:, :, 0] = embed
        seq_pos = polymer.membership(Scale.RESIDUE)
        output = self.transformer(polymer.coordinates, features, seq_pos=seq_pos)
        atom_features = output[:, :, 0]
        residue_features = polymer.reduce(atom_features, Scale.RESIDUE, Reduction.MEAN)
        return self.out_proj(residue_features)


def main():
    # Paths
    model_path = "/scratch/users/hmblair/ciffy/equi_react/equi_split/best_model.pt"
    h5_path = "/scratch/users/hmblair/profiles/pdb130/train2_pdb130.hdf5"
    structure_dir = "/scratch/users/hmblair/structures/pdb130"
    output_dir = "/scratch/users/hmblair/ciffy/equi_react/equi_split/figures"

    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"Val metrics: {checkpoint['val_metrics']}")

    model = EquivariantReactivityModel().to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    print("Model loaded")

    # Build reactivity index
    index = ReactivityIndex()
    with h5py.File(h5_path, "r") as f:
        ids = [x.decode() if isinstance(x, bytes) else str(x) for x in f["id_strings"][0]]
        sequences = [x.decode() if isinstance(x, bytes) else str(x) for x in f["sequences"][0]]
        reactivity = f["r_norm"][:]
        for i, (cid, seq) in enumerate(zip(ids, sequences)):
            index.add(cid, seq, reactivity[i])
    print(f"Index: {len(index)} entries")

    # Create dataset
    dataset = PolymerDataset(
        structure_dir,
        scale=Scale.CHAIN,
        molecule_types=Molecule.RNA,
        min_residues=10,
        max_chains=2,
    )
    print(f"Dataset: {len(dataset)} chains")

    # Collect predictions
    all_pred_shape = []
    all_target_shape = []
    all_pred_dms = []
    all_target_dms = []
    sample_plots = []

    n_plotted = 0
    for idx in range(len(dataset)):
        if n_plotted >= 6:
            break

        polymer = dataset[idx]
        if polymer is None:
            continue
        polymer = polymer.strip()
        if polymer.empty():
            continue
        polymer = polymer.torch().to(device)

        match = index.match(polymer)
        if match is None:
            continue

        target = match.reactivity.clone()

        # Normalize target
        for c in range(2):
            mask = ~torch.isnan(target[:, c])
            if mask.sum() > 0:
                target[mask, c] = normalize(target[mask, c])

        with torch.no_grad():
            pred = model(polymer)
            pred = normalize(pred, dim=0)

        # Get valid masks
        shape_mask = ~torch.isnan(target[:, 0])
        dms_mask = ~torch.isnan(target[:, 1])

        if shape_mask.sum() >= 10 and dms_mask.sum() >= 10:
            pred_np = pred.cpu().numpy()
            target_np = target.cpu().numpy()

            # Collect for scatter plot
            all_pred_shape.extend(pred_np[shape_mask.cpu().numpy(), 0])
            all_target_shape.extend(target_np[shape_mask.cpu().numpy(), 0])
            all_pred_dms.extend(pred_np[dms_mask.cpu().numpy(), 1])
            all_target_dms.extend(target_np[dms_mask.cpu().numpy(), 1])

            # Save individual sample plot
            if n_plotted < 6:
                sample_plots.append({
                    "name": match.name,
                    "pred_shape": pred_np[:, 0],
                    "target_shape": target_np[:, 0],
                    "pred_dms": pred_np[:, 1],
                    "target_dms": target_np[:, 1],
                })
                n_plotted += 1
                print(f"Sample {n_plotted}: {match.name}")

    # Plot scatter: pred vs target (all samples)
    hmbp.quick_scatter(
        all_target_shape,
        all_pred_shape,
        title="SHAPE: Predicted vs Target",
        xlabel="Target (normalized)",
        ylabel="Predicted (normalized)",
        path=f"{output_dir}/scatter_shape.png",
    )
    print("Saved scatter_shape.png")

    hmbp.quick_scatter(
        all_target_dms,
        all_pred_dms,
        title="DMS: Predicted vs Target",
        xlabel="Target (normalized)",
        ylabel="Predicted (normalized)",
        path=f"{output_dir}/scatter_dms.png",
    )
    print("Saved scatter_dms.png")

    # Plot individual samples: pred and target as lines
    for i, sample in enumerate(sample_plots[:3]):
        x = np.arange(len(sample["pred_shape"]))

        # SHAPE
        hmbp.quick_lines(
            [sample["target_shape"], sample["pred_shape"]],
            x,
            labels=["Target", "Predicted"],
            title=f"{sample['name']} - SHAPE",
            xlabel="Residue",
            ylabel="Reactivity (normalized)",
            path=f"{output_dir}/sample_{i+1}_shape.png",
        )

        # DMS
        hmbp.quick_lines(
            [sample["target_dms"], sample["pred_dms"]],
            x,
            labels=["Target", "Predicted"],
            title=f"{sample['name']} - DMS",
            xlabel="Residue",
            ylabel="Reactivity (normalized)",
            path=f"{output_dir}/sample_{i+1}_dms.png",
        )
        print(f"Saved sample_{i+1} plots")

    print("Done!")


if __name__ == "__main__":
    main()
