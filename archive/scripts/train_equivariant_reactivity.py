"""
Train SO(3)-equivariant transformer to predict SHAPE and DMS reactivity.

See EQUIVARIANT_REACTIVITY.md for experiment conclusions and findings.

Key findings:
- dropout=0 required (causes NaN gradients with equivariant ops)
- Use --val-split 0.2 (pdb130/pdb240 have distribution mismatch)
- rank=0 is most efficient (same performance, fewer params)
- Best config: hidden-mult=8, hidden-layers=2, k-neighbors=64

This script replicates the SGNM training but uses an equivariant transformer
with l=0 (scalar) outputs instead of a trainable Gaussian Network Model.

Data loading uses ReactivityIndex to match structure sequences to reactivity
profiles. Sequences come from FASTA files, reactivities from HDF5 files.

Usage:
    # Local test (CPU) - requires local data paths
    python scripts/train_equivariant_reactivity.py \\
        --data /path/to/structures \\
        --h5-train /path/to/wt-profiles.h5 \\
        --fasta-train /path/to/ref.fasta \\
        --epochs 1

    # Sherlock GPU training (uses default sherlock paths)
    rex sherlock -d --gpu scripts/train_equivariant_reactivity.py -- --name experiment1

    # Sherlock with different rank values for comparison
    rex sherlock -d --gpu scripts/train_equivariant_reactivity.py -- --rank 0 --name rank0
    rex sherlock -d --gpu scripts/train_equivariant_reactivity.py -- --rank 1 --name rank1
    rex sherlock -d --gpu scripts/train_equivariant_reactivity.py -- --rank null --name full
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import ciffy
from ciffy import Scale, Molecule, Reduction
from ciffy.nn import PolymerDataset, PolymerEmbedding, configure_precision
from ciffy.nn.geometric import (
    EquivariantTransformer,
    Repr,
    build_knn_graph,
)
from ciffy.rna import ReactivityIndex


# ============================================================================
# Data Loading
# ============================================================================

def load_fasta(fasta_path: str) -> dict[str, str]:
    """Load sequences from FASTA file.

    Handles two ID formats:
    - Simple: >1GID_A
    - With suffix: >5TZS_z_wt (pdb)  -> extracts 5TZS_z

    Args:
        fasta_path: Path to FASTA file.

    Returns:
        Dictionary mapping ID -> sequence.
    """
    sequences = {}
    current_id = None
    current_seq = []

    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                # Save previous sequence
                if current_id is not None:
                    sequences[current_id] = "".join(current_seq)

                # Parse header - handle both formats
                header = line[1:].split()[0]  # Take first part before space
                if "_wt" in header:
                    # Format: 5TZS_z_wt -> 5TZS_z
                    header = header.split("_wt")[0]
                current_id = header
                current_seq = []
            else:
                current_seq.append(line)

        # Save last sequence
        if current_id is not None:
            sequences[current_id] = "".join(current_seq)

    return sequences


def build_reactivity_index(h5_path: str, fasta_path: str | None = None) -> ReactivityIndex:
    """Build ReactivityIndex from HDF5 reactivity data.

    Supports two HDF5 formats:
    1. Old format (e.g., train2_pdb130.hdf5): has embedded sequences
       - Keys: id_strings, sequences, r_norm
       - fasta_path is ignored
    2. New format (e.g., wt-profiles.h5): needs external FASTA
       - Keys: ids, PDB130-2A3/reactivity, etc.
       - fasta_path is required

    Args:
        h5_path: Path to HDF5 file with reactivity data.
        fasta_path: Path to FASTA file with sequences (for new format only).

    Returns:
        ReactivityIndex with sequence-matched entries.
    """
    index = ReactivityIndex()

    with h5py.File(h5_path, "r") as f:
        # Old format with embedded sequences
        if "sequences" in f:
            ids = f["id_strings"][0]
            ids = [x.decode() if isinstance(x, bytes) else str(x) for x in ids]
            sequences = f["sequences"][0]
            sequences = [x.decode() if isinstance(x, bytes) else str(x) for x in sequences]
            reactivity = f["r_norm"][:]  # (N, seq_len, 2) - SHAPE and DMS

            for i, (cid, seq) in enumerate(zip(ids, sequences)):
                reac = reactivity[i]  # (seq_len, 2)
                index.add(cid, seq, reac)

            print(f"  Built ReactivityIndex: {len(index)} entries (old format)")
            return index

        # New format - requires FASTA
        if fasta_path is None:
            raise ValueError(
                "New format HDF5 requires fasta_path. "
                "Use old format (train2_pdb130.hdf5) or provide FASTA."
            )

        T7_OFFSET = 7  # T7 promoter length to skip
        sequences = load_fasta(fasta_path)
        print(f"  Loaded {len(sequences)} sequences from FASTA")

        ids = f["ids"][:]
        ids = [x.decode() if isinstance(x, bytes) else str(x) for x in ids]

        # Detect format (PDB130 vs PDB240)
        if "PDB130-2A3/reactivity" in f:
            reac_shape = f["PDB130-2A3/reactivity"][:]
            reac_dms = f["PDB130-DMS/reactivity"][:]
        elif "PDB240-2A3/reactivity" in f:
            reac_shape = f["PDB240-2A3/reactivity"][:]
            reac_dms = f["PDB240-DMS/reactivity"][:]
        else:
            raise ValueError(f"Unknown HDF5 format in {h5_path}")

        n_matched = 0
        for i, cid in enumerate(ids):
            if cid not in sequences:
                continue
            seq = sequences[cid]
            seq_len = len(seq)
            reac_len = reac_shape.shape[1]

            # Determine alignment based on length difference
            # Case 1: Reactivity has T7 prefix (reac_len = seq_len + 7) -> slice reactivity
            # Case 2: Sequence has T7 prefix (seq_len = reac_len + 7) -> slice sequence
            # Case 3: Same length -> no slicing needed
            if reac_len == seq_len + T7_OFFSET:
                # Reactivity has T7, slice it off
                shape_i = reac_shape[i, T7_OFFSET:T7_OFFSET + seq_len]
                dms_i = reac_dms[i, T7_OFFSET:T7_OFFSET + seq_len]
            elif seq_len == reac_len + T7_OFFSET:
                # Sequence has T7, slice it off
                seq = seq[T7_OFFSET:]
                seq_len = len(seq)
                shape_i = reac_shape[i, :seq_len]
                dms_i = reac_dms[i, :seq_len]
            elif reac_len == seq_len:
                # Same length, no adjustment needed
                shape_i = reac_shape[i]
                dms_i = reac_dms[i]
            else:
                # Length mismatch we can't handle
                print(f"Warning: Skipping {cid} - length mismatch (reac={reac_len}, seq={seq_len})")
                continue

            # Stack as (seq_len, 2) - SHAPE and DMS
            reac = np.stack([shape_i, dms_i], axis=1)
            index.add(cid, seq, reac)
            n_matched += 1

    print(f"  Built ReactivityIndex: {n_matched} entries (matched {n_matched}/{len(ids)} IDs)")
    return index


# ============================================================================
# Model
# ============================================================================

class SimpleReactivityModel(nn.Module):
    """Simple MLP baseline for predicting SHAPE and DMS reactivity.

    Uses only residue-level features (no coordinates/equivariance).
    """

    def __init__(
        self,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        hidden_layers: int = 3,
        dropout: float = 0.1,
        **kwargs,  # Ignore equivariant-specific args
    ):
        super().__init__()

        # Residue-level embedding
        self.embedding = PolymerEmbedding(
            scale=Scale.RESIDUE,
            residue_dim=embed_dim,
            dropout=dropout,
        )

        # Simple MLP
        layers = []
        in_dim = embed_dim
        for _ in range(hidden_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, 2))
        self.mlp = nn.Sequential(*layers)

    def forward(self, polymer: "ciffy.Polymer") -> torch.Tensor:
        """Forward pass."""
        n_res = polymer.size(Scale.RESIDUE)
        if n_res == 0:
            return polymer.coordinates.new_zeros(0, 2)

        # Get residue embeddings
        embed = self.embedding(polymer)  # (n_res, embed_dim)

        # Apply MLP
        return self.mlp(embed)  # (n_res, 2)


class EquivariantReactivityModel(nn.Module):
    """Equivariant transformer for predicting SHAPE and DMS reactivity.

    Operates at atom level using all atom coordinates and features, then
    reduces to residue level for final predictions.

    Args:
        embed_dim: Dimension of atom embedding.
        hidden_mult: Multiplicity for hidden representation.
        hidden_layers: Number of transformer layers.
        k_neighbors: Number of k-NN neighbors.
        edge_dim: Dimension of radial basis functions.
        nheads: Number of attention heads.
        rank: Filter rank for equivariant basis (0, 1, 2, or None for full).
        lvals: Angular momentum values for hidden layers (default [0, 1]).
        dropout: Dropout probability.
    """

    def __init__(
        self,
        embed_dim: int = 32,
        hidden_mult: int = 16,
        hidden_layers: int = 4,
        k_neighbors: int = 16,
        edge_dim: int = 16,
        nheads: int = 4,
        rank: int | None = 0,
        lvals: list[int] | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        if lvals is None:
            lvals = [0, 1]  # Scalars and vectors

        self.lvals = lvals

        # All reprs must have same lvals for fused K/V attention
        # Input mult is smaller, hidden/output use hidden_mult
        in_repr = Repr(lvals=lvals, mult=embed_dim)
        hidden_repr = Repr(lvals=lvals, mult=hidden_mult)
        out_repr = Repr(lvals=lvals, mult=hidden_mult)

        # Atom-level embedding (atom type + residue type + element)
        self.embedding = PolymerEmbedding(
            scale=Scale.ATOM,
            atom_dim=embed_dim // 2,
            residue_dim=embed_dim // 4,
            element_dim=embed_dim // 4,
            dropout=dropout,
        )

        self.transformer = EquivariantTransformer(
            in_repr=in_repr,
            out_repr=out_repr,
            hidden_repr=hidden_repr,
            hidden_layers=hidden_layers,
            edge_dim=edge_dim,
            edge_hidden_dim=edge_dim * 2,
            k_neighbors=k_neighbors,
            nheads=nheads,
            dropout=dropout,
            attn_dropout=dropout,
            transition=True,
            residual_scale=0.5,
            qk_norm=True,
            seq_pos_dim=8,
            rank=rank,
        )

        # Output projection: extract l=0 scalars and project to 2 outputs
        self.out_proj = nn.Linear(hidden_mult, 2)

        self.in_repr = in_repr
        self.out_repr = out_repr
        self.repr_dim = in_repr.dim()  # Total dimension of repr (1 + 3 = 4 for l=0,1)

    def forward(
        self,
        polymer: "ciffy.Polymer",
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            polymer: Input polymer (torch backend, on device).

        Returns:
            Predicted reactivities (n_residues, 2) - [SHAPE, DMS].
        """
        n_atoms = polymer.size(Scale.ATOM)

        if n_atoms == 0:
            return polymer.coordinates.new_zeros(0, 2)

        # Get atom-level embeddings: (n_atoms, embed_dim)
        embed = self.embedding(polymer)

        # Expand to full repr dimension: (n_atoms, embed_dim, repr_dim)
        # Only l=0 (first element) is filled, rest is zero
        features = embed.new_zeros(n_atoms, embed.size(1), self.repr_dim)
        features[:, :, 0] = embed  # Put scalars in l=0 slot

        # Use residue membership as sequence position for atoms
        seq_pos = polymer.membership(Scale.RESIDUE)

        # Apply equivariant transformer at atom level
        # Output: (n_atoms, hidden_mult, repr_dim)
        output = self.transformer(polymer.coordinates, features, seq_pos=seq_pos)

        # Extract l=0 (scalar) component: (n_atoms, hidden_mult)
        atom_features = output[:, :, 0]

        # Reduce from atoms to residues (mean pooling)
        residue_features = polymer.reduce(atom_features, Scale.RESIDUE, Reduction.MEAN)

        # Project to 2 outputs (SHAPE, DMS)
        reactivities = self.out_proj(residue_features)

        return reactivities


# ============================================================================
# Loss Functions
# ============================================================================

def pearson_correlation(pred: torch.Tensor, target: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Compute Pearson correlation coefficient.

    Args:
        pred: Predictions of shape (..., N, ...).
        target: Targets of same shape.
        dim: Dimension along which to compute correlation.

    Returns:
        Correlation coefficient(s).
    """
    pred_centered = pred - pred.mean(dim=dim, keepdim=True)
    target_centered = target - target.mean(dim=dim, keepdim=True)

    numerator = (pred_centered * target_centered).sum(dim=dim)
    denominator = torch.sqrt(
        (pred_centered ** 2).sum(dim=dim) * (target_centered ** 2).sum(dim=dim)
    )

    return numerator / (denominator + 1e-8)


def correlation_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute negative correlation loss (higher correlation = lower loss).

    Args:
        pred: Predictions (N, 2) for SHAPE and DMS.
        target: Targets (N, 2).

    Returns:
        Scalar loss (negative mean correlation).
    """
    # Mask NaN values in target
    mask = ~torch.isnan(target)

    # Compute correlation per channel
    corrs = []
    for c in range(pred.size(1)):
        c_mask = mask[:, c]
        if c_mask.sum() < 3:  # Need at least 3 points for correlation
            continue
        corr = pearson_correlation(pred[c_mask, c], target[c_mask, c])
        corrs.append(corr)

    if len(corrs) == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    # Return negative mean correlation (minimize to maximize correlation)
    mean_corr = torch.stack(corrs).mean()
    return -mean_corr


def normalize(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Normalize tensor to [0, 1] range along given dimension."""
    x_min = x.min(dim=dim, keepdim=True).values
    x_max = x.max(dim=dim, keepdim=True).values
    return (x - x_min) / (x_max - x_min + 1e-8)


# ============================================================================
# Training
# ============================================================================

def train_epoch(
    model: nn.Module,
    dataset: PolymerDataset,
    index: ReactivityIndex,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 10.0,
    debug: bool = False,
) -> dict:
    """Train for one epoch.

    Args:
        model: The model to train.
        dataset: PolymerDataset of RNA structures.
        index: ReactivityIndex for looking up reactivity by sequence.
        optimizer: Optimizer.
        device: Device.
        debug: If True, print debug info for first few unmatched samples.

    Returns:
        Dictionary of metrics.
    """
    model.train()

    total_loss = 0.0
    total_corr_shape = 0.0
    total_corr_dms = 0.0
    n_samples = 0
    n_residues = 0
    n_unmatched = 0

    for idx in range(len(dataset)):
        polymer = dataset[idx]
        if polymer is None:
            continue

        # Strip unresolved residues
        polymer = polymer.strip()
        if polymer.empty():
            continue

        # Move to device
        polymer = polymer.torch().to(device)

        # Match reactivity by sequence
        match = index.match(polymer)
        if match is None:
            n_unmatched += 1
            if debug and n_unmatched <= 5:
                seq = polymer.sequence_str()
                print(f"  [DEBUG] No match for polymer {idx}: seq={seq[:50]}... (len={len(seq)})")
            continue

        # Get aligned reactivity (already on correct device via convert_backend)
        target = match.reactivity  # (n_res, 2) - SHAPE and DMS

        if debug and n_samples < 3:
            print(f"  [DEBUG] Match found for polymer {idx}: target shape={target.shape}, name={match.name}")

        # Normalize target
        for c in range(2):
            mask = ~torch.isnan(target[:, c])
            if mask.sum() > 0:
                target[mask, c] = normalize(target[mask, c])

        # Forward pass
        optimizer.zero_grad()
        pred = model(polymer)

        # Normalize predictions
        pred = normalize(pred, dim=0)

        # Debug check before loss
        if debug and n_samples < 3:
            print(f"  [DEBUG] Before loss: pred_nan={pred.isnan().any()}, pred_range=[{pred.min():.3f}, {pred.max():.3f}]")

        # Compute loss - use MSE for stability, then correlation for metrics
        # Mask NaN values in target
        target_valid = target.clone()
        target_valid[torch.isnan(target_valid)] = 0.0
        valid_mask = ~torch.isnan(target)

        if valid_mask.sum() < 3:
            continue

        # MSE loss only on valid positions
        mse_loss = F.mse_loss(pred[valid_mask], target_valid[valid_mask])
        loss = mse_loss

        if loss.isnan() or loss.isinf():
            if debug:
                print(f"  [DEBUG] NaN/Inf loss for polymer {idx}: pred_nan={pred.isnan().any()}, target_nan={target.isnan().all()}")
            continue

        # Backward pass
        loss.backward()

        # Check for NaN/Inf gradients
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if grad_norm.isnan() or grad_norm.isinf():
            if debug:
                print(f"  [DEBUG] NaN/Inf gradient norm for polymer {idx}, skipping update")
            optimizer.zero_grad()
            continue

        optimizer.step()

        # Track metrics
        n_res = polymer.size(Scale.RESIDUE)
        with torch.no_grad():
            total_loss += loss.item() * n_res
            n_residues += n_res
            n_samples += 1

            # Compute correlations
            for c, (name, total) in enumerate([
                ("shape", total_corr_shape),
                ("dms", total_corr_dms),
            ]):
                mask = ~torch.isnan(target[:, c])
                if mask.sum() >= 3:
                    corr = pearson_correlation(pred[mask, c], target[mask, c])
                    if c == 0:
                        total_corr_shape += corr.item()
                    else:
                        total_corr_dms += corr.item()

        if n_samples % 10 == 0:
            avg_loss = total_loss / max(n_residues, 1)
            avg_shape = total_corr_shape / max(n_samples, 1)
            avg_dms = total_corr_dms / max(n_samples, 1)
            print(f"  [{n_samples}] Loss: {avg_loss:.4f}, SHAPE corr: {avg_shape:.4f}, DMS corr: {avg_dms:.4f}")

    return {
        "loss": total_loss / max(n_residues, 1),
        "corr_shape": total_corr_shape / max(n_samples, 1),
        "corr_dms": total_corr_dms / max(n_samples, 1),
        "n_samples": n_samples,
        "n_residues": n_residues,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataset: PolymerDataset,
    index: ReactivityIndex,
    device: torch.device,
) -> dict:
    """Evaluate model on dataset.

    Args:
        model: The model to evaluate.
        dataset: PolymerDataset of RNA structures.
        index: ReactivityIndex for looking up reactivity by sequence.
        device: Device.

    Returns:
        Dictionary of metrics.
    """
    model.eval()

    total_corr_shape = 0.0
    total_corr_dms = 0.0
    n_samples = 0

    for idx in range(len(dataset)):
        polymer = dataset[idx]
        if polymer is None:
            continue

        polymer = polymer.strip()
        if polymer.empty():
            continue

        polymer = polymer.torch().to(device)

        # Match reactivity by sequence
        match = index.match(polymer)
        if match is None:
            continue

        target = match.reactivity  # (n_res, 2)

        # Normalize target
        for c in range(2):
            mask = ~torch.isnan(target[:, c])
            if mask.sum() > 0:
                target[mask, c] = normalize(target[mask, c])

        pred = model(polymer)
        pred = normalize(pred, dim=0)

        n_samples += 1
        for c in range(2):
            mask = ~torch.isnan(target[:, c])
            if mask.sum() >= 3:
                corr = pearson_correlation(pred[mask, c], target[mask, c])
                if c == 0:
                    total_corr_shape += corr.item()
                else:
                    total_corr_dms += corr.item()

    return {
        "corr_shape": total_corr_shape / max(n_samples, 1),
        "corr_dms": total_corr_dms / max(n_samples, 1),
        "n_samples": n_samples,
    }


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train equivariant reactivity model")

    # Data
    parser.add_argument("--data", type=str, default=None,
                        help="Structure directory (overrides default paths)")
    parser.add_argument("--h5-train", type=str, default=None,
                        help="Training reactivity HDF5 file")
    parser.add_argument("--fasta-train", type=str, default=None,
                        help="Training sequences FASTA file")
    parser.add_argument("--h5-val", type=str, default=None,
                        help="Validation reactivity HDF5 file")
    parser.add_argument("--fasta-val", type=str, default=None,
                        help="Validation sequences FASTA file")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: /scratch/users/hmblair/ciffy/equi_react/<name>)")

    # Model
    parser.add_argument("--rank", type=str, default="0",
                        help="Filter rank: 0, 1, 2, or null/none for full")
    parser.add_argument("--hidden-mult", type=int, default=16,
                        help="Hidden representation multiplicity")
    parser.add_argument("--hidden-layers", type=int, default=4,
                        help="Number of transformer layers")
    parser.add_argument("--k-neighbors", type=int, default=16,
                        help="Number of k-NN neighbors")
    parser.add_argument("--edge-dim", type=int, default=16,
                        help="RBF edge dimension")
    parser.add_argument("--nheads", type=int, default=4,
                        help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="Dropout rate (default 0.0 to avoid NaN gradients with equivariant ops)")
    parser.add_argument("--model", type=str, default="equivariant",
                        choices=["equivariant", "simple"],
                        help="Model type: equivariant (SO(3)-equivariant transformer) or simple (MLP baseline)")

    # Training
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (default 1e-4 for stability with equivariant ops)")
    parser.add_argument("--grad-clip", type=float, default=10.0,
                        help="Gradient clipping threshold")
    parser.add_argument("--val-split", type=float, default=0.0,
                        help="Fraction of training data to use for validation (0 = use separate val set)")
    parser.add_argument("--name", type=str, default="equi_react",
                        help="Experiment name")

    return parser.parse_args()


def main():
    args = parse_args()

    # Configure precision for faster training
    # Disabled - causing numerical issues with equivariant ops
    # configure_precision()

    # Parse rank
    if args.rank.lower() in ("null", "none", "full"):
        rank = None
    else:
        rank = int(args.rank)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Filter rank: {rank}")

    # Default data paths (Sherlock)
    if args.data:
        train_dir = args.data
        val_dir = args.data
    else:
        # Sherlock paths for structures (separate directories for pdb130 and pdb240)
        train_dir = "/scratch/users/hmblair/structures/pdb130"
        val_dir = "/scratch/users/hmblair/structures/pdb240"

    # HDF5 paths
    # HDF5 paths - use old format with embedded sequences by default
    h5_train = args.h5_train or "/scratch/users/hmblair/profiles/pdb130/train2_pdb130.hdf5"
    h5_val = args.h5_val or "/scratch/users/hmblair/profiles/pdb240/wt-profiles.h5"

    # FASTA paths (only needed for new format HDF5)
    fasta_train = args.fasta_train
    fasta_val = args.fasta_val or "/scratch/users/hmblair/profiles/pdb240/ref.fasta"

    # Create output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(f"/scratch/users/hmblair/ciffy/equi_react/{args.name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build ReactivityIndex from HDF5 + FASTA data
    print("Building ReactivityIndex...")
    print(f"  Train: {h5_train} + {fasta_train}")
    print(f"  Val: {h5_val} + {fasta_val}")

    try:
        train_index = build_reactivity_index(h5_train, fasta_train)
    except Exception as e:
        print(f"Warning: Could not build training index: {e}")
        train_index = None

    try:
        val_index = build_reactivity_index(h5_val, fasta_val)
    except Exception as e:
        print(f"Warning: Could not build validation index: {e}")
        val_index = None

    # Create datasets
    print("Creating datasets...")
    if args.val_split > 0:
        # Use train/val split from same dataset
        print(f"  Using {args.val_split:.0%} of training data for validation")
        try:
            full_dataset = PolymerDataset(
                train_dir,
                scale=Scale.CHAIN,
                molecule_types=Molecule.RNA,
                min_residues=10,
                max_chains=2,
            )
            # Split dataset
            n_total = len(full_dataset)
            n_val = int(n_total * args.val_split)
            n_train = n_total - n_val

            # Use random split with fixed seed for reproducibility
            generator = torch.Generator().manual_seed(42)
            train_dataset, val_dataset = torch.utils.data.random_split(
                full_dataset, [n_train, n_val], generator=generator
            )
            # Use same index for both train and val
            val_index = train_index
            print(f"  Training: {len(train_dataset)} chains")
            print(f"  Validation: {len(val_dataset)} chains")
        except Exception as e:
            print(f"Warning: Could not create dataset: {e}")
            train_dataset = None
            val_dataset = None
    else:
        # Use separate train and val directories
        try:
            train_dataset = PolymerDataset(
                train_dir,
                scale=Scale.CHAIN,
                molecule_types=Molecule.RNA,
                min_residues=10,
                max_chains=2,
            )
            print(f"  Training: {len(train_dataset)} chains")
        except Exception as e:
            print(f"Warning: Could not create training dataset: {e}")
            train_dataset = None

        try:
            val_dataset = PolymerDataset(
                val_dir,
                scale=Scale.CHAIN,
                molecule_types=Molecule.RNA,
                min_residues=10,
            )
            print(f"  Validation: {len(val_dataset)} chains")
        except Exception as e:
            print(f"Warning: Could not create validation dataset: {e}")
            val_dataset = None

    # Create model
    print(f"Creating model ({args.model})...")
    if args.model == "simple":
        model = SimpleReactivityModel(
            embed_dim=64,
            hidden_dim=128,
            hidden_layers=args.hidden_layers,
            dropout=args.dropout,
        ).to(device)
    else:
        model = EquivariantReactivityModel(
            hidden_mult=args.hidden_mult,
            hidden_layers=args.hidden_layers,
            k_neighbors=args.k_neighbors,
            edge_dim=args.edge_dim,
            nheads=args.nheads,
            rank=rank,
            dropout=args.dropout,
        ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Training loop
    print("\nStarting training...")
    best_val_corr = -float("inf")

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Train
        if train_dataset and train_index and len(train_index) > 0:
            train_metrics = train_epoch(
                model, train_dataset, train_index, optimizer, device,
                grad_clip=args.grad_clip,
                debug=(epoch == 0),  # Debug only on first epoch
            )
            print(f"  Train: loss={train_metrics['loss']:.4f}, "
                  f"SHAPE={train_metrics['corr_shape']:.4f}, "
                  f"DMS={train_metrics['corr_dms']:.4f}, "
                  f"n={train_metrics['n_samples']}")

        # Validate
        if val_dataset and val_index and len(val_index) > 0:
            val_metrics = evaluate(model, val_dataset, val_index, device)
            print(f"  Val: SHAPE={val_metrics['corr_shape']:.4f}, "
                  f"DMS={val_metrics['corr_dms']:.4f}, "
                  f"n={val_metrics['n_samples']}")

            # Save best model
            mean_corr = (val_metrics["corr_shape"] + val_metrics["corr_dms"]) / 2
            if mean_corr > best_val_corr:
                best_val_corr = mean_corr
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "epoch": epoch,
                        "rank": rank,
                        "val_metrics": val_metrics,
                    },
                    output_dir / "best_model.pt",
                )
                print(f"  Saved best model (corr={mean_corr:.4f})")

        # Save checkpoint
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "rank": rank,
            },
            output_dir / "checkpoint.pt",
        )

    print("\nTraining complete!")
    print(f"Best validation correlation: {best_val_corr:.4f}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
