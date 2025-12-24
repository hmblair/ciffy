#!/usr/bin/env python3
"""
Prototype: Residue-Pair VAE for RNA Conformational Diversity

This script demonstrates learning a minimal parametrization of adjacent
RNA residue pairs using a VAE on Cartesian coordinates.

Usage:
    python scripts/pair_vae_prototype.py
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from dataclasses import dataclass

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    raise ImportError("PyTorch is required for this script")

import ciffy
from ciffy.biochemistry import Residue
from ciffy.types import Scale


# =============================================================================
# Data Extraction
# =============================================================================

def get_residue_indices(poly: ciffy.Polymer) -> np.ndarray:
    """Get residue index for each atom."""
    return poly.reduce(
        np.arange(poly.size(Scale.RESIDUE)),
        Scale.RESIDUE,
        Scale.ATOM,
    )


def extract_adjacent_pairs(
    poly: ciffy.Polymer,
    residue_5prime: Residue,
    residue_3prime: Residue,
) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    """
    Extract all adjacent pairs of specified residue types.

    Returns:
        coords_5prime: (n_pairs, n_atoms_5, 3) coordinates of 5' residue
        coords_3prime: (n_pairs, n_atoms_3, 3) coordinates of 3' residue
        atoms_5prime: list of atom indices for 5' residue
        atoms_3prime: list of atom indices for 3' residue
    """
    from ciffy.backend import to_numpy
    from ciffy.operations.reduction import Reduction

    # Get polymer-only atoms
    poly = poly.poly()

    # Get sequence and residue boundaries
    sequence = to_numpy(poly.sequence)  # (n_residues,)
    n_residues = len(sequence)

    # Find adjacent pairs of the target types
    pair_indices = []
    for i in range(n_residues - 1):
        if sequence[i] == residue_5prime.value and sequence[i + 1] == residue_3prime.value:
            pair_indices.append((i, i + 1))

    if len(pair_indices) == 0:
        raise ValueError(
            f"No adjacent {residue_5prime.name}-{residue_3prime.name} pairs found"
        )

    # Collate atoms and coordinates per residue
    per_res_atoms = poly.reduce(poly.atoms, Scale.RESIDUE, Reduction.COLLATE)
    per_res_coords = poly.reduce(poly.coordinates, Scale.RESIDUE, Reduction.COLLATE)

    # Find common atoms for each residue type
    atoms_5_sets = []
    atoms_3_sets = []
    for i5, i3 in pair_indices:
        atoms_5_sets.append(set(to_numpy(per_res_atoms[i5]).tolist()))
        atoms_3_sets.append(set(to_numpy(per_res_atoms[i3]).tolist()))

    common_atoms_5 = sorted(set.intersection(*atoms_5_sets))
    common_atoms_3 = sorted(set.intersection(*atoms_3_sets))

    n_atoms_5 = len(common_atoms_5)
    n_atoms_3 = len(common_atoms_3)

    # Build output arrays
    coords_5 = np.zeros((len(pair_indices), n_atoms_5, 3), dtype=np.float32)
    coords_3 = np.zeros((len(pair_indices), n_atoms_3, 3), dtype=np.float32)

    atom_to_col_5 = {a: c for c, a in enumerate(common_atoms_5)}
    atom_to_col_3 = {a: c for c, a in enumerate(common_atoms_3)}

    for pair_idx, (i5, i3) in enumerate(pair_indices):
        # 5' residue
        res_atoms = to_numpy(per_res_atoms[i5])
        res_coords = to_numpy(per_res_coords[i5])
        for atom_idx, coord in zip(res_atoms, res_coords):
            if atom_idx in atom_to_col_5:
                coords_5[pair_idx, atom_to_col_5[atom_idx]] = coord

        # 3' residue
        res_atoms = to_numpy(per_res_atoms[i3])
        res_coords = to_numpy(per_res_coords[i3])
        for atom_idx, coord in zip(res_atoms, res_coords):
            if atom_idx in atom_to_col_3:
                coords_3[pair_idx, atom_to_col_3[atom_idx]] = coord

    return coords_5, coords_3, common_atoms_5, common_atoms_3


def align_pair_to_frame(
    coords_5: np.ndarray,
    coords_3: np.ndarray,
    atoms_5: list[int],
    atoms_3: list[int],
    residue_5: Residue,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Align each pair to a canonical frame defined by the 5' residue.

    Frame is defined by C1', N9 (for purines) or N1 (for pyrimidines), and C4.
    - Origin: C1'
    - X-axis: C1' -> N9/N1
    - Y-axis: in plane with C4
    - Z-axis: right-hand rule

    Returns:
        aligned_5: (n_pairs, n_atoms_5, 3)
        aligned_3: (n_pairs, n_atoms_3, 3)
    """
    n_pairs = coords_5.shape[0]

    # Get frame atom indices for 5' residue
    # Try to find C1', N9 (purine) or N1 (pyrimidine), C4
    try:
        c1p_idx = atoms_5.index(residue_5.C1p.value)
    except (ValueError, AttributeError):
        raise ValueError(f"C1' not found in {residue_5.name}")

    # N9 for purines (A, G), N1 for pyrimidines (C, U)
    try:
        n_idx = atoms_5.index(residue_5.N9.value)  # Purine
    except (ValueError, AttributeError):
        try:
            n_idx = atoms_5.index(residue_5.N1.value)  # Pyrimidine
        except (ValueError, AttributeError):
            raise ValueError(f"Neither N9 nor N1 found in {residue_5.name}")

    try:
        c4_idx = atoms_5.index(residue_5.C4.value)
    except (ValueError, AttributeError):
        raise ValueError(f"C4 not found in {residue_5.name}")

    aligned_5 = np.zeros_like(coords_5)
    aligned_3 = np.zeros_like(coords_3)

    for i in range(n_pairs):
        # Get frame atoms
        origin = coords_5[i, c1p_idx]
        n_pos = coords_5[i, n_idx]
        c4_pos = coords_5[i, c4_idx]

        # Build orthonormal frame
        x_axis = n_pos - origin
        x_axis = x_axis / np.linalg.norm(x_axis)

        y_temp = c4_pos - origin
        z_axis = np.cross(x_axis, y_temp)
        z_axis = z_axis / np.linalg.norm(z_axis)

        y_axis = np.cross(z_axis, x_axis)

        # Rotation matrix (columns are basis vectors)
        R = np.column_stack([x_axis, y_axis, z_axis])

        # Transform both residues
        aligned_5[i] = (coords_5[i] - origin) @ R
        aligned_3[i] = (coords_3[i] - origin) @ R

    return aligned_5, aligned_3


@dataclass
class PairDataset:
    """Dataset of aligned residue pairs."""
    coords_5: np.ndarray  # (n_pairs, n_atoms_5, 3)
    coords_3: np.ndarray  # (n_pairs, n_atoms_3, 3)
    atoms_5: list[int]
    atoms_3: list[int]
    residue_5: Residue
    residue_3: Residue

    @property
    def n_pairs(self) -> int:
        return self.coords_5.shape[0]

    @property
    def n_atoms_5(self) -> int:
        return self.coords_5.shape[1]

    @property
    def n_atoms_3(self) -> int:
        return self.coords_3.shape[1]

    @property
    def n_atoms_total(self) -> int:
        return self.n_atoms_5 + self.n_atoms_3

    def get_combined(self) -> np.ndarray:
        """Get combined coordinates (n_pairs, n_atoms_5 + n_atoms_3, 3)."""
        return np.concatenate([self.coords_5, self.coords_3], axis=1)

    def split_combined(self, combined: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Split combined coords back to 5' and 3' parts."""
        return combined[:, :self.n_atoms_5], combined[:, self.n_atoms_5:]

    def to_tensors(self, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
        """Convert to PyTorch tensors."""
        coords_5 = torch.from_numpy(self.coords_5).float().to(device)
        coords_3 = torch.from_numpy(self.coords_3).float().to(device)
        return coords_5, coords_3


def extract_pairs_from_structures(
    cif_paths: list[Path],
    residue_5: Residue = Residue.A,
    residue_3: Residue = Residue.A,
    min_coverage: float = 0.9,
    verbose: bool = True,
) -> PairDataset:
    """
    Extract all pairs of specified types from multiple structures.

    Handles atom mismatches by finding atoms present in at least min_coverage
    fraction of pairs, then filtering out pairs missing those atoms.
    """
    from ciffy.backend import to_numpy
    from ciffy.operations.reduction import Reduction
    from collections import Counter

    # First pass: collect all pairs with their atoms
    all_pairs = []  # List of (coords_5, coords_3, atoms_5, atoms_3)

    for path in cif_paths:
        if verbose:
            print(f"Processing {path.name}...", end=" ")

        try:
            poly = ciffy.load(str(path)).poly()
            sequence = to_numpy(poly.sequence)
            n_residues = len(sequence)

            # Find matching pairs
            pair_indices = []
            for i in range(n_residues - 1):
                if sequence[i] == residue_5.value and sequence[i + 1] == residue_3.value:
                    pair_indices.append((i, i + 1))

            if len(pair_indices) == 0:
                if verbose:
                    print("no pairs")
                continue

            # Get per-residue atoms and coords
            per_res_atoms = poly.reduce(poly.atoms, Scale.RESIDUE, Reduction.COLLATE)
            per_res_coords = poly.reduce(poly.coordinates, Scale.RESIDUE, Reduction.COLLATE)

            for i5, i3 in pair_indices:
                atoms_5 = to_numpy(per_res_atoms[i5]).tolist()
                atoms_3 = to_numpy(per_res_atoms[i3]).tolist()
                coords_5 = to_numpy(per_res_coords[i5])
                coords_3 = to_numpy(per_res_coords[i3])
                all_pairs.append((coords_5, coords_3, atoms_5, atoms_3))

            if verbose:
                print(f"{len(pair_indices)} pairs")

        except Exception as e:
            if verbose:
                print(f"error: {e}")
            continue

    if len(all_pairs) == 0:
        raise ValueError(f"No {residue_5.name}-{residue_3.name} pairs found")

    if verbose:
        print(f"\nCollected {len(all_pairs)} raw pairs")

    # Count atom occurrences
    atom_counts_5 = Counter()
    atom_counts_3 = Counter()
    for _, _, a5, a3 in all_pairs:
        atom_counts_5.update(a5)
        atom_counts_3.update(a3)

    # Find atoms present in at least min_coverage of pairs
    n_pairs_raw = len(all_pairs)
    min_count = int(n_pairs_raw * min_coverage)

    common_atoms_5 = sorted([a for a, c in atom_counts_5.items() if c >= min_count])
    common_atoms_3 = sorted([a for a, c in atom_counts_3.items() if c >= min_count])

    if verbose:
        print(f"Atoms with >={min_coverage*100:.0f}% coverage: {len(common_atoms_5)} (5') + {len(common_atoms_3)} (3')")

    if len(common_atoms_5) == 0 or len(common_atoms_3) == 0:
        raise ValueError("No atoms found with sufficient coverage")

    # Filter pairs to those that have all common atoms
    common_set_5 = set(common_atoms_5)
    common_set_3 = set(common_atoms_3)

    filtered_pairs = [
        p for p in all_pairs
        if common_set_5.issubset(set(p[2])) and common_set_3.issubset(set(p[3]))
    ]

    if verbose:
        print(f"Pairs with all common atoms: {len(filtered_pairs)}")

    if len(filtered_pairs) == 0:
        raise ValueError("No pairs have all common atoms")

    # Build dense arrays with only common atoms
    n_pairs = len(filtered_pairs)
    n_atoms_5 = len(common_atoms_5)
    n_atoms_3 = len(common_atoms_3)

    coords_5_out = np.zeros((n_pairs, n_atoms_5, 3), dtype=np.float32)
    coords_3_out = np.zeros((n_pairs, n_atoms_3, 3), dtype=np.float32)

    atom_to_col_5 = {a: c for c, a in enumerate(common_atoms_5)}
    atom_to_col_3 = {a: c for c, a in enumerate(common_atoms_3)}

    for idx, (c5, c3, a5, a3) in enumerate(filtered_pairs):
        for atom_idx, coord in zip(a5, c5):
            if atom_idx in atom_to_col_5:
                coords_5_out[idx, atom_to_col_5[atom_idx]] = coord
        for atom_idx, coord in zip(a3, c3):
            if atom_idx in atom_to_col_3:
                coords_3_out[idx, atom_to_col_3[atom_idx]] = coord

    # Align to canonical frame
    coords_5_out, coords_3_out = align_pair_to_frame(
        coords_5_out, coords_3_out, common_atoms_5, common_atoms_3, residue_5
    )

    return PairDataset(
        coords_5=coords_5_out,
        coords_3=coords_3_out,
        atoms_5=common_atoms_5,
        atoms_3=common_atoms_3,
        residue_5=residue_5,
        residue_3=residue_3,
    )


# =============================================================================
# VAE Architecture
# =============================================================================

class PairEncoder(nn.Module):
    """
    Encodes a residue pair to latent distribution parameters.

    Input: (batch, n_atoms_total, 3) flattened to (batch, n_atoms_total * 3)
    Output: mu (batch, latent_dim), logvar (batch, latent_dim)
    """

    def __init__(self, n_atoms: int, latent_dim: int = 8, hidden_dim: int = 128):
        super().__init__()
        input_dim = n_atoms * 3

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, n_atoms, 3)
        x_flat = x.flatten(1)  # (batch, n_atoms * 3)
        h = self.encoder(x_flat)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class PairDecoder(nn.Module):
    """
    Decodes latent vector to residue pair coordinates.

    Input: (batch, latent_dim)
    Output: (batch, n_atoms_total, 3)
    """

    def __init__(self, n_atoms: int, latent_dim: int = 8, hidden_dim: int = 128):
        super().__init__()
        output_dim = n_atoms * 3

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        self.n_atoms = n_atoms

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x_flat = self.decoder(z)  # (batch, n_atoms * 3)
        return x_flat.reshape(-1, self.n_atoms, 3)


class PairVAE(nn.Module):
    """
    Variational Autoencoder for residue pairs.
    """

    def __init__(self, n_atoms: int, latent_dim: int = 8, hidden_dim: int = 128):
        super().__init__()
        self.encoder = PairEncoder(n_atoms, latent_dim, hidden_dim)
        self.decoder = PairDecoder(n_atoms, latent_dim, hidden_dim)
        self.latent_dim = latent_dim
        self.n_atoms = n_atoms

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


def compute_pairwise_distances(coords: torch.Tensor) -> torch.Tensor:
    """Compute pairwise distance matrix. coords: (batch, n_atoms, 3)"""
    # (batch, n_atoms, 1, 3) - (batch, 1, n_atoms, 3) -> (batch, n_atoms, n_atoms, 3)
    diff = coords.unsqueeze(2) - coords.unsqueeze(1)
    return torch.sqrt((diff ** 2).sum(-1) + 1e-8)  # (batch, n_atoms, n_atoms)


def vae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
    distance_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    VAE loss = reconstruction loss + distance loss + beta * KL divergence.

    Distance loss encourages the pairwise distances to match, which
    implicitly encourages correct bond lengths and angles.
    """
    # Reconstruction loss (MSE on coordinates)
    recon_loss = nn.functional.mse_loss(recon, target, reduction='mean')

    # Distance matrix loss - compare pairwise distances
    dist_recon = compute_pairwise_distances(recon)
    dist_target = compute_pairwise_distances(target)
    dist_loss = nn.functional.mse_loss(dist_recon, dist_target, reduction='mean')

    # KL divergence: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    total_loss = recon_loss + distance_weight * dist_loss + beta * kl_loss

    return total_loss, recon_loss, dist_loss, kl_loss


# =============================================================================
# Training
# =============================================================================

def train_vae(
    dataset: PairDataset,
    latent_dim: int = 8,
    hidden_dim: int = 128,
    batch_size: int = 64,
    n_epochs: int = 100,
    lr: float = 1e-3,
    beta: float = 0.1,
    use_kl: bool = True,
    device: str = "cpu",
    verbose: bool = True,
) -> PairVAE:
    """
    Train a VAE on the pair dataset.
    """
    # Prepare data
    combined = dataset.get_combined()  # (n_pairs, n_atoms_total, 3)
    X = torch.from_numpy(combined).float().to(device)

    train_dataset = TensorDataset(X)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Create model
    n_atoms = dataset.n_atoms_total
    model = PairVAE(n_atoms, latent_dim, hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if verbose:
        print(f"\nTraining VAE:")
        print(f"  Dataset: {dataset.n_pairs} pairs, {n_atoms} atoms/pair")
        print(f"  Latent dim: {latent_dim}, Hidden dim: {hidden_dim}")
        print(f"  Beta: {beta}, LR: {lr}")
        print()

    # Training loop
    model.train()
    for epoch in range(n_epochs):
        total_loss = 0.0
        total_recon = 0.0
        total_dist = 0.0
        total_kl = 0.0

        for (batch,) in train_loader:
            optimizer.zero_grad()

            recon, mu, logvar = model(batch)
            loss, recon_loss, dist_loss, kl_loss = vae_loss(
                recon, batch, mu, logvar,
                beta=beta if use_kl else 0.0,
                distance_weight=0.0,  # Disable distance loss for now
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_dist += dist_loss.item()
            total_kl += kl_loss.item()

        n_batches = len(train_loader)
        if verbose and (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1:3d}/{n_epochs}: "
                f"Loss={total_loss/n_batches:.4f}, "
                f"Coord={total_recon/n_batches:.4f}, "
                f"Dist={total_dist/n_batches:.4f}, "
                f"KL={total_kl/n_batches:.4f}"
            )

    return model


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_reconstruction(
    model: PairVAE,
    dataset: PairDataset,
    device: str = "cpu",
) -> dict:
    """
    Evaluate reconstruction quality.
    """
    model.eval()

    combined = dataset.get_combined()
    X = torch.from_numpy(combined).float().to(device)

    with torch.no_grad():
        recon, mu, logvar = model(X)

    # Compute per-atom RMSD
    diff = recon - X
    per_atom_rmsd = torch.sqrt((diff ** 2).sum(dim=-1).mean(dim=0))  # (n_atoms,)
    overall_rmsd = torch.sqrt((diff ** 2).sum(dim=-1).mean())

    # Compute per-pair RMSD
    per_pair_rmsd = torch.sqrt((diff ** 2).sum(dim=-1).mean(dim=1))  # (n_pairs,)

    return {
        "overall_rmsd": overall_rmsd.item(),
        "per_atom_rmsd": per_atom_rmsd.cpu().numpy(),
        "per_pair_rmsd": per_pair_rmsd.cpu().numpy(),
        "mean_mu": mu.mean().item(),
        "std_mu": mu.std().item(),
        "mean_logvar": logvar.mean().item(),
    }


def compute_bond_lengths(coords: np.ndarray, bond_pairs: list[tuple[int, int]]) -> np.ndarray:
    """Compute bond lengths for given atom pairs."""
    lengths = []
    for i, j in bond_pairs:
        diff = coords[:, j] - coords[:, i]
        dist = np.linalg.norm(diff, axis=-1)
        lengths.append(dist)
    return np.array(lengths).T  # (n_pairs, n_bonds)


# =============================================================================
# Main
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train a VAE on RNA residue pairs")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/Users/hmblair/academic/data/rna-libraries/pdb/structures/pdb130",
        help="Directory containing CIF files",
    )
    parser.add_argument(
        "--n-structures",
        type=int,
        default=20,
        help="Number of structures to use (default: 20)",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=8,
        help="Latent dimension (default: 8)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Number of training epochs (default: 200)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Residue-Pair VAE Prototype")
    print("=" * 60)

    # Find structures
    data_dir = Path(args.data_dir)
    cif_files = sorted(data_dir.glob("*.cif"))[:args.n_structures]

    if not cif_files:
        print(f"No CIF files found in {data_dir}")
        return

    print(f"\nUsing {len(cif_files)} CIF files from {data_dir.name}/")

    # Extract AA pairs
    print("\n" + "=" * 60)
    print("Extracting A-A pairs...")
    print("=" * 60)

    dataset = extract_pairs_from_structures(cif_files, Residue.A, Residue.A)

    print(f"\nDataset summary:")
    print(f"  Total pairs: {dataset.n_pairs}")
    print(f"  Atoms per 5' residue: {dataset.n_atoms_5}")
    print(f"  Atoms per 3' residue: {dataset.n_atoms_3}")
    print(f"  Total atoms per pair: {dataset.n_atoms_total}")

    # Show coordinate statistics
    combined = dataset.get_combined()
    print(f"\nCoordinate statistics (after frame alignment):")
    print(f"  Mean: {combined.mean():.3f}")
    print(f"  Std:  {combined.std():.3f}")
    print(f"  Min:  {combined.min():.3f}")
    print(f"  Max:  {combined.max():.3f}")

    # Train VAE
    print("\n" + "=" * 60)
    print("Training VAE...")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = train_vae(
        dataset,
        latent_dim=args.latent_dim,
        hidden_dim=256,
        batch_size=64,
        n_epochs=args.epochs,
        lr=1e-3,
        beta=0.0,  # Pure autoencoder - no KL
        use_kl=False,
        device=device,
    )

    # Evaluate
    print("\n" + "=" * 60)
    print("Evaluation...")
    print("=" * 60)

    results = evaluate_reconstruction(model, dataset, device)

    print(f"\nReconstruction quality:")
    print(f"  Overall RMSD: {results['overall_rmsd']:.3f} Å")
    print(f"  Per-atom RMSD range: {results['per_atom_rmsd'].min():.3f} - {results['per_atom_rmsd'].max():.3f} Å")
    print(f"  Per-pair RMSD: mean={results['per_pair_rmsd'].mean():.3f}, max={results['per_pair_rmsd'].max():.3f} Å")

    print(f"\nLatent space statistics:")
    print(f"  Mean of mu: {results['mean_mu']:.3f}")
    print(f"  Std of mu: {results['std_mu']:.3f}")
    print(f"  Mean of logvar: {results['mean_logvar']:.3f}")

    # Check if we meet the success criterion
    print("\n" + "=" * 60)
    if results['overall_rmsd'] < 0.3:
        print("SUCCESS: Reconstruction RMSD < 0.3 Å")
    else:
        print(f"Target not met: RMSD = {results['overall_rmsd']:.3f} Å (target < 0.3 Å)")
    print("=" * 60)

    return model, dataset, results


if __name__ == "__main__":
    main()
