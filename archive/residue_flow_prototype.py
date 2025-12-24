#!/usr/bin/env python3
"""
PCA + Flow for Residue Conformations

Uses PCA for dimensionality reduction and a normalizing flow for density
estimation in the reduced space. This enables both compression and valid
sampling from a low-dimensional latent space.

Architecture:
    coords (N, 22, 3) → align → PCA (66 → k) → Flow → z ~ N(0, I)

The reconstruction error is bounded by PCA truncation error. The flow
is exactly invertible, adding no additional error.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import ciffy
from ciffy.biochemistry import Residue
from ciffy.backend import to_numpy
from ciffy.types import Scale
from ciffy.operations.reduction import Reduction


# =============================================================================
# Data Extraction
# =============================================================================

def extract_residues(
    cif_paths: list[Path],
    residue_type: Residue,
    min_coverage: float = 0.9,
    verbose: bool = True,
) -> tuple[np.ndarray, list[int]]:
    """
    Extract all instances of a residue type from multiple structures.

    Args:
        cif_paths: List of paths to CIF files
        residue_type: Residue enum (e.g., Residue.A for adenosine)
        min_coverage: Minimum fraction of instances an atom must appear in
        verbose: Print progress

    Returns:
        coords: (n_instances, n_atoms, 3) coordinate array
        atoms: List of atom type indices
    """
    from collections import Counter

    all_instances = []

    for path in cif_paths:
        if verbose:
            print(f"Processing {path.name}...", end=" ")

        try:
            poly = ciffy.load(str(path)).poly()
            seq = to_numpy(poly.sequence)
            indices = [i for i in range(len(seq)) if seq[i] == residue_type.value]

            if not indices:
                if verbose:
                    print("no matches")
                continue

            per_res_atoms = poly.reduce(poly.atoms, Scale.RESIDUE, Reduction.COLLATE)
            per_res_coords = poly.reduce(poly.coordinates, Scale.RESIDUE, Reduction.COLLATE)

            for idx in indices:
                atoms = to_numpy(per_res_atoms[idx]).tolist()
                coords = to_numpy(per_res_coords[idx])
                all_instances.append((coords, atoms))

            if verbose:
                print(f"{len(indices)} residues")

        except Exception as e:
            if verbose:
                print(f"error: {e}")

    if not all_instances:
        raise ValueError(f"No {residue_type.name} residues found")

    if verbose:
        print(f"\nCollected {len(all_instances)} raw instances")

    # Find atoms present in most instances
    atom_counts = Counter()
    for _, atoms in all_instances:
        atom_counts.update(atoms)

    min_count = int(len(all_instances) * min_coverage)
    common_atoms = sorted([a for a, c in atom_counts.items() if c >= min_count])

    if verbose:
        print(f"Atoms with >={min_coverage*100:.0f}% coverage: {len(common_atoms)}")

    # Filter and build dense array
    common_set = set(common_atoms)
    filtered = [inst for inst in all_instances if common_set.issubset(set(inst[1]))]

    if verbose:
        print(f"Instances with all common atoms: {len(filtered)}")

    coords_out = np.zeros((len(filtered), len(common_atoms), 3), dtype=np.float32)
    atom_to_col = {a: c for c, a in enumerate(common_atoms)}

    for i, (coords, atoms) in enumerate(filtered):
        for atom_idx, coord in zip(atoms, coords):
            if atom_idx in atom_to_col:
                coords_out[i, atom_to_col[atom_idx]] = coord

    return coords_out, common_atoms


def align_to_frame(coords: np.ndarray, atoms: list[int], residue: Residue) -> np.ndarray:
    """
    Align each residue to a canonical frame.

    Frame: origin at C1', x-axis toward N9/N1, z-axis from plane with C4.
    """
    n_instances = coords.shape[0]

    # Get frame atom indices
    c1p_idx = atoms.index(residue.C1p.value)
    c4_idx = atoms.index(residue.C4.value)

    try:
        n_idx = atoms.index(residue.N9.value)  # Purines
    except (ValueError, AttributeError):
        n_idx = atoms.index(residue.N1.value)  # Pyrimidines

    aligned = np.zeros_like(coords)

    for i in range(n_instances):
        origin = coords[i, c1p_idx]
        n_pos = coords[i, n_idx]
        c4_pos = coords[i, c4_idx]

        x_axis = n_pos - origin
        x_axis /= np.linalg.norm(x_axis)

        y_temp = c4_pos - origin
        z_axis = np.cross(x_axis, y_temp)
        z_axis /= np.linalg.norm(z_axis)

        y_axis = np.cross(z_axis, x_axis)

        R = np.column_stack([x_axis, y_axis, z_axis])
        aligned[i] = (coords[i] - origin) @ R

    return aligned


def check_bond_lengths(coords: np.ndarray, atoms: list[int], residue: Residue) -> dict:
    """Check C1'-N9 glycosidic bond length statistics."""
    try:
        c1p_idx = atoms.index(residue.C1p.value)
        n9_idx = atoms.index(residue.N9.value)
        dists = np.linalg.norm(coords[:, c1p_idx] - coords[:, n9_idx], axis=-1)
        return {
            "C1'-N9 mean": dists.mean(),
            "C1'-N9 std": dists.std(),
        }
    except (ValueError, AttributeError):
        return {}


# =============================================================================
# Flow Components
# =============================================================================

class ActNorm(nn.Module):
    """Activation normalization with data-dependent initialization."""

    def __init__(self, dim: int):
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.initialized = False

    def initialize(self, x: torch.Tensor):
        with torch.no_grad():
            self.bias.copy_(-x.mean(dim=0))
            self.log_scale.copy_(-torch.log(x.std(dim=0).clamp(min=1e-6)))
        self.initialized = True

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.initialized:
            self.initialize(x)
        y = (x + self.bias) * torch.exp(self.log_scale)
        log_det = self.log_scale.sum().expand(x.shape[0])
        return y, log_det

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        return y * torch.exp(-self.log_scale) - self.bias


class CouplingNetwork(nn.Module):
    """MLP for computing scale and translation in coupling layers."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        # Initialize output near zero for stability
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AffineCoupling(nn.Module):
    """
    Affine coupling layer.

    Splits input, transforms one half conditioned on the other:
        y_a = x_a
        y_b = x_b * exp(s(x_a)) + t(x_a)
    """

    def __init__(self, dim: int, hidden_dim: int = 64, even_mask: bool = True):
        super().__init__()
        self.dim = dim
        self.register_buffer("mask", torch.arange(dim) % 2 == (0 if even_mask else 1))

        n_masked = int(self.mask.sum())
        n_unmasked = dim - n_masked
        self.net = CouplingNetwork(n_masked, 2 * n_unmasked, hidden_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_a = x[:, self.mask]
        x_b = x[:, ~self.mask]

        st = self.net(x_a)
        s, t = st.chunk(2, dim=-1)
        s = torch.tanh(s) * 0.5  # Bound scale for stability

        y_b = x_b * torch.exp(s) + t
        log_det = s.sum(dim=-1)

        y = torch.empty_like(x)
        y[:, self.mask] = x_a
        y[:, ~self.mask] = y_b
        return y, log_det

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        y_a = y[:, self.mask]
        y_b = y[:, ~self.mask]

        st = self.net(y_a)
        s, t = st.chunk(2, dim=-1)
        s = torch.tanh(s) * 0.5

        x_b = (y_b - t) * torch.exp(-s)

        x = torch.empty_like(y)
        x[:, self.mask] = y_a
        x[:, ~self.mask] = x_b
        return x


# =============================================================================
# PCA + Flow Model
# =============================================================================

class PCAFlow(nn.Module):
    """
    PCA for dimensionality reduction + normalizing flow for density estimation.

    The model is exactly invertible: decode(encode(x)) = x (up to PCA truncation).

    Args:
        V: PCA components matrix (k, d) where k is latent dim, d is coord dim
        mean: Mean coordinates (d,)
        n_layers: Number of flow layers (ActNorm + Coupling pairs)
        hidden_dim: Hidden dimension in coupling networks
    """

    def __init__(
        self,
        V: torch.Tensor,
        mean: torch.Tensor,
        n_layers: int = 8,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.k = V.shape[0]  # Latent dimension
        self.d = V.shape[1]  # Coordinate dimension (n_atoms * 3)
        self.n_atoms = self.d // 3

        # PCA parameters (fixed)
        self.register_buffer("V", V)
        self.register_buffer("mean", mean)

        # Flow layers
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(ActNorm(self.k))
            self.layers.append(AffineCoupling(self.k, hidden_dim, even_mask=(i % 2 == 0)))

    def coords_to_pca(self, x: torch.Tensor) -> torch.Tensor:
        """Project coordinates to PCA space."""
        flat = x.reshape(-1, self.d)
        return (flat - self.mean) @ self.V.T

    def pca_to_coords(self, pca: torch.Tensor) -> torch.Tensor:
        """Reconstruct coordinates from PCA (approximate due to truncation)."""
        flat = pca @ self.V + self.mean
        return flat.reshape(-1, self.n_atoms, 3)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode: coordinates → latent z.

        Returns (z, log_det) where log_det is the log Jacobian determinant.
        """
        h = self.coords_to_pca(x)
        log_det = torch.zeros(h.shape[0], device=h.device)

        for layer in self.layers:
            h, ld = layer(h)
            log_det = log_det + ld

        return h, log_det

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Decode: latent z → coordinates."""
        h = z
        for layer in reversed(self.layers):
            h = layer.inverse(h)
        return self.pca_to_coords(h)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode coordinates to latent space."""
        z, _ = self.forward(x)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to coordinates."""
        return self.inverse(z)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Compute log probability of coordinates."""
        z, log_det = self.forward(x)
        log_pz = -0.5 * (z ** 2 + np.log(2 * np.pi)).sum(dim=-1)
        return log_pz + log_det

    def sample(self, n_samples: int) -> torch.Tensor:
        """Sample new coordinates from the learned distribution."""
        z = torch.randn(n_samples, self.k, device=self.V.device)
        return self.decode(z)


# =============================================================================
# Training
# =============================================================================

def train_pca_flow(
    coords: np.ndarray,
    latent_dim: int = 12,
    n_layers: int = 8,
    hidden_dim: int = 64,
    n_epochs: int = 200,
    batch_size: int = 128,
    lr: float = 1e-3,
    device: str = "cpu",
    verbose: bool = True,
) -> tuple[PCAFlow, dict]:
    """
    Train a PCA + Flow model on residue coordinates.

    Returns:
        model: Trained PCAFlow model
        info: Dictionary with PCA components, mean, and training stats
    """
    # Compute PCA
    coords_flat = coords.reshape(len(coords), -1)
    mean = coords_flat.mean(axis=0)
    centered = coords_flat - mean
    _, s, Vt = np.linalg.svd(centered, full_matrices=False)

    # Variance explained
    var_explained = (s[:latent_dim] ** 2).sum() / (s ** 2).sum()
    pca_rmsd = np.sqrt(((centered @ Vt[:latent_dim].T @ Vt[:latent_dim] - centered) ** 2).mean())

    if verbose:
        print(f"PCA: {latent_dim} dims capture {var_explained*100:.1f}% variance")
        print(f"PCA reconstruction RMSD: {pca_rmsd:.4f} A")

    # Create model
    V = torch.from_numpy(Vt[:latent_dim]).float()
    mean_t = torch.from_numpy(mean).float()
    model = PCAFlow(V, mean_t, n_layers, hidden_dim).to(device)

    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model: {n_params:,} parameters")

    # Training
    X = torch.from_numpy(coords).float().to(device)
    loader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=True)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    model.train()
    for epoch in range(n_epochs):
        total_loss = 0.0
        for (batch,) in loader:
            optimizer.zero_grad()
            loss = -model.log_prob(batch).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        if verbose and (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}: NLL = {total_loss/len(loader):.2f}")

    # Evaluate
    model.eval()
    with torch.no_grad():
        z = model.encode(X)
        X_recon = model.decode(z)
        # RMSD = sqrt(mean of all squared coordinate differences)
        flow_rmsd = torch.sqrt(((X_recon - X) ** 2).mean()).item()

    if verbose:
        print(f"Flow reconstruction RMSD: {flow_rmsd:.4f} A")

    info = {
        "V": Vt[:latent_dim],
        "mean": mean,
        "var_explained": var_explained,
        "pca_rmsd": pca_rmsd,
        "flow_rmsd": flow_rmsd,
    }

    return model, info


# =============================================================================
# Main
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="PCA + Flow for residue conformations")
    parser.add_argument("--data-dir", type=str,
                        default="/Users/hmblair/academic/data/rna-libraries/pdb/structures/pdb130")
    parser.add_argument("--n-structures", type=int, default=100)
    parser.add_argument("--latent-dim", type=int, default=12)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()

    print("=" * 60)
    print("PCA + Flow for Residue Conformations")
    print("=" * 60)

    # Load data
    data_dir = Path(args.data_dir)
    cif_files = sorted(data_dir.glob("*.cif"))[:args.n_structures]
    print(f"\nLoading {len(cif_files)} structures...")

    coords, atoms = extract_residues(cif_files, Residue.A)
    coords = align_to_frame(coords, atoms, Residue.A)
    print(f"Dataset: {coords.shape[0]} adenosines, {coords.shape[1]} atoms")

    # Original bond lengths
    bond_stats = check_bond_lengths(coords, atoms, Residue.A)
    c1n9_mean = bond_stats["C1'-N9 mean"]
    c1n9_std = bond_stats["C1'-N9 std"]
    print(f"\nOriginal C1'-N9 bond: {c1n9_mean:.3f} +/- {c1n9_std:.3f} A")

    # Train
    print("\n" + "=" * 60)
    print("Training...")
    print("=" * 60 + "\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, info = train_pca_flow(
        coords,
        latent_dim=args.latent_dim,
        n_layers=args.n_layers,
        n_epochs=args.epochs,
        device=device,
    )

    # Sample and check
    print("\n" + "=" * 60)
    print("Sampling...")
    print("=" * 60)

    with torch.no_grad():
        samples = model.sample(50).cpu().numpy()

    sample_stats = check_bond_lengths(samples, atoms, Residue.A)
    s_mean = sample_stats["C1'-N9 mean"]
    s_std = sample_stats["C1'-N9 std"]
    print(f"\nSampled C1'-N9 bond: {s_mean:.3f} +/- {s_std:.3f} A")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"\n  Latent dim:     {args.latent_dim}")
    print(f"  Var explained:  {info['var_explained']*100:.1f}%")
    print(f"  PCA RMSD:       {info['pca_rmsd']:.4f} A")
    print(f"  Flow RMSD:      {info['flow_rmsd']:.4f} A")
    print(f"  Extra error:    {info['flow_rmsd'] - info['pca_rmsd']:.4f} A")

    return model, coords, atoms, info


if __name__ == "__main__":
    main()
