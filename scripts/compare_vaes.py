"""Compare VAE architectures on residue reconstruction.

Trains ResidueVAE (MLP), AttentionResidueVAE, and InvariantResidueVAE
on the same data and compares reconstruction quality.
"""

import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")


def load_training_data(residue, data_dir: str, max_files: int = 100):
    """Load aligned residue data for training."""
    from ciffy.nn.flow.residue.data import extract_residues_with_links
    from ciffy.biochemistry import Residue
    import numpy as np

    if isinstance(residue, str):
        residue = getattr(Residue, residue.upper())

    cif_files = sorted(Path(data_dir).glob("*.cif"))[:max_files]
    print(f"Loading {len(cif_files)} CIF files for {residue.name}...")

    coords, transforms, atoms = extract_residues_with_links(
        cif_paths=cif_files,
        residue_type=residue,
        min_coverage=0.9,
    )

    # Combine coords and transforms: [coords_flat, transforms]
    n_samples = len(coords)
    n_atoms = len(atoms)
    coords_flat = coords.reshape(n_samples, -1)
    data = np.concatenate([coords_flat, transforms], axis=1)

    print(f"  Extracted {n_samples} residues, {n_atoms} atoms each")
    return data, atoms, residue


def create_models(residue, atoms, n_features):
    """Create all three VAE architectures."""
    from ciffy.nn.vae.residue.model import ResidueVAE
    from ciffy.nn.vae.residue.attention import AttentionResidueVAE
    from ciffy.nn.vae.residue.invariant import InvariantResidueVAE

    atom_indices = atoms.tolist()
    n_atoms = len(atom_indices)
    n_atom_types = max(atom_indices) + 1

    # Common config
    latent_dim = 12

    models = {}

    # 1. MLP VAE
    models["MLP"] = ResidueVAE(
        input_dim=n_features,
        latent_dim=latent_dim,
        hidden_dims=[256, 128],
        residue=residue,
        atom_indices=atom_indices,
    )

    # 2. Attention VAE
    models["Attention"] = AttentionResidueVAE(
        n_atom_types=n_atom_types,
        n_atoms=n_atoms,
        latent_dim=latent_dim,
        d_model=64,
        n_heads=4,
        n_encoder_layers=2,
        decoder_hidden_dims=[256, 128],
        residue=residue,
        atom_indices=atom_indices,
    )

    # 3. Invariant VAE
    models["Invariant"] = InvariantResidueVAE(
        n_atom_types=n_atom_types,
        n_atoms=n_atoms,
        latent_dim=latent_dim,
        d_model=64,
        d_dist=32,
        n_heads=4,
        n_encoder_layers=2,
        decoder_hidden_dims=[256, 128],
        residue=residue,
        atom_indices=atom_indices,
    )

    for name, model in models.items():
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  {name}: {n_params:,} parameters")

    return models


def train_epoch_mlp(model, loader, optimizer, beta=1.0, free_bits=0.5):
    """Train one epoch for MLP VAE."""
    model.train()
    total_loss = 0
    total_recon = 0

    for (batch,) in loader:
        optimizer.zero_grad()

        recon, mu, logvar = model(batch)

        recon_loss = F.mse_loss(recon, batch)
        kl_per_dim = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar)
        kl_per_dim = torch.clamp(kl_per_dim - free_bits, min=0.0)
        kl_loss = kl_per_dim.sum(dim=-1).mean()

        loss = recon_loss + beta * kl_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon_loss.item()

    return total_loss / len(loader), total_recon / len(loader)


def train_epoch_attention(model, loader, optimizer, n_atoms, beta=1.0, free_bits=0.5):
    """Train one epoch for Attention VAE."""
    model.train()
    total_loss = 0
    total_recon = 0

    for (batch,) in loader:
        optimizer.zero_grad()

        # Unpack to coords and transforms
        n_coord_dims = n_atoms * 3
        coords = batch[:, :n_coord_dims].reshape(-1, n_atoms, 3)
        transforms = batch[:, n_coord_dims:]

        # All atoms present
        mask = torch.ones(coords.shape[0], n_atoms, dtype=torch.bool, device=coords.device)

        recon_coords, recon_transforms, mu, logvar = model(coords, mask)

        coord_loss = F.mse_loss(recon_coords, coords)
        transform_loss = F.mse_loss(recon_transforms, transforms)
        recon_loss = coord_loss + transform_loss

        kl_per_dim = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar)
        kl_per_dim = torch.clamp(kl_per_dim - free_bits, min=0.0)
        kl_loss = kl_per_dim.sum(dim=-1).mean()

        loss = recon_loss + beta * kl_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon_loss.item()

    return total_loss / len(loader), total_recon / len(loader)


def train_epoch_invariant(model, loader, optimizer, n_atoms, atom_indices, beta=1.0, free_bits=0.5):
    """Train one epoch for Invariant VAE with Kabsch-aligned RMSD loss."""
    from ciffy.operations.metrics import rmsd

    model.train()
    total_loss = 0
    total_recon = 0

    # Precompute atom types tensor
    atom_types_base = torch.tensor(atom_indices, dtype=torch.long)

    for (batch,) in loader:
        optimizer.zero_grad()

        # Unpack to coords and transforms
        n_coord_dims = n_atoms * 3
        coords = batch[:, :n_coord_dims].reshape(-1, n_atoms, 3)
        transforms = batch[:, n_coord_dims:]

        batch_size = coords.shape[0]
        atom_types = atom_types_base.unsqueeze(0).expand(batch_size, -1).to(coords.device)
        mask = torch.ones(batch_size, n_atoms, dtype=torch.bool, device=coords.device)

        recon_coords, recon_transforms, mu, logvar = model(atom_types, coords, mask)

        # Use Kabsch-aligned RMSD for coordinate loss (handles rotation invariance)
        coord_rmsd = rmsd(recon_coords, coords, eps=1e-8)  # (batch,)
        coord_loss = (coord_rmsd ** 2).mean()  # MSE after alignment

        transform_loss = F.mse_loss(recon_transforms, transforms)
        recon_loss = coord_loss + transform_loss

        kl_per_dim = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar)
        kl_per_dim = torch.clamp(kl_per_dim - free_bits, min=0.0)
        kl_loss = kl_per_dim.sum(dim=-1).mean()

        loss = recon_loss + beta * kl_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon_loss.item()

    return total_loss / len(loader), total_recon / len(loader)


def evaluate_all(models, data, atoms, n_atoms):
    """Evaluate all models on the same data."""
    from ciffy.operations.metrics import rmsd as kabsch_rmsd

    atom_indices = atoms.tolist()
    atom_types_base = torch.tensor(atom_indices, dtype=torch.long)

    results = {}

    n_coord_dims = n_atoms * 3
    coords = data[:, :n_coord_dims].reshape(-1, n_atoms, 3)
    transforms = data[:, n_coord_dims:]

    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            if name == "MLP":
                recon, mu, logvar = model(data)
                recon_coords = recon[:, :n_coord_dims].reshape(-1, n_atoms, 3)
                recon_transforms = recon[:, n_coord_dims:]
            elif name == "Attention":
                mask = torch.ones(coords.shape[0], n_atoms, dtype=torch.bool)
                recon_coords, recon_transforms, mu, logvar = model(coords, mask)
            else:  # Invariant
                batch_size = coords.shape[0]
                atom_types = atom_types_base.unsqueeze(0).expand(batch_size, -1)
                mask = torch.ones(batch_size, n_atoms, dtype=torch.bool)
                recon_coords, recon_transforms, mu, logvar = model(atom_types, coords, mask)

            # Use Kabsch-aligned RMSD for all models (fair comparison)
            coord_rmsd_vals = kabsch_rmsd(recon_coords, coords)  # (batch,)
            coord_rmsd = coord_rmsd_vals.mean().item()
            coord_mse = (coord_rmsd_vals ** 2).mean().item()

            transform_mse = F.mse_loss(recon_transforms, transforms).item()

            # Latent statistics
            z_std = mu.std().item()

            results[name] = {
                "coord_mse": coord_mse,
                "coord_rmsd": coord_rmsd,
                "transform_mse": transform_mse,
                "z_std": z_std,
            }

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/Users/hmblair/academic/data/structures/rna")
    parser.add_argument("--residue", default="A")
    parser.add_argument("--max-files", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    print("=" * 70)
    print("VAE Architecture Comparison")
    print("=" * 70)

    # Load data
    data, atoms, residue = load_training_data(args.residue, args.data_dir, args.max_files)
    n_atoms = len(atoms)
    n_features = data.shape[1]

    # Create data loader
    dataset = TensorDataset(torch.tensor(data, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Create models
    print("\nCreating models...")
    models = create_models(residue, atoms, n_features)

    # Create optimizers
    optimizers = {name: torch.optim.AdamW(model.parameters(), lr=args.lr)
                  for name, model in models.items()}

    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    print("-" * 70)

    atom_indices = atoms.tolist()

    for epoch in range(args.epochs):
        # Beta warmup
        beta = min(1.0, epoch / 25)

        losses = {}
        times = {}

        for name, model in models.items():
            start = time.time()

            if name == "MLP":
                loss, recon = train_epoch_mlp(model, loader, optimizers[name], beta)
            elif name == "Attention":
                loss, recon = train_epoch_attention(model, loader, optimizers[name], n_atoms, beta)
            else:  # Invariant
                loss, recon = train_epoch_invariant(model, loader, optimizers[name], n_atoms, atom_indices, beta)

            times[name] = time.time() - start
            losses[name] = recon

        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"Epoch {epoch:3d} | " +
                  " | ".join(f"{name}: {losses[name]:.4f}" for name in models) +
                  f" | beta={beta:.2f}")

    # Final evaluation
    print("\n" + "=" * 70)
    print("Final Evaluation")
    print("=" * 70)

    data_tensor = torch.tensor(data, dtype=torch.float32)
    results = evaluate_all(models, data_tensor, atoms, n_atoms)

    print(f"\n{'Model':<12} {'Coord MSE':>12} {'Coord RMSD':>12} {'Transform MSE':>14} {'Latent σ':>10}")
    print("-" * 62)
    for name in models:
        r = results[name]
        print(f"{name:<12} {r['coord_mse']:>12.6f} {r['coord_rmsd']:>12.4f} {r['transform_mse']:>14.6f} {r['z_std']:>10.4f}")

    # Relative comparison
    print("\n" + "-" * 62)
    baseline = results["MLP"]["coord_mse"]
    print("Relative to MLP:")
    for name in models:
        ratio = results[name]["coord_mse"] / baseline
        print(f"  {name}: {ratio:.2%} of MLP coord MSE")

    print("\n" + "=" * 70)
    print("Done!")


if __name__ == "__main__":
    main()
