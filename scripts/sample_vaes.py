"""Train VAEs on all RNA residue types and sample mixed chains."""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from ciffy.geometry import FrameIndices
from ciffy.geometry import GeometryConstraints


RESIDUE_TYPES = ["A", "C", "G", "U"]


def load_training_data(residue, data_dir: str, max_files: int = 500):
    """Load aligned residue data for training."""
    from ciffy.nn.flow.residue.data import extract_residues_with_links
    from ciffy.biochemistry import Residue

    if isinstance(residue, str):
        residue = getattr(Residue, residue.upper())

    cif_files = sorted(Path(data_dir).glob("*.cif"))[:max_files]

    coords, transforms, atoms = extract_residues_with_links(
        cif_paths=cif_files,
        residue_type=residue,
        min_coverage=0.9,
        verbose=False,
    )

    n_samples = len(coords)
    n_atoms = len(atoms)
    coords_flat = coords.reshape(n_samples, -1)
    data = np.concatenate([coords_flat, transforms], axis=1)

    return data, atoms, residue, coords, transforms


def create_models_for_residue(residue, atoms, n_features):
    """Create all three VAE architectures for a single residue type."""
    from ciffy.nn.vae.residue.model import ResidueVAE
    from ciffy.nn.vae.residue.attention import AttentionResidueVAE
    from ciffy.nn.vae.residue.invariant import InvariantResidueVAE

    atom_indices = atoms.tolist()
    n_atoms = len(atom_indices)
    n_atom_types = max(atom_indices) + 1
    latent_dim = 12

    models = {}

    models["MLP"] = ResidueVAE(
        input_dim=n_features,
        latent_dim=latent_dim,
        hidden_dims=[256, 128],
        residue=residue,
        atom_indices=atom_indices,
    )

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

    return models


def train_mlp(model, loader, optimizer, constraints, beta=1.0, gamma=0.1, free_bits=0.5, n_geom_samples=16):
    """Train MLP VAE with reconstruction + KL + geometry losses."""
    model.train()
    device = next(model.parameters()).device

    for (batch,) in loader:
        optimizer.zero_grad()
        batch = batch.to(device)

        # Reconstruction loss
        recon, mu, logvar = model(batch)
        recon_loss = F.mse_loss(recon, batch)

        # KL loss
        kl = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar)
        kl = torch.clamp(kl - free_bits, min=0.0).sum(-1).mean()

        # Geometry loss on samples from prior
        z = torch.randn(n_geom_samples, model.latent_dim, device=device)
        coords, transforms = model.decode(z)

        geom_loss = constraints.total_loss(coords, transforms)

        loss = recon_loss + beta * kl + gamma * geom_loss
        loss.backward()
        optimizer.step()


def train_attention(model, loader, optimizer, n_atoms, constraints, beta=1.0, gamma=0.1, free_bits=0.5, n_geom_samples=16):
    """Train Attention VAE with reconstruction + KL + geometry losses."""
    model.train()
    device = next(model.parameters()).device

    for (batch,) in loader:
        optimizer.zero_grad()
        batch = batch.to(device)
        n_coord_dims = n_atoms * 3
        coords = batch[:, :n_coord_dims].reshape(-1, n_atoms, 3)
        transforms = batch[:, n_coord_dims:]
        mask = torch.ones(coords.shape[0], n_atoms, dtype=torch.bool, device=device)

        # Reconstruction loss
        recon_coords, recon_transforms, mu, logvar = model(coords, mask)
        recon_loss = F.mse_loss(recon_coords, coords) + F.mse_loss(recon_transforms, transforms)

        # KL loss
        kl = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar)
        kl = torch.clamp(kl - free_bits, min=0.0).sum(-1).mean()

        # Geometry loss on samples from prior
        z = torch.randn(n_geom_samples, model.latent_dim, device=device)
        sample_coords, sample_transforms = model.decode(z)

        geom_loss = constraints.total_loss(sample_coords, sample_transforms)

        loss = recon_loss + beta * kl + gamma * geom_loss
        loss.backward()
        optimizer.step()


def train_invariant(model, loader, optimizer, n_atoms, atom_indices, constraints, beta=1.0, gamma=0.1, free_bits=0.5, n_geom_samples=16):
    """Train Invariant VAE with reconstruction + KL + geometry losses."""
    from ciffy.operations.metrics import rmsd
    model.train()
    device = next(model.parameters()).device
    atom_types_base = torch.tensor(atom_indices, dtype=torch.long, device=device)

    for (batch,) in loader:
        optimizer.zero_grad()
        batch = batch.to(device)
        n_coord_dims = n_atoms * 3
        coords = batch[:, :n_coord_dims].reshape(-1, n_atoms, 3)
        transforms = batch[:, n_coord_dims:]

        batch_size = coords.shape[0]
        atom_types = atom_types_base.unsqueeze(0).expand(batch_size, -1)
        mask = torch.ones(batch_size, n_atoms, dtype=torch.bool, device=device)

        # Reconstruction loss
        recon_coords, recon_transforms, mu, logvar = model(atom_types, coords, mask)
        coord_rmsd = rmsd(recon_coords, coords, eps=1e-8)
        coord_loss = (coord_rmsd ** 2).mean()
        recon_loss = coord_loss + F.mse_loss(recon_transforms, transforms)

        # KL loss
        kl = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar)
        kl = torch.clamp(kl - free_bits, min=0.0).sum(-1).mean()

        # Geometry loss on samples from prior
        z = torch.randn(n_geom_samples, model.latent_dim, device=device)
        sample_coords, sample_transforms = model.decode(z)

        geom_loss = constraints.total_loss(sample_coords, sample_transforms)

        loss = recon_loss + beta * kl + gamma * geom_loss
        loss.backward()
        optimizer.step()


def train_consolidated(model, residue_data, optimizer, beta=1.0, gamma=0.1, free_bits=0.5, n_geom_samples=16):
    """Train Consolidated VAE with reconstruction + KL + geometry losses."""
    from ciffy.operations.metrics import rmsd
    model.train()
    device = next(model.parameters()).device

    for res_name, rd in residue_data.items():
        residue = rd["residue"]
        n_atoms = rd["n_atoms"]
        atom_indices = rd["atom_indices"]
        constraints = rd["constraints"]

        # Get data
        data = torch.tensor(rd["data"], dtype=torch.float32, device=device)
        n_coord_dims = n_atoms * 3
        coords = data[:, :n_coord_dims].reshape(-1, n_atoms, 3)
        transforms = data[:, n_coord_dims:]

        batch_size = coords.shape[0]
        atom_types = torch.tensor(atom_indices, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
        mask = torch.ones(batch_size, n_atoms, dtype=torch.bool, device=device)

        optimizer.zero_grad()

        # Reconstruction loss
        recon_coords, recon_transforms, mu, logvar = model(atom_types, coords, mask, residue)
        coord_rmsd = rmsd(recon_coords, coords, eps=1e-8)
        coord_loss = (coord_rmsd ** 2).mean()
        recon_loss = coord_loss + F.mse_loss(recon_transforms, transforms)

        # KL loss
        kl = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar)
        kl = torch.clamp(kl - free_bits, min=0.0).sum(-1).mean()

        # Geometry loss on samples from prior
        z = torch.randn(n_geom_samples, model.latent_dim, device=device)
        sample_coords, sample_transforms = model.decode(z, residue)

        geom_loss = constraints.total_loss(sample_coords, sample_transforms)

        loss = recon_loss + beta * kl + gamma * geom_loss
        loss.backward()
        optimizer.step()


def sample_and_save(polymer_model, sequence, output_path, name):
    """Sample a chain using PolymerModel and save as CIF."""
    # Sample using PolymerModel (handles positioning correctly)
    polymer = polymer_model.sample_from_sequence(sequence, n_samples=1)

    # Save
    output_file = output_path / f"{name}_chain.cif"
    polymer.write(str(output_file))
    print(f"  Saved: {output_file}")

    return polymer


def main():
    import argparse
    from ciffy.biochemistry import Residue
    from ciffy.nn.polymer import PolymerModel

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/Users/hmblair/academic/data/structures/rna")
    parser.add_argument("--max-files", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-residues", type=int, default=20)
    parser.add_argument("--output-dir", default="outputs/sampled_chains")
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)

    print("=" * 70)
    print("VAE Chain Sampling (All Residue Types)")
    print("=" * 70)

    # Load data for all residue types
    print("\nLoading training data...")
    residue_data = {}
    for res_name in RESIDUE_TYPES:
        print(f"  {res_name}:", end=" ")
        data, atoms, residue, raw_coords, raw_transforms = load_training_data(
            res_name, args.data_dir, args.max_files
        )
        # Create GeometryConstraints for bond/angle losses
        constraints = GeometryConstraints.from_residue(residue, atoms.tolist())
        # Create FrameIndices for chain assembly (still needed for sampling)
        frame_indices = FrameIndices.from_atoms(atoms, residue)

        residue_data[res_name] = {
            "data": data,
            "atoms": atoms,
            "residue": residue,
            "raw_coords": raw_coords,
            "raw_transforms": raw_transforms,
            "n_atoms": len(atoms),
            "n_features": data.shape[1],
            "atom_indices": atoms.tolist(),
            "constraints": constraints,
            "frame_indices": frame_indices,
        }
        print(f"{len(data)} samples, {len(atoms)} atoms, {constraints.n_bonds} bonds, {constraints.n_angles} angles")

    # Create models for each residue type
    print("\nCreating models...")
    all_models = {arch: {} for arch in ["MLP", "Attention", "Invariant"]}
    all_optimizers = {arch: {} for arch in ["MLP", "Attention", "Invariant"]}

    for res_name, rd in residue_data.items():
        models = create_models_for_residue(rd["residue"], rd["atoms"], rd["n_features"])
        for arch, model in models.items():
            all_models[arch][rd["residue"]] = model
            all_optimizers[arch][rd["residue"]] = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Create consolidated model (single model for all residue types)
    from ciffy.nn.vae.residue import ConsolidatedResidueVAE, ConsolidatedVAEConfig
    residue_atoms = {rd["residue"]: rd["atom_indices"] for rd in residue_data.values()}
    consolidated_config = ConsolidatedVAEConfig(latent_dim=12)
    consolidated_model = ConsolidatedResidueVAE(residue_atoms, consolidated_config)
    consolidated_optimizer = torch.optim.AdamW(consolidated_model.parameters(), lr=1e-3)
    print(f"  Consolidated: {sum(p.numel() for p in consolidated_model.parameters()):,} parameters (shared encoder)")

    # Create data loaders
    loaders = {}
    for res_name, rd in residue_data.items():
        dataset = TensorDataset(torch.tensor(rd["data"], dtype=torch.float32))
        loaders[res_name] = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Training
    print(f"\nTraining for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        beta = min(1.0, epoch / 25)

        for res_name, rd in residue_data.items():
            loader = loaders[res_name]
            n_atoms = rd["n_atoms"]
            atom_indices = rd["atom_indices"]
            residue = rd["residue"]
            constraints = rd["constraints"]

            train_mlp(
                all_models["MLP"][residue],
                loader,
                all_optimizers["MLP"][residue],
                constraints,
                beta=beta,
            )
            train_attention(
                all_models["Attention"][residue],
                loader,
                all_optimizers["Attention"][residue],
                n_atoms,
                constraints,
                beta=beta,
            )
            train_invariant(
                all_models["Invariant"][residue],
                loader,
                all_optimizers["Invariant"][residue],
                n_atoms,
                atom_indices,
                constraints,
                beta=beta,
            )

        # Train consolidated model (all residue types in one pass)
        train_consolidated(
            consolidated_model,
            residue_data,
            consolidated_optimizer,
            beta=beta,
        )

        if epoch % 20 == 0 or epoch == args.epochs - 1:
            print(f"  Epoch {epoch}/{args.epochs}")

    # Generate random mixed sequence
    np.random.seed(42)
    sequence = "".join(np.random.choice(list("acgu"), args.n_residues))
    print(f"\nSampling {args.n_residues}-residue chains with sequence: {sequence}")

    # Sample chains using PolymerModel with all residue models
    for arch in ["MLP", "Attention", "Invariant"]:
        print(f"\n{arch} VAE:")
        # Set all models to eval mode
        for model in all_models[arch].values():
            model.eval()
        polymer_model = PolymerModel(all_models[arch])
        sample_and_save(polymer_model, sequence, output_path, arch)

    # Build atom filter (needed for fragment assembly)
    atom_filter = {rd["residue"].value: rd["atom_indices"] for rd in residue_data.values()}

    # Sample from consolidated model using PolymerModel interface
    print(f"\nConsolidated VAE:")
    consolidated_model.eval()
    polymer_model = PolymerModel(consolidated_model.as_residue_models())
    sample_and_save(polymer_model, sequence, output_path, "Consolidated")

    # Save fragment assembly chain from real data
    print(f"\nFragment assembly (from data):")
    from ciffy.geometry import position_next_residue
    from ciffy import from_sequence

    # Build chain by picking random residues of each type from training data
    gt_coords_list = []
    gt_transforms_list = []
    gt_residues = []

    for char in sequence:
        res_name = char.upper()
        rd = residue_data[res_name]
        idx = np.random.randint(len(rd["raw_coords"]))
        gt_coords_list.append(rd["raw_coords"][idx])
        gt_transforms_list.append(rd["raw_transforms"][idx])
        gt_residues.append(rd["residue"])

    # Position residues sequentially
    positioned_coords = []
    for i in range(len(gt_coords_list)):
        coords_i = gt_coords_list[i]
        residue_i = gt_residues[i]
        rd = residue_data[residue_i.name]
        indices = FrameIndices.from_atoms(rd["atoms"], residue_i)

        if i == 0:
            positioned_coords.append(coords_i)
        else:
            prev_transform = gt_transforms_list[i - 1]
            prev_residue = gt_residues[i - 1]
            prev_rd = residue_data[prev_residue.name]
            prev_indices = FrameIndices.from_atoms(prev_rd["atoms"], prev_residue)
            positioned = position_next_residue(
                positioned_coords[-1], coords_i, prev_transform, prev_indices
            )
            positioned_coords.append(positioned)

    # Create polymer and save
    polymer = from_sequence(sequence, atoms=atom_filter)
    all_coords = np.concatenate(positioned_coords, axis=0).astype(np.float32)
    polymer.coordinates = all_coords

    output_file = output_path / "FragmentAssembly_chain.cif"
    polymer.write(str(output_file))
    print(f"  Saved: {output_file}")

    print("\n" + "=" * 70)
    print(f"Done! Chains saved to {output_path}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
