"""
Consolidated VAE with shared encoder and per-residue decoders.

Architecture:
- Single rotation-invariant encoder handles all residue types (A, C, G, U)
- Separate decoder heads for each residue type (different atom counts)
- Shared latent space enables learning common RNA backbone dynamics

Benefits:
- 4x more training data per encoder
- Shared representations for common backbone atoms
- Simpler deployment (1 model instead of 4)

Example with PolymerModel:
    >>> from ciffy.nn.vae.residue import ConsolidatedResidueVAE
    >>> from ciffy.nn.polymer import PolymerModel
    >>>
    >>> # Train consolidated model
    >>> model = ConsolidatedResidueVAE(residue_atoms)
    >>> # ... training ...
    >>>
    >>> # Use with PolymerModel for chain sampling
    >>> polymer_model = PolymerModel(model.as_residue_models())
    >>> polymer = polymer_model.sample_from_sequence("acgu")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from ciffy.biochemistry import Residue, AtomGroup

from .invariant import InvariantAttentionEncoder
from ciffy.nn.blocks import InputNorm, ResidualBlock, build_mlp_stack


@dataclass
class ConsolidatedVAEConfig:
    """Configuration for ConsolidatedResidueVAE."""

    latent_dim: int = 12
    d_model: int = 64
    d_dist: int = 32
    n_heads: int = 4
    n_encoder_layers: int = 2
    hidden_dims: list[int] = field(default_factory=lambda: [256, 128])
    dropout: float = 0.1
    encoder_type: str = "flat"  # "flat" (MLP on padded data) or "invariant" (attention on distances)
    use_input_norm: bool = True  # Learn input normalization (improves reconstruction)
    use_residual: bool = True  # Residual connections in decoder


class ConsolidatedResidueVAE(nn.Module):
    """VAE with shared encoder and per-residue decoders.

    The encoder is rotation/translation invariant (uses only pairwise distances).
    Each residue type has its own decoder head to handle different atom counts.

    Example:
        >>> from ciffy.biochemistry import Residue
        >>>
        >>> # Define atom indices for each residue type
        >>> residue_atoms = {
        ...     Residue.A: [1, 2, 3, ...],  # 22 atoms
        ...     Residue.C: [1, 2, 3, ...],  # 20 atoms
        ...     Residue.G: [1, 2, 3, ...],  # 23 atoms
        ...     Residue.U: [1, 2, 3, ...],  # 20 atoms
        ... }
        >>>
        >>> model = ConsolidatedResidueVAE(residue_atoms)
        >>>
        >>> # Encode any residue type (internal batched method)
        >>> z, mu, logvar = model.encode_batch(atom_types, coords, mask, return_distribution=True)
        >>>
        >>> # Decode for specific residue type
        >>> coords, transform = model.decode(z, Residue.A)
    """

    def __init__(
        self,
        residue_atoms: dict["Residue", list[int]],
        config: ConsolidatedVAEConfig | None = None,
    ):
        """Initialize consolidated VAE.

        Args:
            residue_atoms: Dict mapping Residue enum to list of atom indices.
            config: Model configuration.
        """
        super().__init__()

        if config is None:
            config = ConsolidatedVAEConfig()

        self.config = config
        self.latent_dim = config.latent_dim
        self.encoder_type = config.encoder_type

        # Store residue info
        self._residue_atoms = residue_atoms
        self._residues = list(residue_atoms.keys())

        # Compute max atoms across all residues (for padding)
        self.max_atoms = max(len(indices) for indices in residue_atoms.values())

        # Compute max atom type index across all residues
        all_atom_indices = []
        for indices in residue_atoms.values():
            all_atom_indices.extend(indices)
        self.n_atom_types = max(all_atom_indices) + 1

        # Input dimension for flat encoder
        self._flat_input_dim = self.max_atoms * 3 + 6
        hidden_dims = config.hidden_dims
        decoder_hidden = list(reversed(hidden_dims))  # Mirror encoder dims

        # Input normalization (optional but recommended)
        self.input_norm = InputNorm(self._flat_input_dim) if config.use_input_norm else None

        # Build encoder based on type
        if config.encoder_type == "flat":
            # Flat MLP encoder on padded coords + transforms
            encoder_layers = []
            in_dim = self._flat_input_dim
            for h_dim in hidden_dims:
                encoder_layers.extend([
                    nn.Linear(in_dim, h_dim),
                    nn.LayerNorm(h_dim),
                    nn.SiLU(),
                ])
                if config.dropout > 0:
                    encoder_layers.append(nn.Dropout(config.dropout))
                in_dim = h_dim

            self.encoder = nn.Sequential(*encoder_layers)
            self.fc_mu = nn.Linear(hidden_dims[-1], config.latent_dim)
            self.fc_logvar = nn.Linear(hidden_dims[-1], config.latent_dim)
            self._uses_mlp_encoder = True
        else:
            # Rotation-invariant attention encoder (separate mu/logvar projection)
            self.encoder = InvariantAttentionEncoder(
                n_atom_types=self.n_atom_types,
                d_model=config.d_model,
                d_dist=config.d_dist,
                n_heads=config.n_heads,
                n_layers=config.n_encoder_layers,
                dropout=config.dropout,
            )
            self.fc_mu = nn.Linear(config.d_model, config.latent_dim)
            self.fc_logvar = nn.Linear(config.d_model, config.latent_dim)
            self._uses_mlp_encoder = False

        # Per-residue decoders with residual blocks and separate heads
        self.decoders = nn.ModuleDict()
        self.decoder_residuals = nn.ModuleDict()
        self.fc_coords = nn.ModuleDict()
        self.fc_transforms = nn.ModuleDict()

        for residue, atom_indices in residue_atoms.items():
            n_atoms = len(atom_indices)
            name = residue.name

            if config.use_residual and len(decoder_hidden) >= 2:
                # Decoder with residual blocks (like ResidueVAE)
                decoder_input = nn.Sequential(
                    nn.Linear(config.latent_dim, decoder_hidden[0]),
                    nn.LayerNorm(decoder_hidden[0]),
                    nn.SiLU(),
                )
                decoder_residual = nn.Sequential(*[
                    ResidualBlock(decoder_hidden[0], decoder_hidden[0], config.dropout)
                    for _ in range(2)
                ])
                decoder_layers = []
                in_dim = decoder_hidden[0]
                for h_dim in decoder_hidden[1:]:
                    decoder_layers.extend([
                        nn.Linear(in_dim, h_dim),
                        nn.LayerNorm(h_dim),
                        nn.SiLU(),
                    ])
                    if config.dropout > 0:
                        decoder_layers.append(nn.Dropout(config.dropout))
                    in_dim = h_dim
                decoder = nn.Sequential(decoder_input, *decoder_layers) if decoder_layers else decoder_input
                self.decoders[name] = decoder
                self.decoder_residuals[name] = decoder_residual
                self._decoder_out_dim = decoder_hidden[-1]
            else:
                # Simple sequential decoder
                decoder = build_mlp_stack(
                    config.latent_dim,
                    decoder_hidden,
                    dropout=config.dropout,
                    zero_init_final=False,
                )
                self.decoders[name] = decoder
                self.decoder_residuals[name] = nn.Identity()
                self._decoder_out_dim = decoder_hidden[-1]

            # Separate output heads for coords and transforms (different scales)
            fc_coord = nn.Linear(self._decoder_out_dim, n_atoms * 3)
            fc_transform = nn.Linear(self._decoder_out_dim, 6)
            nn.init.zeros_(fc_coord.weight)
            nn.init.zeros_(fc_coord.bias)
            nn.init.zeros_(fc_transform.weight)
            nn.init.zeros_(fc_transform.bias)
            self.fc_coords[name] = fc_coord
            self.fc_transforms[name] = fc_transform

        # Register atom indices as buffers for each residue
        for residue, atom_indices in residue_atoms.items():
            self.register_buffer(
                f"_atom_indices_{residue.name}",
                torch.tensor(atom_indices, dtype=torch.long),
            )

    @property
    def device(self) -> torch.device:
        """Device where model parameters reside."""
        return self.fc_mu.weight.device

    @property
    def residues(self) -> list["Residue"]:
        """List of supported residue types."""
        return self._residues

    def get_atom_indices(self, residue: "Residue") -> torch.Tensor:
        """Get atom indices for a residue type."""
        return getattr(self, f"_atom_indices_{residue.name}")

    def encode_batch(
        self,
        atom_types: torch.Tensor,
        coords: torch.Tensor,
        mask: torch.Tensor,
        transforms: torch.Tensor | None = None,
        return_distribution: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode batched residues to latent space.

        Args:
            atom_types: (batch, max_atoms) atom type indices (ignored for flat encoder).
            coords: (batch, max_atoms, 3) coordinates.
            mask: (batch, max_atoms) boolean mask (ignored for flat encoder).
            transforms: (batch, 6) SE(3) transforms (required for flat encoder).
            return_distribution: If True, return (z, mu, logvar).

        Returns:
            z or (z, mu, logvar).
        """
        if self._uses_mlp_encoder:
            # Flat MLP encoder: concatenate coords + transforms
            batch_size = coords.shape[0]
            coords_flat = coords.reshape(batch_size, -1)
            if transforms is None:
                transforms = torch.zeros(batch_size, 6, device=coords.device)
            x = torch.cat([coords_flat, transforms], dim=-1)

            # Apply input normalization
            if self.input_norm is not None:
                x = self.input_norm(x)

            h = self.encoder(x)
            mu = self.fc_mu(h)
            logvar = self.fc_logvar(h)

            if self.training:
                std = torch.exp(0.5 * logvar)
                z = mu + std * torch.randn_like(std)
            else:
                z = mu

            if return_distribution:
                return z, mu, logvar
            return z
        else:
            # Invariant attention encoder: use atom types, coords, mask
            h = self.encoder(atom_types, coords, mask)
            mu = self.fc_mu(h)
            logvar = self.fc_logvar(h)

            if self.training:
                std = torch.exp(0.5 * logvar)
                z = mu + std * torch.randn_like(std)
            else:
                z = mu

            if return_distribution:
                return z, mu, logvar
            return z

    def decode(
        self, z: torch.Tensor, residue: "Residue"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode latent to coordinates for a specific residue type.

        Args:
            z: (batch, latent_dim) latent vectors.
            residue: Target residue type.

        Returns:
            coords: (batch, n_atoms, 3) in canonical frame.
            transform: (batch, 6) SE(3) transform parameters.
        """
        name = residue.name
        decoder = self.decoders[name]
        residual = self.decoder_residuals[name]

        # Apply decoder with residual blocks
        if self.config.use_residual:
            # decoder_input is the first part, then residual, then rest
            h = decoder[0](z)  # First layer (input projection)
            h = residual(h)  # Residual blocks
            h = decoder[1:](h) if len(decoder) > 1 else h  # Remaining layers
        else:
            h = decoder(z)

        # Apply separate output heads
        coords_flat = self.fc_coords[name](h)
        transform = self.fc_transforms[name](h)

        # Unnormalize output to original scale
        if self.input_norm is not None:
            n_atoms = len(self._residue_atoms[residue])
            # Reconstruct full output and unnormalize
            output = torch.zeros(z.shape[0], self._flat_input_dim, device=z.device)
            output[:, :n_atoms * 3] = coords_flat
            output[:, self.max_atoms * 3:self.max_atoms * 3 + 6] = transform
            output = self.input_norm.inverse(output)
            coords_flat = output[:, :n_atoms * 3]
            transform = output[:, self.max_atoms * 3:self.max_atoms * 3 + 6]

        n_atoms = len(self._residue_atoms[residue])
        coords = coords_flat.reshape(-1, n_atoms, 3)
        return coords, transform

    def forward(
        self,
        atom_types: torch.Tensor,
        coords: torch.Tensor,
        mask: torch.Tensor,
        residue: "Residue",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for training.

        Args:
            atom_types: (batch, max_atoms) atom type indices.
            coords: (batch, max_atoms, 3) input coordinates.
            mask: (batch, max_atoms) boolean mask.
            residue: Residue type for decoding.

        Returns:
            recon_coords, transform, mu, logvar.
        """
        z, mu, logvar = self.encode_batch(atom_types, coords, mask, return_distribution=True)
        recon_coords, transform = self.decode(z, residue)
        return recon_coords, transform, mu, logvar

    def sample(
        self, residue: "Residue", n_samples: int = 1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample from prior for a specific residue type.

        Args:
            residue: Target residue type.
            n_samples: Number of samples.

        Returns:
            coords: (n_samples, n_atoms, 3) sampled coordinates.
            transform: (n_samples, 6) sampled transforms.
        """
        z = torch.randn(n_samples, self.latent_dim, device=self.device)
        return self.decode(z, residue)

    def as_residue_models(self) -> dict["Residue", "ConsolidatedResidueView"]:
        """Return dict of residue views for use with PolymerModel.

        Creates lightweight wrapper objects that present a per-residue interface
        while sharing the underlying consolidated model. This allows seamless
        integration with PolymerModel.

        Returns:
            Dict mapping Residue to ConsolidatedResidueView.

        Example:
            >>> model = ConsolidatedResidueVAE(residue_atoms)
            >>> polymer_model = PolymerModel(model.as_residue_models())
            >>> polymer = polymer_model.sample_from_sequence("acgu")
        """
        return {
            residue: ConsolidatedResidueView(self, residue)
            for residue in self._residues
        }

    def save(self, path: str | Path) -> None:
        """Save model to directory.

        Args:
            path: Directory to save to.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save model state
        torch.save(self.state_dict(), path / "model.pt")

        # Save config
        config = {
            "latent_dim": self.config.latent_dim,
            "d_model": self.config.d_model,
            "d_dist": self.config.d_dist,
            "n_heads": self.config.n_heads,
            "n_encoder_layers": self.config.n_encoder_layers,
            "hidden_dims": self.config.hidden_dims,
            "dropout": self.config.dropout,
            "encoder_type": self.config.encoder_type,
            "use_input_norm": self.config.use_input_norm,
            "use_residual": self.config.use_residual,
            "residue_atoms": {
                res.name: indices for res, indices in self._residue_atoms.items()
            },
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "ConsolidatedResidueVAE":
        """Load model from directory.

        Args:
            path: Directory containing saved model.
            device: Device to load model to.

        Returns:
            Loaded ConsolidatedResidueVAE.
        """
        from ciffy.biochemistry import Residue

        path = Path(path)

        # Load config
        with open(path / "config.json") as f:
            config_dict = json.load(f)

        # Reconstruct residue_atoms with Residue keys
        residue_atoms = {
            getattr(Residue, name): indices
            for name, indices in config_dict["residue_atoms"].items()
        }

        # Create config (handle legacy decoder_hidden_dims key)
        hidden_dims = config_dict.get("hidden_dims", config_dict.get("decoder_hidden_dims", [256, 128]))
        config = ConsolidatedVAEConfig(
            latent_dim=config_dict["latent_dim"],
            d_model=config_dict["d_model"],
            d_dist=config_dict["d_dist"],
            n_heads=config_dict["n_heads"],
            n_encoder_layers=config_dict["n_encoder_layers"],
            hidden_dims=hidden_dims,
            dropout=config_dict["dropout"],
            encoder_type=config_dict.get("encoder_type", "flat"),
            use_input_norm=config_dict.get("use_input_norm", True),
            use_residual=config_dict.get("use_residual", True),
        )

        # Create model and load state
        model = cls(residue_atoms, config)
        model.load_state_dict(torch.load(path / "model.pt", map_location=device))
        model.to(device)

        return model


class ConsolidatedResidueView(nn.Module):
    """Wrapper presenting a single-residue view of ConsolidatedResidueVAE.

    This class implements the ResidueGenerativeCore protocol, allowing the
    consolidated model to be used with PolymerModel. Each view wraps the
    same underlying model but presents a residue-specific interface.

    This is a lightweight wrapper - it doesn't copy any parameters, just
    delegates to the consolidated model with the appropriate residue type.
    """

    def __init__(self, model: ConsolidatedResidueVAE, residue: "Residue"):
        """Initialize view for a specific residue type.

        Args:
            model: The underlying consolidated model.
            residue: The residue type this view represents.
        """
        super().__init__()
        # Store reference to parent model (not as submodule to avoid double-counting params)
        self._model = model
        self._residue = residue
        self._atom_indices = model._residue_atoms[residue]
    @property
    def latent_dim(self) -> int:
        """Latent space dimension."""
        return self._model.latent_dim

    @property
    def n_atoms(self) -> int:
        """Number of atoms for this residue type."""
        return len(self._atom_indices)

    @property
    def residue(self) -> "Residue":
        """The residue type this view represents."""
        return self._residue

    @property
    def device(self) -> torch.device:
        """Device where model parameters reside."""
        return self._model.device

    @property
    def atoms(self) -> "AtomGroup":
        """AtomGroup subset containing the atoms used by this residue."""
        if not hasattr(self, '_atoms_group') or self._atoms_group is None:
            self._atoms_group = self._residue.subset(set(self._atom_indices))
        return self._atoms_group

    def encode(
        self,
        coords: torch.Tensor,
        next_coords: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode coordinates to latent space (ResidueGenerativeCore protocol).

        Args:
            coords: (n_atoms, 3) or (batch, n_atoms, 3) coordinates.
            next_coords: Ignored (for protocol compatibility).

        Returns:
            (latent_dim,) or (batch, latent_dim) latent vectors.
        """
        # Handle single sample
        single = coords.dim() == 2
        if single:
            coords = coords.unsqueeze(0)

        batch_size = coords.shape[0]
        device = coords.device

        # For flat encoder, need to pad coords to max_atoms
        if self._model.encoder_type == "flat":
            max_atoms = self._model.max_atoms
            padded_coords = torch.zeros(batch_size, max_atoms, 3, device=device)
            padded_coords[:, :self.n_atoms, :] = coords
            coords = padded_coords

        # Build atom types and mask
        atom_indices = self._model.get_atom_indices(self._residue).to(device)
        atom_types = atom_indices.unsqueeze(0).expand(batch_size, -1)
        mask = torch.ones(batch_size, self.n_atoms, dtype=torch.bool, device=device)

        # Pad mask if needed for flat encoder
        if self._model.encoder_type == "flat":
            max_atoms = self._model.max_atoms
            padded_mask = torch.zeros(batch_size, max_atoms, dtype=torch.bool, device=device)
            padded_mask[:, :self.n_atoms] = mask
            mask = padded_mask

        # Zero transforms for encoding (flat encoder expects them)
        transforms = torch.zeros(batch_size, 6, device=device)

        z = self._model.encode_batch(atom_types, coords, mask, transforms=transforms, return_distribution=False)

        if single:
            return z.squeeze(0)
        return z

    def decode(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode latents to coordinates and transform.

        Args:
            z: (batch, latent_dim) latent vectors.

        Returns:
            coords: (batch, n_atoms, 3) coordinates.
            transform: (batch, 6) SE(3) transform parameters.
        """
        return self._model.decode(z, self._residue)

    def sample(self, n_samples: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample from prior.

        Args:
            n_samples: Number of samples.

        Returns:
            coords: (n_samples, n_atoms, 3) sampled coordinates.
            transform: (n_samples, 6) sampled transforms.
        """
        return self._model.sample(self._residue, n_samples)

    def save(self, path: "str | Path") -> None:
        """No-op save - ConsolidatedResidueView shares underlying model.

        The consolidated model is saved once, and views are reconstructed
        via as_residue_models() on load.
        """
        pass  # Parent ConsolidatedResidueVAE handles saving

    def __repr__(self) -> str:
        return f"ConsolidatedResidueView(residue={self._residue.name}, n_atoms={self.n_atoms})"


def _test_consolidated_vae():
    """Test the consolidated VAE."""
    from ciffy.biochemistry import Residue
    from ciffy.nn.flow.residue.data import extract_residues_with_links
    from pathlib import Path

    print("=" * 60)
    print("ConsolidatedResidueVAE Test")
    print("=" * 60)

    # Load atom indices for each residue type
    data_dir = Path("/Users/hmblair/academic/data/structures/rna")
    cif_files = sorted(data_dir.glob("*.cif"))[:10]

    residue_atoms = {}
    for res_name in ["A", "C", "G", "U"]:
        residue = getattr(Residue, res_name)
        coords, transforms, atoms = extract_residues_with_links(
            cif_paths=cif_files,
            residue_type=residue,
            min_coverage=0.9,
            verbose=False,
        )
        residue_atoms[residue] = atoms.tolist()
        print(f"  {res_name}: {len(atoms)} atoms")

    # Create model
    model = ConsolidatedResidueVAE(residue_atoms)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")
    print(f"  Encoder: {sum(p.numel() for p in model.encoder.parameters()):,}")
    print(f"  Latent proj: {sum(p.numel() for p in model.fc_mu.parameters()) + sum(p.numel() for p in model.fc_logvar.parameters()):,}")
    for name, decoder in model.decoders.items():
        print(f"  Decoder {name}: {sum(p.numel() for p in decoder.parameters()):,}")

    # Test forward pass for each residue type
    print("\nTesting forward pass...")
    for residue in model.residues:
        n_atoms = len(residue_atoms[residue])
        batch_size = 4

        # Create dummy input
        atom_indices = model.get_atom_indices(residue)
        atom_types = atom_indices.unsqueeze(0).expand(batch_size, -1)
        coords = torch.randn(batch_size, n_atoms, 3)
        mask = torch.ones(batch_size, n_atoms, dtype=torch.bool)

        # Forward
        recon_coords, transform, mu, logvar = model(atom_types, coords, mask, residue)
        print(f"  {residue.name}: input {coords.shape} -> output {recon_coords.shape}")

    # Test sampling
    print("\nTesting sampling...")
    model.eval()
    for residue in model.residues:
        coords, transform = model.sample(residue, n_samples=8)
        print(f"  {residue.name}: sampled {coords.shape}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    _test_consolidated_vae()
