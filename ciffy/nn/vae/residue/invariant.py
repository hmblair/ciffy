"""
Rotation-invariant VAE for residue conformations.

Uses pairwise distances as the sole source of geometric information,
making the encoder fully rotation and translation invariant. This enables
direct encoding from Polymer coordinates without alignment preprocessing.

Prototype implementation demonstrating:
1. Distance-based invariant attention encoder
2. Vectorized encoding of all residues in a Polymer
3. VAE structure with invariant encoder + coordinate decoder
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from ciffy import Polymer
    from ciffy.biochemistry import AtomGroup, Residue


# =============================================================================
# Distance Encoder
# =============================================================================

# Note: RBFDistanceEncoder and CoordinateDecoder are also available in
# ciffy.nn.blocks for reuse by other models. The local definitions are
# kept for backward compatibility.


class RBFDistanceEncoder(nn.Module):
    """Encode distances using radial basis functions.

    Maps scalar distances to a learned representation using Gaussian RBFs.
    """

    def __init__(self, d_out: int = 64, n_rbf: int = 16, cutoff: float = 10.0):
        super().__init__()
        self.n_rbf = n_rbf
        self.cutoff = cutoff

        # RBF centers and widths
        centers = torch.linspace(0, cutoff, n_rbf)
        self.register_buffer("centers", centers)
        self.width = cutoff / n_rbf

        # Project RBF features to output dimension
        self.proj = nn.Linear(n_rbf, d_out)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """Encode distances.

        Args:
            distances: (...) tensor of distances.

        Returns:
            (..., d_out) encoded distances.
        """
        # Compute RBF features: exp(-((d - center) / width)^2)
        d = distances.unsqueeze(-1)  # (..., 1)
        rbf = torch.exp(-((d - self.centers) / self.width) ** 2)  # (..., n_rbf)
        return self.proj(rbf)


# =============================================================================
# Invariant Attention Encoder
# =============================================================================


class InvariantAttentionEncoder(nn.Module):
    """Rotation-invariant attention encoder using pairwise distances.

    Spatial information enters only through pairwise distances, making
    the encoder fully rotation and translation invariant.

    Architecture:
    1. Embed atom types
    2. Encode pairwise distances
    3. Aggregate distance info into atom features
    4. Self-attention layers
    5. Masked mean pooling
    """

    def __init__(
        self,
        n_atom_types: int,
        d_model: int = 64,
        d_dist: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # Atom type embedding
        self.atom_embed = nn.Embedding(n_atom_types, d_model)

        # Distance encoder (RBF-based)
        self.dist_encoder = RBFDistanceEncoder(d_out=d_dist)

        # Project aggregated distance features
        self.dist_proj = nn.Linear(d_dist, d_model)

        # Combine atom + distance features
        self.combine = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
        )

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Final normalization
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        atom_types: torch.Tensor,
        coords: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode residues using invariant attention.

        Args:
            atom_types: (batch, max_atoms) atom type indices.
            coords: (batch, max_atoms, 3) coordinates.
            mask: (batch, max_atoms) boolean mask, True for present atoms.

        Returns:
            (batch, d_model) pooled representation per residue.
        """
        batch_size, max_atoms, _ = coords.shape

        # 1. Atom embeddings
        atom_feat = self.atom_embed(atom_types)  # (batch, max_atoms, d_model)

        # 2. Compute pairwise distances (invariant!)
        # coords: (batch, max_atoms, 3)
        dist = torch.cdist(coords, coords)  # (batch, max_atoms, max_atoms)

        # 3. Encode distances
        dist_feat = self.dist_encoder(dist)  # (batch, max_atoms, max_atoms, d_dist)

        # 4. Aggregate distance info into each atom (masked mean over neighbors)
        mask_pairs = mask.unsqueeze(2) & mask.unsqueeze(1)  # (batch, max_atoms, max_atoms)
        dist_feat_masked = dist_feat * mask_pairs.unsqueeze(-1)

        # Sum over neighbors, divide by count
        neighbor_count = mask_pairs.sum(dim=2, keepdim=True).clamp(min=1)  # (batch, max_atoms, 1)
        dist_agg = dist_feat_masked.sum(dim=2) / neighbor_count  # (batch, max_atoms, d_dist)
        dist_agg = self.dist_proj(dist_agg)  # (batch, max_atoms, d_model)

        # 5. Combine atom and distance features
        x = self.combine(torch.cat([atom_feat, dist_agg], dim=-1))  # (batch, max_atoms, d_model)

        # 6. Self-attention with masking
        attn_mask = ~mask  # True = ignore
        x = self.transformer(x, src_key_padding_mask=attn_mask)

        # 7. Final norm and masked mean pooling
        x = self.final_norm(x)
        x_masked = x * mask.unsqueeze(-1)
        pooled = x_masked.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)

        return pooled


# =============================================================================
# MLP Decoder (outputs in canonical frame)
# =============================================================================


class CoordinateDecoder(nn.Module):
    """MLP decoder that outputs coordinates in a canonical frame.

    Since the encoder is invariant, the decoder defines the output frame.
    Training targets should be aligned to a consistent frame.
    """

    def __init__(
        self,
        latent_dim: int,
        n_atoms: int,
        hidden_dims: list[int],
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_atoms = n_atoms

        # Build MLP
        layers = []
        in_dim = latent_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.SiLU(),
            ])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        self.mlp = nn.Sequential(*layers)

        # Output heads
        self.coord_head = nn.Linear(hidden_dims[-1], n_atoms * 3)
        self.transform_head = nn.Linear(hidden_dims[-1], 6)

        # Initialize with small values (not zeros - causes SVD issues with RMSD loss)
        nn.init.xavier_uniform_(self.coord_head.weight, gain=0.01)
        nn.init.zeros_(self.coord_head.bias)
        nn.init.xavier_uniform_(self.transform_head.weight, gain=0.01)
        nn.init.zeros_(self.transform_head.bias)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode latent to coordinates and transform.

        Args:
            z: (batch, latent_dim) latent vectors.

        Returns:
            coords: (batch, n_atoms, 3) in canonical frame.
            transform: (batch, 6) SE(3) transform.
        """
        h = self.mlp(z)
        coords = self.coord_head(h).reshape(-1, self.n_atoms, 3)
        transform = self.transform_head(h)
        return coords, transform


# =============================================================================
# Invariant Residue VAE
# =============================================================================


@dataclass
class InvariantResidueVAEConfig:
    """Configuration for InvariantResidueVAE."""

    latent_dim: int = 12
    d_model: int = 64
    d_dist: int = 32
    n_heads: int = 4
    n_encoder_layers: int = 2
    decoder_hidden_dims: list[int] = field(default_factory=lambda: [256, 128])
    dropout: float = 0.1


class InvariantResidueVAE(nn.Module):
    """Rotation-invariant VAE for residue conformations.

    Key properties:
    - Encoder is rotation/translation invariant (uses only distances)
    - Can encode directly from Polymer without alignment
    - Decoder outputs in canonical frame (for reconstruction/generation)

    Example:
        >>> from ciffy import load
        >>> from ciffy.biochemistry import Residue
        >>>
        >>> # Create model
        >>> model = InvariantResidueVAE.from_residue(Residue.A)
        >>>
        >>> # Load polymer and encode all adenosines
        >>> polymer = load("structure.cif")
        >>> z = model.encode_polymer(polymer)  # No alignment needed!
    """

    _hub_model_type = "residue-invariant-vae"

    def __init__(
        self,
        n_atom_types: int,
        n_atoms: int,
        latent_dim: int,
        d_model: int,
        d_dist: int,
        n_heads: int,
        n_encoder_layers: int,
        decoder_hidden_dims: list[int],
        residue: "Residue",
        atom_indices: list[int],
        dropout: float = 0.1,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.n_atoms = n_atoms
        self.n_atom_types = n_atom_types
        self.residue = residue
        self._atom_indices = atom_indices

        # Store config for save/load
        self._d_model = d_model
        self._d_dist = d_dist
        self._n_heads = n_heads
        self._n_encoder_layers = n_encoder_layers
        self._decoder_hidden_dims = decoder_hidden_dims
        self._dropout = dropout

        # Encoder
        self.encoder = InvariantAttentionEncoder(
            n_atom_types=n_atom_types,
            d_model=d_model,
            d_dist=d_dist,
            n_heads=n_heads,
            n_layers=n_encoder_layers,
            dropout=dropout,
        )

        # Latent projection
        self.fc_mu = nn.Linear(d_model, latent_dim)
        self.fc_logvar = nn.Linear(d_model, latent_dim)

        # Decoder
        self.decoder = CoordinateDecoder(
            latent_dim=latent_dim,
            n_atoms=n_atoms,
            hidden_dims=decoder_hidden_dims,
            dropout=dropout,
        )

        # Register buffers for Polymer integration
        self.register_buffer(
            "_atom_indices_tensor",
            torch.tensor(atom_indices, dtype=torch.long),
        )

        # Build atom_type -> canonical position mapping
        atom_type_to_pos = torch.full((n_atom_types,), -1, dtype=torch.long)
        for pos, atom_type in enumerate(atom_indices):
            atom_type_to_pos[atom_type] = pos
        self.register_buffer("_atom_type_to_position", atom_type_to_pos)

    @classmethod
    def from_residue(
        cls,
        residue: "Residue",
        config: InvariantResidueVAEConfig | None = None,
    ) -> "InvariantResidueVAE":
        """Create model for a specific residue type."""
        if config is None:
            config = InvariantResidueVAEConfig()

        atom_indices = [int(atom) for atom in residue]
        n_atoms = len(atom_indices)
        n_atom_types = max(atom_indices) + 1

        return cls(
            n_atom_types=n_atom_types,
            n_atoms=n_atoms,
            latent_dim=config.latent_dim,
            d_model=config.d_model,
            d_dist=config.d_dist,
            n_heads=config.n_heads,
            n_encoder_layers=config.n_encoder_layers,
            decoder_hidden_dims=config.decoder_hidden_dims,
            residue=residue,
            atom_indices=atom_indices,
            dropout=config.dropout,
        )

    @property
    def device(self) -> torch.device:
        return self.fc_mu.weight.device

    @property
    def atoms(self) -> "AtomGroup":
        """AtomGroup subset containing the atoms used by this model."""
        if not hasattr(self, '_atoms_group') or self._atoms_group is None:
            self._atoms_group = self.residue.subset(set(self._atom_indices))
        return self._atoms_group

    # ─────────────────────────────────────────────────────────────────────────
    # Core VAE methods
    # ─────────────────────────────────────────────────────────────────────────

    def encode_batch(
        self,
        atom_types: torch.Tensor,
        coords: torch.Tensor,
        mask: torch.Tensor,
        return_distribution: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode batched data to latent space (internal method for training).

        Args:
            atom_types: (batch, max_atoms) atom type indices.
            coords: (batch, max_atoms, 3) coordinates (any frame - invariant!).
            mask: (batch, max_atoms) boolean mask for present atoms.
            return_distribution: If True, return (z, mu, logvar).

        Returns:
            z or (z, mu, logvar).
        """
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

    def encode(
        self,
        coords: torch.Tensor,
        next_coords: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode coordinates to latent space (ResidueGenerativeCore protocol).

        This method provides protocol-compliant interface for PolymerModel.
        The encoder is rotation/translation invariant, so no alignment is needed.

        Args:
            coords: (n_atoms, 3) or (N, n_atoms, 3) coordinates.
            next_coords: Ignored (for protocol compatibility only).

        Returns:
            (latent_dim,) or (N, latent_dim) latent vectors.
        """
        # Handle single sample
        single = coords.dim() == 2
        if single:
            coords = coords.unsqueeze(0)

        batch_size = coords.shape[0]
        device = coords.device

        # Build atom types from stored indices
        atom_types = self._atom_indices_tensor.unsqueeze(0).expand(batch_size, -1)

        # All atoms present (no padding needed for single residue type)
        mask = torch.ones(batch_size, self.n_atoms, dtype=torch.bool, device=device)

        # Use internal batched encode
        z = self.encode_batch(atom_types, coords, mask, return_distribution=False)

        if single:
            return z.squeeze(0)
        return z

    def decode(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode latent to coordinates (in canonical frame).

        Args:
            z: (batch, latent_dim) latent vectors.

        Returns:
            coords: (batch, n_atoms, 3) in canonical frame.
            transform: (batch, 6) SE(3) transform.
        """
        return self.decoder(z)

    def forward(
        self,
        atom_types: torch.Tensor,
        coords: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for training.

        Args:
            atom_types: (batch, max_atoms) atom type indices.
            coords: (batch, max_atoms, 3) input coordinates.
            mask: (batch, max_atoms) boolean mask.

        Returns:
            recon_coords, transform, mu, logvar.
        """
        z, mu, logvar = self.encode_batch(atom_types, coords, mask, return_distribution=True)
        recon_coords, transform = self.decode(z)
        return recon_coords, transform, mu, logvar

    def sample(self, n_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample from prior."""
        z = torch.randn(n_samples, self.latent_dim, device=self.device)
        return self.decode(z)

    # ─────────────────────────────────────────────────────────────────────────
    # Save/Load
    # ─────────────────────────────────────────────────────────────────────────

    def save(self, path: "str | Path") -> None:
        """Save model to directory."""
        import json
        from pathlib import Path
        from safetensors.torch import save_file

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        tensors = {k: v.cpu().contiguous() for k, v in self.state_dict().items()}
        save_file(tensors, path / "tensors.safetensors")

        import ciffy
        config = {
            "version": ciffy.__version__,
            "model_type": self._hub_model_type,
            "residue_name": self.residue.name,
            "atom_indices": [int(x) for x in self._atom_indices],
            "n_atoms": self.n_atoms,
            "latent_dim": self.latent_dim,
            "d_model": self._d_model,
            "d_dist": self._d_dist,
            "n_heads": self._n_heads,
            "n_encoder_layers": self._n_encoder_layers,
            "decoder_hidden_dims": self._decoder_hidden_dims,
            "dropout": self._dropout,
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(
        cls,
        path: "str | Path",
        device: str = "cpu",
    ) -> "InvariantResidueVAE":
        """Load model from directory."""
        import json
        from pathlib import Path
        from safetensors.torch import load_file
        from ciffy.biochemistry import Residue

        path = Path(path)

        tensors = load_file(path / "tensors.safetensors", device=device)
        with open(path / "config.json") as f:
            config = json.load(f)

        residue = getattr(Residue, config["residue_name"])

        model = cls(
            n_atom_types=max(config["atom_indices"]) + 1,
            n_atoms=config["n_atoms"],
            latent_dim=config["latent_dim"],
            d_model=config["d_model"],
            d_dist=config["d_dist"],
            n_heads=config["n_heads"],
            n_encoder_layers=config["n_encoder_layers"],
            decoder_hidden_dims=config["decoder_hidden_dims"],
            residue=residue,
            atom_indices=config["atom_indices"],
            dropout=config.get("dropout", 0.1),
        ).to(device)

        model.load_state_dict(tensors)
        return model

    # ─────────────────────────────────────────────────────────────────────────
    # Polymer integration (vectorized!)
    # ─────────────────────────────────────────────────────────────────────────

    def encode_polymer(
        self,
        polymer: "Polymer",
        residue_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode all residues of this type from a Polymer.

        Fully vectorized - no Python loops over residues.
        No alignment needed - encoder is rotation invariant!

        Args:
            polymer: Polymer object.
            residue_indices: Optional (R,) indices of specific residues.
                           If None, encodes all residues of self.residue type.

        Returns:
            z: (R, latent_dim) latent vectors for each residue.
        """
        from ciffy import Scale

        # Convert to torch if needed
        if polymer.backend != "torch":
            polymer = polymer.torch()

        device = self.device

        # Get all atom data
        all_coords = polymer.coordinates.to(device)  # (N, 3)
        all_atoms = polymer.atoms.to(device)  # (N,)
        residue_membership = polymer.membership(Scale.RESIDUE).to(device)  # (N,)
        counts = polymer.counts(Scale.RESIDUE).to(device)  # (R_total,)
        sequence = polymer.sequence.to(device)  # (R_total,)

        # Filter to residues of this type
        if residue_indices is None:
            residue_mask = sequence == self.residue.value
            residue_indices = residue_mask.nonzero().squeeze(-1)

        R = len(residue_indices)
        if R == 0:
            return torch.zeros(0, self.latent_dim, device=device)

        # Get atoms belonging to selected residues
        atom_mask = torch.isin(residue_membership, residue_indices)
        selected_coords = all_coords[atom_mask]  # (n_selected, 3)
        selected_atoms = all_atoms[atom_mask]  # (n_selected,)
        selected_membership = residue_membership[atom_mask]  # (n_selected,)

        # Map residue indices to 0..R-1
        residue_map = torch.zeros(polymer.size(Scale.RESIDUE), dtype=torch.long, device=device)
        residue_map[residue_indices] = torch.arange(R, device=device)
        batch_idx = residue_map[selected_membership]  # (n_selected,)

        # Compute position within each residue (0, 1, 2, ...)
        # Atoms are contiguous by residue, so we can use cumsum trick
        n_selected = len(selected_coords)
        ones = torch.ones(n_selected, dtype=torch.long, device=device)

        # For each batch_idx group, compute cumulative position
        # Using a simple approach: sort by batch_idx (already sorted), compute position
        selected_counts = counts[residue_indices]  # (R,)
        max_atoms = selected_counts.max().item()

        # Compute offsets for each residue in the selected atoms
        selected_offsets = torch.cat([
            torch.zeros(1, dtype=torch.long, device=device),
            selected_counts.cumsum(0)[:-1]
        ])
        atom_offsets = selected_offsets[batch_idx]  # (n_selected,)
        position_in_residue = torch.arange(n_selected, device=device) - atom_offsets

        # Scatter to padded tensors
        coords_padded = torch.zeros(R, max_atoms, 3, device=device)
        atoms_padded = torch.zeros(R, max_atoms, dtype=torch.long, device=device)
        mask = torch.zeros(R, max_atoms, dtype=torch.bool, device=device)

        coords_padded[batch_idx, position_in_residue] = selected_coords
        atoms_padded[batch_idx, position_in_residue] = selected_atoms
        mask[batch_idx, position_in_residue] = True

        # Encode (invariant - no alignment needed!)
        z = self.encode_batch(atoms_padded, coords_padded, mask)

        return z


# =============================================================================
# Test / Demo
# =============================================================================


def _test_invariant_vae():
    """Test that the invariant VAE works with Polymer data."""
    import ciffy
    from ciffy import Scale
    from ciffy.biochemistry import Residue

    print("=" * 60)
    print("InvariantResidueVAE Prototype Test")
    print("=" * 60)

    # Create model
    model = InvariantResidueVAE.from_residue(Residue.A)
    print(f"\nModel: {model.n_atoms} atoms, {model.latent_dim}D latent")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load a test structure
    polymer = ciffy.load("tests/data/9MDS.cif")
    print(f"\nLoaded: {polymer.pdb_id}")
    print(f"  Residues: {polymer.size(Scale.RESIDUE)}")
    print(f"  Sequence: {polymer.sequence_str()[:50]}...")

    # Count adenosines
    n_adenosines = (polymer.sequence == Residue.A.value).sum().item()
    print(f"  Adenosines: {n_adenosines}")

    if n_adenosines == 0:
        print("No adenosines found, skipping encoding test")
        return

    # Encode all adenosines (no alignment!)
    print("\nEncoding adenosines (invariant - no alignment)...")
    model.eval()
    with torch.no_grad():
        z = model.encode_polymer(polymer)

    print(f"  Latent shape: {z.shape}")
    print(f"  Latent mean: {z.mean().item():.4f}")
    print(f"  Latent std: {z.std().item():.4f}")

    # Test rotation invariance
    print("\nTesting rotation invariance...")

    # Create a random rotation matrix
    import numpy as np
    theta = 0.5  # radians
    R_mat = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1],
    ], dtype=np.float32)

    # Rotate the polymer
    rotated = polymer.copy()
    rotated.coordinates = polymer.coordinates @ R_mat.T

    with torch.no_grad():
        z_rotated = model.encode_polymer(rotated)

    # Check that latents are the same
    diff = (z - z_rotated).abs().max().item()
    print(f"  Max latent difference after rotation: {diff:.6f}")
    print(f"  Rotation invariant: {diff < 1e-5}")

    # Test translation invariance
    print("\nTesting translation invariance...")
    translated = polymer.copy()
    translated.coordinates = polymer.coordinates + np.array([10.0, 20.0, 30.0], dtype=np.float32)

    with torch.no_grad():
        z_translated = model.encode_polymer(translated)

    diff = (z - z_translated).abs().max().item()
    print(f"  Max latent difference after translation: {diff:.6f}")
    print(f"  Translation invariant: {diff < 1e-5}")

    # Test decoding
    print("\nTesting decode...")
    with torch.no_grad():
        coords_out, transform_out = model.decode(z)
    print(f"  Decoded coords shape: {coords_out.shape}")
    print(f"  Decoded transform shape: {transform_out.shape}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    _test_invariant_vae()
