"""
Attention-based VAE for residue conformations.

Uses a Transformer encoder to handle variable/missing atoms naturally
via attention masking, while the MLP decoder outputs the full canonical
atom set for structure prediction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ciffy.nn.hub import HubMixin

if TYPE_CHECKING:
    from ciffy.biochemistry import Residue, AtomGroup
    from ciffy.geometry import FrameIndices


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class AttentionResidueVAEConfig:
    """Configuration for AttentionResidueVAE.

    Args:
        latent_dim: Dimensionality of latent space.
        d_model: Transformer hidden dimension.
        n_heads: Number of attention heads.
        n_encoder_layers: Number of transformer encoder layers.
        decoder_hidden_dims: Hidden layer dimensions for MLP decoder.
        dropout: Dropout probability.
        beta: KL weight for training (beta-VAE).
        free_bits: Min nats per dim before KL penalty (prevents collapse).
    """

    latent_dim: int = 12
    d_model: int = 64
    n_heads: int = 4
    n_encoder_layers: int = 2
    decoder_hidden_dims: list[int] = field(default_factory=lambda: [256, 128])
    dropout: float = 0.1
    beta: float = 1.0
    free_bits: float = 0.5


# =============================================================================
# Attention Encoder
# =============================================================================


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for atom indices."""

    def __init__(self, d_model: int, max_atoms: int = 64):
        super().__init__()
        pe = torch.zeros(max_atoms, d_model)
        position = torch.arange(0, max_atoms, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Add positional encoding based on atom indices.

        Args:
            x: (batch, n_atoms, d_model) features.
            indices: (batch, n_atoms) atom type indices.

        Returns:
            (batch, n_atoms, d_model) features with positional encoding.
        """
        # Use atom indices for position (not sequence position)
        return x + self.pe[indices]


class AttentionEncoder(nn.Module):
    """Transformer encoder for residue coordinates.

    Encodes variable-length atom coordinates to a fixed-size representation
    using self-attention with masking for missing atoms.
    """

    def __init__(
        self,
        n_atom_types: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # Atom type embedding
        self.atom_embed = nn.Embedding(n_atom_types, d_model)

        # Coordinate encoder (3D coords → d_model features)
        self.coord_encoder = nn.Sequential(
            nn.Linear(3, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Positional encoding based on atom type index
        self.pos_encoding = PositionalEncoding(d_model, max_atoms=n_atom_types)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Layer norm before pooling
        self.pre_pool_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        coords: torch.Tensor,
        atom_types: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode coordinates with attention.

        Args:
            coords: (batch, n_atoms, 3) atom coordinates.
            atom_types: (batch, n_atoms) atom type indices (0 to n_atom_types-1).
            mask: (batch, n_atoms) boolean mask, True for present atoms.

        Returns:
            (batch, d_model) pooled representation.
        """
        batch_size, n_atoms, _ = coords.shape

        # Embed atom types and coordinates
        atom_features = self.atom_embed(atom_types)  # (batch, n_atoms, d_model)
        coord_features = self.coord_encoder(coords)  # (batch, n_atoms, d_model)

        # Combine: atom identity + spatial position
        x = atom_features + coord_features

        # Add positional encoding based on atom type
        x = self.pos_encoding(x, atom_types)

        # Transformer with attention mask
        # src_key_padding_mask expects True for positions to IGNORE
        attn_mask = ~mask  # Invert: True where atoms are missing
        x = self.transformer(x, src_key_padding_mask=attn_mask)

        # Pre-pool normalization
        x = self.pre_pool_norm(x)

        # Mean pooling over present atoms only
        mask_expanded = mask.unsqueeze(-1).float()  # (batch, n_atoms, 1)
        x_masked = x * mask_expanded
        pooled = x_masked.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)

        return pooled


# =============================================================================
# MLP Decoder
# =============================================================================


class MLPDecoder(nn.Module):
    """MLP decoder that outputs full canonical atom set.

    Always outputs all atoms regardless of which atoms were present
    during encoding - this is important for structure prediction.
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
        self.output_dim = n_atoms * 3 + 6  # coords + SE(3) transform

        # Build decoder MLP
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

        self.decoder = nn.Sequential(*layers)

        # Separate heads for coords and transform
        self.coord_head = nn.Linear(hidden_dims[-1], n_atoms * 3)
        self.transform_head = nn.Linear(hidden_dims[-1], 6)

        # Initialize near zero for stable training
        nn.init.zeros_(self.coord_head.weight)
        nn.init.zeros_(self.coord_head.bias)
        nn.init.zeros_(self.transform_head.weight)
        nn.init.zeros_(self.transform_head.bias)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode latent to coordinates and transform.

        Args:
            z: (batch, latent_dim) latent vectors.

        Returns:
            coords: (batch, n_atoms, 3) atom coordinates.
            transform: (batch, 6) SE(3) transform [axis-angle, translation].
        """
        h = self.decoder(z)
        coords_flat = self.coord_head(h)
        transform = self.transform_head(h)

        coords = coords_flat.reshape(-1, self.n_atoms, 3)
        return coords, transform


# =============================================================================
# AttentionResidueVAE
# =============================================================================


class AttentionResidueVAE(nn.Module, HubMixin):
    """
    Attention-based VAE for residue conformations.

    Uses a Transformer encoder to handle variable/missing atoms via attention
    masking, while outputting the full canonical atom set for structure
    prediction.

    This model implements ResidueGenerativeCore protocol and works with
    PolymerModel for full-polymer encoding/decoding.

    Key features:
    - Attention encoder handles missing atoms naturally
    - Always decodes to full atom set (for structure prediction)
    - Learns SE(3) transforms for chain assembly
    - Smooth latent space for gradient-based optimization (design)

    Attributes:
        latent_dim: Dimensionality of latent space.
        n_atoms: Number of atoms in canonical set (decoder output).
        residue: The residue type this model handles.

    Example:
        >>> from ciffy.biochemistry import Residue
        >>> from ciffy.nn.vae.residue import AttentionResidueVAE
        >>>
        >>> # Create model
        >>> model = AttentionResidueVAE.from_residue(Residue.A)
        >>>
        >>> # Encode with missing atoms (mask indicates present atoms)
        >>> coords = torch.randn(10, 22, 3)  # batch of 10
        >>> mask = torch.ones(10, 22, dtype=torch.bool)
        >>> mask[:, 5] = False  # atom 5 missing in all samples
        >>> z = model.encode_masked(coords, mask)
        >>>
        >>> # Decode to full atom set
        >>> coords_out, transform = model.decode(z)  # Always (10, 22, 3)
    """

    _hub_model_type = "attention-residue-vae"

    def __init__(
        self,
        n_atom_types: int,
        n_atoms: int,
        latent_dim: int,
        d_model: int,
        n_heads: int,
        n_encoder_layers: int,
        decoder_hidden_dims: list[int],
        residue: "Residue",
        atom_indices: list[int],
        dropout: float = 0.1,
    ):
        """
        Initialize AttentionResidueVAE.

        Args:
            n_atom_types: Total number of atom types in residue.
            n_atoms: Number of atoms in canonical set.
            latent_dim: Latent space dimensionality.
            d_model: Transformer hidden dimension.
            n_heads: Number of attention heads.
            n_encoder_layers: Number of transformer layers.
            decoder_hidden_dims: MLP decoder hidden dims.
            residue: Residue type this model handles.
            atom_indices: List of atom type indices in canonical order.
            dropout: Dropout probability.
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.n_atoms = n_atoms
        self.n_atom_types = n_atom_types
        self.residue = residue
        self._atom_indices = atom_indices
        self._d_model = d_model
        self._n_heads = n_heads
        self._n_encoder_layers = n_encoder_layers
        self._decoder_hidden_dims = decoder_hidden_dims
        self._dropout = dropout

        # Cached properties
        self._atoms_group: "AtomGroup | None" = None
        self._frame_indices: "FrameIndices | None" = None

        # Encoder
        self.encoder = AttentionEncoder(
            n_atom_types=n_atom_types,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_encoder_layers,
            dropout=dropout,
        )

        # Latent projection
        self.fc_mu = nn.Linear(d_model, latent_dim)
        self.fc_logvar = nn.Linear(d_model, latent_dim)

        # Decoder
        self.decoder = MLPDecoder(
            latent_dim=latent_dim,
            n_atoms=n_atoms,
            hidden_dims=decoder_hidden_dims,
            dropout=dropout,
        )

        # Register atom indices as buffer for save/load
        self.register_buffer(
            "_atom_indices_tensor",
            torch.tensor(atom_indices, dtype=torch.long),
        )

    @classmethod
    def from_residue(
        cls,
        residue: "Residue",
        config: AttentionResidueVAEConfig | None = None,
    ) -> "AttentionResidueVAE":
        """Create model for a specific residue type.

        Args:
            residue: Residue type (e.g., Residue.A).
            config: Model configuration. Uses defaults if None.

        Returns:
            Configured AttentionResidueVAE.
        """
        if config is None:
            config = AttentionResidueVAEConfig()

        # Get all atoms for this residue
        atom_indices = [int(atom) for atom in residue]
        n_atoms = len(atom_indices)
        n_atom_types = max(atom_indices) + 1

        return cls(
            n_atom_types=n_atom_types,
            n_atoms=n_atoms,
            latent_dim=config.latent_dim,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_encoder_layers=config.n_encoder_layers,
            decoder_hidden_dims=config.decoder_hidden_dims,
            residue=residue,
            atom_indices=atom_indices,
            dropout=config.dropout,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Properties (ResidueGenerativeCore protocol)
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def device(self) -> torch.device:
        """Device where model parameters reside."""
        return self.fc_mu.weight.device

    @property
    def atoms(self) -> "AtomGroup":
        """AtomGroup subset containing the atoms used by this model."""
        if self._atoms_group is None:
            self._atoms_group = self.residue.subset(set(self._atom_indices))
        return self._atoms_group

    @property
    def frame_indices(self) -> "FrameIndices | None":
        """FrameIndices for glycosidic frame alignment (cached)."""
        if self._frame_indices is None:
            from ciffy.geometry import FrameIndices

            atoms_array = np.array(self._atom_indices, dtype=np.int64)
            try:
                self._frame_indices = FrameIndices.from_atoms(atoms_array, self.residue)
            except ValueError:
                return None
        return self._frame_indices

    # ─────────────────────────────────────────────────────────────────────────
    # Encoding Methods
    # ─────────────────────────────────────────────────────────────────────────

    def encode_masked(
        self,
        coords: torch.Tensor,
        mask: torch.Tensor,
        return_distribution: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode coordinates with attention masking for missing atoms.

        Args:
            coords: (batch, n_atoms, 3) atom coordinates.
            mask: (batch, n_atoms) boolean mask, True for present atoms.
            return_distribution: If True, return (z, mu, logvar).

        Returns:
            z: (batch, latent_dim) latent vectors (sampled if training).
            Or (z, mu, logvar) if return_distribution=True.
        """
        # Get atom type indices
        atom_types = self._atom_indices_tensor.unsqueeze(0).expand(coords.shape[0], -1)

        # Encode with attention
        h = self.encoder(coords, atom_types, mask)

        # Project to latent distribution
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        # Reparameterization
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + std * eps
        else:
            z = mu  # Use mean at inference

        if return_distribution:
            return z, mu, logvar
        return z

    def encode(
        self,
        coords: torch.Tensor,
        next_coords: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode coordinates to latent space (all atoms present).

        This method assumes all atoms are present (for compatibility with
        ResidueGenerativeCore protocol). For missing atoms, use encode_masked().

        Args:
            coords: (n_atoms, 3) or (batch, n_atoms, 3) coordinates.
            next_coords: Ignored (for protocol compatibility).

        Returns:
            (latent_dim,) or (batch, latent_dim) latent vectors.
        """
        single = coords.dim() == 2
        if single:
            coords = coords.unsqueeze(0)

        # All atoms present
        mask = torch.ones(
            coords.shape[0], coords.shape[1],
            dtype=torch.bool, device=coords.device
        )

        z = self.encode_masked(coords, mask)

        if single:
            z = z.squeeze(0)
        return z

    def encode_distribution(
        self,
        coords: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode to latent distribution parameters.

        Args:
            coords: (batch, n_atoms, 3) coordinates.
            mask: (batch, n_atoms) boolean mask. If None, all atoms present.

        Returns:
            mu: (batch, latent_dim) distribution mean.
            logvar: (batch, latent_dim) log variance.
        """
        if mask is None:
            mask = torch.ones(
                coords.shape[0], coords.shape[1],
                dtype=torch.bool, device=coords.device
            )

        _, mu, logvar = self.encode_masked(coords, mask, return_distribution=True)
        return mu, logvar

    # ─────────────────────────────────────────────────────────────────────────
    # Decoding Methods
    # ─────────────────────────────────────────────────────────────────────────

    def decode(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode latent vectors to coordinates and transforms.

        Always outputs the full canonical atom set regardless of which
        atoms were present during encoding.

        Args:
            z: (batch, latent_dim) latent vectors.

        Returns:
            coords: (batch, n_atoms, 3) residue coordinates.
            transforms: (batch, 6) SE(3) transforms [axis-angle, translation].
        """
        return self.decoder(z)

    def sample(self, n_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample new conformations from prior.

        Returns:
            coords: (n_samples, n_atoms, 3) sampled coordinates.
            transforms: (n_samples, 6) sampled SE(3) transforms.
        """
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim, device=self.device)
            return self.decode(z)

    # ─────────────────────────────────────────────────────────────────────────
    # Forward Pass (for training)
    # ─────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        coords: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for training.

        Args:
            coords: (batch, n_atoms, 3) input coordinates.
            mask: (batch, n_atoms) boolean mask for present atoms.

        Returns:
            recon_coords: (batch, n_atoms, 3) reconstructed coordinates.
            transform: (batch, 6) predicted transforms.
            mu: (batch, latent_dim) latent mean.
            logvar: (batch, latent_dim) latent log-variance.
        """
        z, mu, logvar = self.encode_masked(coords, mask, return_distribution=True)
        recon_coords, transform = self.decode(z)
        return recon_coords, transform, mu, logvar

    # ─────────────────────────────────────────────────────────────────────────
    # Save/Load
    # ─────────────────────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save model to directory."""
        import json
        from safetensors.torch import save_file

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        tensors = {k: v.cpu().contiguous() for k, v in self.state_dict().items()}
        save_file(tensors, path / "tensors.safetensors")

        import ciffy
        config = {
            "version": ciffy.__version__,
            "model_type": "attention-residue-vae",
            "residue_name": self.residue.name,
            "atom_indices": [int(x) for x in self._atom_indices],
            "n_atom_types": self.n_atom_types,
            "n_atoms": self.n_atoms,
            "latent_dim": self.latent_dim,
            "d_model": self._d_model,
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
        path: str | Path,
        device: str = "cpu",
    ) -> "AttentionResidueVAE":
        """Load model from directory.

        Args:
            path: Directory containing saved model.
            device: Device to load model to.

        Returns:
            Loaded AttentionResidueVAE.
        """
        import json
        from safetensors.torch import load_file
        from ciffy.biochemistry import Residue

        path = Path(path)

        tensors = load_file(path / "tensors.safetensors", device=device)
        with open(path / "config.json") as f:
            config = json.load(f)

        residue = getattr(Residue, config["residue_name"])

        model = cls(
            n_atom_types=config["n_atom_types"],
            n_atoms=config["n_atoms"],
            latent_dim=config["latent_dim"],
            d_model=config["d_model"],
            n_heads=config["n_heads"],
            n_encoder_layers=config["n_encoder_layers"],
            decoder_hidden_dims=config["decoder_hidden_dims"],
            residue=residue,
            atom_indices=config["atom_indices"],
            dropout=config.get("dropout", 0.1),
        ).to(device)

        model.load_state_dict(tensors)
        return model


def _attention_residue_vae_repr(self) -> str:
    return (
        f"AttentionResidueVAE({self.residue.name}, "
        f"atoms={self.n_atoms}, "
        f"latent_dim={self.latent_dim}, "
        f"d_model={self._d_model})"
    )


AttentionResidueVAE.__repr__ = _attention_residue_vae_repr


__all__ = ["AttentionResidueVAE", "AttentionResidueVAEConfig"]
