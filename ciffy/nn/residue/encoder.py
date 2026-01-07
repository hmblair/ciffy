"""Residue-level encoder for polymer latent representation."""

from __future__ import annotations

import torch
import torch.nn as nn

from ciffy import Scale
from ciffy.biochemistry.linking import GLYCOSIDIC_FRAME
from ciffy.geometry.transforms import compute_relative_transform
from ciffy.nn import PolymerEmbedding
from ciffy.nn.layers.transformer import Transformer, RMSNorm
from ciffy.polymer import Polymer

from .packing import pack_by_residue


class ResidueEncoder(nn.Module):
    """
    Encodes polymer residues to per-residue latent vectors.

    Uses packed attention within residues to aggregate atom-level features,
    then adds inter-residue transform information before projecting to
    latent space.

    Uses modern transformer architecture (RMSNorm, SwiGLU, Flash Attention)
    with distance-based attention bias instead of positional encodings.

    Args:
        latent_dim: Dimension of the latent space per residue.
        d_model: Hidden dimension for transformer layers.
        n_heads: Number of attention heads.
        n_layers: Number of transformer encoder layers.
        atom_dim: Dimension for atom type embeddings. Defaults to d_model // 2.
        residue_dim: Dimension for residue type embeddings. Defaults to d_model // 2.
        dropout: Dropout probability.

    Example:
        >>> encoder = ResidueEncoder(latent_dim=32, d_model=128)
        >>> z = encoder(polymer)  # (n_residues, 32)
        >>> z, mu, logvar = encoder(polymer, return_distribution=True)
    """

    def __init__(
        self,
        latent_dim: int = 32,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        atom_dim: int | None = None,
        residue_dim: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.d_model = d_model
        self.n_heads = n_heads

        # Default embedding dimensions
        if atom_dim is None:
            atom_dim = d_model // 2
        if residue_dim is None:
            residue_dim = d_model // 2

        self._atom_dim = atom_dim
        self._residue_dim = residue_dim

        # Atom embeddings
        self.embedding = PolymerEmbedding(
            scale=Scale.ATOM,
            atom_dim=atom_dim,
            residue_dim=residue_dim,
        )
        embed_dim = self.embedding.output_dim

        # Project to model dimension
        self.encoder_proj = nn.Sequential(
            nn.Linear(embed_dim, d_model),
            RMSNorm(d_model),
            nn.SiLU(),
        )

        # Transformer for within-residue attention (no RoPE, uses distance bias)
        self.transformer = Transformer(
            d_model=d_model,
            num_layers=n_layers,
            num_heads=n_heads,
            dropout=dropout,
            use_rope=False,  # Use distance-based bias instead
        )

        # Distance-based attention bias (per head)
        self.distance_bias = nn.Linear(1, n_heads, bias=False)
        nn.init.normal_(self.distance_bias.weight, std=0.1)

        # Transform encoder (always included)
        self.transform_encoder = nn.Sequential(
            nn.Linear(6, d_model),
            RMSNorm(d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Latent projection
        self.to_mu = nn.Linear(d_model, latent_dim)
        self.to_logvar = nn.Linear(d_model, latent_dim)

    @property
    def output_dim(self) -> int:
        """Latent dimension per residue."""
        return self.latent_dim

    def _compute_transforms(self, polymer: Polymer) -> torch.Tensor:
        """Compute inter-residue SE(3) transforms from polymer coordinates."""
        n_residues = polymer.size(Scale.RESIDUE)
        device = polymer.coordinates.device

        aligned, Rs = polymer.align()
        origins = polymer.gather([GLYCOSIDIC_FRAME.origin])[:, 0]

        transforms = torch.zeros(n_residues, 6, device=device)
        for i in range(n_residues - 1):
            transforms[i] = compute_relative_transform(
                origins[i], Rs[i],
                origins[i + 1], Rs[i + 1],
            )
        return transforms

    def forward(
        self,
        polymer: Polymer,
        transforms: torch.Tensor | None = None,
        return_distribution: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode polymer to per-residue latent vectors.

        Args:
            polymer: Input polymer (torch backend).
            transforms: Optional (n_residues, 6) inter-residue transforms.
                       If None, computed automatically from polymer.align().
            return_distribution: If True, return (z, mu, logvar).

        Returns:
            z: (n_residues, latent_dim) latent vectors.
            If return_distribution: (z, mu, logvar) tuple.
        """
        # Embed atoms
        atom_emb = self.embedding(polymer)
        x = self.encoder_proj(atom_emb)

        # Pack for within-residue attention
        counts = polymer.counts(Scale.RESIDUE)
        x_packed, mask = pack_by_residue(x, counts)

        # Pack coordinates and compute pairwise distances
        coords_packed, _ = pack_by_residue(polymer.coordinates, counts)
        diff = coords_packed.unsqueeze(2) - coords_packed.unsqueeze(1)
        dists = diff.norm(dim=-1, keepdim=True)  # (n_res, max_atoms, max_atoms, 1)

        # Convert distances to per-head attention bias: (n_res, n_heads, max_atoms, max_atoms)
        attn_bias = self.distance_bias(dists).permute(0, 3, 1, 2)

        # Mask padded positions in attention (set to -inf for keys)
        # mask shape: (n_res, max_atoms) where True = valid
        pad_mask = ~mask  # True = padded (to be masked out)
        attn_bias = attn_bias.masked_fill(pad_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        # Apply transformer with distance-based attention bias
        # Transformer expects: x (batch, seq, dim), mask (batch, seq), attn_bias (batch, heads, seq, seq)
        x_packed = self.transformer(x_packed, mask=pad_mask, attn_bias=attn_bias)

        # Mean pool within residues (mask already applied by transformer)
        x_packed = x_packed * mask.unsqueeze(-1).float()
        pooled = x_packed.sum(dim=1) / counts.unsqueeze(-1).float()

        # Add transform features
        if transforms is None:
            transforms = self._compute_transforms(polymer)
        transform_feat = self.transform_encoder(transforms)
        pooled = pooled + transform_feat

        # Project to latent space
        mu = self.to_mu(pooled)
        logvar = self.to_logvar(pooled)

        # Reparameterization
        if self.training:
            z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        else:
            z = mu

        if return_distribution:
            return z, mu, logvar
        return z
