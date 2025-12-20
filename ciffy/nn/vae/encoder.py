"""
Transformer-based encoder for backbone dihedral angles.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = None

from .distributions import sincos_encode, MAX_DIHEDRALS_PER_RESIDUE
from ..transformer import Transformer
from ..embedding import PolymerEmbedding
from ...types import Scale

if TYPE_CHECKING:
    from ...polymer import Polymer


class DihedralEncoder(nn.Module if TORCH_AVAILABLE else object):
    """
    Encodes backbone dihedral sequences into a global latent vector.

    Architecture:
        1. Sin/cos encode input dihedrals: (L, D) -> (L, 2*D)
        2. Get residue embeddings from PolymerEmbedding
        3. Concatenate dihedrals + residue embeddings and project to hidden dim
        4. Modern Transformer encoder (Pre-LN, RoPE, SwiGLU, RMSNorm)
        5. Mean pool over sequence -> (hidden_dim,)
        6. Project to latent mean (mu) and log-variance (logvar)

    Uses residue identity information via PolymerEmbedding, allowing the model
    to learn sequence-dependent conformational preferences.

    Args:
        latent_dim: Dimension of latent space z
        hidden_dim: Transformer hidden dimension
        residue_dim: Dimension of residue embeddings (default: hidden_dim // 4)
        num_layers: Number of transformer encoder layers
        num_heads: Number of attention heads
        dropout: Dropout probability
        max_seq_len: Maximum sequence length for positional encoding

    Example:
        >>> encoder = DihedralEncoder(latent_dim=64, hidden_dim=256)
        >>> dihedrals = torch.randn(50, 7)  # (seq_len, num_dihedrals)
        >>> mask = torch.ones(50, 7, dtype=torch.bool)  # valid dihedral mask
        >>> mu, logvar = encoder(dihedrals, mask, polymer)
    """

    def __init__(
        self,
        latent_dim: int = 64,
        hidden_dim: int = 256,
        residue_dim: Optional[int] = None,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for DihedralEncoder. "
                "Install with: pip install torch"
            )
        super().__init__()

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

        # Default residue_dim to hidden_dim // 4
        if residue_dim is None:
            residue_dim = hidden_dim // 4
        self.residue_dim = residue_dim

        # Residue embedding (provides sequence identity information)
        self.residue_embedding = PolymerEmbedding(
            scale=Scale.RESIDUE,
            residue_dim=residue_dim,
        )

        # Input dimension: sin/cos encoded dihedrals + residue embeddings
        dihedral_dim = 2 * MAX_DIHEDRALS_PER_RESIDUE  # 14
        input_dim = dihedral_dim + residue_dim

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Modern Transformer encoder
        self.transformer = Transformer(
            d_model=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )

        # Latent projection (mean and log-variance)
        self.to_mu = nn.Linear(hidden_dim, latent_dim)
        self.to_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(
        self,
        dihedrals: "torch.Tensor",
        dihedral_mask: "torch.Tensor",
        polymer: "Polymer",
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        """
        Encode dihedral sequence to latent distribution parameters.

        Args:
            dihedrals: (L, D) tensor of dihedral angles in radians,
                      where D = MAX_DIHEDRALS_PER_RESIDUE (7)
            dihedral_mask: (L, D) boolean mask where True = valid dihedral
            polymer: Polymer object for extracting residue embeddings

        Returns:
            mu: (latent_dim,) mean of latent distribution
            logvar: (latent_dim,) log-variance of latent distribution

        Raises:
            ValueError: If input shapes are invalid.
            RuntimeError: If tensors contain NaN values.
        """
        # Validate input shapes
        if dihedrals.dim() != 2:
            raise ValueError(
                f"DihedralEncoder: dihedrals must be 2D (seq, num_dihedrals), "
                f"got {dihedrals.dim()}D with shape {tuple(dihedrals.shape)}"
            )

        L, D = dihedrals.shape

        if L <= 0:
            raise ValueError(
                f"DihedralEncoder: sequence length must be positive, got {L}"
            )

        if L > self.max_seq_len:
            raise ValueError(
                f"DihedralEncoder: sequence length {L} exceeds max_seq_len {self.max_seq_len}. "
                f"Increase max_seq_len in model config."
            )

        if D != MAX_DIHEDRALS_PER_RESIDUE:
            raise ValueError(
                f"DihedralEncoder: expected {MAX_DIHEDRALS_PER_RESIDUE} dihedrals per residue, "
                f"got {D}. Input shape: {tuple(dihedrals.shape)}"
            )

        # Validate mask shape
        if dihedral_mask.shape != dihedrals.shape:
            raise ValueError(
                f"DihedralEncoder: dihedral_mask shape {tuple(dihedral_mask.shape)} "
                f"doesn't match dihedrals shape {tuple(dihedrals.shape)}"
            )

        # Check for NaN in input
        if torch.isnan(dihedrals).any():
            nan_count = torch.isnan(dihedrals).sum().item()
            raise RuntimeError(
                f"DihedralEncoder: dihedrals contain {nan_count} NaN values. "
                f"Check your dihedral computation or data loading."
            )

        # Sin/cos encode: (L, D) -> (L, 2*D)
        encoded = sincos_encode(dihedrals)

        # Get residue embeddings: (L, residue_dim)
        res_emb = self.residue_embedding(polymer)

        # Concatenate dihedrals + residue embeddings: (L, 2*D + residue_dim)
        h = torch.cat([encoded, res_emb], dim=-1)

        # Project to hidden dim: (L, hidden_dim)
        h = self.input_proj(h)

        # Add batch dimension for transformer: (1, L, hidden_dim)
        h = h.unsqueeze(0)

        # Create sequence mask from dihedral mask (valid if any dihedral is valid)
        seq_mask = dihedral_mask.any(dim=-1)  # (L,)

        # Create attention mask for transformer (True = masked/ignored)
        attn_mask = ~seq_mask.unsqueeze(0)  # (1, L)

        # Transformer encoding
        h = self.transformer(h, mask=attn_mask)

        # Remove batch dimension: (L, hidden_dim)
        h = h.squeeze(0)

        # Mean pool over sequence (respecting mask)
        h = h * seq_mask.unsqueeze(-1).float()
        seq_length = seq_mask.float().sum().clamp(min=1)
        h_pooled = h.sum(dim=0) / seq_length  # (hidden_dim,)

        # Project to latent parameters
        mu = self.to_mu(h_pooled)
        logvar = self.to_logvar(h_pooled)

        return mu, logvar
