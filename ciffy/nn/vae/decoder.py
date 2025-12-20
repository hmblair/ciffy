"""
Transformer-based decoder for backbone dihedral angles.
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

from .distributions import MAX_DIHEDRALS_PER_RESIDUE
from ..transformer import Transformer
from ..embedding import PolymerEmbedding
from ...types import Scale

if TYPE_CHECKING:
    from ...polymer import Polymer


class DihedralDecoder(nn.Module if TORCH_AVAILABLE else object):
    """
    Decodes global latent vector into backbone dihedral sequence.

    Architecture:
        1. Project latent z to hidden dimension
        2. Broadcast to target sequence length
        3. Get residue embeddings from PolymerEmbedding
        4. Concatenate latent + residue embeddings and project
        5. Modern Transformer (Pre-LN, RoPE, SwiGLU, RMSNorm)
        6. Project to dihedral parameters (mu and log_kappa for von Mises)

    Uses parallel (non-autoregressive) decoding - all positions decoded
    simultaneously. Uses residue identity via PolymerEmbedding for
    sequence-appropriate conformations.

    Args:
        latent_dim: Dimension of latent space z
        hidden_dim: Transformer hidden dimension
        residue_dim: Dimension of residue embeddings (default: hidden_dim // 4)
        num_layers: Number of transformer decoder layers
        num_heads: Number of attention heads
        dropout: Dropout probability
        max_seq_len: Maximum sequence length for positional encoding

    Example:
        >>> decoder = DihedralDecoder(latent_dim=64, hidden_dim=256)
        >>> z = torch.randn(64)  # (latent_dim,)
        >>> mu, kappa = decoder(z, polymer, dihedral_mask)
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
                "PyTorch is required for DihedralDecoder. "
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

        # Input projection: latent + residue embedding -> hidden
        input_dim = latent_dim + residue_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Modern Transformer decoder (non-autoregressive)
        self.transformer = Transformer(
            d_model=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )

        # Output heads for each dihedral angle
        # Predict mu (mean) and log_kappa (log concentration) for von Mises
        self.to_mu = nn.Linear(hidden_dim, MAX_DIHEDRALS_PER_RESIDUE)
        self.to_log_kappa = nn.Linear(hidden_dim, MAX_DIHEDRALS_PER_RESIDUE)

        # Minimum kappa for numerical stability (kappa near 0 = uniform distribution)
        self.min_kappa = 0.1

    def forward(
        self,
        z: "torch.Tensor",
        polymer: "Polymer",
        dihedral_mask: "torch.Tensor",
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        """
        Decode latent vector to dihedral distribution parameters.

        Args:
            z: (latent_dim,) latent vector
            polymer: Polymer object for extracting residue embeddings
            dihedral_mask: (L, D) boolean mask where True = valid dihedral

        Returns:
            mu: (L, D) predicted mean angles in radians
            kappa: (L, D) predicted concentration parameters (> 0)

        Raises:
            ValueError: If input shapes are invalid.
            RuntimeError: If tensors contain NaN values.
        """
        # Validate latent vector
        if z.dim() != 1:
            raise ValueError(
                f"DihedralDecoder: z must be 1D (latent_dim,), "
                f"got {z.dim()}D with shape {tuple(z.shape)}"
            )

        if z.shape[0] != self.latent_dim:
            raise ValueError(
                f"DihedralDecoder: z dimension {z.shape[0]} "
                f"doesn't match latent_dim {self.latent_dim}"
            )

        # Check for NaN in latent
        if torch.isnan(z).any():
            nan_count = torch.isnan(z).sum().item()
            raise RuntimeError(
                f"DihedralDecoder: latent vector contains {nan_count} NaN values. "
                f"This indicates a problem in the encoder or reparameterization."
            )

        # Validate mask shape
        if dihedral_mask.dim() != 2:
            raise ValueError(
                f"DihedralDecoder: dihedral_mask must be 2D (seq, num_dihedrals), "
                f"got {dihedral_mask.dim()}D with shape {tuple(dihedral_mask.shape)}"
            )

        L = dihedral_mask.shape[0]

        if L <= 0:
            raise ValueError(
                f"DihedralDecoder: sequence length must be positive, got {L}"
            )

        if L > self.max_seq_len:
            raise ValueError(
                f"DihedralDecoder: sequence length {L} exceeds max_seq_len {self.max_seq_len}. "
                f"Increase max_seq_len in model config."
            )

        # Get residue embeddings: (L, residue_dim)
        res_emb = self.residue_embedding(polymer)

        # Broadcast latent to sequence: (L, latent_dim)
        z_broadcast = z.unsqueeze(0).expand(L, -1)

        # Concatenate latent + residue embeddings: (L, latent_dim + residue_dim)
        h = torch.cat([z_broadcast, res_emb], dim=-1)

        # Project to hidden dim: (L, hidden_dim)
        h = self.input_proj(h)

        # Add batch dimension for transformer: (1, L, hidden_dim)
        h = h.unsqueeze(0)

        # Create sequence mask from dihedral mask (valid if any dihedral is valid)
        seq_mask = dihedral_mask.any(dim=-1)  # (L,)

        # Create attention mask for transformer (True = masked/ignored)
        attn_mask = ~seq_mask.unsqueeze(0)  # (1, L)

        # Transformer decoding
        h = self.transformer(h, mask=attn_mask)

        # Remove batch dimension: (L, hidden_dim)
        h = h.squeeze(0)

        # Project to dihedral parameters
        mu = self.to_mu(h)  # (L, D)
        log_kappa = self.to_log_kappa(h)
        kappa = torch.exp(log_kappa) + self.min_kappa  # Ensure positive

        return mu, kappa

    def sample(
        self,
        z: "torch.Tensor",
        polymer: "Polymer",
        dihedral_mask: "torch.Tensor",
        temperature: float = 1.0,
    ) -> "torch.Tensor":
        """
        Sample dihedrals from the decoded distribution.

        Uses a Gaussian approximation to the von Mises distribution.
        Temperature controls sharpness: lower = more deterministic.

        Args:
            z: (latent_dim,) latent vector
            polymer: Polymer object for extracting residue embeddings
            dihedral_mask: (L, D) boolean mask where True = valid dihedral
            temperature: Sampling temperature (1.0 = standard, <1.0 = sharper)

        Returns:
            (L, D) sampled dihedral angles in radians, range [-pi, pi]
        """
        mu, kappa = self.forward(z, polymer, dihedral_mask)

        # Apply temperature: divide kappa by temperature
        kappa_tempered = kappa / temperature

        # Sample from von Mises using Gaussian approximation
        std = 1.0 / torch.sqrt(kappa_tempered)
        samples = mu + std * torch.randn_like(mu)

        # Wrap to [-pi, pi]
        samples = torch.atan2(torch.sin(samples), torch.cos(samples))

        return samples
