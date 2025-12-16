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
        5. Modern Transformer (Pre-LN, RoPE, SwiGLU)
        6. Project to dihedral parameters (mu and log_kappa for von Mises)

    The decoder uses parallel (non-autoregressive) decoding, meaning all
    positions are decoded simultaneously. This is efficient and appropriate
    for VAE architectures where the latent z already captures the full
    sequence information.

    Uses residue identity information via PolymerEmbedding, allowing the model
    to generate sequence-appropriate conformations.

    Uses modern transformer architecture:
        - Pre-LN for stable training
        - RoPE for better length generalization
        - SwiGLU activation
        - RMSNorm

    Args:
        latent_dim: Dimension of latent space z
        hidden_dim: Transformer hidden dimension
        residue_dim: Dimension of residue embeddings (default: hidden_dim // 4)
        num_layers: Number of transformer decoder layers
        num_heads: Number of attention heads
        dropout: Dropout probability
        max_seq_len: Maximum sequence length for positional encoding
        use_rope: Whether to use Rotary Position Embeddings (default True)
        use_swiglu: Whether to use SwiGLU activation (default True)

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
        use_rope: bool = True,
        use_swiglu: bool = True,
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
        self.use_rope = use_rope

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

        # Learnable positional encoding (only used if not using RoPE)
        if not use_rope:
            self.pos_encoding = nn.Embedding(max_seq_len, hidden_dim)
        else:
            self.pos_encoding = None

        # Modern Transformer decoder (non-autoregressive, uses encoder architecture)
        self.transformer = Transformer(
            d_model=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            use_rope=use_rope,
            use_swiglu=use_swiglu,
            use_rmsnorm=True,
            max_seq_len=max_seq_len,
            bias=False,
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
        """
        L = dihedral_mask.shape[0]

        # Get residue embeddings: (L, residue_dim)
        res_emb = self.residue_embedding(polymer)

        # Broadcast latent to sequence: (L, latent_dim)
        z_broadcast = z.unsqueeze(0).expand(L, -1)

        # Concatenate latent + residue embeddings: (L, latent_dim + residue_dim)
        h = torch.cat([z_broadcast, res_emb], dim=-1)

        # Project to hidden dim: (L, hidden_dim)
        h = self.input_proj(h)

        # Add positional encoding if not using RoPE
        if self.pos_encoding is not None:
            positions = torch.arange(L, device=z.device)
            pos_emb = self.pos_encoding(positions)  # (L, hidden_dim)
            h = h + pos_emb

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

        Uses a Gaussian approximation to the von Mises distribution,
        which is accurate for large kappa values. The temperature parameter
        controls the sharpness of the distribution: lower temperature gives
        more deterministic (mode-seeking) samples.

        Args:
            z: (latent_dim,) latent vector
            polymer: Polymer object for extracting residue embeddings
            dihedral_mask: (L, D) boolean mask where True = valid dihedral
            temperature: Sampling temperature. 1.0 = standard sampling,
                        <1.0 = sharper/more deterministic,
                        >1.0 = more diverse/random

        Returns:
            (L, D) sampled dihedral angles in radians, range [-pi, pi]
        """
        mu, kappa = self.forward(z, polymer, dihedral_mask)

        # Apply temperature: divide kappa by temperature
        # Higher kappa = sharper distribution, so dividing by temp > 1 makes it broader
        kappa_tempered = kappa / temperature

        # Sample from von Mises using Gaussian approximation
        # For large kappa, von Mises ~ Normal(mu, 1/sqrt(kappa))
        std = 1.0 / torch.sqrt(kappa_tempered)
        samples = mu + std * torch.randn_like(mu)

        # Wrap to [-pi, pi]
        samples = torch.atan2(torch.sin(samples), torch.cos(samples))

        return samples
