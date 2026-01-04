"""Transformer-based denoiser for coordinate diffusion.

This module provides the CoordinateDenoiser, a transformer that predicts noise
on atom coordinates. It takes noisy coordinates, a timestep, and a polymer
(for sequence conditioning) as input, and outputs predicted noise.

Example:
    >>> from ciffy.nn.diffusion import CoordinateDenoiser, CoordinateDenoiserConfig
    >>>
    >>> config = CoordinateDenoiserConfig(d_model=256, num_layers=6)
    >>> denoiser = CoordinateDenoiser(config)
    >>>
    >>> noisy_coords = torch.randn(batch, n_atoms, 3)
    >>> t = torch.randint(0, 1000, (batch,))
    >>> predicted_noise = denoiser(noisy_coords, t, polymer)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

if TYPE_CHECKING:
    import torch
    import torch.nn as nn
    from ciffy import Polymer

from ciffy.biochemistry import Scale

from ..layers.embedding import PolymerEmbedding
from ..layers.transformer import Transformer
from .process import TimestepEmbedding


@dataclass
class CoordinateDenoiserConfig:
    """Configuration for the coordinate denoiser transformer.

    Attributes:
        d_model: Transformer hidden dimension.
        num_layers: Number of transformer blocks.
        num_heads: Number of attention heads.
        d_ff: Feedforward hidden dimension (None = auto from SwiGLU).
        dropout: Dropout probability.
        max_seq_len: Maximum sequence length for RoPE.
        num_timesteps: Number of diffusion timesteps (for embedding).
        residue_dim: Dimension for residue type embedding (split with element_dim).
        element_dim: Dimension for element type embedding (split with residue_dim).
    """

    d_model: int = 256
    num_layers: int = 6
    num_heads: int = 8
    d_ff: Optional[int] = None
    dropout: float = 0.1
    max_seq_len: int = 4096  # Larger for atom-level sequences
    num_timesteps: int = 1000
    # Embedding dimensions (should sum to d_model for clean addition)
    residue_dim: int = 128
    element_dim: int = 128


class CoordinateDenoiser(nn.Module):
    """Transformer-based denoiser for coordinate diffusion.

    Takes noisy atom coordinates, timestep, and polymer as input.
    Outputs predicted noise at each atom position.

    Architecture:
        1. Project noisy coords: (n_atoms, 3) -> (n_atoms, d_model)
        2. Add sequence embedding via PolymerEmbedding (residue + element)
        3. Add timestep embedding: broadcast across atoms
        4. Apply Transformer layers with RoPE
        5. Project back: (n_atoms, d_model) -> (n_atoms, 3)

    Example:
        >>> config = CoordinateDenoiserConfig(d_model=256, num_layers=6)
        >>> denoiser = CoordinateDenoiser(config)
        >>> noisy_coords = torch.randn(batch, n_atoms, 3)
        >>> t = torch.randint(0, 1000, (batch,))
        >>> predicted_noise = denoiser(noisy_coords, t, polymer)
    """

    def __init__(self, config: CoordinateDenoiserConfig) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for CoordinateDenoiser")
        super().__init__()

        self.config = config

        # Input projection: 3 -> d_model
        self.input_proj = nn.Linear(3, config.d_model)

        # Sequence embedding via PolymerEmbedding (at atom level)
        # Uses residue type (expanded to atoms) + element type
        self.polymer_embed = PolymerEmbedding(
            scale=Scale.ATOM,
            residue_dim=config.residue_dim,
            element_dim=config.element_dim,
        )

        # Project polymer embedding to d_model if dimensions don't match
        embed_dim = config.residue_dim + config.element_dim
        if embed_dim != config.d_model:
            self.embed_proj = nn.Linear(embed_dim, config.d_model)
        else:
            self.embed_proj = None

        # Timestep embedding (sinusoidal + MLP)
        self.timestep_embed = TimestepEmbedding(
            max_index=config.num_timesteps,
            embedding_dim=config.d_model,
        )

        # Transformer backbone
        self.transformer = Transformer(
            d_model=config.d_model,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
            dropout=config.dropout,
            max_seq_len=config.max_seq_len,
        )

        # Output projection: d_model -> 3
        self.output_proj = nn.Linear(config.d_model, 3)

        # Initialize output projection to near-zero for stability
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        noisy_coords: "torch.Tensor",
        timestep: "torch.Tensor",
        polymer: "Polymer",
        mask: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        """Predict noise from noisy coordinates, timestep, and polymer.

        Args:
            noisy_coords: Noisy atom coordinates (batch, n_atoms, 3) or (n_atoms, 3).
            timestep: Diffusion timestep indices (batch,) or scalar.
            polymer: Polymer object for sequence conditioning (must be torch backend).
            mask: Optional padding mask (batch, n_atoms). True = masked/padding.

        Returns:
            Predicted noise with same shape as noisy_coords.
        """
        # Handle unbatched input
        unbatched = noisy_coords.dim() == 2
        if unbatched:
            noisy_coords = noisy_coords.unsqueeze(0)
            if isinstance(timestep, int):
                timestep = torch.tensor([timestep], device=noisy_coords.device)
            elif timestep.dim() == 0:
                timestep = timestep.unsqueeze(0)

        batch, n_atoms, _ = noisy_coords.shape

        # Project noisy coordinates
        h = self.input_proj(noisy_coords)  # (B, N, d_model)

        # Get polymer embedding (residue + element at atom level)
        seq_emb = self.polymer_embed(polymer)  # (N, embed_dim)
        if self.embed_proj is not None:
            seq_emb = self.embed_proj(seq_emb)  # (N, d_model)

        # Expand to batch if needed
        if seq_emb.dim() == 2:
            seq_emb = seq_emb.unsqueeze(0).expand(batch, -1, -1)

        h = h + seq_emb

        # Add timestep embedding (broadcast across atoms)
        t_emb = self.timestep_embed(timestep)  # (B, d_model)
        h = h + t_emb.unsqueeze(1)  # (B, N, d_model)

        # Apply transformer
        h = self.transformer(h, mask=mask)  # (B, N, d_model)

        # Project to noise prediction
        noise_pred = self.output_proj(h)  # (B, N, 3)

        # Remove batch dimension if input was unbatched
        if unbatched:
            noise_pred = noise_pred.squeeze(0)

        return noise_pred


__all__ = [
    "CoordinateDenoiserConfig",
    "CoordinateDenoiser",
]
