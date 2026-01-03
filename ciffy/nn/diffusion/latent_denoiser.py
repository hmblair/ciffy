"""Transformer-based denoiser for latent diffusion.

This module provides the LatentDenoiser, a transformer that predicts noise
in the latent space of a PolymerFlowModel. It takes noisy latent vectors,
a timestep, and residue sequence as input, and outputs predicted noise.

Example:
    >>> from ciffy.nn.diffusion import LatentDenoiser, LatentDenoiserConfig
    >>>
    >>> config = LatentDenoiserConfig(d_model=256, num_layers=6)
    >>> denoiser = LatentDenoiser(config)
    >>>
    >>> noisy_z = torch.randn(batch, n_residues, 12)
    >>> t = torch.randint(0, 1000, (batch,))
    >>> sequence = torch.randint(0, NUM_RESIDUES, (batch, n_residues))
    >>> predicted_noise = denoiser(noisy_z, t, sequence)
"""

from __future__ import annotations

from dataclasses import dataclass, field
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

from ciffy.biochemistry import NUM_RESIDUES

from ..layers.transformer import Transformer, AdaLNTransformer
from .process import TimestepEmbedding


@dataclass
class LatentDenoiserConfig:
    """Configuration for the latent denoiser transformer.

    Attributes:
        latent_dim: Dimension of per-residue latents (default: 12).
        d_model: Transformer hidden dimension.
        num_layers: Number of transformer blocks.
        num_heads: Number of attention heads.
        d_ff: Feedforward hidden dimension (None = auto from SwiGLU).
        dropout: Dropout probability.
        max_seq_len: Maximum sequence length for RoPE.
        num_timesteps: Number of diffusion timesteps (for embedding).
        num_residue_types: Vocabulary size for residue embedding.
        use_adaln: Use Adaptive Layer Normalization for timestep conditioning.
            If True, uses AdaLN (DiT-style). If False, uses additive conditioning.
    """

    latent_dim: int = 12
    d_model: int = 256
    num_layers: int = 6
    num_heads: int = 8
    d_ff: Optional[int] = None
    dropout: float = 0.1
    max_seq_len: int = 2048
    num_timesteps: int = 1000
    num_residue_types: int = field(default_factory=lambda: NUM_RESIDUES)
    use_adaln: bool = True


class LatentDenoiser(nn.Module):
    """Transformer-based denoiser for residue latent diffusion.

    Takes noisy latent vectors, timestep, and residue sequence as input.
    Outputs predicted noise at each residue position.

    Architecture:
        1. Project noisy latents: (n_res, latent_dim) -> (n_res, d_model)
        2. Add sequence embedding: nn.Embedding(num_residues, d_model)
        3. Add timestep embedding: broadcast across sequence
        4. Apply Transformer layers with RoPE
        5. Project back: (n_res, d_model) -> (n_res, latent_dim)

    Example:
        >>> config = LatentDenoiserConfig(d_model=256, num_layers=6)
        >>> denoiser = LatentDenoiser(config)
        >>> noisy_z = torch.randn(batch, n_residues, 12)
        >>> t = torch.randint(0, 1000, (batch,))
        >>> sequence = torch.randint(0, NUM_RESIDUES, (batch, n_residues))
        >>> predicted_noise = denoiser(noisy_z, t, sequence)
    """

    def __init__(self, config: LatentDenoiserConfig) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for LatentDenoiser")
        super().__init__()

        self.config = config
        self.use_adaln = config.use_adaln

        # Input projection: latent_dim -> d_model
        self.input_proj = nn.Linear(config.latent_dim, config.d_model)

        # Sequence embedding (residue type)
        self.sequence_embed = nn.Embedding(config.num_residue_types, config.d_model)

        # Timestep embedding (sinusoidal + MLP)
        self.timestep_embed = TimestepEmbedding(
            max_index=config.num_timesteps,
            embedding_dim=config.d_model,
        )

        # Transformer backbone - choose based on conditioning mode
        if config.use_adaln:
            # AdaLN conditioning: timestep modulates each layer's normalization
            self.transformer = AdaLNTransformer(
                d_model=config.d_model,
                cond_dim=config.d_model,  # timestep embedding dim
                num_layers=config.num_layers,
                num_heads=config.num_heads,
                d_ff=config.d_ff,
                dropout=config.dropout,
                max_seq_len=config.max_seq_len,
            )
        else:
            # Additive conditioning: timestep added to input
            self.transformer = Transformer(
                d_model=config.d_model,
                num_layers=config.num_layers,
                num_heads=config.num_heads,
                d_ff=config.d_ff,
                dropout=config.dropout,
                max_seq_len=config.max_seq_len,
            )

        # Output projection: d_model -> latent_dim
        self.output_proj = nn.Linear(config.d_model, config.latent_dim)

        # Initialize output projection to near-zero for stability
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        noisy_latents: "torch.Tensor",
        timestep: "torch.Tensor",
        sequence: "torch.Tensor",
        mask: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        """Predict noise from noisy latents, timestep, and sequence.

        Args:
            noisy_latents: Noisy per-residue latent vectors (batch, n_res, latent_dim).
            timestep: Diffusion timestep indices (batch,) or scalar.
            sequence: Residue type indices (batch, n_res).
            mask: Optional padding mask (batch, n_res). True = masked/padding.

        Returns:
            Predicted noise with same shape as noisy_latents (batch, n_res, latent_dim).
        """
        batch, n_res, _ = noisy_latents.shape

        # Project noisy latents
        h = self.input_proj(noisy_latents)  # (B, L, d_model)

        # Get embeddings
        seq_emb = self.sequence_embed(sequence)  # (B, L, d_model)
        t_emb = self.timestep_embed(timestep)  # (B, d_model)

        # Apply transformer with appropriate conditioning
        if self.use_adaln:
            # Per-position AdaLN: combine timestep (global) + sequence (per-position)
            # This gives each layer both "what timestep" and "what residue type" info
            cond = t_emb.unsqueeze(1) + seq_emb  # (B, L, d_model)
            h = self.transformer(h, cond, mask=mask)  # (B, L, d_model)
        else:
            # Additive: add both embeddings to input
            h = h + seq_emb
            h = h + t_emb.unsqueeze(1)  # (B, L, d_model)
            h = self.transformer(h, mask=mask)  # (B, L, d_model)

        # Project to noise prediction
        noise_pred = self.output_proj(h)  # (B, L, latent_dim)

        return noise_pred


__all__ = [
    "LatentDenoiserConfig",
    "LatentDenoiser",
]
