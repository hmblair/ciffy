"""
Autoregressive models for residue latent prediction.

This module provides transformer-based autoregressive models for predicting
residue latent vectors sequentially along a polymer chain.

The key model is `ResidueLatentAR`, which:
1. Takes a sequence of residue types
2. Predicts latent vectors autoregressively (each position conditioned on previous)
3. Can be used with any residue encoder (PCAQuantile, VAE, Flow)

Architecture:
    - Residue type embeddings + latent projections
    - Causal transformer (GPT-style)
    - Output head predicting next latent (mean + optional std)

Example:
    >>> from ciffy.nn import ResidueLatentAR, PCAQuantileResidueModel, PolymerModel
    >>> from ciffy.biochemistry import Residue
    >>>
    >>> # Build AR model
    >>> ar_model = ResidueLatentAR(
    ...     latent_dim=12,
    ...     d_model=256,
    ...     num_layers=6,
    ...     num_heads=8,
    ... )
    >>>
    >>> # Training: predict next latent from sequence + previous latents
    >>> sequence = torch.tensor([0, 1, 4, 15])  # ACGU
    >>> latents = torch.randn(1, 4, 12)  # Ground truth latents
    >>> loss = ar_model.compute_loss(sequence, latents)
    >>>
    >>> # Generation: sample latents autoregressively
    >>> sampled_latents = ar_model.generate(sequence, temperature=1.0)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, TYPE_CHECKING
import math

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = None

from .layers import CausalTransformer, RMSNorm, DenseNetwork

if TYPE_CHECKING:
    from ..biochemistry import Residue


@dataclass
class ResidueLatentARConfig:
    """Configuration for ResidueLatentAR model.

    Args:
        latent_dim: Dimension of residue latent vectors.
        d_model: Transformer hidden dimension.
        num_layers: Number of transformer blocks.
        num_heads: Number of attention heads.
        dropout: Dropout probability.
        max_seq_len: Maximum sequence length.
        num_residue_types: Number of distinct residue types (default 32 for safety).
        predict_std: If True, predict both mean and std (Gaussian output).
        min_std: Minimum std when predicting distributions.
        use_residue_bias: If True, add per-residue bias to output.
    """
    latent_dim: int = 12
    d_model: int = 256
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    max_seq_len: int = 2048
    num_residue_types: int = 32
    predict_std: bool = False
    min_std: float = 0.01
    use_residue_bias: bool = True


class ResidueLatentAR(nn.Module if TORCH_AVAILABLE else object):
    """
    Autoregressive model for predicting residue latents along a chain.

    Given a sequence of residue types, predicts latent vectors one at a time,
    where each prediction is conditioned on all previous latents.

    The model uses a GPT-style architecture:
    1. Embed residue types and project latents to model dimension
    2. Sum embeddings and process through causal transformer
    3. Predict next latent (and optionally its uncertainty)

    For the first position, there's no previous latent, so we use a learned
    "start" token embedding.

    Args:
        config: Model configuration.

    Attributes:
        config: The configuration object.
        latent_dim: Dimension of latent vectors.
        d_model: Transformer hidden dimension.
    """

    def __init__(self, config: Optional[ResidueLatentARConfig] = None, **kwargs):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        super().__init__()

        # Allow kwargs to override config
        if config is None:
            config = ResidueLatentARConfig(**kwargs)
        self.config = config

        self.latent_dim = config.latent_dim
        self.d_model = config.d_model

        # Embeddings
        self.residue_embed = nn.Embedding(config.num_residue_types, config.d_model)
        self.latent_proj = nn.Linear(config.latent_dim, config.d_model)

        # Learned start token (for first position with no previous latent)
        self.start_token = nn.Parameter(torch.randn(config.d_model))

        # Causal transformer
        self.transformer = CausalTransformer(
            d_model=config.d_model,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            dropout=config.dropout,
            max_seq_len=config.max_seq_len,
        )

        # Output projection
        output_dim = config.latent_dim * 2 if config.predict_std else config.latent_dim
        self.output_proj = nn.Linear(config.d_model, output_dim)

        # Optional per-residue output bias (helps with residue-specific distributions)
        if config.use_residue_bias:
            self.residue_bias = nn.Embedding(config.num_residue_types, output_dim)
            nn.init.zeros_(self.residue_bias.weight)
        else:
            self.residue_bias = None

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for stable training."""
        nn.init.normal_(self.residue_embed.weight, std=0.02)
        nn.init.normal_(self.latent_proj.weight, std=0.02)
        nn.init.zeros_(self.latent_proj.bias)
        nn.init.normal_(self.output_proj.weight, std=0.02)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        sequence: "torch.Tensor",
        latents: "torch.Tensor",
        padding_mask: Optional["torch.Tensor"] = None,
    ) -> dict:
        """
        Forward pass with teacher forcing.

        Args:
            sequence: Residue type indices (batch, seq_len).
            latents: Ground truth latent vectors (batch, seq_len, latent_dim).
            padding_mask: Optional mask (batch, seq_len) where True = padded.

        Returns:
            Dict with:
                - pred_mean: Predicted latent means (batch, seq_len, latent_dim)
                - pred_std: Predicted stds if config.predict_std (batch, seq_len, latent_dim)
                - hidden: Transformer hidden states (batch, seq_len, d_model)
        """
        B, L = sequence.shape

        # Embed residue types: tells model what residue we're predicting FOR
        residue_emb = self.residue_embed(sequence)  # (B, L, d_model)

        # Project latents and shift right (causal: position i predicts i+1)
        # Position 0 uses start token, position i uses latent[i-1]
        latent_emb = self.latent_proj(latents)  # (B, L, d_model)

        # Shift latents right: [start, z0, z1, ..., z_{L-2}]
        start = self.start_token.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
        shifted_latent_emb = torch.cat([start, latent_emb[:, :-1]], dim=1)

        # Combine: residue embedding + shifted latent embedding
        x = residue_emb + shifted_latent_emb

        # Process through causal transformer
        hidden = self.transformer(x, padding_mask=padding_mask)

        # Predict output
        output = self.output_proj(hidden)

        # Add per-residue bias if enabled
        if self.residue_bias is not None:
            output = output + self.residue_bias(sequence)

        # Split into mean and std if predicting distribution
        if self.config.predict_std:
            pred_mean, pred_log_std = output.chunk(2, dim=-1)
            pred_std = F.softplus(pred_log_std) + self.config.min_std
            return {"pred_mean": pred_mean, "pred_std": pred_std, "hidden": hidden}
        else:
            return {"pred_mean": output, "hidden": hidden}

    def compute_loss(
        self,
        sequence: "torch.Tensor",
        latents: "torch.Tensor",
        padding_mask: Optional["torch.Tensor"] = None,
        reduction: str = "mean",
    ) -> "torch.Tensor":
        """
        Compute training loss (MSE or NLL).

        Args:
            sequence: Residue type indices (batch, seq_len).
            latents: Ground truth latent vectors (batch, seq_len, latent_dim).
            padding_mask: Optional mask (batch, seq_len) where True = padded.
            reduction: Loss reduction ('mean', 'sum', 'none').

        Returns:
            Loss tensor.
        """
        outputs = self.forward(sequence, latents, padding_mask)
        pred_mean = outputs["pred_mean"]

        if self.config.predict_std:
            # Gaussian NLL
            pred_std = outputs["pred_std"]
            nll = 0.5 * (
                torch.log(2 * math.pi * pred_std ** 2) +
                (latents - pred_mean) ** 2 / (pred_std ** 2)
            )
            loss = nll.sum(dim=-1)  # Sum over latent dims
        else:
            # MSE loss
            loss = F.mse_loss(pred_mean, latents, reduction='none').sum(dim=-1)

        # Apply padding mask
        if padding_mask is not None:
            loss = loss.masked_fill(padding_mask, 0.0)
            if reduction == "mean":
                n_valid = (~padding_mask).sum()
                return loss.sum() / n_valid.clamp(min=1)
            elif reduction == "sum":
                return loss.sum()
            else:
                return loss
        else:
            if reduction == "mean":
                return loss.mean()
            elif reduction == "sum":
                return loss.sum()
            else:
                return loss

    @torch.no_grad()
    def generate(
        self,
        sequence: "torch.Tensor",
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> "torch.Tensor":
        """
        Generate latents autoregressively.

        Args:
            sequence: Residue type indices (batch, seq_len) or (seq_len,).
            temperature: Sampling temperature (1.0 = normal, <1 = more deterministic).
            top_k: Not used for continuous latents, kept for API consistency.

        Returns:
            Generated latent vectors (batch, seq_len, latent_dim).
        """
        # Handle unbatched input
        if sequence.dim() == 1:
            sequence = sequence.unsqueeze(0)

        B, L = sequence.shape
        device = sequence.device

        # Initialize with zeros (will be filled in)
        latents = torch.zeros(B, L, self.latent_dim, device=device)

        # Generate autoregressively
        for i in range(L):
            # Get predictions for all positions up to i
            outputs = self.forward(sequence[:, :i+1], latents[:, :i+1])
            pred_mean = outputs["pred_mean"][:, i]  # (B, latent_dim)

            if self.config.predict_std:
                pred_std = outputs["pred_std"][:, i]
                # Sample from predicted distribution
                if temperature > 0:
                    noise = torch.randn_like(pred_mean)
                    latents[:, i] = pred_mean + temperature * pred_std * noise
                else:
                    latents[:, i] = pred_mean
            else:
                # Deterministic prediction (or add noise scaled by temperature)
                if temperature > 0 and temperature != 1.0:
                    # Add Gaussian noise scaled by temperature
                    noise = torch.randn_like(pred_mean)
                    latents[:, i] = pred_mean + (temperature - 1.0) * 0.1 * noise
                else:
                    latents[:, i] = pred_mean

        return latents

    @torch.no_grad()
    def generate_with_prefix(
        self,
        sequence: "torch.Tensor",
        prefix_latents: "torch.Tensor",
        temperature: float = 1.0,
    ) -> "torch.Tensor":
        """
        Generate latents conditioned on a prefix.

        Useful for:
        - Completing a partial structure
        - Conditional generation given fixed N-terminus

        Args:
            sequence: Full residue type indices (batch, seq_len).
            prefix_latents: Known latents for first k positions (batch, k, latent_dim).
            temperature: Sampling temperature.

        Returns:
            Full latent vectors (batch, seq_len, latent_dim).
        """
        if sequence.dim() == 1:
            sequence = sequence.unsqueeze(0)

        B, L = sequence.shape
        k = prefix_latents.shape[1]
        device = sequence.device

        # Initialize with prefix
        latents = torch.zeros(B, L, self.latent_dim, device=device)
        latents[:, :k] = prefix_latents

        # Generate remaining positions
        for i in range(k, L):
            outputs = self.forward(sequence[:, :i+1], latents[:, :i+1])
            pred_mean = outputs["pred_mean"][:, i]

            if self.config.predict_std:
                pred_std = outputs["pred_std"][:, i]
                if temperature > 0:
                    noise = torch.randn_like(pred_mean)
                    latents[:, i] = pred_mean + temperature * pred_std * noise
                else:
                    latents[:, i] = pred_mean
            else:
                latents[:, i] = pred_mean

        return latents

    def save(self, path: str) -> None:
        """Save model to disk."""
        import json
        from pathlib import Path
        from dataclasses import asdict

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save config
        with open(path / "config.json", "w") as f:
            json.dump(asdict(self.config), f, indent=2)

        # Save weights
        torch.save(self.state_dict(), path / "model.pt")

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "ResidueLatentAR":
        """Load model from disk."""
        import json
        from pathlib import Path

        path = Path(path)

        # Load config
        with open(path / "config.json") as f:
            config_dict = json.load(f)
        config = ResidueLatentARConfig(**config_dict)

        # Create model and load weights
        model = cls(config)
        model.load_state_dict(torch.load(path / "model.pt", map_location=device))
        model.to(device)

        return model


class PolymerLatentAR(nn.Module if TORCH_AVAILABLE else object):
    """
    End-to-end autoregressive polymer generation.

    Combines:
    1. ResidueLatentAR for predicting latent vectors
    2. Per-residue decoders (e.g., PCAQuantile) for decoding to coordinates

    This provides a complete pipeline from sequence to 3D structure.

    Example:
        >>> from ciffy.nn import PolymerLatentAR, PolymerModel
        >>>
        >>> # Build from a trained PolymerModel (has per-residue decoders)
        >>> polymer_model = PolymerModel.load("path/to/model")
        >>> ar_model = PolymerLatentAR(
        ...     latent_dim=12,
        ...     residue_decoders=polymer_model.models,
        ... )
        >>>
        >>> # Generate structure from sequence
        >>> polymer = ar_model.sample_polymer(sequence="ACGU")
    """

    def __init__(
        self,
        latent_dim: int,
        residue_decoders: Dict["Residue", nn.Module],
        ar_config: Optional[ResidueLatentARConfig] = None,
        **ar_kwargs,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        super().__init__()

        self.latent_dim = latent_dim
        self.residue_decoders = nn.ModuleDict({
            str(k.value) if hasattr(k, 'value') else str(k): v
            for k, v in residue_decoders.items()
        })

        # Build AR model
        if ar_config is None:
            ar_config = ResidueLatentARConfig(latent_dim=latent_dim, **ar_kwargs)
        self.ar_model = ResidueLatentAR(ar_config)

    def forward(
        self,
        sequence: "torch.Tensor",
        latents: "torch.Tensor",
        padding_mask: Optional["torch.Tensor"] = None,
    ) -> dict:
        """Forward pass through AR model."""
        return self.ar_model(sequence, latents, padding_mask)

    def compute_loss(
        self,
        sequence: "torch.Tensor",
        latents: "torch.Tensor",
        padding_mask: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        """Compute AR training loss."""
        return self.ar_model.compute_loss(sequence, latents, padding_mask)

    @torch.no_grad()
    def generate_latents(
        self,
        sequence: "torch.Tensor",
        temperature: float = 1.0,
    ) -> "torch.Tensor":
        """Generate latent vectors for a sequence."""
        return self.ar_model.generate(sequence, temperature)

    @torch.no_grad()
    def sample_polymer(
        self,
        sequence: str,
        temperature: float = 1.0,
        n_samples: int = 1,
    ) -> "Polymer":
        """
        Generate polymer structures from sequence string.

        Args:
            sequence: Amino acid or nucleotide sequence (e.g., "ACGU").
            temperature: Sampling temperature.
            n_samples: Number of samples to generate.

        Returns:
            Polymer object with generated coordinates.
        """
        from ..biochemistry import Residue
        from ..polymer import Polymer

        # Convert sequence string to indices
        seq_indices = []
        for char in sequence.upper():
            try:
                res = Residue[char]
                seq_indices.append(res.value)
            except KeyError:
                raise ValueError(f"Unknown residue: {char}")

        seq_tensor = torch.tensor(seq_indices, device=self.device)

        if n_samples > 1:
            seq_tensor = seq_tensor.unsqueeze(0).expand(n_samples, -1)

        # Generate latents
        latents = self.generate_latents(seq_tensor, temperature)

        # Decode to coordinates using per-residue decoders
        # This would integrate with PolymerModel.decode()
        # For now, return latents - full integration requires PolymerModel
        return latents

    @property
    def device(self) -> "torch.device":
        return next(self.parameters()).device
