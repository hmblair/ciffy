"""
Shared neural network building blocks.

This module provides reusable components for residue-level models:
- InputNorm: Learnable input normalization (like ActNorm)
- ResidualBlock: MLP block with skip connections
- MLPEncoder: General MLP encoder to latent space
- CoordinateDecoder: Decodes latents to coordinates + transforms
- build_mlp_stack: Helper to construct MLP layers
"""

from __future__ import annotations

import torch
import torch.nn as nn


def build_mlp_stack(
    input_dim: int,
    hidden_dims: list[int],
    output_dim: int | None = None,
    dropout: float = 0.0,
    final_activation: bool = False,
    zero_init_final: bool = True,
) -> nn.Sequential:
    """Build an MLP with LayerNorm and SiLU activations.

    Standard architecture used across all residue models:
    Linear -> LayerNorm -> SiLU -> [Dropout] -> repeat -> [Linear output]

    Args:
        input_dim: Input feature dimension.
        hidden_dims: List of hidden layer dimensions.
        output_dim: If provided, adds final linear layer to this dimension.
        dropout: Dropout probability (0 to disable).
        final_activation: If True, add activation after final layer.
        zero_init_final: If True, initialize final layer weights to zero.

    Returns:
        nn.Sequential containing the MLP layers.

    Example:
        >>> # Encoder: 72 -> 256 -> 128 (no output projection)
        >>> encoder = build_mlp_stack(72, [256, 128])
        >>>
        >>> # Decoder with output: 12 -> 128 -> 256 -> 75
        >>> decoder = build_mlp_stack(12, [128, 256], output_dim=75)
    """
    layers: list[nn.Module] = []
    in_dim = input_dim

    for h_dim in hidden_dims:
        layers.append(nn.Linear(in_dim, h_dim))
        layers.append(nn.LayerNorm(h_dim))
        layers.append(nn.SiLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        in_dim = h_dim

    if output_dim is not None:
        final_layer = nn.Linear(in_dim, output_dim)
        if zero_init_final:
            nn.init.zeros_(final_layer.weight)
            nn.init.zeros_(final_layer.bias)
        layers.append(final_layer)

        if final_activation:
            layers.append(nn.SiLU())

    return nn.Sequential(*layers)


class InputNorm(nn.Module):
    """
    Learnable input normalization (like BatchNorm but with data-dependent init).

    On first forward pass, initializes to normalize input to zero mean, unit std.
    After initialization, scale and bias become learnable parameters.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.register_buffer("initialized", torch.tensor(False))

    def initialize(self, x: torch.Tensor) -> None:
        with torch.no_grad():
            mean = x.mean(dim=0)
            std = x.std(dim=0, correction=0).clamp(min=1e-6)
            self.bias.copy_(-mean / std)
            self.scale.copy_(1.0 / std)
            self.initialized.fill_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.initialized:
            self.initialize(x)
        return x * self.scale + self.bias

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Unnormalize output back to original scale."""
        return (y - self.bias) / self.scale


class ResidualBlock(nn.Module):
    """Residual block with LayerNorm and SiLU activation."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class MLPEncoder(nn.Module):
    """MLP encoder to latent space with mu/logvar heads.

    Standard VAE encoder architecture used across residue models.
    Outputs distribution parameters for reparameterization.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.0,
    ):
        """Initialize encoder.

        Args:
            input_dim: Input feature dimension.
            latent_dim: Latent space dimension.
            hidden_dims: Hidden layer dimensions. Defaults to [256, 128].
            dropout: Dropout probability.
        """
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        if hidden_dims is None:
            hidden_dims = [256, 128]

        # Shared encoder backbone
        self.backbone = build_mlp_stack(
            input_dim, hidden_dims, dropout=dropout, zero_init_final=False
        )
        self._hidden_dim = hidden_dims[-1]

        # Latent projection heads
        self.fc_mu = nn.Linear(self._hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(self._hidden_dim, latent_dim)

    def forward(
        self, x: torch.Tensor, return_distribution: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode input to latent space.

        Args:
            x: (batch, input_dim) input features.
            return_distribution: If True, return (z, mu, logvar).

        Returns:
            z or (z, mu, logvar) depending on return_distribution.
        """
        h = self.backbone(x)
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


class CoordinateDecoder(nn.Module):
    """MLP decoder from latent space to coordinates + SE(3) transform.

    Outputs coordinates in a canonical frame plus a 6D transform
    (axis-angle rotation + translation) for positioning the next residue.
    """

    def __init__(
        self,
        latent_dim: int,
        n_atoms: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_atoms = n_atoms
        self.output_dim = n_atoms * 3 + 6  # coords + transform

        if hidden_dims is None:
            hidden_dims = [256, 128]

        # Build decoder MLP
        self.net = build_mlp_stack(
            latent_dim,
            hidden_dims,
            output_dim=self.output_dim,
            dropout=dropout,
            zero_init_final=True,
        )

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode latent to coordinates and transform.

        Args:
            z: (batch, latent_dim) latent vectors.

        Returns:
            coords: (batch, n_atoms, 3) coordinates in canonical frame.
            transform: (batch, 6) SE(3) transform [axis-angle, translation].
        """
        out = self.net(z)

        coords_flat = out[:, : self.n_atoms * 3]
        transform = out[:, self.n_atoms * 3 :]

        coords = coords_flat.reshape(-1, self.n_atoms, 3)
        return coords, transform


class RBFDistanceEncoder(nn.Module):
    """
    Radial basis function encoding for pairwise distances.

    Maps scalar distances to a learned representation using Gaussian RBFs
    centered at evenly-spaced points from 0 to cutoff.
    """

    def __init__(
        self,
        d_out: int = 64,
        n_rbf: int = 16,
        cutoff: float = 10.0,
    ):
        super().__init__()
        self.n_rbf = n_rbf
        self.cutoff = cutoff

        # RBF centers evenly spaced from 0 to cutoff
        centers = torch.linspace(0, cutoff, n_rbf)
        self.register_buffer("centers", centers)

        # Width based on cutoff and number of RBFs
        self.width = cutoff / n_rbf

        # Project to output dimension
        self.proj = nn.Linear(n_rbf, d_out)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """Encode distances to feature vectors.

        Args:
            distances: (...,) tensor of distances.

        Returns:
            (..., d_out) encoded features.
        """
        # Compute RBF features: exp(-((d - center) / width)^2)
        d = distances.unsqueeze(-1)  # (..., 1)
        rbf = torch.exp(-((d - self.centers) / self.width) ** 2)

        return self.proj(rbf)


__all__ = [
    "build_mlp_stack",
    "InputNorm",
    "ResidualBlock",
    "MLPEncoder",
    "CoordinateDecoder",
    "RBFDistanceEncoder",
]
