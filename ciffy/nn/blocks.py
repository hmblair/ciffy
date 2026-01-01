"""
Shared neural network building blocks.

This module provides reusable components for residue-level models:
- InputNorm: Learnable input normalization (like ActNorm)
- ResidualBlock: MLP block with skip connections
- CoordinateDecoder: Decodes latents to coordinates + transforms
"""

from __future__ import annotations

import torch
import torch.nn as nn


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


class CoordinateDecoder(nn.Module):
    """
    MLP decoder from latent space to coordinates + SE(3) transform.

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

        layers.append(nn.Linear(in_dim, self.output_dim))
        self.net = nn.Sequential(*layers)

        # Initialize final layer to output near-zero
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

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
    "InputNorm",
    "ResidualBlock",
    "CoordinateDecoder",
    "RBFDistanceEncoder",
]
