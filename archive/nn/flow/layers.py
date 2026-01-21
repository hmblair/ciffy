"""Normalizing flow layers and architectures.

This module provides generic building blocks for normalizing flows:

Coupling Layers:
    - AffineCouplingLayer: Simple affine coupling (RealNVP)
    - NeuralSplineLayer: Rational quadratic spline coupling (NSF)

Complete Flows:
    - RealNVP: Stack of affine coupling layers
    - NeuralSplineFlow: Stack of spline coupling layers

Example:
    >>> from ciffy.nn.flow import RealNVP, NeuralSplineFlow
    >>>
    >>> # Create a RealNVP flow
    >>> flow = RealNVP(dim=16, n_layers=8, hidden_dim=128)
    >>> x = torch.randn(32, 16)
    >>>
    >>> # Forward: data -> latent
    >>> z, log_det = flow(x)
    >>>
    >>> # Inverse: latent -> data
    >>> x_recon = flow.inverse(z)  # Exact reconstruction
    >>>
    >>> # Density estimation
    >>> log_prob = flow.log_prob(x)
    >>>
    >>> # Sampling
    >>> samples = flow.sample((100,), device="cuda")

References:
    - Dinh et al. (2017) "Density estimation using Real-NVP"
    - Durkan et al. (2019) "Neural Spline Flows"
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import MLP


# =============================================================================
# Spline Transformation
# =============================================================================


def _searchsorted(bin_locations: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """Find bin indices for inputs given bin locations."""
    return torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1


def rational_quadratic_spline(
    inputs: torch.Tensor,
    widths: torch.Tensor,
    heights: torch.Tensor,
    derivatives: torch.Tensor,
    inverse: bool = False,
    left: float = -5.0,
    right: float = 5.0,
    bottom: float = -5.0,
    top: float = 5.0,
    min_bin_width: float = 1e-3,
    min_bin_height: float = 1e-3,
    min_derivative: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Rational quadratic spline transformation.

    Implements the monotonic rational-quadratic spline from Neural Spline Flows.
    This is more expressive than affine transformations while remaining invertible.

    Args:
        inputs: Input tensor to transform.
        widths: Unnormalized bin widths (will be softmaxed).
        heights: Unnormalized bin heights (will be softmaxed).
        derivatives: Unnormalized derivatives at knots (will be softplus'd).
        inverse: If True, compute inverse transformation.
        left, right: Horizontal bounds of the spline.
        bottom, top: Vertical bounds of the spline.
        min_bin_width: Minimum bin width for numerical stability.
        min_bin_height: Minimum bin height for numerical stability.
        min_derivative: Minimum derivative for numerical stability.

    Returns:
        outputs: Transformed values.
        logabsdet: Log absolute determinant of Jacobian (per element).

    Reference:
        Durkan et al. (2019) "Neural Spline Flows"
    """
    num_bins = widths.shape[-1]

    # Normalize widths and heights to sum to interval size
    widths = F.softmax(widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths

    heights = F.softmax(heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights

    # Derivatives must be positive
    derivatives = min_derivative + F.softplus(derivatives)

    # Compute bin boundaries
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, (1, 0), value=0.0)
    cumwidths = (right - left) * cumwidths + left
    cumwidths[..., 0] = left
    cumwidths[..., -1] = right

    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, (1, 0), value=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom
    cumheights[..., -1] = top

    # Pad derivatives
    derivatives = F.pad(derivatives, (1, 1), value=1.0)

    # Find which bin each input falls into
    if inverse:
        bin_idx = _searchsorted(cumheights, inputs)
    else:
        bin_idx = _searchsorted(cumwidths, inputs)

    bin_idx = bin_idx.clamp(0, num_bins - 1)

    # Gather bin parameters
    input_cumwidths = cumwidths.gather(-1, bin_idx[..., None]).squeeze(-1)
    input_bin_widths = widths.gather(-1, bin_idx[..., None]).squeeze(-1)

    input_cumheights = cumheights.gather(-1, bin_idx[..., None]).squeeze(-1)
    input_bin_heights = heights.gather(-1, bin_idx[..., None]).squeeze(-1)

    input_delta = input_bin_heights / input_bin_widths

    input_derivatives = derivatives.gather(-1, bin_idx[..., None]).squeeze(-1)
    input_derivatives_plus_one = derivatives.gather(
        -1, (bin_idx + 1)[..., None]
    ).squeeze(-1)

    # Compute spline
    if inverse:
        a = (input_cumheights - inputs) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        ) + input_bin_heights * (input_delta - input_derivatives)
        b = input_bin_heights * input_derivatives - (input_cumheights - inputs) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        )
        c = -input_delta * (input_cumheights - inputs)

        discriminant = b.pow(2) - 4 * a * c
        root = (2 * c) / (-b - torch.sqrt(discriminant))

        outputs = root * input_bin_widths + input_cumwidths

        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
            * theta_one_minus_theta
        )
        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * root.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - root).pow(2)
        )
        logabsdet = -torch.log(derivative_numerator) + 2 * torch.log(denominator)
    else:
        theta = (inputs - input_cumwidths) / input_bin_widths
        theta_one_minus_theta = theta * (1 - theta)

        numerator = input_bin_heights * (
            input_delta * theta.pow(2) + input_derivatives * theta_one_minus_theta
        )
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
            * theta_one_minus_theta
        )
        outputs = input_cumheights + numerator / denominator

        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * theta.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - theta).pow(2)
        )
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

    return outputs, logabsdet


# =============================================================================
# Coupling Layers
# =============================================================================


class AffineCouplingLayer(nn.Module):
    """
    Affine coupling layer for RealNVP flows.

    Splits input into two halves. The first half is unchanged, while the
    second half undergoes an affine transformation parameterized by a
    neural network that takes the first half as input.

    Args:
        dim: Input dimension.
        hidden_dim: Hidden dimension for the parameter network.
        n_layers: Number of hidden layers in the parameter network.

    Example:
        >>> layer = AffineCouplingLayer(dim=16, hidden_dim=64)
        >>> x = torch.randn(32, 16)
        >>> z, log_det = layer(x)
        >>> x_recon = layer.inverse(z)
    """

    def __init__(self, dim: int, hidden_dim: int = 128, n_layers: int = 2):
        super().__init__()
        self.dim = dim
        self.split_dim = dim // 2

        self.net = MLP(
            in_dim=self.split_dim,
            out_dim=2 * (dim - self.split_dim),
            hidden_dims=[hidden_dim] * n_layers,
        )

        # Initialize to identity transform
        nn.init.zeros_(self.net.layers[-1].weight)
        nn.init.zeros_(self.net.layers[-1].bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: x -> z.

        Args:
            x: Input tensor of shape (..., dim).

        Returns:
            z: Transformed tensor of shape (..., dim).
            log_det: Log determinant of shape (...,).
        """
        x1, x2 = x[..., : self.split_dim], x[..., self.split_dim :]

        params = self.net(x1)
        log_scale, shift = params.chunk(2, dim=-1)
        log_scale = torch.tanh(log_scale) * 2  # Bound for stability

        z2 = x2 * torch.exp(log_scale) + shift
        z = torch.cat([x1, z2], dim=-1)

        return z, log_scale.sum(dim=-1)

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """
        Inverse pass: z -> x.

        Args:
            z: Latent tensor of shape (..., dim).

        Returns:
            x: Reconstructed tensor of shape (..., dim).
        """
        z1, z2 = z[..., : self.split_dim], z[..., self.split_dim :]

        params = self.net(z1)
        log_scale, shift = params.chunk(2, dim=-1)
        log_scale = torch.tanh(log_scale) * 2

        x2 = (z2 - shift) * torch.exp(-log_scale)
        return torch.cat([z1, x2], dim=-1)


class NeuralSplineLayer(nn.Module):
    """
    Neural spline coupling layer.

    More expressive than affine coupling, using rational quadratic splines
    for the transformation. This allows modeling more complex distributions
    while remaining analytically invertible.

    Args:
        dim: Input dimension.
        hidden_dim: Hidden dimension for the parameter network.
        n_layers: Number of hidden layers in the parameter network.
        num_bins: Number of spline bins.
        tail_bound: Bounds for the spline transformation.

    Example:
        >>> layer = NeuralSplineLayer(dim=16, hidden_dim=64, num_bins=8)
        >>> x = torch.randn(32, 16)
        >>> z, log_det = layer(x)
        >>> x_recon = layer.inverse(z)
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int = 128,
        n_layers: int = 2,
        num_bins: int = 8,
        tail_bound: float = 5.0,
    ):
        super().__init__()
        self.dim = dim
        self.split_dim = dim // 2
        self.num_bins = num_bins
        self.tail_bound = tail_bound

        # Output: num_bins widths + num_bins heights + (num_bins + 1) derivatives
        output_dim = (dim - self.split_dim) * (3 * num_bins + 1)

        self.net = MLP(
            in_dim=self.split_dim,
            out_dim=output_dim,
            hidden_dims=[hidden_dim] * n_layers,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward: x -> z"""
        x1, x2 = x[..., : self.split_dim], x[..., self.split_dim :]

        params = self.net(x1)
        params = params.view(*x2.shape, 3 * self.num_bins + 1)

        widths = params[..., : self.num_bins]
        heights = params[..., self.num_bins : 2 * self.num_bins]
        derivatives = params[..., 2 * self.num_bins :]

        z2, logabsdet = rational_quadratic_spline(
            x2,
            widths,
            heights,
            derivatives,
            inverse=False,
            left=-self.tail_bound,
            right=self.tail_bound,
            bottom=-self.tail_bound,
            top=self.tail_bound,
        )

        z = torch.cat([x1, z2], dim=-1)
        return z, logabsdet.sum(dim=-1)

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Inverse: z -> x"""
        z1, z2 = z[..., : self.split_dim], z[..., self.split_dim :]

        params = self.net(z1)
        params = params.view(*z2.shape, 3 * self.num_bins + 1)

        widths = params[..., : self.num_bins]
        heights = params[..., self.num_bins : 2 * self.num_bins]
        derivatives = params[..., 2 * self.num_bins :]

        x2, _ = rational_quadratic_spline(
            z2,
            widths,
            heights,
            derivatives,
            inverse=True,
            left=-self.tail_bound,
            right=self.tail_bound,
            bottom=-self.tail_bound,
            top=self.tail_bound,
        )

        x = torch.cat([z1, x2], dim=-1)
        return x


# =============================================================================
# Complete Flow Architectures
# =============================================================================


class RealNVP(nn.Module):
    """
    RealNVP normalizing flow.

    Stacks multiple affine coupling layers with alternating permutations.
    Simple, stable, and effective for many applications.

    Args:
        dim: Input/output dimension.
        n_layers: Number of coupling layers.
        hidden_dim: Hidden dimension for coupling networks.

    Example:
        >>> flow = RealNVP(dim=16, n_layers=8, hidden_dim=128)
        >>> x = torch.randn(32, 16)
        >>>
        >>> # Forward pass
        >>> z, log_det = flow(x)
        >>>
        >>> # Inverse (exact reconstruction)
        >>> x_recon = flow.inverse(z)
        >>> assert torch.allclose(x, x_recon, atol=1e-5)
        >>>
        >>> # Density estimation
        >>> log_prob = flow.log_prob(x)
        >>>
        >>> # Sampling
        >>> samples = flow.sample((100,), device=x.device)

    Reference:
        Dinh et al. (2017) "Density estimation using Real-NVP"
    """

    def __init__(self, dim: int, n_layers: int = 8, hidden_dim: int = 128):
        super().__init__()
        self.dim = dim

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(AffineCouplingLayer(dim, hidden_dim))
            perm = torch.arange(dim - 1, -1, -1)
            self.register_buffer(f"perm_{i}", perm)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: x -> z.

        Args:
            x: Input tensor of shape (..., dim).

        Returns:
            z: Latent tensor of shape (..., dim).
            log_det: Log determinant of Jacobian, shape (...,).
        """
        log_det = torch.zeros(x.shape[:-1], device=x.device)
        z = x

        for i, layer in enumerate(self.layers):
            z, ld = layer(z)
            log_det = log_det + ld
            z = z[..., getattr(self, f"perm_{i}")]

        return z, log_det

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """
        Inverse pass: z -> x.

        Args:
            z: Latent tensor of shape (..., dim).

        Returns:
            x: Reconstructed tensor of shape (..., dim).
        """
        x = z
        for i in range(len(self.layers) - 1, -1, -1):
            perm = getattr(self, f"perm_{i}")
            x = x[..., torch.argsort(perm)]
            x = self.layers[i].inverse(x)
        return x

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability under the flow.

        Args:
            x: Input tensor of shape (..., dim).

        Returns:
            Log probability of shape (...,).
        """
        z, log_det = self(x)
        log_pz = -0.5 * (z.pow(2).sum(dim=-1) + self.dim * math.log(2 * math.pi))
        return log_pz + log_det

    def log_prob_decomposed(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute log probability with separate components.

        Useful for diagnosing training issues:
        - If log_det -> -inf, the flow is collapsing (losing volume)
        - If log_pz -> -inf, z is moving away from Gaussian center

        Returns:
            log_prob: Total log probability (log_pz + log_det).
            log_pz: Log probability under base Gaussian.
            log_det: Log determinant of Jacobian.
        """
        z, log_det = self(x)
        log_pz = -0.5 * (z.pow(2).sum(dim=-1) + self.dim * math.log(2 * math.pi))
        return log_pz + log_det, log_pz, log_det

    def sample(self, shape: tuple, device: torch.device) -> torch.Tensor:
        """
        Sample from the flow.

        Args:
            shape: Batch shape for samples.
            device: Device for samples.

        Returns:
            Samples of shape (*shape, dim).
        """
        z = torch.randn(*shape, self.dim, device=device)
        return self.inverse(z)


class NeuralSplineFlow(nn.Module):
    """
    Neural Spline Flow.

    More expressive than RealNVP, using rational quadratic splines instead
    of affine transformations. Better for complex multimodal distributions.

    Args:
        dim: Input/output dimension.
        n_layers: Number of coupling layers.
        hidden_dim: Hidden dimension for coupling networks.
        num_bins: Number of spline bins.
        tail_bound: Bounds for the spline transformation.

    Example:
        >>> flow = NeuralSplineFlow(dim=16, n_layers=8)
        >>> x = torch.randn(32, 16)
        >>> z, log_det = flow(x)
        >>> log_prob = flow.log_prob(x)

    Reference:
        Durkan et al. (2019) "Neural Spline Flows"
    """

    def __init__(
        self,
        dim: int,
        n_layers: int = 8,
        hidden_dim: int = 128,
        num_bins: int = 8,
        tail_bound: float = 5.0,
    ):
        super().__init__()
        self.dim = dim

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(
                NeuralSplineLayer(
                    dim,
                    hidden_dim,
                    n_layers=2,
                    num_bins=num_bins,
                    tail_bound=tail_bound,
                )
            )
            perm = torch.arange(dim - 1, -1, -1)
            self.register_buffer(f"perm_{i}", perm)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward: x -> z, returns (z, log_det)."""
        log_det_total = torch.zeros(x.shape[:-1], device=x.device)
        z = x

        for i, layer in enumerate(self.layers):
            z, log_det = layer(z)
            log_det_total = log_det_total + log_det

            perm = getattr(self, f"perm_{i}")
            z = z[..., perm]

        return z, log_det_total

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Inverse: z -> x."""
        x = z

        for i in range(len(self.layers) - 1, -1, -1):
            perm = getattr(self, f"perm_{i}")
            inv_perm = torch.argsort(perm)
            x = x[..., inv_perm]

            x = self.layers[i].inverse(x)

        return x

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Compute log p(x) under the flow."""
        z, log_det = self(x)
        log_pz = -0.5 * (z.pow(2).sum(dim=-1) + self.dim * math.log(2 * math.pi))
        return log_pz + log_det

    def sample(self, shape: tuple, device: torch.device) -> torch.Tensor:
        """Sample from the flow."""
        z = torch.randn(*shape, self.dim, device=device)
        return self.inverse(z)


__all__ = [
    "rational_quadratic_spline",
    "AffineCouplingLayer",
    "NeuralSplineLayer",
    "RealNVP",
    "NeuralSplineFlow",
]
