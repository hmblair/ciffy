"""
Angular distributions and utilities for periodic angle handling.

Provides sin/cos encoding, von Mises distribution, and angular loss functions
for representing and modeling dihedral angles in VAE architectures.
"""

from __future__ import annotations

import math

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = None

from ...types.dihedral import (
    NUM_PROTEIN_BACKBONE_DIHEDRALS,
    NUM_RNA_BACKBONE_DIHEDRALS,
    MAX_DIHEDRALS_PER_RESIDUE,
)


def sincos_encode(angles: "torch.Tensor") -> "torch.Tensor":
    """
    Encode angles using sin/cos representation.

    This handles the periodic nature of angles by mapping each angle to
    a 2D representation on the unit circle. The encoding is continuous
    across the periodic boundary at +/- pi.

    Args:
        angles: (..., D) tensor of angles in radians

    Returns:
        (..., 2*D) tensor with interleaved [sin(a1), cos(a1), sin(a2), cos(a2), ...]
    """
    sin_enc = torch.sin(angles)
    cos_enc = torch.cos(angles)
    # Stack and flatten: (..., D, 2) -> (..., 2*D)
    return torch.stack([sin_enc, cos_enc], dim=-1).flatten(start_dim=-2)


def sincos_decode(encoded: "torch.Tensor") -> "torch.Tensor":
    """
    Decode sin/cos representation back to angles.

    Args:
        encoded: (..., 2*D) tensor with interleaved sin/cos pairs

    Returns:
        (..., D) tensor of angles in radians, range [-pi, pi]
    """
    # Reshape to (..., D, 2)
    encoded = encoded.view(*encoded.shape[:-1], -1, 2)
    sin_vals = encoded[..., 0]
    cos_vals = encoded[..., 1]
    return torch.atan2(sin_vals, cos_vals)


def angular_distance(pred: "torch.Tensor", target: "torch.Tensor") -> "torch.Tensor":
    """
    Compute angular distance handling periodicity.

    Returns the shortest angle between pred and target, accounting for
    the periodic boundary at +/- pi.

    Args:
        pred: Predicted angles in radians
        target: Target angles in radians

    Returns:
        Absolute angular distance in radians, range [0, pi]
    """
    diff = pred - target
    # Wrap to [-pi, pi] and take absolute value
    return torch.atan2(torch.sin(diff), torch.cos(diff)).abs()


class VonMisesNLL(nn.Module if TORCH_AVAILABLE else object):
    """
    Negative log-likelihood loss for von Mises distribution.

    The von Mises distribution is the circular analog of the Gaussian:
        p(theta | mu, kappa) = exp(kappa * cos(theta - mu)) / (2 * pi * I_0(kappa))

    where I_0 is the modified Bessel function of order 0.

    This is the appropriate distribution for modeling periodic quantities
    like dihedral angles, as it respects the circular topology.

    Args:
        reduction: How to reduce the loss ('mean', 'sum', or 'none')

    Example:
        >>> nll = VonMisesNLL(reduction='mean')
        >>> loss = nll(mu_pred, kappa_pred, target, mask)
    """

    def __init__(self, reduction: str = "mean"):
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for VonMisesNLL. "
                "Install with: pip install torch"
            )
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got {reduction}")
        self.reduction = reduction

    def forward(
        self,
        mu: "torch.Tensor",
        kappa: "torch.Tensor",
        target: "torch.Tensor",
        mask: "torch.Tensor | None" = None,
    ) -> "torch.Tensor":
        """
        Compute von Mises negative log-likelihood.

        NLL = -kappa * cos(target - mu) + log(2*pi) + log(I_0(kappa))

        Args:
            mu: Predicted mean angles in radians
            kappa: Predicted concentration parameter (> 0, higher = more peaked)
            target: Target angles in radians
            mask: Optional boolean mask (True = valid, False = ignore)

        Returns:
            Negative log-likelihood loss
        """
        # Compute NLL per element
        # Use torch.special.i0e for numerical stability: i0e(x) = I_0(x) * exp(-|x|)
        # log(I_0(kappa)) = log(i0e(kappa)) + kappa
        cos_diff = torch.cos(target - mu)
        log_i0 = torch.log(torch.special.i0e(kappa) + 1e-8) + kappa
        nll = -kappa * cos_diff + math.log(2 * math.pi) + log_i0

        if self.reduction == "none":
            if mask is not None:
                nll = nll * mask.float()
            return nll

        if mask is not None:
            nll = nll * mask.float()
            total = nll.sum()
            if self.reduction == "mean":
                count = mask.float().sum().clamp(min=1)
                return total / count
            return total

        if self.reduction == "mean":
            return nll.mean()
        return nll.sum()
