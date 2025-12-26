"""
Evaluation metrics for normalizing flow models.

Provides functions for assessing flow model quality on held-out data:
- Negative log-likelihood (NLL)
- Latent space distribution statistics (moments)
- KL divergence estimates

These metrics help diagnose generalization and sampling quality.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from ciffy.backend import Array

if TYPE_CHECKING:
    from .residue.model import PCAFlow


@dataclass
class LatentMoments:
    """Statistics of encoded latent vectors.

    For a well-trained flow on in-distribution data, encoded points
    should be approximately N(0, I):
    - mean ≈ 0
    - std ≈ 1
    - skewness ≈ 0 (symmetric)
    - kurtosis ≈ 0 (excess kurtosis, Gaussian = 0)

    Deviations indicate distribution shift or poor generalization.
    """

    mean: np.ndarray  # (latent_dim,) per-dimension means
    std: np.ndarray   # (latent_dim,) per-dimension stds
    skewness: np.ndarray  # (latent_dim,) per-dimension skewness
    kurtosis: np.ndarray  # (latent_dim,) per-dimension excess kurtosis

    @property
    def mean_abs_mean(self) -> float:
        """Mean absolute deviation of means from 0."""
        return float(np.abs(self.mean).mean())

    @property
    def mean_abs_std_dev(self) -> float:
        """Mean absolute deviation of stds from 1."""
        return float(np.abs(self.std - 1.0).mean())

    @property
    def mean_abs_skewness(self) -> float:
        """Mean absolute skewness (should be near 0 for Gaussian)."""
        return float(np.abs(self.skewness).mean())

    @property
    def mean_abs_kurtosis(self) -> float:
        """Mean absolute excess kurtosis (should be near 0 for Gaussian)."""
        return float(np.abs(self.kurtosis).mean())

    def gaussianity_score(self) -> float:
        """
        Combined score measuring how Gaussian the latent distribution is.

        Returns a value in [0, 1] where 1 = perfectly Gaussian.
        Uses exponential decay based on moment deviations.
        """
        # Weight different moment deviations
        deviation = (
            self.mean_abs_mean +
            self.mean_abs_std_dev +
            0.5 * self.mean_abs_skewness +
            0.25 * self.mean_abs_kurtosis
        )
        return float(np.exp(-deviation))

    def __repr__(self) -> str:
        return (
            f"LatentMoments(mean={self.mean_abs_mean:.4f}, "
            f"std_dev={self.mean_abs_std_dev:.4f}, "
            f"skew={self.mean_abs_skewness:.4f}, "
            f"kurt={self.mean_abs_kurtosis:.4f}, "
            f"gaussianity={self.gaussianity_score():.3f})"
        )


@dataclass
class FlowMetrics:
    """Comprehensive evaluation metrics for a flow model.

    Attributes:
        nll: Mean negative log-likelihood (lower is better).
        moments: Latent space distribution statistics.
        n_samples: Number of samples used for evaluation.
    """

    nll: float
    moments: LatentMoments
    n_samples: int

    @property
    def gaussianity(self) -> float:
        """How Gaussian the encoded distribution is (0-1, higher is better)."""
        return self.moments.gaussianity_score()

    def __repr__(self) -> str:
        return (
            f"FlowMetrics(nll={self.nll:.4f}, "
            f"gaussianity={self.gaussianity:.3f}, "
            f"n={self.n_samples})"
        )


def compute_nll(
    flow: "PCAFlow",
    data: Array,
    batch_size: int = 1024,
) -> float:
    """
    Compute mean negative log-likelihood on data.

    Lower NLL indicates better density estimation. Comparing train vs test
    NLL reveals overfitting.

    Args:
        flow: Trained PCAFlow model.
        data: (N, d) data array (numpy or torch).
        batch_size: Batch size for evaluation.

    Returns:
        Mean negative log-likelihood.
    """
    flow.eval()
    device = flow.V.device

    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data).float()
    data = data.to(device)

    n_samples = len(data)
    total_nll = 0.0

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch = data[i:i + batch_size]
            log_prob = flow.log_prob(batch)
            total_nll -= log_prob.sum().item()

    return total_nll / n_samples


def compute_latent_moments(
    flow: "PCAFlow",
    data: Array,
    batch_size: int = 1024,
) -> LatentMoments:
    """
    Compute statistics of encoded latent vectors.

    For a well-trained flow, encoded data should be approximately N(0, I).
    Deviations from this indicate distribution shift or poor generalization.

    Args:
        flow: Trained PCAFlow model.
        data: (N, d) data array (numpy or torch).
        batch_size: Batch size for encoding.

    Returns:
        LatentMoments with per-dimension statistics.
    """
    flow.eval()
    device = flow.V.device

    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data).float()
    data = data.to(device)

    # Encode all data
    encoded = []
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            z = flow.encode(batch)
            encoded.append(z.cpu())

    z = torch.cat(encoded, dim=0).numpy()

    # Compute per-dimension statistics
    mean = z.mean(axis=0)
    std = z.std(axis=0)

    # Skewness: E[(X - μ)^3] / σ^3
    centered = z - mean
    skewness = (centered ** 3).mean(axis=0) / (std ** 3 + 1e-8)

    # Excess kurtosis: E[(X - μ)^4] / σ^4 - 3
    kurtosis = (centered ** 4).mean(axis=0) / (std ** 4 + 1e-8) - 3.0

    return LatentMoments(
        mean=mean,
        std=std,
        skewness=skewness,
        kurtosis=kurtosis,
    )


def compute_flow_metrics(
    flow: "PCAFlow",
    data: Array,
    batch_size: int = 1024,
) -> FlowMetrics:
    """
    Compute comprehensive evaluation metrics for a flow model.

    This is the main evaluation function that computes:
    - Negative log-likelihood (density estimation quality)
    - Latent space moments (distribution matching)

    Args:
        flow: Trained PCAFlow model.
        data: (N, d) data array (numpy or torch).
        batch_size: Batch size for evaluation.

    Returns:
        FlowMetrics with NLL and latent moment statistics.

    Example:
        >>> metrics = compute_flow_metrics(flow, test_data)
        >>> print(f"Test NLL: {metrics.nll:.4f}")
        >>> print(f"Gaussianity: {metrics.gaussianity:.3f}")
    """
    nll = compute_nll(flow, data, batch_size)
    moments = compute_latent_moments(flow, data, batch_size)

    n_samples = len(data) if isinstance(data, np.ndarray) else data.shape[0]

    return FlowMetrics(
        nll=nll,
        moments=moments,
        n_samples=n_samples,
    )


def estimate_kl_divergence(
    flow: "PCAFlow",
    data: Array,
    batch_size: int = 1024,
) -> float:
    """
    Estimate KL divergence between encoded distribution and N(0, I).

    Uses the analytical KL for a fitted Gaussian:
    KL(N(μ, Σ) || N(0, I)) = 0.5 * (tr(Σ) + μᵀμ - k - log|Σ|)

    This assumes the encoded distribution is approximately Gaussian,
    which is reasonable for flow models trained on continuous data.

    Args:
        flow: Trained PCAFlow model.
        data: (N, d) data array.
        batch_size: Batch size for encoding.

    Returns:
        Estimated KL divergence (nats). Lower is better.
    """
    moments = compute_latent_moments(flow, data, batch_size)

    k = len(moments.mean)
    mu = moments.mean
    sigma_sq = moments.std ** 2

    # KL(N(μ, diag(σ²)) || N(0, I))
    # = 0.5 * (Σσ² + μᵀμ - k - Σlog(σ²))
    kl = 0.5 * (
        sigma_sq.sum() +
        (mu ** 2).sum() -
        k -
        np.log(sigma_sq + 1e-8).sum()
    )

    return float(kl)
