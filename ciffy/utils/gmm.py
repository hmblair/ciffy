"""
Gaussian Mixture Model for sampling from empirical distributions.

Provides a simple GMM implementation for sampling phi/psi angles from
Ramachandran distributions fitted to PDB data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class GaussianMixtureModel:
    """
    2D Gaussian Mixture Model for sampling.

    Attributes:
        means: (K, D) array of component means.
        covariances: (K, D, D) array of covariance matrices.
        weights: (K,) array of mixture weights (sum to 1).

    Example:
        >>> # Create a simple 2-component GMM
        >>> gmm = GaussianMixtureModel(
        ...     means=np.array([[-1.0, -0.7], [-2.1, 2.3]]),
        ...     covariances=np.array([[[0.1, 0], [0, 0.1]], [[0.2, 0], [0, 0.2]]]),
        ...     weights=np.array([0.6, 0.4])
        ... )
        >>> samples = gmm.sample(100)
        >>> samples.shape
        (100, 2)
    """

    means: np.ndarray  # (K, D)
    covariances: np.ndarray  # (K, D, D)
    weights: np.ndarray  # (K,)

    def __post_init__(self):
        """Validate parameters."""
        self.means = np.asarray(self.means, dtype=np.float64)
        self.covariances = np.asarray(self.covariances, dtype=np.float64)
        self.weights = np.asarray(self.weights, dtype=np.float64)

        k = len(self.weights)
        if self.means.shape[0] != k:
            raise ValueError(f"means has {self.means.shape[0]} components but weights has {k}")
        if self.covariances.shape[0] != k:
            raise ValueError(f"covariances has {self.covariances.shape[0]} components but weights has {k}")
        if not np.isclose(self.weights.sum(), 1.0):
            raise ValueError(f"weights must sum to 1, got {self.weights.sum()}")

    @property
    def n_components(self) -> int:
        """Number of mixture components."""
        return len(self.weights)

    @property
    def n_features(self) -> int:
        """Dimensionality of the data."""
        return self.means.shape[1]

    def sample(
        self,
        n: int,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """
        Sample n points from the mixture.

        Args:
            n: Number of samples to generate.
            rng: Random number generator for reproducibility.

        Returns:
            (n, D) array of samples.
        """
        if rng is None:
            rng = np.random.default_rng()

        # Select components according to weights
        components = rng.choice(self.n_components, size=n, p=self.weights)

        # Sample from each component's Gaussian
        samples = np.empty((n, self.n_features), dtype=np.float64)
        for i in range(n):
            k = components[i]
            samples[i] = rng.multivariate_normal(self.means[k], self.covariances[k])

        return samples

    @classmethod
    def fit(
        cls,
        data: np.ndarray,
        n_components: int,
        max_iter: int = 100,
        tol: float = 1e-4,
        rng: np.random.Generator | None = None,
    ) -> "GaussianMixtureModel":
        """
        Fit GMM to data using the EM algorithm.

        Args:
            data: (N, D) array of data points.
            n_components: Number of mixture components (K).
            max_iter: Maximum EM iterations.
            tol: Convergence tolerance on log-likelihood.
            rng: Random number generator for initialization.

        Returns:
            Fitted GaussianMixtureModel.
        """
        if rng is None:
            rng = np.random.default_rng()

        data = np.asarray(data, dtype=np.float64)
        n_samples, n_features = data.shape
        k = n_components

        # Initialize with k-means++ style
        means = cls._init_means_kmeans_pp(data, k, rng)
        covariances = np.array([np.eye(n_features) * np.var(data) for _ in range(k)])
        weights = np.ones(k) / k

        prev_log_likelihood = -np.inf

        for iteration in range(max_iter):
            # E-step: compute responsibilities
            responsibilities = cls._e_step(data, means, covariances, weights)

            # M-step: update parameters
            means, covariances, weights = cls._m_step(data, responsibilities)

            # Check convergence
            log_likelihood = cls._log_likelihood(data, means, covariances, weights)
            if abs(log_likelihood - prev_log_likelihood) < tol:
                break
            prev_log_likelihood = log_likelihood

        return cls(means=means, covariances=covariances, weights=weights)

    @staticmethod
    def _init_means_kmeans_pp(
        data: np.ndarray,
        k: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Initialize means using k-means++ algorithm."""
        n_samples = len(data)
        means = np.empty((k, data.shape[1]), dtype=np.float64)

        # First center: random point
        means[0] = data[rng.integers(n_samples)]

        for i in range(1, k):
            # Compute distances to nearest center
            dists = np.min([np.sum((data - means[j]) ** 2, axis=1) for j in range(i)], axis=0)
            # Sample proportional to squared distance
            probs = dists / dists.sum()
            means[i] = data[rng.choice(n_samples, p=probs)]

        return means

    @staticmethod
    def _e_step(
        data: np.ndarray,
        means: np.ndarray,
        covariances: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        """E-step: compute responsibilities."""
        n_samples = len(data)
        k = len(weights)

        # Compute log-probabilities for each component
        log_probs = np.empty((n_samples, k), dtype=np.float64)
        for j in range(k):
            log_probs[:, j] = (
                np.log(weights[j] + 1e-10)
                + GaussianMixtureModel._log_gaussian(data, means[j], covariances[j])
            )

        # Normalize to get responsibilities (softmax)
        log_sum = np.logaddexp.reduce(log_probs, axis=1, keepdims=True)
        responsibilities = np.exp(log_probs - log_sum)

        return responsibilities

    @staticmethod
    def _m_step(
        data: np.ndarray,
        responsibilities: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """M-step: update parameters."""
        n_samples, n_features = data.shape
        k = responsibilities.shape[1]

        # Effective number of points per component
        nk = responsibilities.sum(axis=0) + 1e-10

        # Update weights
        weights = nk / n_samples

        # Update means
        means = np.empty((k, n_features), dtype=np.float64)
        for j in range(k):
            means[j] = (responsibilities[:, j:j+1] * data).sum(axis=0) / nk[j]

        # Update covariances
        covariances = np.empty((k, n_features, n_features), dtype=np.float64)
        for j in range(k):
            diff = data - means[j]
            covariances[j] = (responsibilities[:, j:j+1] * diff).T @ diff / nk[j]
            # Add small regularization for numerical stability
            covariances[j] += np.eye(n_features) * 1e-6

        return means, covariances, weights

    @staticmethod
    def _log_gaussian(
        data: np.ndarray,
        mean: np.ndarray,
        cov: np.ndarray,
    ) -> np.ndarray:
        """Compute log probability under multivariate Gaussian."""
        n_features = len(mean)
        diff = data - mean

        # Use pseudo-inverse for numerical stability
        try:
            cov_inv = np.linalg.inv(cov)
            log_det = np.log(np.linalg.det(cov) + 1e-10)
        except np.linalg.LinAlgError:
            cov_inv = np.linalg.pinv(cov)
            log_det = np.log(np.abs(np.linalg.det(cov)) + 1e-10)

        mahal = np.sum(diff @ cov_inv * diff, axis=1)
        log_prob = -0.5 * (n_features * np.log(2 * np.pi) + log_det + mahal)

        return log_prob

    @staticmethod
    def _log_likelihood(
        data: np.ndarray,
        means: np.ndarray,
        covariances: np.ndarray,
        weights: np.ndarray,
    ) -> float:
        """Compute total log-likelihood."""
        k = len(weights)
        log_probs = np.empty((len(data), k), dtype=np.float64)
        for j in range(k):
            log_probs[:, j] = (
                np.log(weights[j] + 1e-10)
                + GaussianMixtureModel._log_gaussian(data, means[j], covariances[j])
            )
        return np.logaddexp.reduce(log_probs, axis=1).sum()

    def save(self, path: str | Path) -> None:
        """
        Save GMM parameters to .npz file.

        Args:
            path: Output file path.
        """
        np.savez(
            path,
            means=self.means,
            covariances=self.covariances,
            weights=self.weights,
        )

    @classmethod
    def load(cls, path: str | Path) -> "GaussianMixtureModel":
        """
        Load GMM parameters from .npz file.

        Args:
            path: Input file path.

        Returns:
            Loaded GaussianMixtureModel.
        """
        data = np.load(path)
        return cls(
            means=data["means"],
            covariances=data["covariances"],
            weights=data["weights"],
        )
