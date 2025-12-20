"""
Energy function framework for Langevin dynamics sampling.

Provides abstract base class and concrete implementations for energy functions
that can be used with gradient-based sampling methods. Supports composition of
multiple energy terms for flexible constraint specification.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..polymer import Polymer
    from ..utils.gmm import GaussianMixtureModel


class EnergyFunction(ABC):
    """
    Abstract base class for differentiable energy functions.

    Energy functions define the probability distribution P(x) ∝ exp(-E(x)/T),
    where T is temperature. Each energy function must implement:
    - energy(): Compute scalar energy for given state
    - gradient(): Compute gradient of energy w.r.t. state (for each dimension)

    Example:
        >>> energy = GMMEnergy(gmm) + ClashEnergy(polymer)
        >>> E = energy.energy(angles)
        >>> grad_E = energy.gradient(angles)
    """

    @abstractmethod
    def energy(self, state: np.ndarray) -> float:
        """
        Compute energy of state.

        Args:
            state: State vector (angles in radians, or other parameterization)

        Returns:
            Scalar energy value
        """
        raise NotImplementedError

    @abstractmethod
    def gradient(self, state: np.ndarray) -> np.ndarray:
        """
        Compute gradient of energy w.r.t. state.

        Args:
            state: State vector

        Returns:
            Gradient array, same shape as state
        """
        raise NotImplementedError


class GMMEnergy(EnergyFunction):
    """
    Energy function from Gaussian Mixture Model log-likelihood.

    Energy: E = -log(GMM(x))

    This encourages sampling from the GMM by penalizing low-probability regions.

    Args:
        gmm: Fitted GaussianMixtureModel

    Example:
        >>> gmm = GaussianMixtureModel(...)
        >>> energy = GMMEnergy(gmm)
        >>> angles = np.array([1.5, -2.0])
        >>> E = energy.energy(angles)  # Returns -log(GMM probability)
    """

    def __init__(self, gmm: "GaussianMixtureModel"):
        """Initialize with GMM."""
        self.gmm = gmm

    def energy(self, state: np.ndarray) -> float:
        """Compute -log(GMM probability) for state."""
        # Reshape to (1, D) for compatibility with GMM methods
        state = np.atleast_1d(state)
        if state.ndim == 1:
            state = state.reshape(1, -1)

        # Compute log probability using GMM's internal method
        log_prob = self._compute_log_prob_gmm(state)[0]

        # Energy is negative log probability
        return -log_prob

    def gradient(self, state: np.ndarray) -> np.ndarray:
        """Compute gradient of -log(GMM) w.r.t. state."""
        state = np.atleast_1d(state)
        if state.ndim == 1:
            state = state.reshape(1, -1)

        # Compute numerical gradient (finite differences)
        # This is a reasonable approximation for GMM gradients
        h = 1e-4
        grad = np.zeros_like(state[0])

        for i in range(len(state[0])):
            state_plus = state[0].copy()
            state_plus[i] += h
            state_minus = state[0].copy()
            state_minus[i] -= h

            E_plus = self.energy(state_plus)
            E_minus = self.energy(state_minus)

            grad[i] = (E_plus - E_minus) / (2 * h)

        return grad

    def _compute_log_prob_gmm(self, data: np.ndarray) -> np.ndarray:
        """Compute log probability of data under this GMM."""
        k = len(self.gmm.weights)
        log_probs = np.empty((len(data), k), dtype=np.float64)

        for j in range(k):
            log_probs[:, j] = (
                np.log(self.gmm.weights[j] + 1e-10)
                + self._log_gaussian(data, self.gmm.means[j], self.gmm.covariances[j])
            )

        # Log-sum-exp trick for numerical stability
        log_sum = np.logaddexp.reduce(log_probs, axis=1)
        return log_sum

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


class ClashEnergy(EnergyFunction):
    """
    Energy function from steric clash penalties using smooth 1/r² repulsion.

    Energy: E = λ * Σ(1/distance²) for all atom pairs

    Uses a smooth inverse-square penalty that:
    - Diverges at short distances (strong repulsion)
    - Decays to zero at long distances (no penalty)
    - Is infinitely differentiable (perfect for Langevin)

    Args:
        polymer_evaluator: PolymerEvaluator object with apply_angles() and get_distances() methods
        lambda_clash: Weight of clash penalty (default 100)
        r_min: Minimum distance cutoff to prevent divergence (default 0.5 Å)

    The evaluator's apply_angles(state) applies candidate angles and reconstructs
    coordinates, while get_distances() returns pairwise distances for clash checking.
    """

    def __init__(
        self,
        polymer_evaluator,
        lambda_clash: float = 100.0,
        r_min: float = 0.5,
    ):
        """
        Initialize clash energy with 1/r² repulsion.

        Args:
            polymer_evaluator: PolymerEvaluator with apply_angles() and get_distances()
            lambda_clash: Weight coefficient for clash penalty
            r_min: Minimum distance cutoff (Angstroms)
        """
        self.polymer_evaluator = polymer_evaluator
        self.lambda_clash = lambda_clash
        self.r_min = r_min

    def energy(self, state: np.ndarray) -> float:
        """
        Compute 1/r² clash penalty for candidate angles.

        E = λ * Σ(1/max(r², r_min²))

        Args:
            state: Dihedral angles to evaluate

        Returns:
            λ * sum_of_inverse_squares
        """
        # Apply angles using evaluator's apply_angles method
        self.polymer_evaluator.apply_angles(state)

        # Get distances from evaluator
        distances = self.polymer_evaluator.get_distances()

        if distances is None or len(distances) == 0:
            return 0.0

        # Clip distances to avoid divergence
        distances = np.atleast_1d(distances)
        distances_clipped = np.maximum(distances, self.r_min)

        # Sum of 1/r²
        inverse_square_sum = np.sum(1.0 / (distances_clipped ** 2))

        return self.lambda_clash * inverse_square_sum

    def gradient(self, state: np.ndarray) -> np.ndarray:
        """
        Compute gradient of clash energy w.r.t. angles.

        Uses numerical gradients (finite differences).
        """
        state = np.atleast_1d(state)
        h = 1e-4
        grad = np.zeros_like(state)

        for i in range(len(state)):
            state_plus = state.copy()
            state_plus[i] += h
            state_minus = state.copy()
            state_minus[i] -= h

            E_plus = self.energy(state_plus)
            E_minus = self.energy(state_minus)

            grad[i] = (E_plus - E_minus) / (2 * h)

        return grad


class CompositeEnergy(EnergyFunction):
    """
    Composite energy function that combines multiple energy terms.

    Energy: E = Σ E_i

    This allows flexible specification of constraints by combining different
    energy functions (e.g., GMM prior + clash penalty).

    Args:
        energy_funcs: List of EnergyFunction instances to combine

    Example:
        >>> gmm_energy = GMMEnergy(gmm)
        >>> clash_energy = ClashEnergy(polymer)
        >>> combined = CompositeEnergy([gmm_energy, clash_energy])
        >>> E = combined.energy(angles)
        >>> grad_E = combined.gradient(angles)
    """

    def __init__(self, energy_funcs: list[EnergyFunction]):
        """Initialize with list of energy functions."""
        if not energy_funcs:
            raise ValueError("At least one energy function required")
        self.energy_funcs = energy_funcs

    def energy(self, state: np.ndarray) -> float:
        """Compute sum of all energy functions."""
        return sum(e.energy(state) for e in self.energy_funcs)

    def gradient(self, state: np.ndarray) -> np.ndarray:
        """Compute sum of all gradients."""
        grad = np.zeros_like(state)
        for e in self.energy_funcs:
            grad = grad + e.gradient(state)
        return grad
