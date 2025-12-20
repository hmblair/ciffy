"""
Langevin dynamics sampler for gradient-based inference.

Implements underdamped Langevin dynamics (also known as stochastic gradient descent
with momentum, or Hamiltonian Monte Carlo with noise). Samples from the posterior
distribution P(x) ∝ exp(-E(x)/T) where E is the energy function and T is temperature.

References:
    - Leimkuhler and Matthews (2015): "Molecular Dynamics: With Deterministic and
      Stochastic Numerical Methods"
    - Betancourt (2017): "A Conceptual Introduction to Hamiltonian Monte Carlo"
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .energy import EnergyFunction


def langevin_dynamics(
    energy: "EnergyFunction",
    initial_state: np.ndarray,
    n_steps: int = 50,
    step_size: float = 0.01,
    temperature: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Sample from P(x) ∝ exp(-E(x)/T) using Langevin dynamics.

    Implements the Euler-Maruyama discretization of overdamped Langevin dynamics:

        x_{t+1} = x_t - (step_size/2) * ∇E(x_t) + √(step_size * T) * noise

    This is an unadjusted Langevin algorithm (ULA) that samples from the target
    distribution in the limit of small step_size.

    Args:
        energy: Energy function E(x) to sample from
        initial_state: Initial state vector x_0
        n_steps: Number of Langevin steps (default 50)
        step_size: Step size for discretization (default 0.01)
        temperature: Temperature T for thermal noise (default 1.0)
        rng: Random number generator for reproducibility

    Returns:
        Final state after n_steps of Langevin dynamics

    Example:
        >>> from ciffy.sampling.energy import CompositeEnergy, GMMEnergy, ClashEnergy
        >>> from ciffy.sampling.langevin import langevin_dynamics
        >>>
        >>> energy = CompositeEnergy([GMMEnergy(gmm), ClashEnergy(polymer)])
        >>> initial_angles = np.array([1.5, -2.0, 3.14])
        >>> final_angles = langevin_dynamics(
        ...     energy, initial_angles, n_steps=50, step_size=0.01, temperature=1.0
        ... )

    References:
        Roberts and Tweedie (1996): "Exponential convergence of Langevin distributions
        and their discrete approximations"

        Welling and Teh (2011): "Bayesian Learning via Stochastic Gradient Descent"
    """
    if rng is None:
        rng = np.random.default_rng()

    state = initial_state.astype(np.float64).copy()
    n_dims = len(state)

    for step in range(n_steps):
        # Compute gradient of energy at current state
        grad_E = energy.gradient(state)

        # Ensure gradient is same shape as state
        grad_E = np.atleast_1d(grad_E)

        # Deterministic drift term (gradient descent)
        drift = -(step_size / 2) * grad_E

        # Stochastic diffusion term (Gaussian noise)
        noise = rng.normal(0, np.sqrt(step_size * temperature), size=n_dims)

        # Update state (Euler-Maruyama step)
        state = state + drift + noise

    return state


def langevin_dynamics_with_adaptation(
    energy: "EnergyFunction",
    initial_state: np.ndarray,
    n_steps: int = 50,
    step_size: float = 0.01,
    temperature: float = 1.0,
    adapt_interval: int = 10,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Langevin dynamics with adaptive step size.

    Adjusts step size during sampling to maintain reasonable acceptance rates
    and exploration efficiency. Can be useful for high-dimensional problems
    or when the energy landscape varies significantly.

    Args:
        energy: Energy function E(x) to sample from
        initial_state: Initial state vector
        n_steps: Number of Langevin steps
        step_size: Initial step size
        temperature: Temperature for thermal noise
        adapt_interval: How often to adapt step size (steps)
        rng: Random number generator

    Returns:
        Final state after n_steps
    """
    if rng is None:
        rng = np.random.default_rng()

    state = initial_state.astype(np.float64).copy()
    n_dims = len(state)

    # Estimate gradient magnitude for adaptation
    initial_grad = energy.gradient(state)
    grad_magnitude = np.linalg.norm(initial_grad)

    current_step_size = step_size

    for step in range(n_steps):
        # Adapt step size every adapt_interval steps
        if step > 0 and step % adapt_interval == 0:
            grad_E = energy.gradient(state)
            grad_magnitude = np.linalg.norm(grad_E)

            # Increase step size if gradient is small (flat region)
            # Decrease step size if gradient is large (steep region)
            if grad_magnitude < 1e-3:
                current_step_size *= 1.1  # Increase by 10%
            elif grad_magnitude > 1e2:
                current_step_size *= 0.9  # Decrease by 10%

        # Compute gradient
        grad_E = energy.gradient(state)
        grad_E = np.atleast_1d(grad_E)

        # Drift and diffusion
        drift = -(current_step_size / 2) * grad_E
        noise = rng.normal(0, np.sqrt(current_step_size * temperature), size=n_dims)

        # Update state
        state = state + drift + noise

    return state


def langevin_dynamics_with_tempering(
    energy: "EnergyFunction",
    initial_state: np.ndarray,
    n_steps: int = 50,
    step_size: float = 0.01,
    temperature_schedule: str = "constant",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Langevin dynamics with temperature schedule.

    Useful for sampling from multi-modal distributions. Can gradually decrease
    temperature (annealing) to focus on high-probability regions, or use other
    schedules for exploration.

    Args:
        energy: Energy function E(x) to sample from
        initial_state: Initial state vector
        n_steps: Number of Langevin steps
        step_size: Step size for discretization
        temperature_schedule: Temperature schedule type
            - "constant": T(t) = 1.0 (default)
            - "linear_anneal": T(t) = 1.0 - (t/n_steps) (linearly decrease to 0)
            - "exponential_anneal": T(t) = exp(-t/n_steps) (exponentially decrease)
        rng: Random number generator

    Returns:
        Final state after n_steps
    """
    if rng is None:
        rng = np.random.default_rng()

    state = initial_state.astype(np.float64).copy()
    n_dims = len(state)

    for step in range(n_steps):
        # Compute temperature according to schedule
        if temperature_schedule == "constant":
            temperature = 1.0
        elif temperature_schedule == "linear_anneal":
            temperature = max(0.01, 1.0 - step / n_steps)
        elif temperature_schedule == "exponential_anneal":
            temperature = np.exp(-step / n_steps)
        else:
            raise ValueError(f"Unknown temperature schedule: {temperature_schedule}")

        # Compute gradient
        grad_E = energy.gradient(state)
        grad_E = np.atleast_1d(grad_E)

        # Drift and diffusion with current temperature
        drift = -(step_size / 2) * grad_E
        noise = rng.normal(0, np.sqrt(step_size * temperature), size=n_dims)

        # Update state
        state = state + drift + noise

    return state
