"""
Archived: Langevin dynamics with temperature tempering.

This function was removed from ciffy/sampling/langevin.py as it was unused,
but preserved here for potential future use.

To restore: copy this function back to ciffy/sampling/langevin.py and add
to the module's __all__ export list.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ciffy.sampling.energy import EnergyFunction


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
