"""Diffusion process utilities for generative modeling.

This module provides utilities for training and sampling from diffusion models,
including noise schedules, forward/reverse diffusion processes, and timestep
embeddings.

Supports both DDPM (stochastic) and DDIM (deterministic) sampling, with the
same trained model. DDIM enables faster sampling with fewer steps.

Key components:

Noise Schedules:
    - NoiseSchedule: Base class for noise schedules
    - LinearNoiseSchedule: Linear interpolation between beta values
    - CosineNoiseSchedule: Cosine schedule (Nichol & Dhariwal, 2021)

Diffusion Process:
    - DiffusionProcess: Forward and reverse diffusion operations
    - Supports DDPM (stochastic) and DDIM (deterministic) sampling
    - Step skipping for faster inference (e.g., 50 steps instead of 1000)

Embeddings:
    - FixedSinusoidalEmbedding: Sinusoidal positional embedding for timesteps
    - TimestepEmbedding: Learnable timestep embedding with MLP

Example (Training - same for both DDPM and DDIM):
    >>> import torch
    >>> from ciffy.nn.diffusion import CosineNoiseSchedule, DiffusionProcess
    >>>
    >>> schedule = CosineNoiseSchedule(num_timesteps=1000)
    >>> process = DiffusionProcess(schedule)
    >>>
    >>> # Training: add noise and predict it
    >>> x = torch.randn(32, 3, 64, 64)
    >>> t = schedule.random_timestep((32,))
    >>> noise, noisy_x = process.forward_diffusion(x, t)
    >>> # loss = mse(model(noisy_x, t), noise)

Example (DDPM Sampling - 1000 steps, stochastic):
    >>> x = torch.randn(4, 3, 64, 64)  # Start from noise
    >>> for t in process.timesteps():
    ...     predicted_noise = model(x, t)
    ...     x = process.ddpm_step(x, predicted_noise, t)

Example (DDIM Sampling - 50 steps, deterministic):
    >>> x = torch.randn(4, 3, 64, 64)  # Start from noise
    >>> timesteps = process.get_sampling_timesteps(num_steps=50)
    >>> for i, t in enumerate(timesteps):
    ...     t_prev = timesteps[i + 1] if i < len(timesteps) - 1 else 0
    ...     predicted_noise = model(x, t)
    ...     x = process.ddim_step(x, predicted_noise, t, t_prev, eta=0.0)

References:
    - Ho, J., et al. (2020). "Denoising Diffusion Probabilistic Models"
    - Song, J., et al. (2020). "Denoising Diffusion Implicit Models"
    - Nichol, A. & Dhariwal, P. (2021). "Improved Denoising Diffusion
      Probabilistic Models"
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import nn

from ..layers.dense_network import DenseNetwork


class FixedSinusoidalEmbedding(nn.Module):
    """Fixed sinusoidal positional embedding for timesteps.

    Uses the sinusoidal encoding from "Attention Is All You Need" but with
    a fixed maximum index. The embedding is precomputed and stored as a buffer.

    Args:
        max_index: Maximum index value (e.g., number of timesteps).
        embedding_dim: Dimension of the embedding vectors.

    Example:
        >>> embed = FixedSinusoidalEmbedding(max_index=1000, embedding_dim=128)
        >>> t = torch.tensor([0, 100, 500, 999])
        >>> embeddings = embed(t)  # shape: (4, 128)
    """

    def __init__(
        self,
        max_index: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()

        if max_index <= 0:
            raise ValueError("max_index must be a positive integer")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be a positive integer")

        self.max_index = max_index
        self.embedding_dim = embedding_dim

        # Precompute the sinusoidal embeddings
        embeddings = self._compute_embeddings(max_index, embedding_dim)
        self.register_buffer("embeddings", embeddings)
        # Type hint for buffer
        self.embeddings: torch.Tensor

    @staticmethod
    def _compute_embeddings(max_index: int, embedding_dim: int) -> torch.Tensor:
        """Compute sinusoidal embeddings for all indices."""
        # Create position indices
        position = torch.arange(max_index).unsqueeze(1).float()

        # Create dimension indices for the frequency
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * (-math.log(10000.0) / embedding_dim)
        )

        # Compute sinusoidal embeddings
        embeddings = torch.zeros(max_index, embedding_dim)
        embeddings[:, 0::2] = torch.sin(position * div_term)
        embeddings[:, 1::2] = torch.cos(position * div_term)

        return embeddings

    def forward(
        self,
        index: int | torch.Tensor,
        shape: torch.Size = torch.Size([]),
    ) -> torch.Tensor:
        """Return the sinusoidal embedding at the specified index.

        Args:
            index: Index or tensor of indices to embed.
            shape: Optional shape to broadcast the embedding to.

        Returns:
            Embedding tensor of shape (*shape, embedding_dim) or
            (batch_size, embedding_dim) if index is a tensor.
        """
        if isinstance(index, int):
            embedding = self.embeddings[index]
            if shape:
                embedding = embedding.expand(*shape, -1)
            return embedding
        else:
            return self.embeddings[index]


class NoiseSchedule(nn.Module):
    """Base class for noise schedules in diffusion processes.

    A noise schedule defines the variance of noise added at each timestep
    during the forward diffusion process. This class computes and stores
    the key quantities:
        - beta_t: Noise variance at timestep t
        - alpha_t: 1 - beta_t
        - alphabar_t: Product of alpha from 0 to t (cumulative)
        - betatilde_t: Posterior variance for reverse process

    Args:
        betas: Tensor of beta values for each timestep, in range [0, 1).

    Raises:
        ValueError: If any beta is outside [0, 1).
    """

    def __init__(
        self,
        betas: torch.Tensor,
    ) -> None:
        super().__init__()

        # Validate betas
        if (betas < 0).any() or (betas >= 1).any():
            raise ValueError("All beta values must be in the range [0, 1).")

        # Type hints for buffers
        self._beta: torch.Tensor
        self._betatilde: torch.Tensor
        self._alpha: torch.Tensor
        self._alphabar: torch.Tensor

        # Compute derived quantities
        _alpha = 1 - betas
        _alphabar = torch.cumprod(_alpha, dim=0)

        # Posterior variance (betatilde) for reverse process
        # betatilde_t = (1 - alphabar_{t-1}) / (1 - alphabar_t) * beta_t
        _betatilde = (1 - _alphabar[:-1]) / (1 - _alphabar[1:]) * betas[1:]
        _betatilde = torch.cat([torch.zeros_like(betas[:1]), _betatilde])

        # Register as buffers for device movement
        self.register_buffer("_beta", betas)
        self.register_buffer("_alpha", _alpha)
        self.register_buffer("_alphabar", _alphabar)
        self.register_buffer("_betatilde", _betatilde)

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        """Return alpha values at timesteps t."""
        return self._alpha[t]

    def alphabar(self, t: torch.Tensor) -> torch.Tensor:
        """Return cumulative alpha product at timesteps t."""
        return self._alphabar[t]

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """Return beta values at timesteps t."""
        return self._beta[t]

    def betatilde(self, t: torch.Tensor) -> torch.Tensor:
        """Return posterior variance at timesteps t."""
        return self._betatilde[t]

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Return noise standard deviation at timesteps t."""
        return torch.sqrt(self.beta(t))

    def __len__(self) -> int:
        """Return number of timesteps in the schedule."""
        return len(self._beta)

    def timesteps(self) -> torch.Tensor:
        """Return timesteps in reverse order for sampling."""
        return torch.arange(len(self), device=self._beta.device).flip(0)

    def random_timestep(
        self,
        size: tuple | torch.Size = torch.Size([]),
    ) -> torch.Tensor:
        """Sample random timesteps uniformly.

        Args:
            size: Shape of the output tensor.

        Returns:
            Tensor of random timestep indices.
        """
        return torch.randint(0, len(self), size, device=self._beta.device)


class LinearNoiseSchedule(NoiseSchedule):
    """Linear noise schedule interpolating between two beta values.

    Creates a linearly spaced sequence of beta values from beta_start
    to beta_end over the specified number of timesteps.

    Args:
        num_timesteps: Number of diffusion timesteps.
        beta_range: Tuple of (beta_start, beta_end). Default: (1e-4, 2e-2).

    Example:
        >>> schedule = LinearNoiseSchedule(1000)
        >>> len(schedule)
        1000
        >>> schedule.beta(torch.tensor([0, 999]))
        tensor([1.0000e-04, 2.0000e-02])
    """

    DEFAULT_BETA_RANGE = (1e-4, 2e-2)

    def __init__(
        self,
        num_timesteps: int,
        beta_range: Tuple[float, float] = DEFAULT_BETA_RANGE,
    ) -> None:
        if num_timesteps <= 0:
            raise ValueError("num_timesteps must be a positive integer")

        betas = torch.linspace(
            beta_range[0],
            beta_range[1],
            num_timesteps,
        )
        super().__init__(betas)


class CosineNoiseSchedule(NoiseSchedule):
    """Cosine noise schedule for improved sample quality.

    Implements the cosine schedule from "Improved Denoising Diffusion
    Probabilistic Models" (Nichol & Dhariwal, 2021), which provides
    better sample quality by maintaining more signal at early timesteps.

    Args:
        num_timesteps: Number of diffusion timesteps.
        epsilon: Small offset to prevent singularities. Default: 0.008.
        beta_clip: Maximum beta value to prevent instabilities. Default: 0.999.

    Example:
        >>> schedule = CosineNoiseSchedule(1000)
        >>> len(schedule)
        1000
    """

    DEFAULT_EPSILON = 0.008
    DEFAULT_BETA_CLIP = 0.999

    def __init__(
        self,
        num_timesteps: int,
        epsilon: float = DEFAULT_EPSILON,
        beta_clip: float = DEFAULT_BETA_CLIP,
    ) -> None:
        if num_timesteps <= 0:
            raise ValueError("num_timesteps must be a positive integer")

        # Compute alphabar using cosine schedule
        timesteps = torch.arange(0, num_timesteps) / num_timesteps
        _alphabar = torch.cos(
            (timesteps + epsilon) / (timesteps[-1] + epsilon) * math.pi / 2
        ) ** 2
        _alphabar = _alphabar / _alphabar[0]

        # Derive alpha from alphabar
        _alpha = _alphabar[1:] / _alphabar[:-1]
        _alpha = torch.cat([_alphabar[:1], _alpha])

        # Compute betas with clipping
        betas = torch.clamp(1 - _alpha, 0, beta_clip)

        super().__init__(betas)


class DiffusionProcess(nn.Module):
    """Forward and reverse diffusion process operations.

    Implements the key operations for training and sampling from
    diffusion models:
        - forward_diffusion: Add noise to data (training)
        - ddpm_step: Stochastic reverse step (DDPM sampling)
        - ddim_step: Deterministic reverse step (DDIM sampling)
        - sample_step: Unified interface for both methods

    DDPM vs DDIM:
        - DDPM: Stochastic, requires all timesteps (e.g., 1000 steps)
        - DDIM: Deterministic (eta=0) or stochastic (eta>0), supports step
          skipping for faster sampling (e.g., 50 steps)
        - Both use the same trained model (same loss function)

    Args:
        schedule: A NoiseSchedule defining the noise levels.

    Example (DDPM - stochastic, all steps):
        >>> schedule = CosineNoiseSchedule(1000)
        >>> process = DiffusionProcess(schedule)
        >>> x = torch.randn(4, 3, 64, 64)
        >>> for t in process.timesteps():
        ...     x = process.ddpm_step(x, model(x, t), t)

    Example (DDIM - deterministic, fewer steps):
        >>> timesteps = process.get_sampling_timesteps(num_steps=50)
        >>> x = torch.randn(4, 3, 64, 64)
        >>> for i, t in enumerate(timesteps):
        ...     t_prev = timesteps[i + 1] if i < len(timesteps) - 1 else 0
        ...     x = process.ddim_step(x, model(x, t), t, t_prev, eta=0.0)
    """

    def __init__(
        self,
        schedule: NoiseSchedule,
    ) -> None:
        super().__init__()
        self.schedule = schedule

    def __len__(self) -> int:
        """Return number of timesteps in the diffusion process."""
        return len(self.schedule)

    def timesteps(self) -> torch.Tensor:
        """Return all timesteps in reverse order for DDPM sampling."""
        return self.schedule.timesteps()

    def get_sampling_timesteps(
        self,
        num_steps: int,
        spacing: str = "uniform",
        max_t: int | None = None,
    ) -> torch.Tensor:
        """Get a subset of timesteps for accelerated sampling.

        Useful for DDIM which can skip steps. For example, with a 1000-step
        schedule, you can sample with only 50 steps.

        Args:
            num_steps: Number of sampling steps (must be <= total timesteps).
            spacing: How to space the timesteps:
                - "uniform": Evenly spaced (default)
                - "trailing": More steps at the end (near t=0)
                - "leading": More steps at the start (near t=T)
            max_t: Maximum starting timestep. If None, uses total_steps - 1.
                Set lower (e.g., 980) to avoid numerical issues with cosine
                schedule where alphabar approaches 0 at high timesteps.

        Returns:
            Tensor of timestep indices in descending order.

        Example:
            >>> process.get_sampling_timesteps(50)  # 50 evenly spaced steps
            tensor([980, 960, 940, ..., 40, 20, 0])
        """
        total_steps = len(self.schedule)
        # Default to 980 for cosine schedule to avoid alphabar ≈ 0 issues
        if max_t is None:
            max_t = min(total_steps - 1, 980)
        max_t = min(max_t, total_steps - 1)

        if num_steps > total_steps:
            raise ValueError(
                f"num_steps ({num_steps}) cannot exceed total timesteps ({total_steps})"
            )
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")

        if spacing == "uniform":
            # Evenly spaced indices from 0 to max_t
            step_ratio = (max_t + 1) / num_steps
            timesteps = torch.arange(num_steps, device=self.schedule._beta.device)
            timesteps = (timesteps * step_ratio).long()
            timesteps = torch.clamp(timesteps, 0, max_t)
            timesteps = timesteps.flip(0)  # Reverse order
        elif spacing == "trailing":
            # More steps near t=0 (quadratic spacing)
            t = torch.linspace(0, 1, num_steps, device=self.schedule._beta.device)
            timesteps = ((1 - t**2) * max_t).long()
        elif spacing == "leading":
            # More steps near t=T (quadratic spacing)
            t = torch.linspace(0, 1, num_steps, device=self.schedule._beta.device)
            timesteps = ((t**2) * max_t).long()
            timesteps = timesteps.flip(0)
        else:
            raise ValueError(f"Unknown spacing: {spacing}. Use 'uniform', 'trailing', or 'leading'")

        return timesteps

    def forward_diffusion(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply forward diffusion (add noise) to input data.

        This is the same for both DDPM and DDIM training.

        Samples q(x_t | x_0) = N(x_t; sqrt(alphabar_t) * x_0, (1 - alphabar_t) * I)

        Args:
            x: Input data tensor of shape (batch, ...).
            timestep: Timestep indices of shape (batch,).

        Returns:
            Tuple of (noise, noisy_x) where:
                - noise: The sampled Gaussian noise
                - noisy_x: The noised input x_t
        """
        # Sample standard normal noise
        noise = torch.randn_like(x)

        # Get alphabar for each sample in the batch
        alphabar = self.schedule.alphabar(timestep)

        # Broadcast alphabar to match x shape: (batch,) -> (batch, 1, 1, ...)
        while alphabar.dim() < x.dim():
            alphabar = alphabar.unsqueeze(-1)

        # Apply forward diffusion: x_t = sqrt(alphabar) * x_0 + sqrt(1 - alphabar) * noise
        noisy_x = torch.sqrt(alphabar) * x + torch.sqrt(1 - alphabar) * noise

        return noise, noisy_x

    def _predict_x0(
        self,
        x_t: torch.Tensor,
        predicted_noise: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Predict x_0 from x_t and predicted noise.

        Uses the reparameterization: x_0 = (x_t - sqrt(1-alphabar) * eps) / sqrt(alphabar)

        Args:
            x_t: Noisy sample at timestep t.
            predicted_noise: Model's noise prediction.
            t: Current timestep.

        Returns:
            Predicted clean sample x_0.
        """
        alphabar = self.schedule.alphabar(t)
        return (x_t - torch.sqrt(1 - alphabar) * predicted_noise) / torch.sqrt(alphabar)

    def ddpm_step(
        self,
        x: torch.Tensor,
        predicted_noise: torch.Tensor,
        timestep: int | torch.Tensor,
    ) -> torch.Tensor:
        """Apply one step of DDPM reverse diffusion (stochastic).

        This is the original DDPM formulation which adds noise at each step.
        Requires stepping through all timesteps (no step skipping).

        Args:
            x: Noisy input at timestep t, shape (batch, ...).
            predicted_noise: Model's noise prediction, same shape as x.
            timestep: Current timestep (scalar).

        Returns:
            Denoised sample x_{t-1}.
        """
        # Convert timestep to tensor if needed
        if isinstance(timestep, int):
            t = torch.tensor(timestep, device=x.device)
        else:
            t = timestep

        # Get schedule values
        alpha = self.schedule.alpha(t)
        alphabar = self.schedule.alphabar(t)
        beta = self.schedule.beta(t)
        sigma = self.schedule.sigma(t)

        # Sample noise (zero at t=0)
        if t == 0:
            z = torch.zeros_like(x)
        else:
            z = torch.randn_like(x)

        # DDPM reverse step
        # x_{t-1} = 1/sqrt(alpha_t) * (x_t - beta_t/sqrt(1-alphabar_t) * eps) + sigma_t * z
        denoised = (
            1 / torch.sqrt(alpha) * (x - beta / torch.sqrt(1 - alphabar) * predicted_noise)
            + sigma * z
        )

        return denoised

    def ddim_step(
        self,
        x: torch.Tensor,
        predicted_noise: torch.Tensor,
        timestep: int | torch.Tensor,
        timestep_prev: int | torch.Tensor,
        eta: float = 0.0,
        clip_x0: tuple[float, float] | None = None,
    ) -> torch.Tensor:
        """Apply one step of DDIM reverse diffusion.

        DDIM allows deterministic sampling (eta=0) and step skipping for
        faster inference. When eta=1, it's equivalent to DDPM.

        Args:
            x: Noisy input at timestep t, shape (batch, ...).
            predicted_noise: Model's noise prediction, same shape as x.
            timestep: Current timestep t.
            timestep_prev: Previous timestep t-1 (can skip steps, e.g., t=100, t_prev=80).
            eta: Stochasticity parameter. 0 = deterministic, 1 = DDPM-like.
            clip_x0: Optional (min, max) bounds to clip the x0 prediction for stability.
                Use (-1, 1) for normalized data, None for latent space (no clipping).

        Returns:
            Sample at timestep t_prev.

        Example:
            >>> timesteps = process.get_sampling_timesteps(50)
            >>> for i, t in enumerate(timesteps):
            ...     t_prev = timesteps[i + 1] if i < len(timesteps) - 1 else 0
            ...     x = process.ddim_step(x, model(x, t), t, t_prev, eta=0.0)
        """
        # Convert timesteps to tensors
        if isinstance(timestep, int):
            t = torch.tensor(timestep, device=x.device)
        else:
            t = timestep
        if isinstance(timestep_prev, int):
            t_prev = torch.tensor(timestep_prev, device=x.device)
        else:
            t_prev = timestep_prev

        # Get alphabar values
        alphabar_t = self.schedule.alphabar(t)
        # Handle t_prev = 0 case (alphabar at t=-1 is defined as 1)
        if t_prev == 0:
            alphabar_t_prev = torch.tensor(1.0, device=x.device)
        else:
            alphabar_t_prev = self.schedule.alphabar(t_prev)

        # Predict x_0 from x_t and predicted noise
        x0_pred = self._predict_x0(x, predicted_noise, t)

        # Optionally clip x0 prediction for stability
        if clip_x0 is not None:
            x0_pred = torch.clamp(x0_pred, clip_x0[0], clip_x0[1])

        # Compute sigma for stochasticity
        # sigma_t = eta * sqrt((1 - alphabar_{t-1}) / (1 - alphabar_t)) * sqrt(1 - alphabar_t / alphabar_{t-1})
        if eta > 0 and t_prev > 0:
            sigma = eta * torch.sqrt(
                (1 - alphabar_t_prev) / (1 - alphabar_t)
                * (1 - alphabar_t / alphabar_t_prev)
            )
        else:
            sigma = torch.tensor(0.0, device=x.device)

        # Direction pointing to x_t
        # The "direction" is the component of predicted noise that remains
        dir_xt = torch.sqrt(1 - alphabar_t_prev - sigma**2) * predicted_noise

        # DDIM update
        # x_{t-1} = sqrt(alphabar_{t-1}) * x0_pred + dir_xt + sigma * z
        if sigma > 0:
            z = torch.randn_like(x)
        else:
            z = torch.zeros_like(x)

        x_prev = torch.sqrt(alphabar_t_prev) * x0_pred + dir_xt + sigma * z

        return x_prev

    def sample_step(
        self,
        x: torch.Tensor,
        predicted_noise: torch.Tensor,
        timestep: int | torch.Tensor,
        timestep_prev: int | torch.Tensor | None = None,
        method: str = "ddim",
        eta: float = 0.0,
    ) -> torch.Tensor:
        """Unified interface for sampling steps.

        Args:
            x: Noisy input at timestep t.
            predicted_noise: Model's noise prediction.
            timestep: Current timestep.
            timestep_prev: Previous timestep (required for DDIM, ignored for DDPM).
            method: Sampling method, either "ddpm" or "ddim".
            eta: Stochasticity for DDIM (0 = deterministic, 1 = DDPM-like).

        Returns:
            Sample at the previous timestep.
        """
        if method == "ddpm":
            return self.ddpm_step(x, predicted_noise, timestep)
        elif method == "ddim":
            if timestep_prev is None:
                # Default to t-1
                t = timestep if isinstance(timestep, int) else timestep.item()
                timestep_prev = max(0, t - 1)
            return self.ddim_step(x, predicted_noise, timestep, timestep_prev, eta=eta)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'ddpm' or 'ddim'")

    # Backward compatibility alias
    def reverse_diffusion(
        self,
        x: torch.Tensor,
        predicted_noise: torch.Tensor,
        timestep: int | torch.Tensor,
    ) -> torch.Tensor:
        """Alias for ddpm_step (backward compatibility)."""
        return self.ddpm_step(x, predicted_noise, timestep)


class TimestepEmbedding(nn.Module):
    """Learnable timestep embedding with MLP.

    Combines a fixed sinusoidal embedding with a learnable MLP projection,
    following the approach in "Denoising Diffusion Probabilistic Models".

    Args:
        max_index: Maximum timestep value.
        embedding_dim: Dimension of the output embedding.
        hidden_mult: Multiplier for hidden layer size. Default: 4.

    Example:
        >>> embed = TimestepEmbedding(max_index=1000, embedding_dim=256)
        >>> t = torch.tensor([0, 100, 500])
        >>> embeddings = embed(t)  # shape: (3, 256)
    """

    def __init__(
        self,
        max_index: int,
        embedding_dim: int,
        hidden_mult: int = 4,
    ) -> None:
        super().__init__()

        # Fixed sinusoidal embedding
        self.embedding = FixedSinusoidalEmbedding(
            max_index=max_index,
            embedding_dim=embedding_dim,
        )

        # Learnable MLP projection
        self.dense = DenseNetwork(
            in_size=embedding_dim,
            out_size=embedding_dim,
            hidden_sizes=[hidden_mult * embedding_dim],
        )

    def forward(
        self,
        index: int | torch.Tensor,
        shape: torch.Size = torch.Size([]),
    ) -> torch.Tensor:
        """Return the learnable timestep embedding.

        Args:
            index: Timestep index or tensor of indices.
            shape: Optional shape to broadcast to.

        Returns:
            Embedding tensor passed through the MLP.
        """
        return self.dense(self.embedding(index, shape))


__all__ = [
    "FixedSinusoidalEmbedding",
    "NoiseSchedule",
    "LinearNoiseSchedule",
    "CosineNoiseSchedule",
    "DiffusionProcess",
    "TimestepEmbedding",
]
