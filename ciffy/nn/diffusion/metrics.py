"""
Evaluation metrics for diffusion models.

Provides functions for assessing diffusion model quality:
- Denoising loss at various timesteps
- Per-timestep loss analysis
- Sample quality metrics (RMSD, validity)
- ELBO estimates for log-likelihood

These metrics help diagnose training progress and sample quality.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np
import torch

from ciffy.backend import Array

if TYPE_CHECKING:
    from .process import DiffusionProcess


@dataclass
class TimestepLossProfile:
    """Per-timestep loss analysis.

    Helps identify which noise levels are harder for the model to denoise.
    Early timesteps (high noise) and late timesteps (fine details) often
    have different characteristics.

    Attributes:
        timesteps: Array of timestep values.
        losses: Mean loss at each timestep.
        stds: Standard deviation of loss at each timestep.
        n_samples: Number of samples per timestep.
    """

    timesteps: np.ndarray
    losses: np.ndarray
    stds: np.ndarray
    n_samples: int

    @property
    def mean_loss(self) -> float:
        """Overall mean loss across all timesteps."""
        return float(np.mean(self.losses))

    @property
    def max_loss_timestep(self) -> int:
        """Timestep with highest loss."""
        return int(self.timesteps[np.argmax(self.losses)])

    @property
    def min_loss_timestep(self) -> int:
        """Timestep with lowest loss."""
        return int(self.timesteps[np.argmin(self.losses)])

    def loss_at(self, t: int) -> float:
        """Get loss at specific timestep (nearest match)."""
        idx = np.argmin(np.abs(self.timesteps - t))
        return float(self.losses[idx])

    def __repr__(self) -> str:
        return (
            f"TimestepLossProfile(mean={self.mean_loss:.4f}, "
            f"max_t={self.max_loss_timestep}, "
            f"min_t={self.min_loss_timestep})"
        )


@dataclass
class SampleQualityMetrics:
    """Quality metrics for generated samples.

    For molecular data, these measure structural validity and accuracy.

    Attributes:
        rmsd_to_reference: Mean RMSD to reference structures (if available).
        bond_length_rmse: RMSE of bond lengths vs ideal.
        bond_angle_rmse: RMSE of bond angles vs ideal (degrees).
        clash_fraction: Fraction of samples with steric clashes.
        validity_rate: Fraction of chemically valid samples.
        n_samples: Number of samples evaluated.
    """

    rmsd_to_reference: float | None = None
    bond_length_rmse: float | None = None
    bond_angle_rmse: float | None = None
    clash_fraction: float | None = None
    validity_rate: float | None = None
    n_samples: int = 0

    def __repr__(self) -> str:
        parts = [f"n={self.n_samples}"]
        if self.rmsd_to_reference is not None:
            parts.append(f"rmsd={self.rmsd_to_reference:.4f}")
        if self.bond_length_rmse is not None:
            parts.append(f"bond_rmse={self.bond_length_rmse:.4f}")
        if self.validity_rate is not None:
            parts.append(f"valid={self.validity_rate:.1%}")
        return f"SampleQualityMetrics({', '.join(parts)})"


@dataclass
class DiffusionMetrics:
    """Comprehensive evaluation metrics for a diffusion model.

    Attributes:
        train_loss: Mean denoising loss on training data.
        test_loss: Mean denoising loss on test data.
        loss_profile: Per-timestep loss breakdown (optional).
        sample_quality: Sample quality metrics (optional).
        elbo: Evidence lower bound estimate (optional).
    """

    train_loss: float
    test_loss: float
    loss_profile: TimestepLossProfile | None = None
    sample_quality: SampleQualityMetrics | None = None
    elbo: float | None = None

    @property
    def generalization_gap(self) -> float:
        """Difference between test and train loss."""
        return self.test_loss - self.train_loss

    def __repr__(self) -> str:
        parts = [
            f"train={self.train_loss:.4f}",
            f"test={self.test_loss:.4f}",
            f"gap={self.generalization_gap:.4f}",
        ]
        if self.elbo is not None:
            parts.append(f"elbo={self.elbo:.2f}")
        return f"DiffusionMetrics({', '.join(parts)})"


def compute_denoising_loss(
    model: torch.nn.Module,
    process: "DiffusionProcess",
    data: Array,
    batch_size: int = 256,
    n_samples: int | None = None,
) -> float:
    """
    Compute mean denoising loss on data.

    Samples random timesteps and computes MSE between predicted and true noise.

    Args:
        model: Denoising model that takes (noisy_x, timestep) and predicts noise.
        process: DiffusionProcess for forward diffusion.
        data: (N, ...) data array.
        batch_size: Batch size for evaluation.
        n_samples: Number of samples to use. None uses all data.

    Returns:
        Mean squared error loss.
    """
    model.eval()
    device = next(model.parameters()).device

    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data).float()
    data = data.to(device)

    if n_samples is not None:
        indices = torch.randperm(len(data))[:n_samples]
        data = data[indices]

    n_total = len(data)
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for i in range(0, n_total, batch_size):
            batch = data[i:i + batch_size]
            t = process.schedule.random_timestep((len(batch),))

            noise, noisy_x = process.forward_diffusion(batch, t)
            predicted_noise = model(noisy_x, t)

            mse = ((predicted_noise - noise) ** 2).mean()
            total_loss += mse.item()
            n_batches += 1

    return total_loss / n_batches


def compute_timestep_loss_profile(
    model: torch.nn.Module,
    process: "DiffusionProcess",
    data: Array,
    n_timesteps: int = 20,
    samples_per_timestep: int = 100,
    batch_size: int = 256,
) -> TimestepLossProfile:
    """
    Compute loss at different timesteps to diagnose model performance.

    Args:
        model: Denoising model.
        process: DiffusionProcess.
        data: Data to evaluate on.
        n_timesteps: Number of timesteps to sample.
        samples_per_timestep: Number of samples at each timestep.
        batch_size: Batch size for evaluation.

    Returns:
        TimestepLossProfile with per-timestep losses.
    """
    model.eval()
    device = next(model.parameters()).device

    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data).float()
    data = data.to(device)

    total_timesteps = len(process.schedule)
    timesteps = np.linspace(0, total_timesteps - 1, n_timesteps, dtype=int)

    losses = []
    stds = []

    with torch.no_grad():
        for t_val in timesteps:
            t_losses = []

            # Sample multiple times at this timestep
            for _ in range(0, samples_per_timestep, batch_size):
                # Random subset of data
                indices = torch.randperm(len(data))[:min(batch_size, samples_per_timestep)]
                batch = data[indices]

                t = torch.full((len(batch),), t_val, device=device, dtype=torch.long)
                noise, noisy_x = process.forward_diffusion(batch, t)
                predicted_noise = model(noisy_x, t)

                per_sample_mse = ((predicted_noise - noise) ** 2).mean(dim=tuple(range(1, noise.ndim)))
                t_losses.extend(per_sample_mse.cpu().numpy())

            losses.append(np.mean(t_losses))
            stds.append(np.std(t_losses))

    return TimestepLossProfile(
        timesteps=timesteps,
        losses=np.array(losses),
        stds=np.array(stds),
        n_samples=samples_per_timestep,
    )


def compute_elbo(
    model: torch.nn.Module,
    process: "DiffusionProcess",
    data: Array,
    n_timestep_samples: int = 100,
    batch_size: int = 256,
) -> float:
    """
    Estimate the Evidence Lower Bound (ELBO) for log-likelihood.

    Uses the variational bound from Ho et al. (2020). Higher is better.
    Note: This is an estimate that becomes more accurate with more samples.

    Args:
        model: Denoising model.
        process: DiffusionProcess.
        data: Data to evaluate on.
        n_timestep_samples: Number of timestep samples for Monte Carlo estimate.
        batch_size: Batch size.

    Returns:
        ELBO estimate (in nats, per data point).
    """
    model.eval()
    device = next(model.parameters()).device

    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data).float()
    data = data.to(device)

    T = len(process.schedule)
    n_data = len(data)
    total_vlb = 0.0
    n_samples = 0

    with torch.no_grad():
        for i in range(0, n_data, batch_size):
            batch = data[i:i + batch_size]
            batch_vlb = 0.0

            # Monte Carlo estimate over timesteps
            for _ in range(n_timestep_samples):
                t = process.schedule.random_timestep((len(batch),))

                noise, noisy_x = process.forward_diffusion(batch, t)
                predicted_noise = model(noisy_x, t)

                # Simple loss term (ignoring constant factors)
                mse = ((predicted_noise - noise) ** 2).sum(dim=tuple(range(1, noise.ndim)))
                batch_vlb += mse.mean().item()

            batch_vlb /= n_timestep_samples
            # Scale by number of timesteps (rough ELBO approximation)
            batch_vlb *= T

            total_vlb += batch_vlb * len(batch)
            n_samples += len(batch)

    # Negative because ELBO is -VLB
    return -total_vlb / n_samples


def compute_sample_rmsd(
    samples: Array,
    references: Array,
) -> float:
    """
    Compute mean RMSD between samples and reference structures.

    Uses Kabsch alignment for fair comparison.

    Args:
        samples: (N, n_atoms, 3) generated samples.
        references: (N, n_atoms, 3) reference structures.

    Returns:
        Mean aligned RMSD.
    """
    from ciffy import rmsd

    if isinstance(samples, torch.Tensor):
        samples = samples.cpu().numpy()
    if isinstance(references, torch.Tensor):
        references = references.cpu().numpy()

    rmsds = rmsd(samples, references)
    return float(np.mean(rmsds))


def evaluate_samples(
    samples: Array,
    references: Array | None = None,
    bond_checker: Callable[[Array], float] | None = None,
    validity_checker: Callable[[Array], float] | None = None,
) -> SampleQualityMetrics:
    """
    Evaluate quality of generated samples.

    Args:
        samples: (N, n_atoms, 3) generated samples.
        references: (N, n_atoms, 3) reference structures for RMSD.
        bond_checker: Function that returns bond length RMSE for samples.
        validity_checker: Function that returns validity rate for samples.

    Returns:
        SampleQualityMetrics with available metrics.
    """
    if isinstance(samples, torch.Tensor):
        samples = samples.cpu().numpy()

    n_samples = len(samples)

    rmsd_val = None
    if references is not None:
        rmsd_val = compute_sample_rmsd(samples, references)

    bond_rmse = None
    if bond_checker is not None:
        bond_rmse = bond_checker(samples)

    validity = None
    if validity_checker is not None:
        validity = validity_checker(samples)

    return SampleQualityMetrics(
        rmsd_to_reference=rmsd_val,
        bond_length_rmse=bond_rmse,
        validity_rate=validity,
        n_samples=n_samples,
    )


def compute_diffusion_metrics(
    model: torch.nn.Module,
    process: "DiffusionProcess",
    train_data: Array,
    test_data: Array,
    batch_size: int = 256,
    compute_profile: bool = True,
    compute_elbo_estimate: bool = False,
) -> DiffusionMetrics:
    """
    Compute comprehensive evaluation metrics for a diffusion model.

    Args:
        model: Denoising model.
        process: DiffusionProcess.
        train_data: Training data for computing train loss.
        test_data: Test data for computing test loss.
        batch_size: Batch size for evaluation.
        compute_profile: Whether to compute per-timestep profile.
        compute_elbo_estimate: Whether to compute ELBO (slower).

    Returns:
        DiffusionMetrics with train/test loss and optional extras.

    Example:
        >>> metrics = compute_diffusion_metrics(model, process, train_data, test_data)
        >>> print(f"Train: {metrics.train_loss:.4f}, Test: {metrics.test_loss:.4f}")
        >>> print(f"Generalization gap: {metrics.generalization_gap:.4f}")
    """
    train_loss = compute_denoising_loss(model, process, train_data, batch_size)
    test_loss = compute_denoising_loss(model, process, test_data, batch_size)

    profile = None
    if compute_profile:
        profile = compute_timestep_loss_profile(model, process, test_data, batch_size=batch_size)

    elbo = None
    if compute_elbo_estimate:
        elbo = compute_elbo(model, process, test_data, batch_size=batch_size)

    return DiffusionMetrics(
        train_loss=train_loss,
        test_loss=test_loss,
        loss_profile=profile,
        elbo=elbo,
    )
