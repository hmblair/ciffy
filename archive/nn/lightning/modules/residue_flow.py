"""LightningModule for residue flow model training.

Special handling for PCA preprocessing which requires all training data upfront.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from ciffy.nn.config import TrainingConfig, SchedulerConfig
from ciffy.geometry.constraints import GeometryConstraints
from .base import BaseCiffyModule


def _default_flow_training_config() -> TrainingConfig:
    """Default training config for residue flow with cosine scheduler and gradient logging."""
    return TrainingConfig(
        log_gradient_norms=True,
        scheduler=SchedulerConfig(scheduler_type="cosine"),
    )

if TYPE_CHECKING:
    from ciffy.biochemistry import Residue
    from ciffy.nn.flow.residue.model import ResidueFlowModel

# Pre-computed constant for Gaussian log-prob
LOG_2PI = math.log(2 * math.pi)


@dataclass
class ResidueFlowModelConfig:
    """Configuration for ResidueFlowModel."""

    latent_dim: int = 12
    n_layers: int = 8
    hidden_dim: int = 64
    bound: float | None = None
    use_rotation: bool = True
    noise_std: float = 0.05
    latent_reg: float = 0.01  # Jacobian regularization: penalize (log_det)^2
    transform_scale: float = 1.0  # Scale factor for transforms in PCA
    # Gaussian regularization to enforce N(0,1) latent marginals
    gauss_reg: float = 0.5  # Weight for moment matching: (μ² + (σ-1)²)
    kurtosis_reg: float = 0.1  # Weight for kurtosis: (κ-3)²
    kurtosis_warmup: float = 0.25  # Fraction of epochs before kurtosis reg starts
    outlier_clip: float = 5.0  # Clip latents for robust moment estimation
    outlier_penalty: float = 0.1  # Penalty weight for latents beyond clip threshold


@dataclass
class ResidueFlowDataConfig:
    """Data configuration for residue flow training."""

    data_dir: str = ""
    cif_patterns: list[str] | None = None
    residue: str = "A"  # Residue name (A, U, G, C, etc.)
    min_coverage: float = 0.9
    train_split: float = 0.8
    batch_size: int = 256


@dataclass
class ResidueFlowFullConfig:
    """Full configuration for residue flow training."""

    model: ResidueFlowModelConfig = field(default_factory=ResidueFlowModelConfig)
    data: ResidueFlowDataConfig = field(default_factory=ResidueFlowDataConfig)
    training: TrainingConfig = field(default_factory=_default_flow_training_config)


class ResidueFlowModule(BaseCiffyModule):
    """LightningModule for training residue flow models.

    Unlike diffusion modules, this handles:
    - PCA computation during setup (requires full dataset)
    - Normalizing flow training with NLL loss
    - Creates a full ResidueFlowModel with metadata for save/load

    The model is created in setup() after PCA is computed from the
    DataModule's training data.

    Example:
        >>> from ciffy.biochemistry import Residue
        >>> from ciffy.nn.lightning import ResidueFlowModule, ResidueDataModule
        >>> import lightning as L
        >>>
        >>> config = ResidueFlowFullConfig()
        >>> dm = ResidueDataModule(cif_paths, residue=Residue.A)
        >>> module = ResidueFlowModule(config, residue=Residue.A)
        >>>
        >>> trainer = L.Trainer(max_epochs=200)
        >>> trainer.fit(module, dm)
        >>>
        >>> # Get the trained model for inference/saving
        >>> model = module.get_model()
        >>> model.save("my_model")
    """

    def __init__(
        self,
        config: ResidueFlowFullConfig,
        residue: "Residue",
    ) -> None:
        """Initialize the residue flow module.

        Args:
            config: Full training configuration.
            residue: Residue type to train on.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["residue"])

        self.config = config
        self.training_config = config.training
        self.residue = residue

        # Model created in setup() after PCA computed
        self._residue_model: "ResidueFlowModel | None" = None
        self.pca_V: torch.Tensor | None = None
        self.pca_mean: torch.Tensor | None = None
        self._geometry_constraints: GeometryConstraints | None = None

    @property
    def model(self) -> torch.nn.Module | None:
        """The underlying PCAFlow model (for training)."""
        if self._residue_model is None:
            return None
        return self._residue_model.flow

    def get_model(self) -> "ResidueFlowModel":
        """Get the trained ResidueFlowModel for inference/saving.

        Returns:
            The trained ResidueFlowModel with all metadata.

        Raises:
            ValueError: If training hasn't been run yet.
        """
        if self._residue_model is None:
            raise ValueError("Model not yet created. Run trainer.fit() first.")
        return self._residue_model

    def setup(self, stage: str) -> None:
        """Create model with PCA from training data.

        This is called after DataModule.setup(), so we can access
        the training data for PCA computation.
        """
        if stage != "fit" or self._residue_model is not None:
            return

        # Get training data and atoms from datamodule
        dm = self.trainer.datamodule
        train_data = dm.train_data
        atoms = dm.atoms

        if train_data is None or len(train_data) == 0:
            raise ValueError("No training data available for PCA computation")

        # Import here to avoid circular imports
        from ciffy.nn.flow.residue.data import compute_pca
        from ciffy.nn.flow.residue.model import PCAFlow, ResidueFlowModel

        config = self.config.model

        # Compute PCA
        V, mean, singular_values, var_explained = compute_pca(
            train_data, n_components=config.latent_dim
        )

        self.pca_V = torch.from_numpy(V).float()
        self.pca_mean = torch.from_numpy(mean).float()

        # Store PCA info for logging later (can't log in setup)
        self._pca_var_explained = var_explained[config.latent_dim - 1]

        # Create PCAFlow
        flow = PCAFlow(
            self.pca_V,
            self.pca_mean,
            n_layers=config.n_layers,
            hidden_dim=config.hidden_dim,
            bound=config.bound,
            use_rotation=config.use_rotation,
        )

        # Wrap in ResidueFlowModel with metadata
        atom_list = atoms.tolist() if isinstance(atoms, np.ndarray) else list(atoms)
        self._residue_model = ResidueFlowModel(
            flow=flow,
            residue=self.residue,
            atom_indices=atom_list,
            n_atoms=len(atoms),
            transform_scale=config.transform_scale,
        )

        # Create geometry constraints for validation
        self._geometry_constraints = GeometryConstraints.from_residue(
            self.residue, atom_list
        )

    def training_step(self, batch: tuple[torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Compute NLL loss for a batch.

        Args:
            batch: Tuple containing (data,) tensor of shape (batch_size, n_features).
            batch_idx: Batch index (unused).

        Returns:
            Negative log-likelihood loss.
        """
        # TensorDataset returns tuple
        data = batch[0] if isinstance(batch, (tuple, list)) else batch

        # Log input data statistics per batch
        self.log("train/data_mean", data.mean(), on_step=True, on_epoch=False)
        self.log("train/data_std", data.std(), on_step=True, on_epoch=False)

        # Add noise regularization during training
        if self.config.model.noise_std > 0:
            data = data + self.config.model.noise_std * torch.randn_like(data)

        # Forward pass through the flow
        z, log_det = self._residue_model.flow(data)

        # Compute NLL: -log p(z) - log |det J|
        log_pz = -0.5 * (z**2 + LOG_2PI).sum(dim=-1)
        nll = -(log_pz + log_det).mean()

        # Jacobian regularization: penalize log_det away from 0
        # This prevents the flow from exploiting the Jacobian to lower NLL
        jac_reg = self.config.model.latent_reg
        if jac_reg > 0:
            jac_loss = (log_det ** 2).mean()
            loss = nll + jac_reg * jac_loss
            self.log("train/jac_loss", jac_loss, on_step=False, on_epoch=True)
            self.log("train/log_det_mean", log_det.mean(), on_step=False, on_epoch=True)
        else:
            loss = nll

        # Gaussian regularization: enforce N(0,1) marginals
        config = self.config.model
        if config.gauss_reg > 0:
            gauss_loss = self._gaussian_regularization_loss(z)
            loss = loss + config.gauss_reg * gauss_loss
            self.log("train/gauss_loss", gauss_loss, on_step=False, on_epoch=True)

        # Kurtosis regularization (after warmup period)
        if config.kurtosis_reg > 0:
            warmup_epochs = int(config.kurtosis_warmup * self.trainer.max_epochs)
            if self.current_epoch >= warmup_epochs:
                kurt_loss = self._kurtosis_regularization_loss(z)
                loss = loss + config.kurtosis_reg * kurt_loss
                self.log("train/kurtosis_loss", kurt_loss, on_step=False, on_epoch=True)

        # Outlier penalty: discourage extreme latent values
        if config.outlier_penalty > 0:
            outlier_loss = self._outlier_penalty_loss(z)
            loss = loss + config.outlier_penalty * outlier_loss
            self.log("train/outlier_loss", outlier_loss, on_step=False, on_epoch=True)

        self.log("train/nll", nll, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/loss", loss, on_step=False, on_epoch=True)

        # Log latent Gaussianity metrics (target: mean=0, std=1, skew=0, kurtosis=0)
        self._log_gaussianity_metrics(z, prefix="train")

        # Log component breakdown for diagnostics
        self.log("train/log_pz", -log_pz.mean(), on_step=False, on_epoch=True)
        self.log("train/log_det", -log_det.mean(), on_step=False, on_epoch=True)

        # Log ActNorm statistics (every 100 steps to reduce overhead)
        if batch_idx % 100 == 0:
            self._log_actnorm_stats()

        return loss

    def _gaussian_regularization_loss(self, z: torch.Tensor) -> torch.Tensor:
        """Compute moment matching loss to enforce N(0,1) marginals.

        Uses clipped latents for robust estimation when outliers are present.
        Penalizes deviation from mean=0 and std=1 per dimension.
        """
        clip = self.config.model.outlier_clip
        z_clipped = torch.clamp(z, -clip, clip)

        batch_mean = z_clipped.mean(dim=0)
        batch_std = z_clipped.std(dim=0, correction=0)

        mean_loss = (batch_mean ** 2).mean()
        std_loss = ((batch_std - 1) ** 2).mean()

        return mean_loss + std_loss

    def _kurtosis_regularization_loss(self, z: torch.Tensor) -> torch.Tensor:
        """Penalize excess kurtosis to encourage Gaussian shape.

        Uses clipped latents for robust estimation.
        For N(0,1), excess kurtosis = 0 (kurtosis = 3).
        """
        clip = self.config.model.outlier_clip
        z_clipped = torch.clamp(z, -clip, clip)

        z_centered = z_clipped - z_clipped.mean(dim=0, keepdim=True)
        z_std = z_clipped.std(dim=0, keepdim=True, correction=0).clamp(min=1e-6)
        z_norm = z_centered / z_std

        # Excess kurtosis: E[(x-μ)^4]/σ^4 - 3
        excess_kurtosis = (z_norm ** 4).mean(dim=0) - 3.0

        return (excess_kurtosis ** 2).mean()

    def _outlier_penalty_loss(self, z: torch.Tensor) -> torch.Tensor:
        """Penalize latent values beyond clip threshold."""
        clip = self.config.model.outlier_clip
        return torch.relu(z.abs() - clip).mean()

    def _log_actnorm_stats(self) -> None:
        """Log ActNorm log_scale statistics for debugging."""
        from ciffy.nn.flow.residue.model import ActNorm

        total_log_scale_sum = 0.0
        for i, layer in enumerate(self._residue_model.flow.layers):
            if isinstance(layer, ActNorm):
                log_scale = layer.log_scale.detach()
                total_log_scale_sum += log_scale.sum().item()
                self.log(f"actnorm/{i}/log_scale_mean", log_scale.mean())
                self.log(f"actnorm/{i}/log_scale_std", log_scale.std())
                self.log(f"actnorm/{i}/log_scale_sum", log_scale.sum())

        self.log("actnorm/total_log_det", total_log_scale_sum)

    def _log_gaussianity_metrics(self, z: torch.Tensor, prefix: str) -> None:
        """Log comprehensive Gaussianity metrics for latent space.

        For a standard Gaussian N(0,1):
        - mean should be 0
        - std should be 1
        - skewness should be 0
        - excess kurtosis should be 0 (kurtosis = 3)
        """
        # Basic statistics
        z_mean = z.mean()
        z_std = z.std()

        # Per-dimension statistics (detect collapsed dimensions)
        z_dim_std = z.std(dim=0)
        z_std_min = z_dim_std.min()
        z_std_max = z_dim_std.max()

        # Skewness: E[(x - μ)³] / σ³
        z_centered = z - z.mean(dim=0, keepdim=True)
        z_normalized = z_centered / (z.std(dim=0, keepdim=True) + 1e-8)
        skewness = (z_normalized ** 3).mean()

        # Excess kurtosis: E[(x - μ)⁴] / σ⁴ - 3
        kurtosis = (z_normalized ** 4).mean() - 3.0

        # Log metrics
        self.log(f"{prefix}/z_mean", z_mean, on_step=False, on_epoch=True)
        self.log(f"{prefix}/z_std", z_std, on_step=False, on_epoch=True)
        self.log(f"{prefix}/z_std_min", z_std_min, on_step=False, on_epoch=True)
        self.log(f"{prefix}/z_std_max", z_std_max, on_step=False, on_epoch=True)
        self.log(f"{prefix}/z_skewness", skewness, on_step=False, on_epoch=True)
        self.log(f"{prefix}/z_kurtosis", kurtosis, on_step=False, on_epoch=True)

    def _log_geometry_metrics(self, coords: torch.Tensor, transforms: torch.Tensor, prefix: str) -> None:
        """Log geometry violation metrics.

        Tracks bond length and angle errors vs ideal values.
        """
        if self._geometry_constraints is None:
            return

        gc = self._geometry_constraints.to(coords.device)
        metrics = gc.compute_error_metrics(coords, transforms)

        for name, value in metrics.items():
            self.log(f"{prefix}/{name}", value, on_step=False, on_epoch=True)

    def validation_step(self, batch: tuple[torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Compute validation NLL."""
        data = batch[0] if isinstance(batch, (tuple, list)) else batch

        z, log_det = self._residue_model.flow(data)
        log_pz = -0.5 * (z**2 + LOG_2PI).sum(dim=-1)
        loss = -(log_pz + log_det).mean()

        self.log("val/nll", loss, prog_bar=True, sync_dist=True)

        # Log validation Gaussianity metrics
        self._log_gaussianity_metrics(z, prefix="val")

        # Log geometry metrics on decoded samples
        with torch.no_grad():
            # Sample from prior and decode
            z_sample = torch.randn(100, self._residue_model.latent_dim, device=data.device)
            coords, transforms = self._residue_model.decode(z_sample)
            self._log_geometry_metrics(coords, transforms, prefix="val")

        return loss


__all__ = [
    "ResidueFlowModelConfig",
    "ResidueFlowDataConfig",
    "ResidueFlowFullConfig",
    "ResidueFlowModule",
]
