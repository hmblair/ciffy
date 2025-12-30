"""LightningModule for residue flow model training.

Special handling for PCA preprocessing which requires all training data upfront.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from lightning import LightningModule
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from ciffy.nn.config import TrainingConfig

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
    training: TrainingConfig = field(default_factory=TrainingConfig)


class ResidueFlowModule(LightningModule):
    """LightningModule for training residue flow models.

    Unlike diffusion modules, this handles:
    - PCA computation during setup (requires full dataset)
    - Normalizing flow training with NLL loss
    - Creates a full ResidueFlowModel with metadata for save/load

    The model is created in setup() after PCA is computed from the
    DataModule's training data.

    Example:
        >>> from ciffy.biochemistry import Residue
        >>> from ciffy.nn.lightning import ResidueFlowModule, FlowDataModule
        >>> import lightning as L
        >>>
        >>> config = ResidueFlowFullConfig()
        >>> dm = FlowDataModule(cif_paths, residue=Residue.A)
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
        self._residue_model = ResidueFlowModel(
            flow=flow,
            residue=self.residue,
            atom_indices=atoms.tolist() if isinstance(atoms, np.ndarray) else list(atoms),
            n_atoms=len(atoms),
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

        # Add noise regularization during training
        if self.config.model.noise_std > 0:
            data = data + self.config.model.noise_std * torch.randn_like(data)

        # Forward pass through the flow
        z, log_det = self._residue_model.flow(data)

        # Compute NLL: -log p(z) - log |det J|
        log_pz = -0.5 * (z**2 + LOG_2PI).sum(dim=-1)
        loss = -(log_pz + log_det).mean()

        self.log("train/nll", loss, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch: tuple[torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Compute validation NLL."""
        data = batch[0] if isinstance(batch, (tuple, list)) else batch

        z, log_det = self._residue_model.flow(data)
        log_pz = -0.5 * (z**2 + LOG_2PI).sum(dim=-1)
        loss = -(log_pz + log_det).mean()

        self.log("val/nll", loss, prog_bar=True, sync_dist=True)

        return loss

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizer and scheduler."""
        config = self.training_config

        optimizer = Adam(self._residue_model.parameters(), lr=config.lr)

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=config.scheduler.min_lr if hasattr(config, "scheduler") else 1e-6,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        """Apply gradient clipping if configured."""
        if self.training_config.grad_clip:
            torch.nn.utils.clip_grad_norm_(
                self._residue_model.parameters(),
                self.training_config.grad_clip,
            )


__all__ = [
    "ResidueFlowModelConfig",
    "ResidueFlowDataConfig",
    "ResidueFlowFullConfig",
    "ResidueFlowModule",
]
