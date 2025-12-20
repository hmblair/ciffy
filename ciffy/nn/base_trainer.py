"""
Base training framework for polymer models.

Provides abstract base classes and configuration dataclasses that can be
extended for specific model types (VAE, diffusion, etc.).

Example:
    >>> from ciffy.nn.base_trainer import BaseConfig, BaseTrainer
    >>>
    >>> @dataclass
    >>> class MyConfig(BaseConfig):
    ...     model: MyModelConfig = field(default_factory=MyModelConfig)
    >>>
    >>> class MyTrainer(BaseTrainer):
    ...     def create_optimizer(self):
    ...         return torch.optim.Adam(self.model.parameters())
    ...
    ...     def create_dataloader(self):
    ...         return DataLoader(self.dataset, batch_size=1)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Protocol, runtime_checkable

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    yaml = None

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    optim = None
    DataLoader = None

if TYPE_CHECKING:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader

from .training import get_device, load_checkpoint, save_checkpoint, set_seed, train_epoch

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Dataclasses
# =============================================================================


@dataclass
class TrainingConfig:
    """Training hyperparameters.

    Attributes:
        epochs: Number of training epochs.
        lr: Learning rate.
        weight_decay: L2 regularization weight.
        grad_clip: Maximum gradient norm for clipping. None to disable.
        device: Device string ('auto', 'cuda', 'cpu', 'mps', or 'cuda:N').
        seed: Random seed for reproducibility. None for no seeding.
        num_workers: Number of DataLoader workers.
    """

    epochs: int = 100
    lr: float = 1e-4
    weight_decay: float = 0.0
    grad_clip: float | None = None
    device: str = "auto"
    seed: int | None = None
    num_workers: int = 0


@dataclass
class OutputConfig:
    """Output and checkpoint configuration.

    Attributes:
        checkpoint_dir: Directory for saving checkpoints.
        sample_dir: Directory for saving generated samples.
        save_every: Save checkpoint every N epochs.
        n_perturbations: Number of latent perturbations for sample generation.
        perturbation_scale: Scale of latent perturbations.
    """

    checkpoint_dir: str = "./checkpoints"
    sample_dir: str = "./samples"
    save_every: int = 10
    n_perturbations: int = 5
    perturbation_scale: float = 1.0


@dataclass
class WandbConfig:
    """Weights & Biases logging configuration.

    Attributes:
        project: Wandb project name. None to disable wandb.
        group: Experiment group name for organizing runs.
        name: Run name. None for auto-generated name.
        enabled: Whether wandb logging is enabled.
    """

    project: str | None = None
    group: str | None = None
    name: str | None = None
    enabled: bool = True


@dataclass
class BaseConfig:
    """Base configuration with nested sections.

    Subclass this to add model-specific and data-specific configuration.

    Example:
        >>> @dataclass
        >>> class VAEConfig(BaseConfig):
        ...     model: VAEModelConfig = field(default_factory=VAEModelConfig)
        ...     data: DataConfig = field(default_factory=DataConfig)
    """

    training: TrainingConfig = field(default_factory=TrainingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BaseConfig":
        """Load configuration from a YAML file.

        Args:
            path: Path to YAML configuration file.

        Returns:
            Configuration instance with values from YAML.

        Raises:
            ImportError: If PyYAML is not installed.
            FileNotFoundError: If config file does not exist.
        """
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML is required for config loading. Install with: pip install pyyaml")

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        return cls._from_dict(raw)

    @classmethod
    def _from_dict(cls, data: dict) -> "BaseConfig":
        """Create config from dictionary, handling nested dataclasses."""
        kwargs = {}

        for f in fields(cls):
            if f.name in data:
                value = data[f.name]
                # If the field type is a dataclass, recursively construct it
                if hasattr(f.type, "__dataclass_fields__"):
                    kwargs[f.name] = cls._dict_to_dataclass(f.type, value)
                else:
                    kwargs[f.name] = value

        return cls(**kwargs)

    @staticmethod
    def _dict_to_dataclass(dc_class: type, data: dict) -> Any:
        """Convert a dictionary to a dataclass instance."""
        if data is None:
            return dc_class()

        valid_fields = {f.name for f in fields(dc_class)}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return dc_class(**filtered)

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return asdict(self)


# =============================================================================
# Logger Protocol
# =============================================================================


@runtime_checkable
class MetricsLogger(Protocol):
    """Protocol for metrics logging (wandb, tensorboard, etc.)."""

    def log(self, metrics: dict[str, float], step: int) -> None:
        """Log metrics for a given step.

        Args:
            metrics: Dictionary of metric names to values.
            step: Training step or epoch number.
        """
        ...

    def finish(self) -> None:
        """Finalize logging (e.g., close wandb run)."""
        ...


# =============================================================================
# Base Trainer
# =============================================================================


class BaseTrainer(ABC):
    """Abstract base trainer for any model type.

    Subclasses must implement:
        - create_optimizer(): Return optimizer for training
        - create_dataloader(): Return DataLoader for training data

    Optional hooks:
        - on_epoch_start(epoch): Called before each epoch
        - on_epoch_end(epoch, metrics): Called after each epoch
        - create_loss_fn(): Return loss function (default uses model.compute_loss)

    Example:
        >>> class VAETrainer(BaseTrainer):
        ...     def create_optimizer(self):
        ...         return torch.optim.AdamW(self.model.parameters(), lr=self.config.training.lr)
        ...
        ...     def create_dataloader(self):
        ...         return DataLoader(self.dataset, batch_size=1, collate_fn=polymer_collate_fn)
        ...
        ...     def on_epoch_start(self, epoch):
        ...         self.model.beta = self.beta_scheduler.get_beta(epoch)
    """

    def __init__(
        self,
        config: BaseConfig,
        model: "nn.Module",
        dataset: Any,
        device: "torch.device | None" = None,
        logger: MetricsLogger | None = None,
        quiet: bool = False,
    ):
        """Initialize the trainer.

        Args:
            config: Training configuration.
            model: PyTorch model to train.
            dataset: Training dataset.
            device: Device to train on. If None, uses config.training.device.
            logger: Optional metrics logger (e.g., WandbLogger).
            quiet: If True, suppress progress bars and reduce logging.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for BaseTrainer")

        self.config = config
        self.model = model
        self.dataset = dataset
        self.quiet = quiet
        self.metrics_logger = logger

        # Setup device
        if device is None:
            self.device = get_device(config.training.device)
        else:
            self.device = device

        # Move model to device
        self.model = self.model.to(self.device)

        # Set random seed
        if config.training.seed is not None:
            set_seed(config.training.seed)

        # Create optimizer and dataloader (subclass implementations)
        self.optimizer = self.create_optimizer()
        self.dataloader = self.create_dataloader()

        # Create loss function
        self.loss_fn = self.create_loss_fn()

        # Setup output directories
        self.checkpoint_dir = Path(config.output.checkpoint_dir)
        self.sample_dir = Path(config.output.sample_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.sample_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.current_epoch = 0
        self.best_loss = float("inf")
        self.best_checkpoint_path = self.checkpoint_dir / "checkpoint_best.pt"

    @abstractmethod
    def create_optimizer(self) -> "optim.Optimizer":
        """Create and return the optimizer.

        Returns:
            PyTorch optimizer for training.
        """
        ...

    @abstractmethod
    def create_dataloader(self) -> "DataLoader":
        """Create and return the training DataLoader.

        Returns:
            DataLoader for iterating over training data.
        """
        ...

    def create_loss_fn(self) -> Callable[["nn.Module", Any], dict[str, "torch.Tensor"]]:
        """Create and return the loss function.

        The loss function receives (model, sample) and returns a dict
        with at least a 'loss' key.

        Default implementation calls model.compute_loss(sample).
        Override for custom preprocessing or validation.

        Returns:
            Loss function callable.
        """
        device = self.device

        def default_loss_fn(model: "nn.Module", sample: Any) -> dict[str, "torch.Tensor"]:
            if hasattr(sample, "to"):
                sample = sample.to(device)
            return model.compute_loss(sample)

        return default_loss_fn

    def on_epoch_start(self, epoch: int) -> None:
        """Hook called at the start of each epoch.

        Override to update hyperparameters (e.g., beta scheduling).

        Args:
            epoch: Current epoch number (0-indexed).
        """
        pass

    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> None:
        """Hook called at the end of each epoch.

        Override to generate samples, log additional metrics, etc.

        Args:
            epoch: Current epoch number (0-indexed).
            metrics: Metrics from the epoch (loss, recon_loss, etc.).
        """
        pass

    def train(
        self,
        resume_path: str | Path | None = None,
        progress_callback: Callable[[int, int, dict], None] | None = None,
    ) -> dict[str, Any]:
        """Run the full training loop.

        Args:
            resume_path: Optional checkpoint path to resume from.
            progress_callback: Optional callback called after each epoch with
                signature: callback(epoch, total_epochs, metrics).

        Returns:
            Dictionary containing:
                - final_loss: Loss from the final epoch
                - best_loss: Best loss achieved during training
                - epochs_trained: Number of epochs completed
                - checkpoint_path: Path to best checkpoint
                - Plus any model-specific metrics
        """
        start_epoch = 0

        # Resume from checkpoint if specified
        if resume_path is not None:
            ckpt = load_checkpoint(Path(resume_path), self.model, self.optimizer)
            start_epoch = ckpt.get("epoch", 0) + 1
            self.best_loss = ckpt.get("metrics", {}).get("loss", float("inf"))
            if not self.quiet:
                logger.info(f"Resumed from epoch {start_epoch}")

        total_epochs = self.config.training.epochs
        metrics: dict[str, float] = {}
        total_samples = 0

        try:
            for epoch in range(start_epoch, total_epochs):
                self.current_epoch = epoch

                # Pre-epoch hook
                self.on_epoch_start(epoch)

                # Train one epoch
                metrics = train_epoch(
                    model=self.model,
                    dataloader=self.dataloader,
                    loss_fn=self.loss_fn,
                    optimizer=self.optimizer,
                    grad_clip=self.config.training.grad_clip,
                    progress_bar=not self.quiet,
                )
                total_samples += int(metrics.get("n_samples", 0))

                # Log metrics
                if not self.quiet:
                    self._log_epoch(epoch, total_epochs, metrics)

                # Log to external logger (wandb, etc.)
                if self.metrics_logger is not None:
                    self.metrics_logger.log(metrics, step=epoch)

                # Progress callback
                if progress_callback is not None:
                    progress_callback(epoch + 1, total_epochs, metrics)

                # Post-epoch hook
                self.on_epoch_end(epoch, metrics)

                # Save periodic checkpoint
                if (epoch + 1) % self.config.output.save_every == 0:
                    self._save_checkpoint(epoch, metrics, is_best=False)

                # Save best checkpoint
                current_loss = metrics.get("loss", float("inf"))
                if current_loss < self.best_loss:
                    self.best_loss = current_loss
                    self._save_checkpoint(epoch, metrics, is_best=True)

            # Save final checkpoint
            self._save_checkpoint(total_epochs - 1, metrics, is_best=False, is_final=True)

            if not self.quiet:
                logger.info("Training complete!")

        finally:
            # Ensure logger is closed
            if self.metrics_logger is not None:
                self.metrics_logger.finish()

        return {
            "final_loss": metrics.get("loss"),
            "best_loss": self.best_loss,
            "final_recon_loss": metrics.get("recon_loss"),
            "final_kl_loss": metrics.get("kl_loss"),
            "epochs_trained": total_epochs - start_epoch,
            "n_samples": total_samples,
            "device": str(self.device),
            "checkpoint_path": str(self.best_checkpoint_path),
            "error": None,
        }

    def _log_epoch(self, epoch: int, total_epochs: int, metrics: dict[str, float]) -> None:
        """Log epoch metrics to console."""
        parts = [f"Epoch {epoch + 1}/{total_epochs}"]

        if "loss" in metrics:
            parts.append(f"Loss: {metrics['loss']:.4f}")
        if "recon_loss" in metrics:
            parts.append(f"Recon: {metrics['recon_loss']:.4f}")
        if "kl_loss" in metrics:
            parts.append(f"KL: {metrics['kl_loss']:.4f}")
        if "n_samples" in metrics:
            parts.append(f"Samples: {int(metrics['n_samples'])}")
        if "n_skipped" in metrics and metrics["n_skipped"] > 0:
            parts.append(f"Skipped: {int(metrics['n_skipped'])}")

        logger.info(" | ".join(parts))

    def _save_checkpoint(
        self,
        epoch: int,
        metrics: dict[str, float],
        is_best: bool = False,
        is_final: bool = False,
    ) -> None:
        """Save a training checkpoint."""
        if is_best:
            path = self.best_checkpoint_path
        elif is_final:
            path = self.checkpoint_dir / "checkpoint_final.pt"
        else:
            path = self.checkpoint_dir / f"checkpoint_epoch{epoch + 1:04d}.pt"

        save_checkpoint(
            path=path,
            model=self.model,
            optimizer=self.optimizer,
            epoch=epoch + 1,
            metrics=metrics,
            config=self.config,
        )


__all__ = [
    "TrainingConfig",
    "OutputConfig",
    "WandbConfig",
    "BaseConfig",
    "MetricsLogger",
    "BaseTrainer",
]
