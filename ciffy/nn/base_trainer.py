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
from .diagnostics import DiagnosticsConfig, TrainingDiagnostics
from .schedulers import create_scheduler, get_current_lr
from .early_stopping import EarlyStopper

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Dataclasses
# =============================================================================


@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration.

    Attributes:
        scheduler_type: Type of scheduler ('cosine', 'linear', 'step', 'none').
        warmup_epochs: Number of warmup epochs (linear ramp from 0 to lr).
        min_lr: Minimum learning rate for cosine/linear decay.
        step_size: Epochs between LR drops (for step scheduler).
        gamma: Multiplicative factor for step scheduler.
    """

    scheduler_type: str = "none"
    warmup_epochs: int = 0
    min_lr: float = 1e-6
    step_size: int = 30
    gamma: float = 0.1


@dataclass
class ValidationConfig:
    """Validation and early stopping configuration.

    Attributes:
        val_every: Validate every N epochs (0 to disable).
        val_fraction: Fraction of training data for validation (if no val dataset).
        early_stopping: Enable early stopping based on validation loss.
        patience: Epochs to wait for improvement before stopping.
        min_delta: Minimum improvement to reset patience counter.
    """

    val_every: int = 0
    val_fraction: float = 0.1
    early_stopping: bool = False
    patience: int = 10
    min_delta: float = 1e-4


@dataclass
class TrainingConfig:
    """Training hyperparameters.

    Attributes:
        epochs: Number of training epochs.
        lr: Learning rate.
        weight_decay: L2 regularization weight.
        grad_clip: Maximum gradient norm for clipping. None to disable.
        device: Device string ('auto', 'cuda', 'cpu', 'mps', or 'cuda:N').
        precision: Training precision ('32-true', '16-mixed', 'bf16-mixed').
        seed: Random seed for reproducibility. None for no seeding.
        num_workers: Number of DataLoader workers.
        scheduler: Learning rate scheduler configuration.
        validation: Validation and early stopping configuration.
    """

    epochs: int = 100
    lr: float = 1e-4
    weight_decay: float = 0.0
    grad_clip: float | None = None
    device: str = "auto"
    precision: str = "32-true"
    seed: int | None = None
    num_workers: int = 0
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)


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
class DataConfig:
    """Base data configuration shared across trainers.

    Attributes:
        data_dir: Directory containing training data (CIF files, etc.).
        batch_size: Training batch size.
        num_workers: Number of DataLoader workers.
    """

    data_dir: str = ""
    batch_size: int = 32
    num_workers: int = 4


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

    Attributes:
        training: Training hyperparameters (epochs, lr, etc.).
        output: Checkpoint and output directory settings.
        wandb: Weights & Biases logging configuration.
        diagnostics: Training diagnostics configuration. If None (default),
            diagnostics are disabled. When enabled, gradient norms, parameter
            statistics, and learning rates are tracked and logged automatically.

    Example:
        >>> @dataclass
        >>> class VAEConfig(BaseConfig):
        ...     model: VAEModelConfig = field(default_factory=VAEModelConfig)
        ...     data: DataConfig = field(default_factory=DataConfig)

    Example with diagnostics:
        >>> config = VAEConfig.from_yaml("config.yaml")
        >>> config.diagnostics = DiagnosticsConfig(
        ...     track_gradients=True,
        ...     track_parameters=True,
        ... )
    """

    training: TrainingConfig = field(default_factory=TrainingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    diagnostics: DiagnosticsConfig | None = None

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
        from typing import get_type_hints

        kwargs = {}

        # Resolve type hints (handles forward references from __future__ annotations)
        try:
            type_hints = get_type_hints(cls)
        except Exception:
            type_hints = {}

        for f in fields(cls):
            if f.name in data:
                value = data[f.name]
                # Get resolved type from type_hints, fallback to f.type
                field_type = type_hints.get(f.name, f.type)
                # If the field type is a dataclass, recursively construct it
                if hasattr(field_type, "__dataclass_fields__"):
                    kwargs[f.name] = cls._dict_to_dataclass(field_type, value)
                else:
                    kwargs[f.name] = value

        return cls(**kwargs)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        **overrides: Any,
    ) -> "BaseConfig":
        """Create config from dictionary with optional overrides.

        This is the standard interface for the trainer registry.

        Args:
            data: Dictionary from YAML config file.
            **overrides: Override specific fields (e.g., device='cuda').
                Overrides are applied to the 'training' sub-config for
                common fields like 'device'.

        Returns:
            Configuration instance.
        """
        # Apply device override to training config
        if "device" in overrides:
            training = data.get("training", {})
            training = {**training, "device": overrides.pop("device")}
            data = {**data, "training": training}

        config = cls._from_dict(data)

        # Apply remaining overrides directly if they're top-level fields
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)

        return config

    @classmethod
    def _dict_to_dataclass(cls, dc_class: type, data: dict) -> Any:
        """Convert a dictionary to a dataclass instance, recursively handling nested dataclasses."""
        from typing import get_type_hints

        if data is None:
            return dc_class()

        # Resolve type hints for this dataclass
        try:
            type_hints = get_type_hints(dc_class)
        except Exception:
            type_hints = {}

        kwargs = {}
        for f in fields(dc_class):
            if f.name in data:
                value = data[f.name]
                field_type = type_hints.get(f.name, f.type)
                # Recursively convert nested dataclasses
                if hasattr(field_type, "__dataclass_fields__") and isinstance(value, dict):
                    kwargs[f.name] = cls._dict_to_dataclass(field_type, value)
                else:
                    kwargs[f.name] = value

        return dc_class(**kwargs)

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

    .. deprecated::
        BaseTrainer is deprecated. New trainers should use PyTorch Lightning
        Fabric directly for device handling and mixed precision support.
        See ResidueFlowTrainer and LatentDiffusionTrainer for examples.

    Subclasses must implement:
        - create_optimizer(): Return optimizer for training
        - create_dataloader(): Return DataLoader for training data

    Optional hooks:
        - on_epoch_start(epoch): Called before each epoch
        - on_epoch_end(epoch, metrics): Called after each epoch
        - create_loss_fn(): Return loss function (default uses model.compute_loss)

    Diagnostics:
        When config.diagnostics is set to a DiagnosticsConfig, the trainer
        automatically tracks gradient norms, parameter statistics, and learning
        rates. These metrics are included in the metrics dict passed to the
        logger (wandb, etc.) each epoch.

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

    Example with diagnostics:
        >>> config.diagnostics = DiagnosticsConfig(track_gradients=True)
        >>> trainer = VAETrainer(config, model, dataset, logger=wandb_logger)
        >>> trainer.train()  # Gradient norms logged to wandb automatically
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

        # Initialize diagnostics if configured
        if config.diagnostics is not None:
            self.diagnostics = TrainingDiagnostics(self.model, config.diagnostics)
        else:
            self.diagnostics = None

        # Create LR scheduler
        self.scheduler = create_scheduler(
            self.optimizer,
            config.training.scheduler,
            config.training.epochs,
        )

        # Create validation dataloader (if configured)
        self.val_dataloader = self.create_val_dataloader()

        # Create early stopper (if configured)
        val_config = config.training.validation
        if val_config.early_stopping and val_config.val_every > 0:
            self.early_stopper = EarlyStopper(
                patience=val_config.patience,
                min_delta=val_config.min_delta,
                mode="min",
            )
        else:
            self.early_stopper = None

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

    def create_val_dataloader(self) -> "DataLoader | None":
        """Create and return the validation DataLoader.

        Default: Returns None (no validation). Override to provide validation data.
        If validation.val_fraction > 0, subclasses can split their training dataset.

        Returns:
            DataLoader for validation, or None to disable validation.
        """
        return None

    def validate(self) -> dict[str, float]:
        """Run validation loop.

        Returns:
            Dictionary of validation metrics (prefixed with 'val_').
        """
        if self.val_dataloader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        n_samples = 0

        try:
            with torch.no_grad():
                for sample in self.val_dataloader:
                    losses = self.loss_fn(self.model, sample)
                    loss = losses.get("loss")
                    if loss is not None:
                        total_loss += loss.item()
                        n_samples += 1
        finally:
            self.model.train()

        if n_samples == 0:
            return {}

        return {"val_loss": total_loss / n_samples}

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
                    diagnostics=self.diagnostics,
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

                # Step LR scheduler and log learning rate
                if self.scheduler is not None:
                    self.scheduler.step()
                    current_lr = get_current_lr(self.scheduler)
                    if current_lr is not None:
                        metrics["learning_rate"] = current_lr

                # Run validation
                val_config = self.config.training.validation
                if val_config.val_every > 0 and (epoch + 1) % val_config.val_every == 0:
                    val_metrics = self.validate()
                    metrics.update(val_metrics)

                    # Log validation metrics
                    if self.metrics_logger is not None and val_metrics:
                        self.metrics_logger.log(val_metrics, step=epoch)

                    # Early stopping check
                    if self.early_stopper is not None and "val_loss" in val_metrics:
                        if self.early_stopper.should_stop(val_metrics["val_loss"], epoch):
                            if not self.quiet:
                                logger.info(
                                    f"Early stopping at epoch {epoch + 1}. "
                                    f"Best val_loss: {self.early_stopper.best_value:.4f}"
                                )
                            break

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
            # Clean up diagnostics (remove activation hooks)
            if self.diagnostics is not None:
                self.diagnostics.cleanup()

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
        if "val_loss" in metrics:
            parts.append(f"Val: {metrics['val_loss']:.4f}")
        if "recon_loss" in metrics:
            parts.append(f"Recon: {metrics['recon_loss']:.4f}")
        if "kl_loss" in metrics:
            parts.append(f"KL: {metrics['kl_loss']:.4f}")
        if "learning_rate" in metrics:
            parts.append(f"LR: {metrics['learning_rate']:.2e}")
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
    "SchedulerConfig",
    "ValidationConfig",
    "TrainingConfig",
    "OutputConfig",
    "DataConfig",
    "WandbConfig",
    "BaseConfig",
    "MetricsLogger",
    "BaseTrainer",
]
