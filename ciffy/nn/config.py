"""
Training configuration framework for polymer models.

Provides configuration dataclasses for training neural network models.
Trainers should use PyTorch Lightning Fabric for device handling.

Example:
    >>> from ciffy.nn.config import BaseConfig, TrainingConfig
    >>>
    >>> @dataclass
    >>> class MyConfig(BaseConfig):
    ...     model: MyModelConfig = field(default_factory=MyModelConfig)
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    yaml = None

from .diagnostics import DiagnosticsConfig

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
    num_devices: int = 1  # Number of GPUs for DDP training
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
        >>> class MyConfig(BaseConfig):
        ...     model: MyModelConfig = field(default_factory=MyModelConfig)
        ...     data: DataConfig = field(default_factory=DataConfig)
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
# Device Utilities
# =============================================================================


def get_device(device: str = "auto") -> str:
    """Resolve device string to actual device.

    Args:
        device: Device specification. 'auto' detects cuda/mps/cpu.

    Returns:
        Resolved device string ('cuda', 'mps', or 'cpu').
    """
    try:
        import torch
    except ImportError:
        return "cpu"

    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device


__all__ = [
    "SchedulerConfig",
    "ValidationConfig",
    "TrainingConfig",
    "OutputConfig",
    "DataConfig",
    "WandbConfig",
    "BaseConfig",
    "MetricsLogger",
    "get_device",
]
