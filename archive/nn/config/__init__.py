"""Configuration framework for ciffy neural network models.

Provides configuration dataclasses for training, validation, and inference.
"""

from .base import (
    BaseConfig,
    DataConfig,
    DiagnosticsConfig,
    MetricsLogger,
    OutputConfig,
    SchedulerConfig,
    TrainingConfig,
    ValidationConfig,
    WandbConfig,
    get_device,
)
from .inference import InferenceConfig

__all__ = [
    # Base configs
    "BaseConfig",
    "DataConfig",
    "DiagnosticsConfig",
    "MetricsLogger",
    "OutputConfig",
    "SchedulerConfig",
    "TrainingConfig",
    "ValidationConfig",
    "WandbConfig",
    "get_device",
    # Inference
    "InferenceConfig",
]
