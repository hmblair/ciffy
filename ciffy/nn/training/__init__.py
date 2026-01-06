"""Training infrastructure for ciffy neural network models.

Provides utilities for training, data splitting, logging, and diagnostics.
"""

from .api import (
    available_models,
    load,
    register_model_type,
    train,
)
from .diagnostics import (
    ActivationTracker,
    GradientTracker,
    LearningRateTracker,
    ParameterTracker,
    TrainingDiagnostics,
    diagnose_gradients,
)
# Re-export DiagnosticsConfig from config for backwards compatibility
from ..config import DiagnosticsConfig
from .loggers import (
    NoOpLogger,
    WandbLogger,
    create_logger,
)
from .split import (
    DataScalingSplit,
    DataSplit,
    create_scaling_split,
    split_by_structure,
)

__all__ = [
    # Training API
    "train",
    "load",
    "available_models",
    "register_model_type",
    # Data splitting
    "DataSplit",
    "DataScalingSplit",
    "split_by_structure",
    "create_scaling_split",
    # Diagnostics
    "GradientTracker",
    "ParameterTracker",
    "ActivationTracker",
    "LearningRateTracker",
    "TrainingDiagnostics",
    "DiagnosticsConfig",
    "diagnose_gradients",
    # Loggers
    "WandbLogger",
    "NoOpLogger",
    "create_logger",
]
