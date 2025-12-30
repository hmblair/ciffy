"""PyTorch Lightning training infrastructure for ciffy models.

This module provides LightningModules, DataModules, and Callbacks for training
ciffy's generative models with PyTorch Lightning.

Example:
    >>> from ciffy.nn.lightning import LatentDiffusionModule, LatentDiffusionDataModule
    >>> from lightning import Trainer
    >>>
    >>> module = LatentDiffusionModule(config)
    >>> datamodule = LatentDiffusionDataModule(data_dir, flow_model)
    >>>
    >>> trainer = Trainer(max_epochs=100)
    >>> trainer.fit(module, datamodule)
"""

from .modules import (
    BaseCiffyModule,
    LatentDiffusionDataConfig,
    LatentDiffusionFullConfig,
    LatentDiffusionModule,
    CoordinateDiffusionDataConfig,
    CoordinateDiffusionFullConfig,
    CoordinateDiffusionModule,
    ResidueFlowDataConfig,
    ResidueFlowFullConfig,
    ResidueFlowModelConfig,
    ResidueFlowModule,
)
from .data import (
    LatentDiffusionDataModule,
    CoordinateDiffusionDataModule,
    FlowDataModule,
)
from .callbacks import (
    EMACallback,
    SampleGenerationCallback,
)

__all__ = [
    # Modules
    "BaseCiffyModule",
    "LatentDiffusionDataConfig",
    "LatentDiffusionFullConfig",
    "LatentDiffusionModule",
    "CoordinateDiffusionDataConfig",
    "CoordinateDiffusionFullConfig",
    "CoordinateDiffusionModule",
    "ResidueFlowDataConfig",
    "ResidueFlowFullConfig",
    "ResidueFlowModelConfig",
    "ResidueFlowModule",
    # Data modules
    "LatentDiffusionDataModule",
    "CoordinateDiffusionDataModule",
    "FlowDataModule",
    # Callbacks
    "EMACallback",
    "SampleGenerationCallback",
]
