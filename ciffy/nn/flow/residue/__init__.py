"""
Residue Flow: PCA + Normalizing Flow for residue conformations.

This module provides tools for learning low-dimensional representations
of RNA residue conformations that enable valid sampling.

Example:
    >>> from ciffy.nn.flow.residue import ResidueFlowModel
    >>> from ciffy.biochemistry import Residue
    >>> from ciffy.nn.lightning import ResidueFlowModule, FlowDataModule
    >>> import lightning as L
    >>>
    >>> # Train using Lightning
    >>> dm = FlowDataModule(cif_paths, Residue.A)
    >>> module = ResidueFlowModule(config, Residue.A)
    >>> trainer = L.Trainer(max_epochs=200)
    >>> trainer.fit(module, dm)
    >>> model = module.get_model()
    >>>
    >>> # Decode gives coords + link transform to next residue
    >>> coords, transform = model.decode(z)
"""

from .model import (
    PCAFlow,
    ResidueFlowModel,
    ResidueFlowConfig,
)
from .data import (
    extract_residues,
    extract_residues_with_links,
    ResidueDataset,
)

__all__ = [
    # Models
    "PCAFlow",
    "ResidueFlowModel",
    "ResidueFlowConfig",
    # Data extraction
    "extract_residues",
    "extract_residues_with_links",
    "ResidueDataset",
]
