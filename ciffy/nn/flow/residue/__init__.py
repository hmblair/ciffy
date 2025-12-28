"""
Residue Flow: PCA + Normalizing Flow for residue conformations.

This module provides tools for learning low-dimensional representations
of RNA residue conformations that enable valid sampling.

Example:
    >>> from ciffy.nn.flow.residue import ResidueFlowModel, FrameIndices
    >>> from ciffy.biochemistry import Residue
    >>>
    >>> # Train a model for adenosine
    >>> model = ResidueFlowModel.from_structures(cif_paths, Residue.A)
    >>>
    >>> # Decode gives coords + link transform to next residue
    >>> coords, transform = model.decode(z)
    >>>
    >>> # Position next residue using precomputed frame indices
    >>> from ciffy.nn.flow.residue import position_next_residue
    >>> indices = FrameIndices.from_atoms(model.atoms.index(), Residue.A)
    >>> coords2 = position_next_residue(coords1, coords2, transform, indices)
"""

from .model import (
    PCAFlow,
    ResidueFlowModel,
    ResidueFlowConfig,
)
from .data import (
    # Frame indices (precomputed for fast frame computation)
    FrameIndices,
    # Frame computation
    compute_glycosidic_frame,
    # Data extraction
    extract_residues,
    align_to_frame,
    extract_residues_with_links,
    position_next_residue,
    ResidueDataset,
)
from .train import train_pca_flow
from .trainer import ResidueFlowTrainer, ResidueFlowTrainingConfig, TrainingResult

# Re-export from geometry for convenience
from ciffy.geometry import (
    compute_frame_from_indices,
    compute_relative_transform,
    apply_relative_transform,
)

__all__ = [
    # Models
    "PCAFlow",
    "ResidueFlowModel",
    "ResidueFlowConfig",
    # Frame indices
    "FrameIndices",
    "compute_frame_from_indices",
    # Frame computation
    "compute_glycosidic_frame",
    # SE(3) transforms
    "compute_relative_transform",
    "apply_relative_transform",
    # Data extraction
    "extract_residues",
    "align_to_frame",
    "extract_residues_with_links",
    "position_next_residue",
    "ResidueDataset",
    # Training
    "train_pca_flow",
    "ResidueFlowTrainer",
    "ResidueFlowTrainingConfig",
    "TrainingResult",
]
