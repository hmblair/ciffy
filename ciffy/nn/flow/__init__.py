"""
Flow models for molecular conformations.

This module provides normalizing flow-based models for learning distributions
over molecular conformations:

- `residue`: Per-residue flow models (ResidueFlowModel)
- `polymer`: Multi-residue orchestration (PolymerFlowModel)

Example:
    >>> from ciffy.nn.flow import ResidueFlowModel, PolymerFlowModel
    >>> from ciffy.biochemistry import Residue
    >>>
    >>> # Train per-residue models
    >>> model_A = ResidueFlowModel.from_structures(paths, Residue.A)
    >>>
    >>> # Combine into polymer model
    >>> polymer = PolymerFlowModel({Residue.A: model_A})
"""

from .residue import (
    PCAFlow,
    ResidueFlowModel,
    ResidueFlowConfig,
    create_atom_subset,
    # Frame computation
    compute_glycosidic_frame,
    compute_o3p_frame,
    compute_p_frame,
    # SE(3) transforms
    compute_relative_transform,
    apply_relative_transform,
    # Data extraction
    extract_residues,
    align_to_frame,
    extract_residues_with_links,
    position_next_residue,
    compute_link_frames,
    # Training
    train_pca_flow,
)

from .polymer import PolymerFlowModel

__all__ = [
    # Residue flow
    "PCAFlow",
    "ResidueFlowModel",
    "ResidueFlowConfig",
    "create_atom_subset",
    # Frame computation
    "compute_glycosidic_frame",
    "compute_o3p_frame",
    "compute_p_frame",
    # SE(3) transforms
    "compute_relative_transform",
    "apply_relative_transform",
    # Data extraction
    "extract_residues",
    "align_to_frame",
    "extract_residues_with_links",
    "position_next_residue",
    "compute_link_frames",
    # Training
    "train_pca_flow",
    # Polymer flow
    "PolymerFlowModel",
]
