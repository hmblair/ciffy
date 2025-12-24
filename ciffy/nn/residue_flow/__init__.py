"""
Residue Flow: PCA + Normalizing Flow for residue conformations.

This module provides tools for learning low-dimensional representations
of RNA residue conformations that enable valid sampling.

Example:
    >>> from ciffy.nn.residue_flow import ResidueFlowModel
    >>> from ciffy.biochemistry import Residue
    >>>
    >>> # Train a model for adenosine
    >>> model = ResidueFlowModel.from_structures(cif_paths, Residue.A)
    >>>
    >>> # Access atom information via IndexEnum interface
    >>> model.atoms.list()   # ['C1p', 'C2p', 'O2p', ...]
    >>> model.atoms.index()  # array([0, 1, 2, ...])
    >>> len(model.atoms)     # 22
    >>>
    >>> # Decode gives coords + link transform to next residue
    >>> coords, transform = model.decode(z)
    >>>
    >>> # Position next residue using the transform
    >>> from ciffy.nn.residue_flow import position_next_residue
    >>> coords2 = position_next_residue(coords, ref_coords, transform, atoms, residue)
    >>>
    >>> # Sample new conformations
    >>> coords, transforms = model.sample(n_samples=100)
"""

from .model import (
    PCAFlow,
    ResidueFlowModel,
    ResidueFlowConfig,
    create_atom_subset,
)
from .data import (
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
)
from .train import train_pca_flow

__all__ = [
    # Models
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
]
