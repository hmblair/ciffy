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
    >>> # Encode/decode
    >>> z = model.encode(coords)
    >>> coords_recon = model.decode(z)
    >>>
    >>> # Sample new conformations
    >>> samples = model.sample(n_samples=100)
"""

from .model import PCAFlow, ResidueFlowModel, ResidueFlowConfig, create_atom_subset
from .data import extract_residues, align_to_frame
from .train import train_pca_flow

__all__ = [
    "PCAFlow",
    "ResidueFlowModel",
    "ResidueFlowConfig",
    "create_atom_subset",
    "extract_residues",
    "align_to_frame",
    "train_pca_flow",
]
