"""
Flow models for molecular conformations.

This module provides normalizing flow-based models for learning distributions
over molecular conformations:

- `ResidueFlowModel`: Per-residue flow model (PCA + normalizing flow)
- `PolymerFlowModel`: Multi-residue orchestration with SE(3) positioning
- `ResidueFlowTrainer`: Training infrastructure for multiple residue types
- `load_pretrained`: Load pre-trained models

Quick Start:
    >>> import ciffy
    >>> from ciffy.nn.flow import PolymerFlowModel
    >>>
    >>> # Load pre-trained model (if available)
    >>> model = ciffy.load_flow_model("rna", device="cuda")
    >>>
    >>> # Encode polymer coordinates
    >>> polymer = ciffy.load("structure.cif").poly()
    >>> latents = model.encode_polymer(polymer)
    >>>
    >>> # Sample new conformations
    >>> samples = model.sample(polymer.sequence, n_samples=10)
    >>>
    >>> # Interpolate between conformations
    >>> path = model.interpolate(polymer1, polymer2, n_steps=20)

Training Custom Models:
    >>> from ciffy.nn.flow import ResidueFlowTrainer, ResidueFlowTrainingConfig
    >>> from ciffy.biochemistry import Residue
    >>>
    >>> config = ResidueFlowTrainingConfig(latent_dim=12, n_epochs=200, device="cuda")
    >>> trainer = ResidueFlowTrainer(config)
    >>>
    >>> # Train on your data
    >>> results = trainer.train_all(cif_paths, [Residue.A, Residue.C, Residue.G, Residue.U])
    >>>
    >>> # Save and convert to PolymerFlowModel
    >>> trainer.save(results, "models/my_rna")
    >>> model = trainer.to_polymer_model(results)
"""

from .residue import (
    PCAFlow,
    ResidueFlowModel,
    ResidueFlowConfig,
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
    ResidueFlowTrainer,
    ResidueFlowTrainingConfig,
    TrainingResult,
)

from .polymer import PolymerFlowModel
from .pretrained import load_pretrained, list_pretrained, is_pretrained_available

__all__ = [
    # Residue flow
    "PCAFlow",
    "ResidueFlowModel",
    "ResidueFlowConfig",
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
    "ResidueFlowTrainer",
    "ResidueFlowTrainingConfig",
    "TrainingResult",
    # Polymer flow
    "PolymerFlowModel",
    # Pre-trained models
    "load_pretrained",
    "list_pretrained",
    "is_pretrained_available",
]
