"""
Flow models for molecular conformations.

This module provides normalizing flow-based models for learning distributions
over molecular conformations:

- `ResidueFlowModel`: Per-residue flow model (PCA + normalizing flow)
- `PolymerFlowModel`: Multi-residue orchestration with SE(3) positioning
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
    >>> from ciffy.nn.lightning import ResidueFlowModule, FlowDataModule
    >>> from ciffy.biochemistry import Residue
    >>> import lightning as L
    >>>
    >>> # Train a model for each residue type
    >>> dm = FlowDataModule(cif_paths, Residue.A)
    >>> module = ResidueFlowModule(config, Residue.A)
    >>> trainer = L.Trainer(max_epochs=200)
    >>> trainer.fit(module, dm)
    >>> model = module.get_model()
    >>> model.save("models/A")
"""

from .residue import (
    PCAFlow,
    ResidueFlowModel,
    ResidueFlowConfig,
    # Frame indices and computation
    FrameIndices,
    compute_frame_from_indices,
    compute_glycosidic_frame,
    # SE(3) transforms
    compute_relative_transform,
    apply_relative_transform,
    # Data extraction
    extract_residues,
    align_to_frame,
    extract_residues_with_links,
    position_next_residue,
    ResidueDataset,
)

from .metrics import (
    LatentMoments,
    FlowMetrics,
    compute_nll,
    compute_latent_moments,
    compute_flow_metrics,
    estimate_kl_divergence,
)

from .polymer import PolymerFlowModel
from .pretrained import load_pretrained, list_pretrained, is_pretrained_available

__all__ = [
    # Residue flow
    "PCAFlow",
    "ResidueFlowModel",
    "ResidueFlowConfig",
    # Frame indices and computation
    "FrameIndices",
    "compute_frame_from_indices",
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
    # Metrics
    "LatentMoments",
    "FlowMetrics",
    "compute_nll",
    "compute_latent_moments",
    "compute_flow_metrics",
    "estimate_kl_divergence",
    # Polymer flow
    "PolymerFlowModel",
    # Pre-trained models
    "load_pretrained",
    "list_pretrained",
    "is_pretrained_available",
]
