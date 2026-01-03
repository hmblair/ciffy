"""
Normalizing flow models for molecular conformations.

This module provides normalizing flow-based models for learning distributions
over molecular conformations:

- `ResidueFlowModel`: Per-residue flow model (PCA + normalizing flow)
- `load_pretrained`: Load pre-trained models

Note: `PolymerModel` (formerly `PolymerFlowModel`) has moved to `ciffy.nn.polymer`
since it works with any residue-level model (Flow, VAE, etc.). Import from there:

    >>> from ciffy.nn import PolymerModel, ResidueGenerativeCore

Quick Start:
    >>> import ciffy
    >>> from ciffy.nn import PolymerModel
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

Training Custom Models:
    >>> from ciffy.nn.lightning import ResidueFlowModule, ResidueDataModule
    >>> from ciffy.biochemistry import Residue
    >>> import lightning as L
    >>>
    >>> # Train a model for each residue type
    >>> dm = ResidueDataModule(cif_paths, Residue.A)
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
    # Frame computation
    compute_glycosidic_frame,
    align_to_frame,
    align_and_compute_transform,
    position_next_residue,
    # SE(3) transforms
    compute_relative_transform,
    apply_relative_transform,
    # Data extraction
    extract_residues,
    extract_residues_with_links,
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

# Re-export from new location (ciffy.nn.polymer) for backwards compatibility
from .polymer import PolymerModel, PolymerFlowModel, ResidueGenerativeCore
from .pretrained import load_pretrained, list_pretrained, is_pretrained_available

__all__ = [
    # Protocol for residue-level models (now in ciffy.nn.polymer)
    "ResidueGenerativeCore",
    # Residue flow
    "PCAFlow",
    "ResidueFlowModel",
    "ResidueFlowConfig",
    # Frame computation
    "compute_glycosidic_frame",
    "align_to_frame",
    "align_and_compute_transform",
    "position_next_residue",
    # SE(3) transforms
    "compute_relative_transform",
    "apply_relative_transform",
    # Data extraction
    "extract_residues",
    "extract_residues_with_links",
    "ResidueDataset",
    # Metrics
    "LatentMoments",
    "FlowMetrics",
    "compute_nll",
    "compute_latent_moments",
    "compute_flow_metrics",
    "estimate_kl_divergence",
    # Polymer model (now in ciffy.nn.polymer, re-exported for backwards compat)
    "PolymerModel",
    "PolymerFlowModel",  # Deprecated alias
    # Pre-trained models
    "load_pretrained",
    "list_pretrained",
    "is_pretrained_available",
]
