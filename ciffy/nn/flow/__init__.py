"""
Normalizing flow utilities.

Note:
    The residue-level flow models (ResidueFlowModel, PCAFlow) and PolymerModel
    have been archived. For new residue-level modeling, use:

        >>> from ciffy.nn.residue import ResidueVAE

    The old code is preserved in archive/nn/flow/ for reference.

Remaining utilities:
    - FlowMetrics: Metrics for evaluating normalizing flow models
    - load_pretrained: Load pre-trained models (if available)
"""

from .metrics import (
    LatentMoments,
    FlowMetrics,
    compute_nll,
    compute_latent_moments,
    compute_flow_metrics,
    estimate_kl_divergence,
)
from .pretrained import load_pretrained, list_pretrained, is_pretrained_available

__all__ = [
    # Metrics
    "LatentMoments",
    "FlowMetrics",
    "compute_nll",
    "compute_latent_moments",
    "compute_flow_metrics",
    "estimate_kl_divergence",
    # Pre-trained models
    "load_pretrained",
    "list_pretrained",
    "is_pretrained_available",
]
