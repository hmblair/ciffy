"""
Normalizing flow layers and utilities.

This module provides generic normalizing flow building blocks:

Complete Flows:
    - RealNVP: Stack of affine coupling layers
    - NeuralSplineFlow: Stack of spline coupling layers

Embeddings:
    - SinusoidalTimeEmbedding: Time embedding for flow matching

Utilities:
    - FlowMetrics: Metrics for evaluating normalizing flow models

Example:
    >>> from ciffy.nn.flow import RealNVP, NeuralSplineFlow
    >>>
    >>> flow = RealNVP(dim=16, n_layers=8)
    >>> z, log_det = flow(x)
    >>> x_recon = flow.inverse(z)  # Exact reconstruction
    >>> log_prob = flow.log_prob(x)
"""

from .layers import (
    RealNVP,
    NeuralSplineFlow,
)
from .embeddings import SinusoidalTimeEmbedding
from .metrics import FlowMetrics

__all__ = [
    "RealNVP",
    "NeuralSplineFlow",
    "SinusoidalTimeEmbedding",
    "FlowMetrics",
]
