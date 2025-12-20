"""Gaussian Network Model utilities.

This module provides functions for computing Gaussian Network Model (GNM)
properties from molecular structures. GNM models molecular dynamics as a
network of harmonic springs connecting nearby atoms.

Functions:
    graph_laplacian: Compute graph Laplacian (Kirchhoff matrix)
    gnm_correlations: Compute Gaussian Network Model correlations
    gnm_variances: Compute GNM position variances
"""
from __future__ import annotations

import torch

__all__ = [
    "graph_laplacian",
    "gnm_correlations",
    "gnm_variances",
]


def graph_laplacian(adj: torch.Tensor) -> torch.Tensor:
    """Compute the graph Laplacian (Kirchhoff matrix).

    The graph Laplacian is defined as L = D - A, where D is the
    degree matrix and A is the adjacency matrix.

    Args:
        adj: Adjacency matrix of shape (N, N).

    Returns:
        Laplacian matrix of shape (N, N).

    Example:
        >>> adj = torch.tensor([[0., 1., 1.], [1., 0., 1.], [1., 1., 0.]])
        >>> L = graph_laplacian(adj)
    """
    deg = torch.diag(adj.sum(1))
    return deg - adj


def gnm_correlations(adj: torch.Tensor) -> torch.Tensor:
    """Compute correlations under a Gaussian Network Model.

    The GNM models molecular dynamics as a network of springs,
    with correlations given by the pseudo-inverse of the Laplacian.

    Args:
        adj: Adjacency/connectivity matrix of shape (N, N).

    Returns:
        Correlation matrix of shape (N, N).

    Example:
        >>> adj = torch.tensor([[0., 1., 1.], [1., 0., 1.], [1., 1., 0.]])
        >>> corr = gnm_correlations(adj)
    """
    lap = graph_laplacian(adj)
    return torch.linalg.pinv(lap, rtol=1e-2)


def gnm_variances(adj: torch.Tensor) -> torch.Tensor:
    """Compute position variances under a Gaussian Network Model.

    Returns the diagonal of the GNM correlation matrix, which
    represents the variance in position for each node.

    Args:
        adj: Adjacency/connectivity matrix of shape (N, N).

    Returns:
        Variance vector of shape (N,).

    Example:
        >>> adj = torch.tensor([[0., 1., 1.], [1., 0., 1.], [1., 1., 0.]])
        >>> var = gnm_variances(adj)
    """
    return torch.diagonal(gnm_correlations(adj))
