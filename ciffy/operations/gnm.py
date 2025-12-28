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

from ..backend import Array, diag, pinv, diagonal

__all__ = [
    "graph_laplacian",
    "gnm_correlations",
    "gnm_variances",
]


def graph_laplacian(adj: Array) -> Array:
    """Compute the graph Laplacian (Kirchhoff matrix).

    The graph Laplacian is defined as L = D - A, where D is the
    degree matrix and A is the adjacency matrix.

    Args:
        adj: Adjacency matrix of shape (N, N). Can be numpy array or torch tensor.

    Returns:
        Laplacian matrix of shape (N, N).

    Example:
        >>> import numpy as np
        >>> adj = np.array([[0., 1., 1.], [1., 0., 1.], [1., 1., 0.]])
        >>> L = graph_laplacian(adj)
    """
    deg = diag(adj.sum(1))
    return deg - adj


def gnm_correlations(adj: Array, rtol: float = 1e-2) -> Array:
    """Compute correlations under a Gaussian Network Model.

    The GNM models molecular dynamics as a network of springs,
    with correlations given by the pseudo-inverse of the Laplacian.

    Args:
        adj: Adjacency/connectivity matrix of shape (N, N).
            Can be numpy array or torch tensor.
        rtol: Relative tolerance for pseudo-inverse computation.

    Returns:
        Correlation matrix of shape (N, N).

    Example:
        >>> import numpy as np
        >>> adj = np.array([[0., 1., 1.], [1., 0., 1.], [1., 1., 0.]])
        >>> corr = gnm_correlations(adj)
    """
    lap = graph_laplacian(adj)
    return pinv(lap, rtol=rtol)


def gnm_variances(adj: Array, rtol: float = 1e-2) -> Array:
    """Compute position variances under a Gaussian Network Model.

    Returns the diagonal of the GNM correlation matrix, which
    represents the variance in position for each node.

    Args:
        adj: Adjacency/connectivity matrix of shape (N, N).
            Can be numpy array or torch tensor.
        rtol: Relative tolerance for pseudo-inverse computation.

    Returns:
        Variance vector of shape (N,).

    Example:
        >>> import numpy as np
        >>> adj = np.array([[0., 1., 1.], [1., 0., 1.], [1., 1., 0.]])
        >>> var = gnm_variances(adj)
    """
    return diagonal(gnm_correlations(adj, rtol=rtol))
