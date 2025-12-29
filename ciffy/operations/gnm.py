"""Gaussian Network Model (GNM) for molecular dynamics analysis.

The GNM models molecular dynamics as a network of harmonic springs connecting
nearby atoms/residues. This module provides the GNM class which computes and
caches GNM properties efficiently.

Example:
    >>> import ciffy
    >>> from ciffy import Scale
    >>> from ciffy.operations import contact_map, GNM
    >>>
    >>> polymer = ciffy.load("structure.cif").poly()
    >>> adj = contact_map(polymer, cutoff=7.0)  # 7Å cutoff at residue level
    >>> gnm = GNM(adj)
    >>>
    >>> gnm.variances           # Position variances (B-factor prediction)
    >>> gnm.correlations        # Full correlation matrix
    >>> gnm.cross_correlations  # Normalized correlations [-1, 1]
    >>> eigenvalues, modes = gnm.modes(k=3)  # Slowest 3 modes
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from ..backend import Array, diag, pinv, diagonal, eigh, sqrt, outer
from ..biochemistry import Scale

if TYPE_CHECKING:
    from ..polymer import Polymer

__all__ = ["GNM", "contact_map"]


def contact_map(
    polymer: "Polymer",
    cutoff: float = 7.0,
    scale: Scale = Scale.RESIDUE,
) -> Array:
    """Build a contact/adjacency matrix from a Polymer.

    Computes pairwise distances at the specified scale and returns a binary
    adjacency matrix where entry (i, j) is 1 if the distance between units
    i and j is less than the cutoff.

    This is the standard way to create an adjacency matrix for GNM analysis.

    Args:
        polymer: Polymer structure to analyze.
        cutoff: Distance cutoff in Angstroms. Pairs closer than this are
            considered in contact. Default 7.0Å is typical for C-alpha GNM.
        scale: Scale at which to compute contacts. Default is RESIDUE,
            which uses residue centroids. Use Scale.ATOM for all-atom contacts.

    Returns:
        Binary adjacency matrix of shape (N, N) where N is the number of
        units at the specified scale. Uses the same backend (numpy/torch)
        as the input polymer.

    Example:
        >>> import ciffy
        >>> from ciffy import Scale
        >>> from ciffy.operations import contact_map, GNM
        >>>
        >>> polymer = ciffy.load("structure.cif").poly()
        >>>
        >>> # Residue-level contact map (default, for coarse-grained GNM)
        >>> adj = contact_map(polymer, cutoff=7.0)
        >>> gnm = GNM(adj)
        >>>
        >>> # Atom-level contact map (for all-atom analysis)
        >>> adj_atom = contact_map(polymer, cutoff=4.0, scale=Scale.ATOM)
    """
    # Compute pairwise distances at the specified scale
    dists = polymer.pairwise_distances(scale)

    # Create binary adjacency: 1 if distance < cutoff, 0 otherwise
    mask = dists < cutoff

    # Convert boolean to float, handling both numpy and torch
    if hasattr(mask, 'astype'):
        # NumPy
        import numpy as np
        adj = mask.astype(dists.dtype)
        np.fill_diagonal(adj, 0)
    else:
        # PyTorch
        adj = mask.to(dists.dtype)
        adj.fill_diagonal_(0)

    return adj


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


class GNM:
    """Gaussian Network Model with cached pseudo-inverse.

    This class wraps GNM computations and caches the pseudo-inverse of the
    Laplacian matrix. This is efficient when you need multiple GNM properties
    (correlations, variances, cross-correlations) since they all derive from
    the same pseudo-inverse.

    Attributes:
        adj: The adjacency matrix used to construct the model.
        laplacian: The graph Laplacian (Kirchhoff matrix).

    Example:
        >>> import numpy as np
        >>> adj = np.array([[0., 1., 1.], [1., 0., 1.], [1., 1., 0.]])
        >>> gnm = GNM(adj)
        >>> gnm.variances       # Position variances
        >>> gnm.correlations    # Full correlation matrix
        >>> gnm.cross_correlations  # Normalized correlations [-1, 1]
        >>> eigenvalues, modes = gnm.modes(k=2)  # Slowest 2 modes
    """

    def __init__(self, adj: Array, rtol: float = 1e-2):
        """Initialize GNM from an adjacency matrix.

        Args:
            adj: Adjacency/connectivity matrix of shape (N, N).
                Can be numpy array or torch tensor.
            rtol: Relative tolerance for pseudo-inverse computation.
        """
        self.adj = adj
        self.rtol = rtol
        self.laplacian = graph_laplacian(adj)
        self._pinv: Array | None = None
        self._eigenvalues: Array | None = None
        self._eigenvectors: Array | None = None

    def _compute_pinv(self) -> Array:
        """Lazily compute and cache the pseudo-inverse."""
        if self._pinv is None:
            self._pinv = pinv(self.laplacian, rtol=self.rtol)
        return self._pinv

    def _compute_eigen(self) -> tuple[Array, Array]:
        """Lazily compute and cache the eigendecomposition."""
        if self._eigenvalues is None or self._eigenvectors is None:
            self._eigenvalues, self._eigenvectors = eigh(self.laplacian)
        return self._eigenvalues, self._eigenvectors

    @property
    def correlations(self) -> Array:
        """GNM correlation matrix (pseudo-inverse of Laplacian).

        Returns:
            Correlation matrix of shape (N, N).
        """
        return self._compute_pinv()

    @property
    def variances(self) -> Array:
        """Position variances (diagonal of correlation matrix).

        Returns:
            Variance vector of shape (N,).
        """
        return diagonal(self._compute_pinv())

    @property
    def cross_correlations(self) -> Array:
        """Normalized cross-correlation matrix.

        Returns correlations normalized by the geometric mean of variances,
        giving values in the range [-1, 1]. This is useful for identifying
        coupled motions between residues.

        Returns:
            Normalized correlation matrix of shape (N, N).
        """
        corr = self._compute_pinv()
        std = sqrt(diagonal(corr))
        return corr / outer(std, std)

    @property
    def eigenvalues(self) -> Array:
        """Eigenvalues of the Laplacian (squared frequencies).

        Returns eigenvalues in ascending order. The first eigenvalue is
        always zero (trivial mode corresponding to rigid-body translation).

        Returns:
            Eigenvalue array of shape (N,).
        """
        eigenvalues, _ = self._compute_eigen()
        return eigenvalues

    def modes(self, k: int | None = None) -> tuple[Array, Array]:
        """Compute GNM normal modes (eigenvectors of Laplacian).

        Returns the slowest k non-trivial modes (skipping the zero eigenvalue
        mode which corresponds to rigid-body translation).

        Args:
            k: Number of modes to return. If None, returns all non-trivial modes.

        Returns:
            Tuple of (eigenvalues, eigenvectors):
                - eigenvalues: Array of shape (k,) containing squared frequencies
                - eigenvectors: Array of shape (N, k) where each column is a mode

        Example:
            >>> gnm = GNM(adj)
            >>> eigenvalues, modes = gnm.modes(k=3)  # Get 3 slowest modes
            >>> modes[:, 0]  # First (slowest) mode
        """
        eigenvalues, eigenvectors = self._compute_eigen()
        # Skip the first (zero) eigenvalue and return up to k modes
        if k is None:
            return eigenvalues[1:], eigenvectors[:, 1:]
        return eigenvalues[1:k + 1], eigenvectors[:, 1:k + 1]
