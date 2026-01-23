"""
Geometry operations on polymer structures.

Functions for computing distances, neighbors, and statistical analysis
of molecular coordinates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..backend import Array, size as arr_size
from ..backend import ops
from ..biochemistry import Scale

if TYPE_CHECKING:
    from ..polymer import Polymer


def pairwise_distances(polymer: "Polymer", scale: Scale | None = None) -> Array:
    """
    Compute pairwise distances.

    If scale is provided, computes distances between centroids
    at that scale. Otherwise, computes atom-atom distances.

    Args:
        polymer: Polymer structure to analyze.
        scale: Optional scale for centroid distances.

    Returns:
        Pairwise distance matrix.
    """
    if scale is None or scale == Scale.ATOM:
        coords = polymer.coordinates
    else:
        coords = polymer.reduce(polymer.coordinates, scale)

    return ops.cdist(coords, coords)


def knn(polymer: "Polymer", k: int, scale: Scale = Scale.ATOM) -> Array:
    """
    Find k-nearest neighbors at the specified scale.

    Args:
        polymer: Polymer structure to analyze.
        k: Number of neighbors per point (excluding self).
        scale: Scale at which to compute (ATOM, RESIDUE, CHAIN).

    Returns:
        Tensor of shape (k, N) where N = size at scale.
        Entry [i, j] is the index of j's i-th nearest neighbor.

    Example:
        >>> from ciffy import operations
        >>> p = ciffy.load("structure.cif", backend="torch")
        >>> neighbors = operations.knn(p, k=16, scale=Scale.ATOM)  # (16, num_atoms)
        >>> # Convert to edge_index for PyG:
        >>> src = torch.arange(p.size()).repeat_interleave(16)
        >>> dst = neighbors.flatten()
        >>> edge_index = torch.stack([src, dst])
    """
    # Compute pairwise distances at the given scale
    dists = pairwise_distances(polymer, scale)

    n = dists.shape[0]
    if k >= n:
        raise ValueError(f"k={k} must be less than number of points ({n})")

    # Use topk to find k+1 smallest (includes self at distance 0)
    _, indices = ops.topk(dists, k + 1, dim=1, largest=False)
    # Exclude self (first column) and transpose to (k, N)
    return indices[:, 1:].T


def adjacency(polymer: "Polymer", dtype: str = 'bool') -> Array:
    """
    Symmetric adjacency matrix of the bond graph.

    Args:
        polymer: Polymer structure to analyze.
        dtype: Data type for the matrix ('bool', 'float32', 'int32').

    Returns:
        (N, N) symmetric matrix where adj[i,j] = True/1 if atoms
        i and j are bonded.

    Note:
        This is O(N^2) memory. For large structures, use polymer.bonds
        property directly or CSR representation.
    """
    import numpy as np

    n_atoms = polymer.size()

    # Handle empty polymer
    if n_atoms == 0:
        return np.zeros((0, 0), dtype=dtype)

    bonds = polymer.bonds

    # Create zero matrix in correct backend
    adj = ops.zeros((n_atoms, n_atoms), like=polymer.coordinates, dtype=dtype)

    # Set symmetric entries
    adj[bonds[:, 0], bonds[:, 1]] = 1
    adj[bonds[:, 1], bonds[:, 0]] = 1

    return adj


def bonded_distances(
    polymer: "Polymer",
    atom1: int | Array,
    atom2: int | Array,
) -> Array:
    """
    Get distances between bonded atoms of specified types.

    Finds all covalent bonds where one atom matches `atom1` and the other
    matches `atom2`, then computes the Euclidean distance for each pair.

    Args:
        polymer: Polymer structure to analyze.
        atom1: First atom type(s) as integer value(s). Can be a single
            int or an array of ints to match multiple atom types.
        atom2: Second atom type(s) as integer value(s).

    Returns:
        1D array of distances between matching bonded pairs.
        Empty array if no matching bonds found.

    Example:
        >>> from ciffy import operations
        >>> from ciffy.biochemistry import Residue
        >>> # Get all phosphodiester bond lengths (O3'-P) for adenosine
        >>> operations.bonded_distances(polymer, Residue.A.O3p, Residue.A.P)
        array([1.59, 1.61, 1.58, ...])
    """
    # Get bonds and atom types
    bonds = polymer.bonds  # (B, 2) array of atom indices
    atoms = polymer.atoms  # (N,) array of atom type values

    # Convert atom type arguments to arrays in the same backend
    def to_values(atom: int | Array) -> Array:
        if isinstance(atom, int):
            return ops.array([atom], like=atoms)
        return ops.convert_backend(atom, like=atoms)

    v1 = to_values(atom1)
    v2 = to_values(atom2)

    # Get atom types at bond endpoints
    atom_i = atoms[bonds[:, 0]]
    atom_j = atoms[bonds[:, 1]]

    # Match: (i in v1 AND j in v2) OR (i in v2 AND j in v1)
    mask1 = ops.isin(atom_i, v1) & ops.isin(atom_j, v2)
    mask2 = ops.isin(atom_i, v2) & ops.isin(atom_j, v1)
    mask = mask1 | mask2

    # Get matching bond indices
    matching_indices = ops.nonzero_1d(mask)

    if arr_size(matching_indices) == 0:
        return ops.empty(0, like=polymer.coordinates)

    matching_bonds = bonds[matching_indices]

    # Compute distances
    coords = polymer.coordinates
    p1 = coords[matching_bonds[:, 0]]
    p2 = coords[matching_bonds[:, 1]]
    diff = p1 - p2

    return ops.norm(diff, axis=1)


def moment(polymer: "Polymer", n: int, scale: Scale) -> Array:
    """
    Compute the n-th moment of coordinates at a scale.

    Args:
        polymer: Polymer structure to analyze.
        n: Moment order (1=mean, 2=variance, 3=skewness).
        scale: Scale at which to compute.

    Returns:
        Moment tensor with one value per scale unit per dimension.
    """
    return polymer.reduce(polymer.coordinates ** n, scale)


def _pc(polymer: "Polymer", scale: Scale) -> tuple[Array, Array]:
    """
    Compute principal components at the specified scale.

    Args:
        polymer: Polymer structure to analyze.
        scale: Scale at which to compute.

    Returns:
        Tuple of (eigenvalues, eigenvectors).

    Note:
        Principal components are only defined up to sign.
        Use pca() for stable, unique orientations.
    """
    cov = polymer.coordinates[:, None, :] * polymer.coordinates[:, :, None]
    cov = polymer.reduce(cov, scale)
    return ops.eigh(cov)


def pca(polymer: "Polymer", scale: Scale) -> tuple["Polymer", Array]:
    """
    Align structure to principal axes at the specified scale.

    Centers the structure and rotates it so that the covariance
    matrix is diagonal. Signs are chosen so that the largest
    two third moments are positive.

    Args:
        polymer: Polymer structure to analyze.
        scale: Scale at which to align.

    Returns:
        Tuple of (aligned polymer, rotation matrices Q).
    """
    aligned, _ = polymer.center(scale)
    _, Q = _pc(aligned, scale)

    Q_exp = aligned.expand(Q, scale)
    aligned.coordinates = (
        Q_exp @ aligned.coordinates[..., None]
    ).squeeze()

    # Ensure stability by fixing signs based on third moments
    signs = ops.sign(moment(aligned, 3, scale))
    signs[:, 0] = signs[:, 1] * signs[:, 2] * ops.det(Q)
    signs_exp = aligned.expand(signs, scale)

    aligned.coordinates = aligned.coordinates * signs_exp
    Q = Q * signs[..., None]

    return aligned, Q


__all__ = [
    "pairwise_distances",
    "knn",
    "adjacency",
    "bonded_distances",
    "moment",
    "pca",
]
