"""Clebsch-Gordan coefficients for SO(3) tensor products.

This module computes real Clebsch-Gordan coefficients for coupling
spherical harmonics in tensor product operations.

The CG coefficients C^{l,m}_{l1,m1,l2,m2} satisfy:
    Y_{l1}^{m1} ⊗ Y_{l2}^{m2} = Σ_{l,m} C^{l,m}_{l1,m1,l2,m2} Y_l^m

where Y_l^m are real spherical harmonics.
"""
from __future__ import annotations

from functools import lru_cache

import torch


def _complex_to_real_matrix(l: int) -> torch.Tensor:
    """Unitary matrix converting complex to real spherical harmonics.

    Returns Q such that Y_real = Q @ Y_complex.

    Args:
        l: Degree of the spherical harmonic.

    Returns:
        Unitary matrix of shape (2l+1, 2l+1).
    """
    dim = 2 * l + 1
    Q = torch.zeros(dim, dim, dtype=torch.complex128)
    SQRT2 = 2**-0.5

    for m in range(-l, 0):
        Q[l + m, l + abs(m)] = SQRT2
        Q[l + m, l - abs(m)] = -1j * SQRT2

    Q[l, l] = 1

    for m in range(1, l + 1):
        Q[l + m, l + abs(m)] = (-1) ** m * SQRT2
        Q[l + m, l - abs(m)] = 1j * (-1) ** m * SQRT2

    return (-1j) ** l * Q


@lru_cache(maxsize=128)
def clebsch_gordan(l1: int, l2: int, l: int) -> torch.Tensor:
    """Compute real Clebsch-Gordan coefficients.

    Returns the CG tensor for coupling irreps l1 and l2 to produce irrep l.
    Valid when |l1-l2| <= l <= l1+l2.

    Args:
        l1: First irrep degree.
        l2: Second irrep degree.
        l: Output irrep degree.

    Returns:
        Tensor of shape (2*l1+1, 2*l2+1, 2*l+1).
    """
    try:
        import wigners
    except ImportError as e:
        raise ImportError(
            "The 'wigners' package is required for Clebsch-Gordan coefficients. "
            "Install with: pip install wigners"
        ) from e

    # Get complex CG coefficients from wigners
    cg_complex = torch.tensor(wigners.clebsch_gordan_array(l1, l2, l))

    # Convert all three indices from complex to real basis
    Q1 = _complex_to_real_matrix(l1)
    Q2 = _complex_to_real_matrix(l2)
    Q3 = _complex_to_real_matrix(l)

    # Transform: C_real[i,j,k] = Q1[i,a] Q2[j,b] Q3*[k,c] C_complex[a,b,c]
    cg_real = torch.einsum(
        "ij,kl,mn,ikm->jln", Q1, Q2, Q3.conj(), cg_complex.to(torch.complex128)
    ).real

    # Normalize
    cg_real = cg_real / (2 * l + 1)

    return cg_real.float()
