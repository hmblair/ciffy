"""
CIF file loading functionality.
"""

from __future__ import annotations
import os
import warnings
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from ..polymer import Polymer

# Sentinel to detect if backend was specified
_NO_BACKEND = object()


def load(file: str, backend: Optional[str] = _NO_BACKEND) -> "Polymer":
    """
    Load a molecular structure from a CIF file.

    Parses the CIF file using the C extension and constructs a Polymer
    object with coordinates, atoms, elements, and structural information.

    Args:
        file: Path to the CIF file.
        backend: Array backend, either "numpy" or "torch".
                 Default is "torch" (deprecated, will change to "numpy" in v0.6.0).

    Returns:
        Polymer object containing the parsed structure.

    Raises:
        OSError: If the file does not exist.
        RuntimeError: If parsing fails.
        ValueError: If backend is not "numpy" or "torch".

    Example:
        >>> polymer = load("1abc.cif", backend="numpy")
        >>> print(polymer)
        PDB 1ABC with 1234 atoms (numpy).
    """
    # Import here to avoid circular imports
    from ..polymer import Polymer
    from ..types import Scale
    from .._c import _load

    # Handle backend parameter
    if backend is _NO_BACKEND:
        warnings.warn(
            "Default backend will change from 'torch' to 'numpy' in v0.6.0. "
            "Pass backend='torch' or backend='numpy' explicitly to silence this warning.",
            DeprecationWarning,
            stacklevel=2
        )
        backend = "torch"

    if backend not in ("numpy", "torch"):
        raise ValueError(f"backend must be 'numpy' or 'torch', got {backend!r}")

    if not os.path.isfile(file):
        raise OSError(f'The file "{file}" does not exist.')

    (
        id,
        coordinates,
        atoms,
        elements,
        residues,
        atoms_per_res,
        atoms_per_chain,
        res_per_chain,
        chain_names,
        strand_names,
        polymer_count,
    ) = _load(file)

    mol_sizes = np.array([len(coordinates)], dtype=np.int64)

    sizes = {
        Scale.RESIDUE: atoms_per_res,
        Scale.CHAIN: atoms_per_chain,
        Scale.MOLECULE: mol_sizes,
    }

    # Create Polymer with NumPy arrays (what the C extension returns)
    polymer = Polymer(
        coordinates,
        atoms.astype(np.int64),
        elements.astype(np.int64),
        residues.astype(np.int64),
        {key: value.astype(np.int64) for key, value in sizes.items()},
        id,
        chain_names,
        strand_names,
        res_per_chain.astype(np.int64),
        polymer_count,
    )

    # Convert to torch if requested
    if backend == "torch":
        return polymer.torch()

    return polymer
