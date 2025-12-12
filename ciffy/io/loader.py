"""
CIF file loading functionality.
"""

from __future__ import annotations
import os
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..polymer import Polymer

def load(
    file: str,
    backend: str | None = None,
    load_descriptions: bool = False,
) -> "Polymer":
    """
    Load a molecular structure from a CIF file.

    Parses the CIF file using the C extension and constructs a Polymer
    object with coordinates, atoms, elements, and structural information.

    Args:
        file: Path to the CIF file.
        backend: Array backend, either "numpy" or "torch". Default is "numpy".
        load_descriptions: If True, parse entity descriptions from CIF file.
            Default is False for performance (descriptions not needed for DL).

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

        >>> polymer = load("1abc.cif", load_descriptions=True)
        >>> print(polymer.descriptions)
        ['RNA (66-MER)', 'CESIUM ION', ...]
    """
    # Import here to avoid circular imports
    from ..polymer import Polymer
    from ..types import Scale
    from .._c import _load

    # Handle backend parameter
    if backend is None:
        backend = "numpy"

    if backend not in ("numpy", "torch"):
        raise ValueError(f"backend must be 'numpy' or 'torch', got {backend!r}")

    if not os.path.isfile(file):
        raise OSError(f'The file "{file}" does not exist.')

    # Load returns a dict with all parsed data
    data = _load(file, load_descriptions=load_descriptions)

    # Extract fields from dict
    id = data["id"]
    coordinates = data["coordinates"]
    atoms = data["atoms"]
    elements = data["elements"]
    residues = data["residues"]
    atoms_per_res = data["atoms_per_res"]
    atoms_per_chain = data["atoms_per_chain"]
    res_per_chain = data["res_per_chain"]
    chain_names = data["chain_names"]
    strand_names = data["strand_names"]
    polymer_count = data["polymer_count"]
    molecule_types = data["molecule_types"]

    mol_sizes = np.array([len(coordinates)], dtype=np.int64)

    sizes = {
        Scale.RESIDUE: atoms_per_res,
        Scale.CHAIN: atoms_per_chain,
        Scale.MOLECULE: mol_sizes,
    }

    # Get descriptions if loaded
    descriptions = data.get("descriptions", None)

    # Create Polymer with NumPy arrays (C extension returns int64 directly)
    polymer = Polymer(
        coordinates,
        atoms,
        elements,
        residues,
        sizes,
        id,
        chain_names,
        strand_names,
        res_per_chain,
        polymer_count,
        molecule_types,
        descriptions,
    )

    # Convert to torch if requested
    if backend == "torch":
        return polymer.torch()

    return polymer
