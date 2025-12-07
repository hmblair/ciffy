"""
CIF file loading functionality.
"""

from __future__ import annotations
import os
from typing import TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from ..polymer import Polymer


def load(file: str) -> "Polymer":
    """
    Load a molecular structure from a CIF file.

    Parses the CIF file using the C extension and constructs a Polymer
    object with coordinates, atoms, elements, and structural information.

    Args:
        file: Path to the CIF file.

    Returns:
        Polymer object containing the parsed structure.

    Raises:
        OSError: If the file does not exist.
        RuntimeError: If parsing fails.

    Example:
        >>> polymer = load("1abc.cif")
        >>> print(polymer)
        PDB 1ABC with 1234 atoms.
    """
    # Import here to avoid circular imports
    from ..polymer import Polymer
    from ..types import Scale
    from .._c import _load

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

    mol_sizes = torch.tensor([len(coordinates)], dtype=torch.long).numpy()

    sizes = {
        Scale.RESIDUE: atoms_per_res,
        Scale.CHAIN: atoms_per_chain,
        Scale.MOLECULE: mol_sizes,
    }

    return Polymer(
        torch.from_numpy(coordinates),
        torch.from_numpy(atoms).long(),
        torch.from_numpy(elements).long(),
        torch.from_numpy(residues).long(),
        {key: torch.from_numpy(value).long() for key, value in sizes.items()},
        id,
        chain_names,
        strand_names,
        torch.from_numpy(res_per_chain),
        polymer_count,
    )
