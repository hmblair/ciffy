"""
File writing functionality for molecular structures.

Supports writing to CIF format.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..polymer import Polymer


def write_cif(polymer: "Polymer", filename: str) -> None:
    """
    Write a polymer structure to mmCIF format.

    Supports all molecule types (protein, RNA, DNA) and includes
    both polymer and non-polymer atoms.

    Args:
        polymer: The polymer structure to write.
        filename: Path to the output file.

    Raises:
        IOError: If the file cannot be written.
        TypeError: If the data has wrong type.

    Example:
        >>> polymer = ciffy.load("structure.cif", backend="numpy")
        >>> polymer.write("output.cif")
    """
    from .._c import _save
    from ..types import Scale
    from ..backend import is_torch

    # Convert to numpy if using torch backend
    if is_torch(polymer.coordinates):
        polymer = polymer.numpy()

    # Ensure arrays are the correct dtype and contiguous
    # Flatten coordinates from (N, 3) to (N*3,) for C interface
    coordinates = np.ascontiguousarray(polymer.coordinates.flatten().astype(np.float32))
    atoms = np.ascontiguousarray(polymer.atoms.astype(np.int32))
    elements = np.ascontiguousarray(polymer.elements.astype(np.int32))
    residues = np.ascontiguousarray(polymer.sequence.astype(np.int32))
    atoms_per_res = np.ascontiguousarray(polymer._sizes[Scale.RESIDUE].astype(np.int32))
    atoms_per_chain = np.ascontiguousarray(polymer._sizes[Scale.CHAIN].astype(np.int32))
    res_per_chain = np.ascontiguousarray(polymer.lengths.astype(np.int32))
    molecule_types = np.ascontiguousarray(polymer.molecule_type.astype(np.int32))

    _save(
        filename,
        polymer.pdb_id,
        coordinates,
        atoms,
        elements,
        residues,
        atoms_per_res,
        atoms_per_chain,
        res_per_chain,
        polymer.names,
        polymer.strands,
        polymer.polymer_count,
        molecule_types,
    )
