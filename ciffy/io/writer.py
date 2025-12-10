"""
File writing functionality for molecular structures.

Supports writing to PDB and CIF formats.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..polymer import Polymer


def write_pdb(polymer: "Polymer", filename: str) -> None:
    """
    Write a polymer structure to a PDB format file.

    Non-polymer atoms (water, ions, ligands) are automatically filtered out.

    Args:
        polymer: The polymer structure to write.
        filename: Path to the output file.

    Raises:
        IOError: If the file cannot be written.
        ValueError: If the polymer contains non-RNA chains.
        KeyError: If an atom type is not recognized.

    Note:
        Currently supports RNA structures only.
    """
    from ..types import Scale, Molecule
    from ..biochemistry import Element, RibonucleicAcid

    # Filter out non-polymer atoms
    polymer = polymer.polymer_only()

    # Validate that all chains are RNA
    for i, mol_type in enumerate(polymer.molecule_type):
        if mol_type.item() != Molecule.RNA.value:
            raise ValueError(
                f"Chain {i} is not RNA (type={mol_type.item()}). "
                "PDB writing currently supports RNA structures only. "
                "Use polymer.subset(RNA) to filter RNA chains first."
            )

    with open(filename, 'w') as file:
        for chain in polymer.chains():
            seq = chain.str()
            atom_idx = 0
            for residue in range(chain.size(Scale.RESIDUE)):
                if residue >= len(seq):
                    import warnings
                    warnings.warn(
                        f"Residue index {residue} exceeds sequence length {len(seq)} "
                        f"in chain '{chain.names[0]}'; using 'X' as placeholder.",
                        UserWarning
                    )
                    residue_name = 'X'
                else:
                    residue_name = seq[residue]

                for _ in range(chain._sizes[Scale.RESIDUE][residue]):
                    element = Element.revdict()[chain.elements[atom_idx].item()]
                    atom_value = chain.atoms[atom_idx].item()

                    if atom_value not in RibonucleicAcid.revdict():
                        element_name = Element.revdict().get(chain.elements[atom_idx].item(), "unknown")
                        raise ValueError(
                            f"Unrecognized atom type (index={atom_value}, element={element_name}) "
                            f"at residue {residue + 1} in chain '{chain.names[0]}'. "
                            f"PDB writing supports standard RNA atoms only. "
                            f"Use write() for CIF format which supports all atom types."
                        )

                    atom_name = RibonucleicAcid.revdict()[atom_value].replace('p', "'")

                    file.write(
                        "ATOM  {:5d} {:4s} {:3s} {:1s}{:4d}    "
                        "{:8.3f}{:8.3f}{:8.3f}  1.00  0.00           {:2s}\n".format(
                            atom_idx + 1,
                            atom_name,
                            residue_name,
                            chain.names[0],
                            residue + 1,
                            chain.coordinates[atom_idx][0],
                            chain.coordinates[atom_idx][1],
                            chain.coordinates[atom_idx][2],
                            element,
                        )
                    )

                    atom_idx += 1


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
        >>> polymer.write_cif("output.cif")
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

    _save(
        filename,
        polymer._id,
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
    )
