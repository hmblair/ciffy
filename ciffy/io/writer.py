"""
PDB file writing functionality.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

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
        KeyError: If an atom type is not recognized.

    Note:
        Currently supports RNA structures only.
    """
    from ..types import Scale
    from ..biochemistry import Element, RibonucleicAcid

    # Filter out non-polymer atoms
    polymer = polymer.polymer_only()

    with open(filename, 'w') as file:
        for chain in polymer.chains():
            atom_idx = 0
            for residue in range(chain.size(Scale.RESIDUE)):
                residue_name = polymer.str()[residue]

                for _ in range(chain._sizes[Scale.RESIDUE][residue]):
                    element = Element.revdict()[chain.elements[atom_idx].item()]
                    atom_value = chain.atoms[atom_idx].item()

                    if atom_value not in RibonucleicAcid.revdict():
                        raise KeyError(
                            f"Unknown atom type: {atom_value}. "
                            "PDB writing currently supports RNA structures only."
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
