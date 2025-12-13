"""
Inter-residue linking definitions for polymer chains.

Defines the atoms involved in linking consecutive residues and
their bond lengths for template positioning.
"""

from dataclasses import dataclass

from ..types import Molecule


@dataclass
class LinkingDefinition:
    """Definition for inter-residue bonding."""
    prev_atom: str      # Atom on residue N (e.g., "O3p" for NA, "C" for protein)
    next_atom: str      # Atom on residue N+1 (e.g., "P" for NA, "N" for protein)
    bond_length: float  # Standard bond length in Angstroms


# Phosphodiester bond: O3' of residue N to P of residue N+1
NUCLEIC_ACID_LINK = LinkingDefinition(
    prev_atom="O3p",     # O3' (using Python name with p for ')
    next_atom="P",
    bond_length=1.60,    # P-O bond ~1.60 Angstroms
)

# Peptide bond: C of residue N to N of residue N+1
PEPTIDE_LINK = LinkingDefinition(
    prev_atom="C",
    next_atom="N",
    bond_length=1.33,    # C-N peptide bond ~1.33 Angstroms
)

# Map molecule type to linking definition
LINKING_BY_TYPE: dict[int, LinkingDefinition] = {
    Molecule.RNA: NUCLEIC_ACID_LINK,
    Molecule.DNA: NUCLEIC_ACID_LINK,
    Molecule.PROTEIN: PEPTIDE_LINK,
}
