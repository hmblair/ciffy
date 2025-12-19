"""
Inter-residue linking definitions for polymer chains.

Defines the atoms involved in linking consecutive residues and
their bond lengths for template positioning.

Supported polymer types:
- RNA, DNA, HYBRID: Phosphodiester linkage (O3' -> P)
- PROTEIN, PROTEIN_D, CYCLIC_PEPTIDE: Peptide bond (C -> N)

Unsupported polymer types (no linking definition):
- POLYSACCHARIDE: Glycosidic bonds vary by sugar type
- PNA: Synthetic backbone with different linkage
- LIGAND, ION, WATER, OTHER, UNKNOWN: Non-polymeric
"""

from dataclasses import dataclass

from ._generated_molecule import Molecule


@dataclass
class LinkingDefinition:
    """
    Definition for inter-residue bonding.

    Attributes:
        prev_atom: Atom name on residue N that forms the bond (e.g., "O3p", "C").
                   Uses Python naming convention (apostrophe -> p).
        next_atom: Atom name on residue N+1 that forms the bond (e.g., "P", "N").
        bond_length: Standard bond length in Angstroms.

    Example:
        For RNA, residue N's O3' binds to residue N+1's P with ~1.6A bond.
    """
    prev_atom: str
    next_atom: str
    bond_length: float


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
# Only polymer types with well-defined inter-residue linkages are included
LINKING_BY_TYPE: dict[int, LinkingDefinition] = {
    # Nucleic acids (phosphodiester linkage)
    Molecule.RNA: NUCLEIC_ACID_LINK,
    Molecule.DNA: NUCLEIC_ACID_LINK,
    Molecule.HYBRID: NUCLEIC_ACID_LINK,
    # Peptides (peptide bond)
    Molecule.PROTEIN: PEPTIDE_LINK,
    Molecule.PROTEIN_D: PEPTIDE_LINK,
    Molecule.CYCLIC_PEPTIDE: PEPTIDE_LINK,
}
