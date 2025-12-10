"""
Molecule type enumeration.

Based on PDB/mmCIF entity types from wwPDB:
- _entity.type: polymer, non-polymer, water, branched
- _entity_poly.type: polypeptide(L/D), polyribonucleotide, polydeoxyribonucleotide, etc.

See: https://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v50.dic/Items/_entity_poly.type.html
"""

from enum import Enum


class Molecule(Enum):
    """
    Types of molecules that can appear in a structure.

    Used to classify chains by their molecular type, enabling
    filtering and type-specific operations.

    Polymer types (from _entity_poly.type):
        PROTEIN: polypeptide(L) - standard L-amino acid chains
        PROTEIN_D: polypeptide(D) - D-amino acid chains (rare)
        RNA: polyribonucleotide
        DNA: polydeoxyribonucleotide
        HYBRID: polydeoxyribonucleotide/polyribonucleotide hybrid
        POLYSACCHARIDE: polysaccharide(D) or polysaccharide(L)
        PNA: peptide nucleic acid (synthetic DNA/RNA mimic)
        CYCLIC_PEPTIDE: cyclic-pseudo-peptide

    Non-polymer types (from _entity.type):
        LIGAND: small molecules, cofactors, drugs
        ION: metal ions (Mg2+, Ca2+, Zn2+, etc.)
        WATER: water molecules (HOH)

    Special:
        OTHER: unclassified or unknown polymer type
        UNKNOWN: residue type not recognized
    """

    # Polymer types
    PROTEIN = 0         # polypeptide(L) - standard proteins
    RNA = 1             # polyribonucleotide
    DNA = 2             # polydeoxyribonucleotide
    HYBRID = 3          # DNA/RNA hybrid
    PROTEIN_D = 4       # polypeptide(D) - D-amino acids
    POLYSACCHARIDE = 5  # carbohydrates
    PNA = 6             # peptide nucleic acid
    CYCLIC_PEPTIDE = 7  # cyclic-pseudo-peptide

    # Non-polymer types
    LIGAND = 8          # small molecules, cofactors
    ION = 9             # metal ions
    WATER = 10          # water (HOH)

    # Special
    OTHER = 11          # other polymer type
    UNKNOWN = 12        # unrecognized


def molecule_type(value: int) -> Molecule:
    """
    Convert an integer value to the corresponding Molecule type.

    Args:
        value: Integer representing molecule type.

    Returns:
        The corresponding Molecule enum value.

    Raises:
        ValueError: If value doesn't correspond to a known molecule type.
    """
    try:
        return Molecule(value)
    except ValueError as e:
        raise ValueError(f"Unknown molecule type value: {value}") from e
