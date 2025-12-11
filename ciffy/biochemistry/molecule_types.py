"""
Molecule type mappings for CIF entity_poly.type field.

Maps the _entity_poly.type strings from mmCIF files to Molecule enum values.
Used by the gperf code generator to create hash tables for fast lookup.
"""

from ..types import Molecule

# Maps _entity_poly.type strings to Molecule enum values
# These are the standard wwPDB entity types
# See: https://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v50.dic/Items/_entity_poly.type.html
ENTITY_POLY_TYPES: dict[str, int] = {
    # RNA
    "polyribonucleotide": Molecule.RNA.value,

    # DNA
    "polydeoxyribonucleotide": Molecule.DNA.value,

    # DNA/RNA hybrid
    "polydeoxyribonucleotide/polyribonucleotide hybrid": Molecule.HYBRID.value,

    # Protein variants
    "polypeptide(L)": Molecule.PROTEIN.value,
    "polypeptide(D)": Molecule.PROTEIN_D.value,

    # Cyclic peptides
    "cyclic-pseudo-peptide": Molecule.CYCLIC_PEPTIDE.value,

    # Hybrid peptides (L and D mixed) - treat as standard protein
    "polypeptide(D)-polypeptide(L)": Molecule.PROTEIN.value,

    # Polysaccharides (carbohydrates)
    "polysaccharide(D)": Molecule.POLYSACCHARIDE.value,
    "polysaccharide(L)": Molecule.POLYSACCHARIDE.value,

    # Peptide nucleic acid
    "peptide nucleic acid": Molecule.PNA.value,

    # Fallback
    "other": Molecule.OTHER.value,
}

__all__ = ["ENTITY_POLY_TYPES"]
