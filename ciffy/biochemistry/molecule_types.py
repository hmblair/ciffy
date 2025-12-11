"""
Molecule type mappings for CIF entity fields.

Maps mmCIF entity type strings to Molecule enum values.
Used by the gperf code generator to create hash tables for fast lookup.

Sources:
- _entity.type: polymer, non-polymer, water, branched
- _entity_poly.type: polyribonucleotide, polypeptide(L), etc.
- _pdbx_entity_nonpoly.comp_id: ion identification
"""

from ..types import Molecule


# Maps _entity.type strings to Molecule enum values
# See: https://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v50.dic/Items/_entity.type.html
ENTITY_TYPES: dict[str, int] = {
    "water": Molecule.WATER.value,
    "branched": Molecule.POLYSACCHARIDE.value,
    "non-polymer": Molecule.LIGAND.value,  # Default; may be refined to ION
    # "polymer" is handled by _entity_poly lookup
}


# Known ion component IDs from _pdbx_entity_nonpoly.comp_id
# Used to distinguish ION from LIGAND for non-polymer entities
ION_COMP_IDS: set[str] = {
    # Alkali metals
    "NA", "K", "LI", "RB", "CS",
    # Alkaline earth metals
    "MG", "CA", "SR", "BA",
    # Transition metals (common in proteins)
    "ZN", "FE", "FE2", "CU", "CU1", "MN", "CO", "NI", "CD",
    # Other metals
    "AL", "PB", "HG",
    # Halide ions
    "CL", "BR", "F", "IOD",
}


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

__all__ = ["ENTITY_TYPES", "ENTITY_POLY_TYPES", "ION_COMP_IDS"]
