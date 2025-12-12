"""
Nucleotide and amino acid atom definitions.

This module re-exports atom enums from the auto-generated file.
To modify atoms, edit ciffy/biochemistry/definitions.py and run:
    python ciffy/src/codegen/generate.py
"""

# Re-export everything from the generated file
from ._generated_atoms import (
    # RNA nucleotides
    Adenosine,
    Cytosine,
    Guanosine,
    Uridine,
    # DNA nucleotides
    Deoxyadenosine,
    Deoxycytidine,
    Deoxyguanosine,
    Thymidine,
    # Modified nucleotides
    GuanosineTriphosphate,
    CytidineTriphosphate,
    Deoxyguanosine2Prime,
    # Amino acids
    Glycine,
    Alanine,
    Valine,
    Leucine,
    Isoleucine,
    Proline,
    Phenylalanine,
    Tryptophan,
    Methionine,
    Cysteine,
    Serine,
    Threonine,
    Asparagine,
    Glutamine,
    AsparticAcid,
    GlutamicAcid,
    Lysine,
    Arginine,
    Histidine,
    Tyrosine,
    # Combined enums
    RibonucleicAcid,
    RibonucleicAcidNoPrefix,
    DeoxyribonucleicAcid,
    ModifiedNucleotides,
    AminoAcids,
)

__all__ = [
    # RNA nucleotides
    "Adenosine",
    "Cytosine",
    "Guanosine",
    "Uridine",
    # DNA nucleotides
    "Deoxyadenosine",
    "Deoxycytidine",
    "Deoxyguanosine",
    "Thymidine",
    # Modified nucleotides
    "GuanosineTriphosphate",
    "CytidineTriphosphate",
    "Deoxyguanosine2Prime",
    # Amino acids
    "Glycine",
    "Alanine",
    "Valine",
    "Leucine",
    "Isoleucine",
    "Proline",
    "Phenylalanine",
    "Tryptophan",
    "Methionine",
    "Cysteine",
    "Serine",
    "Threonine",
    "Asparagine",
    "Glutamine",
    "AsparticAcid",
    "GlutamicAcid",
    "Lysine",
    "Arginine",
    "Histidine",
    "Tyrosine",
    # Combined enums
    "RibonucleicAcid",
    "RibonucleicAcidNoPrefix",
    "DeoxyribonucleicAcid",
    "ModifiedNucleotides",
    "AminoAcids",
]
