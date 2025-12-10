"""
Nucleotide and amino acid atom definitions.

This module re-exports atom enums from the auto-generated file.
To modify atoms, edit ciffy/biochemistry/atoms.py and run:
    python ciffy/src/codegen/generate.py

The atom definitions (as simple lists) are in atoms.py.
The generated IndexEnum classes are in _generated_atoms.py.
"""

# Re-export everything from the generated file
from ._generated_atoms import (
    # Nucleotides
    Adenosine,
    Cytosine,
    Guanosine,
    Uridine,
    GuanosineTriphosphate,
    CytidineTriphosphate,
    Deoxyguanosine,
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
    ModifiedNucleotides,
    AminoAcids,
)

__all__ = [
    # Nucleotides
    "Adenosine",
    "Cytosine",
    "Guanosine",
    "Uridine",
    "GuanosineTriphosphate",
    "CytidineTriphosphate",
    "Deoxyguanosine",
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
    "ModifiedNucleotides",
    "AminoAcids",
]
