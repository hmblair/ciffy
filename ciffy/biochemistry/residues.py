"""
Residue definitions for nucleotides and amino acids.

This module re-exports from the auto-generated file.
To modify residues, edit ciffy/biochemistry/definitions.py and run:
    python ciffy/src/codegen/generate.py
"""

# Re-export everything from the generated file
from ._generated_residues import (
    Residue,
    RESIDUE_MOLECULE_TYPE,
    CIF_RESIDUE_NAMES,
    RESIDUE_CIF_NAMES,
    RESIDUE_ABBREV,
)

__all__ = [
    "Residue",
    "RESIDUE_MOLECULE_TYPE",
    "CIF_RESIDUE_NAMES",
    "RESIDUE_CIF_NAMES",
    "RESIDUE_ABBREV",
]
