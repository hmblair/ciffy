"""
Biochemistry constants for structure analysis.

Defines atom groupings for backbone, nucleobase, phosphate, and sidechain atoms.
"""

from typing import Callable
from ..utils import IndexEnum
from .nucleotides import (
    Adenosine, Cytosine, Guanosine, Uridine,
    Glycine, Alanine, Valine, Leucine, Isoleucine, Proline,
    Phenylalanine, Tryptophan, Methionine, Cysteine, Serine, Threonine,
    Asparagine, Glutamine, AsparticAcid, GlutamicAcid, Lysine, Arginine,
    Histidine, Tyrosine,
)


def _filter_nucleotide_atoms(predicate: Callable[[str], bool]) -> dict[str, int]:
    """
    Filter nucleotide atoms across all four bases using a predicate.

    Args:
        predicate: Function that takes an atom name and returns True to include.

    Returns:
        Dictionary mapping prefixed atom names to their indices.
    """
    result = {}
    nucleotides = [
        ("A_", Adenosine),
        ("C_", Cytosine),
        ("G_", Guanosine),
        ("U_", Uridine),
    ]
    for prefix, nucleotide in nucleotides:
        for name, value in nucleotide.dict().items():
            if predicate(name):
                result[prefix + name] = value
    return result


def _filter_amino_acid_atoms(predicate: Callable[[str], bool]) -> dict[str, int]:
    """
    Filter amino acid atoms across all 20 residues using a predicate.

    Args:
        predicate: Function that takes an atom name and returns True to include.

    Returns:
        Dictionary mapping prefixed atom names to their indices.
    """
    result = {}
    amino_acids = [
        ("G_", Glycine), ("A_", Alanine), ("V_", Valine), ("L_", Leucine),
        ("I_", Isoleucine), ("P_", Proline), ("F_", Phenylalanine),
        ("W_", Tryptophan), ("M_", Methionine), ("C_", Cysteine),
        ("S_", Serine), ("T_", Threonine), ("N_", Asparagine),
        ("Q_", Glutamine), ("D_", AsparticAcid), ("E_", GlutamicAcid),
        ("K_", Lysine), ("R_", Arginine), ("H_", Histidine), ("Y_", Tyrosine),
    ]
    for prefix, amino_acid in amino_acids:
        for name, value in amino_acid.dict().items():
            if predicate(name):
                result[prefix + name] = value
    return result


# Protein backbone atoms
_PROTEIN_BACKBONE = {'N', 'CA', 'C', 'O', 'OXT', 'H', 'H2', 'H3', 'HA', 'HA2', 'HA3'}

# Backbone atoms: contain 'p' (sugar) or 'P' (phosphate)
Backbone = IndexEnum(
    "Backbone",
    _filter_nucleotide_atoms(lambda n: 'p' in n or 'P' in n)
)

# Nucleobase atoms: neither 'p' nor 'P'
Nucleobase = IndexEnum(
    "Nucleobase",
    _filter_nucleotide_atoms(lambda n: 'p' not in n and 'P' not in n)
)

# Phosphate atoms: contain uppercase 'P'
Phosphate = IndexEnum(
    "Phosphate",
    _filter_nucleotide_atoms(lambda n: 'P' in n)
)

# Sidechain atoms: amino acid atoms not in backbone
Sidechain = IndexEnum(
    "Sidechain",
    _filter_amino_acid_atoms(lambda n: n not in _PROTEIN_BACKBONE)
)
