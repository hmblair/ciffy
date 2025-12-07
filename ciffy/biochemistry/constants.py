"""
Biochemistry constants for RNA structure analysis.

Defines atom groupings for backbone, nucleobase, and phosphate atoms,
as well as reference frames for structural alignment.
"""

from typing import Callable
import torch
from ..utils import IndexEnum
from .nucleotides import Adenosine, Cytosine, Guanosine, Uridine


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

# Coarse-grained representation: N1/N3 atoms for base pairing
COARSE = torch.tensor([
    Adenosine.N1.value,
    Cytosine.N3.value,
    Guanosine.N1.value,
    Uridine.N3.value,
])

# Reference frame atoms for structural alignment
FRAME1 = torch.tensor([
    Adenosine.C2.value,
    Cytosine.C2.value,
    Guanosine.C2.value,
    Uridine.C2.value,
])

FRAME2 = torch.tensor([
    Adenosine.C4.value,
    Cytosine.C4.value,
    Guanosine.C4.value,
    Uridine.C4.value,
])

FRAME3 = torch.tensor([
    Adenosine.C6.value,
    Cytosine.C6.value,
    Guanosine.C6.value,
    Uridine.C6.value,
])

FRAMES = torch.cat([FRAME1, FRAME2, FRAME3])
