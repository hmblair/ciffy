"""
Scale enumeration for hierarchical structure levels.
"""

from enum import Enum


class Scale(Enum):
    """
    Hierarchical levels in a molecular structure.

    Defines the granularity at which operations can be performed:
    - ATOM: Individual atoms
    - RESIDUE: Amino acids or nucleotides
    - CHAIN: Polymer chains (e.g., RNA strands)
    - MOLECULE: Complete molecular assemblies

    Operations like reduce(), expand(), and center() accept a Scale to
    specify at which level to aggregate or distribute values.
    """

    ATOM = 0
    RESIDUE = 1
    CHAIN = 2
    MOLECULE = 3
