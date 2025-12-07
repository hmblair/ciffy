"""
Molecule type enumeration.
"""

from enum import Enum


class Molecule(Enum):
    """
    Types of molecules that can appear in a structure.

    Used to classify chains by their molecular type, enabling
    filtering and type-specific operations.
    """

    PROTEIN = 0
    RNA = 1
    DNA = 2
    WATER = 3
    ION = 4
    OTHER = 5
    MISSING = 6


def molecule_type(value: int) -> Molecule:
    """
    Convert an integer value to the corresponding Molecule type.

    Args:
        value: Integer representing molecule type from CIF parsing.

    Returns:
        The corresponding Molecule enum value.

    Raises:
        ValueError: If value doesn't correspond to a known molecule type.
    """
    mapping = {
        0: Molecule.PROTEIN,
        1: Molecule.RNA,
        2: Molecule.DNA,
        3: Molecule.OTHER,
        4: Molecule.MISSING,
    }
    if value in mapping:
        return mapping[value]
    raise ValueError(f"Unknown molecule type value: {value}")
