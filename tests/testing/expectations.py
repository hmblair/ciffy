"""Helpers for deriving expected values in tests.

Instead of hardcoding expected values like [Residue.A.value, Residue.C.value, ...],
use these helpers to derive them from input strings.
"""

from typing import List


def expected_sequence_values(sequence_str: str) -> List[int]:
    """Derive expected sequence values from a sequence string.

    Args:
        sequence_str: Sequence like "acgu" (RNA), "MGKLF" (protein), or "acgt" (DNA)

    Returns:
        List of Residue enum values

    Example:
        >>> expected_sequence_values("acgu")
        [0, 1, 2, 3]  # Residue.A.value, Residue.C.value, Residue.G.value, Residue.U.value
    """
    from ciffy.biochemistry import Residue

    if not sequence_str:
        return []

    # Determine molecule type from case
    is_lowercase = sequence_str[0].islower()

    # Check for DNA vs RNA in lowercase sequences
    is_dna = is_lowercase and "t" in sequence_str.lower()

    values = []
    for char in sequence_str:
        if is_lowercase:
            if is_dna:
                # DNA
                mapping = {
                    "a": Residue.DA,
                    "c": Residue.DC,
                    "g": Residue.DG,
                    "t": Residue.DT,
                }
            else:
                # RNA
                mapping = {
                    "a": Residue.A,
                    "c": Residue.C,
                    "g": Residue.G,
                    "u": Residue.U,
                }
            values.append(mapping[char].value)
        else:
            # Protein - one-letter codes
            mapping = {
                "A": Residue.ALA,
                "C": Residue.CYS,
                "D": Residue.ASP,
                "E": Residue.GLU,
                "F": Residue.PHE,
                "G": Residue.GLY,
                "H": Residue.HIS,
                "I": Residue.ILE,
                "K": Residue.LYS,
                "L": Residue.LEU,
                "M": Residue.MET,
                "N": Residue.ASN,
                "P": Residue.PRO,
                "Q": Residue.GLN,
                "R": Residue.ARG,
                "S": Residue.SER,
                "T": Residue.THR,
                "V": Residue.VAL,
                "W": Residue.TRP,
                "Y": Residue.TYR,
            }
            values.append(mapping[char].value)

    return values


def assert_sequence_matches(polymer, sequence_str: str) -> None:
    """Assert polymer sequence matches the expected string.

    This derives expected values from the string rather than hardcoding.

    Args:
        polymer: Polymer to check
        sequence_str: Expected sequence string (e.g., "acgu", "MGKLF")

    Raises:
        AssertionError: If sequence doesn't match
    """
    expected = expected_sequence_values(sequence_str)
    actual = list(polymer.sequence)

    assert actual == expected, (
        f"Sequence mismatch for '{sequence_str}':\n"
        f"  expected: {expected}\n"
        f"  actual:   {actual}"
    )
