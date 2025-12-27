"""
Name conversion utilities for code generation.

Converts between CIF atom/residue names and valid Python identifiers.
CIF uses prime notation (O3', C5') which cannot appear in Python identifiers,
so we convert: ' -> p (e.g., O3' -> O3p).

The reverse mapping for backbone atoms is defined in config.BACKBONE_PYTHON_TO_CIF,
enabling round-trip conversion when generating C code that needs original CIF names.

Functions:
    clean_atom_name: Remove quotes from CIF atom names
    sanitize_identifier: Convert special chars to Python-safe equivalents
    to_class_name: Convert residue ID to class name (e.g., "5MU" -> "X5MU")
    to_python_name: Convert atom name to Python identifier
"""

from __future__ import annotations


def clean_atom_name(name: str) -> str:
    """Remove outer double quotes from CIF atom names."""
    if name.startswith('"') and name.endswith('"'):
        return name[1:-1]
    return name


def sanitize_identifier(name: str) -> str:
    """
    Apply common substitutions to make a string a valid Python identifier.

    CIF atom names use prime notation (O3', C5', etc.) which is not valid
    in Python identifiers. This function converts them to Python-safe names.

    Replacements:
        ' -> p  (apostrophe/prime, e.g., O3' -> O3p, C5' -> C5p)
        " -> "" (remove quotes)
        * -> s  (star, e.g., HN* -> HNs)

    The reverse mapping (Python -> CIF) is defined in config.BACKBONE_PYTHON_TO_CIF
    for backbone atoms. For arbitrary atoms, replace 'p' suffix with apostrophe.

    Does NOT handle leading digits (caller should check).
    """
    return name.replace("'", "p").replace('"', "").replace("*", "s")


def to_class_name(comp_id: str) -> str:
    """
    Convert CCD component ID to Python class name (UPPERCASE).

    Uses uppercase to match biochemistry convention where residue codes
    are always uppercase (e.g., ALA, CCC, PSU).

    Examples:
        "A" -> "A"
        "5MU" -> "X5MU"
        "ALA" -> "ALA"
    """
    name = sanitize_identifier(comp_id).replace("-", "_").replace("+", "PLUS")
    if name[0].isdigit():
        name = "X" + name
    return name.upper()


def to_python_name(cif_name: str) -> str:
    """
    Convert CIF atom name to valid Python identifier.

    Examples:
        "O3'" -> "O3p"
        "HN*" -> "HNs"
        "1H2" -> "X1H2"
    """
    name = sanitize_identifier(cif_name)
    if name and name[0].isdigit():
        name = "X" + name
    return name
