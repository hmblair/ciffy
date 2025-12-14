"""
Name conversion utilities for code generation.

Provides functions to convert CIF names to valid Python identifiers.
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

    Replacements:
        ' -> p  (apostrophe, e.g., O3' -> O3p)
        ` -> p  (backtick)
        " -> "" (remove quotes)
        * -> s  (star, e.g., HN* -> HNs)

    Does NOT handle leading digits (caller should check).
    """
    return name.replace("'", "p").replace("`", "p").replace('"', "").replace("*", "s")


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
