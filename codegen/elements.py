"""
Parse element data from PubChem periodic table.

This module parses the PubChem periodic table CSV to extract element
symbols and atomic numbers, replacing the hard-coded ELEMENTS dictionary.

Data source: https://pubchem.ncbi.nlm.nih.gov/rest/pug/periodictable/CSV
License: Public Domain (US Government work)
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Iterator


def parse_elements_csv(csv_path: Path) -> dict[str, int]:
    """
    Parse PubChem periodic table CSV.

    Returns a dictionary mapping element symbol (uppercase) to atomic number.

    Args:
        csv_path: Path to the PubChem periodic table CSV file.

    Returns:
        Dictionary mapping element symbol to atomic number.
        Example: {"H": 1, "HE": 2, "LI": 3, ...}
    """
    elements: dict[str, int] = {}

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            symbol = row['Symbol'].upper()
            atomic_number = int(row['AtomicNumber'])
            elements[symbol] = atomic_number

    return elements


def load_elements() -> dict[str, int]:
    """
    Load elements from PubChem, downloading if necessary.

    Returns:
        Dictionary mapping element symbol to atomic number.
    """
    from .cli import get_elements_path

    elements_path = get_elements_path()
    return parse_elements_csv(elements_path)


# For convenience, expose a function to get all element symbols
def get_element_symbols() -> list[str]:
    """Get list of all element symbols in order of atomic number."""
    elements = load_elements()
    return sorted(elements.keys(), key=lambda s: elements[s])
