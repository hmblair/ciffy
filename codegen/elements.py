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


class ElementsParseError(Exception):
    """Error parsing the PubChem elements CSV."""
    pass


def parse_elements_csv(csv_path: Path) -> dict[str, int]:
    """
    Parse PubChem periodic table CSV.

    Returns a dictionary mapping element symbol (uppercase) to atomic number.

    Args:
        csv_path: Path to the PubChem periodic table CSV file.

    Returns:
        Dictionary mapping element symbol to atomic number.
        Example: {"H": 1, "HE": 2, "LI": 3, ...}

    Raises:
        ElementsParseError: If the file cannot be opened or has unexpected format.
    """
    elements: dict[str, int] = {}

    try:
        f = open(csv_path, 'r', encoding='utf-8')
    except FileNotFoundError:
        raise ElementsParseError(
            f"Elements CSV not found: {csv_path}\n"
            f"Run 'python -m codegen.cli --download' to download it, or download manually from:\n"
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/periodictable/CSV"
        ) from None
    except OSError as e:
        raise ElementsParseError(f"Error opening elements file {csv_path}: {e}") from None

    with f:
        reader = csv.DictReader(f)

        # Verify expected columns exist
        if reader.fieldnames is None:
            raise ElementsParseError(
                f"Elements CSV appears empty or malformed: {csv_path}"
            )

        required_cols = {'Symbol', 'AtomicNumber'}
        missing = required_cols - set(reader.fieldnames)
        if missing:
            raise ElementsParseError(
                f"Elements CSV missing required columns: {missing}\n"
                f"Found columns: {reader.fieldnames}\n"
                f"The PubChem CSV format may have changed. File: {csv_path}"
            )

        for line_num, row in enumerate(reader, start=2):  # start=2 for 1-indexed + header
            try:
                symbol = row['Symbol'].upper()
                atomic_number = int(row['AtomicNumber'])
                elements[symbol] = atomic_number
            except (KeyError, ValueError, TypeError) as e:
                raise ElementsParseError(
                    f"Error parsing line {line_num} of elements CSV: {e}\n"
                    f"Row data: {row}"
                ) from None

    if not elements:
        raise ElementsParseError(f"No elements found in CSV file: {csv_path}")

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
