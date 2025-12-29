"""
Premade dataset definitions for common benchmarks.

Contains PDB ID lists for:
- CASP15 (2022): Protein and RNA structure prediction targets
- CASP16 (2024): Protein and RNA structure prediction targets

Use --type rna/protein to filter by molecule type when downloading.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DatasetPreset:
    """Definition of a premade dataset.

    Attributes:
        name: Short identifier for the dataset.
        description: Human-readable description.
        pdb_ids: List of PDB IDs in the dataset.
        url: URL with more information about the dataset.
    """

    name: str
    description: str
    pdb_ids: list[str]
    url: str


# CASP15 targets (2022)
# Source: https://predictioncenter.org/casp15/targetlist.cgi
CASP15 = DatasetPreset(
    name="casp15",
    description="CASP15 (2022) structure prediction targets",
    url="https://predictioncenter.org/casp15/targetlist.cgi",
    pdb_ids=[
        "7ROA", "7QIH", "7QR4", "7QR3", "8ORK", "7UYX", "7UTD", "8S95",
        "8FZA", "8B43", "7SQ4", "7QVB", "7TIL", "8BBT", "7UZT", "7UX8",
        "8H2N", "8TVZ", "8BTZ", "8XBP", "8A8C", "8ECX", "8DYS", "7UBZ",
        "7Z8Y", "7ZJ4", "8FEF", "7PTK", "7PTL", "9ERT", "9ERU", "7UWW",
        "9ERW", "9ETJ", "8EM5", "9ETL", "8UYS", "8D5V", "7R1L", "8ZCX",
        "8PBV", "8PKO", "8SXA", "7PZT", "8SX8", "8JVN", "8SXB", "8SX7",
        "8SWN", "8FJP", "7PBR", "7PBL", "7PBP", "8ON4", "8OND", "8SMQ",
        "8UFN", "8TN8", "8IFX", "8OUY", "8AD2", "8C6Z", "7YR7", "7YR6",
        "8RJW", "8OKH",
    ],
)

# CASP16 targets (2024)
# Source: https://predictioncenter.org/casp16/targetlist.cgi
CASP16 = DatasetPreset(
    name="casp16",
    description="CASP16 (2024) structure prediction targets",
    url="https://predictioncenter.org/casp16/targetlist.cgi",
    pdb_ids=[
        "8BWD", "8BWL", "8UO6", "8VYL", "9CFN", "9C2K", "9ENR", "9DCF",
        "9C4O", "9B0L", "9CP0", "9NB9", "9FYA", "9FYG", "9ELY", "9DXD",
        "9CQD", "9CQB", "9CQA", "9CBU", "9CBX", "9CN2", "9CBN", "9H1G",
        "9N8W", "9BZC", "9BZ1", "8VQV", "8VVJ", "9C3C", "9HIO", "9CI3",
        "8QPQ", "8QFB", "9GVQ", "9C75", "9ISV", "9J3R", "9J3T", "9MCW",
        "9MEE", "9J6Y", "9F5M", "9SFA", "9FJL",
    ],
)

# Registry of all available presets
PRESETS: dict[str, DatasetPreset] = {
    "casp15": CASP15,
    "casp16": CASP16,
}


def get_preset(name: str) -> DatasetPreset:
    """Get a dataset preset by name.

    Args:
        name: Name of the preset (case-insensitive).

    Returns:
        The DatasetPreset object.

    Raises:
        ValueError: If the preset name is not found.
    """
    key = name.lower()
    if key not in PRESETS:
        available = ", ".join(PRESETS.keys())
        raise ValueError(f"Unknown preset '{name}'. Available: {available}")
    return PRESETS[key]


def list_presets() -> list[str]:
    """List all available preset names."""
    return list(PRESETS.keys())


__all__ = [
    "DatasetPreset",
    "PRESETS",
    "CASP15",
    "CASP16",
    "get_preset",
    "list_presets",
]
