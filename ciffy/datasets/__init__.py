"""
Dataset downloading and management utilities.

Provides tools for downloading molecular structure datasets from public
databases like RCSB PDB.

Example:
    >>> from ciffy.datasets import search_structures, download_structures
    >>>
    >>> # Download RNA structures
    >>> result = download_rna_dataset("data/rna/", max_resolution=3.0, max_count=100)
    >>>
    >>> # Or search and download any polymer type
    >>> pdb_ids = search_structures(polymer_types=["RNA", "DNA"], max_resolution=2.5)
    >>> result = download_structures(pdb_ids[:50], "data/nucleic_acids/")
"""

from .pdb import (
    DownloadResult,
    EXPERIMENTAL_METHODS,
    POLYMER_TYPES,
    search_structures,
    search_rna_structures,
    download_structure,
    download_structures,
    download_rna_dataset,
    download_cli,
)

__all__ = [
    "DownloadResult",
    "EXPERIMENTAL_METHODS",
    "POLYMER_TYPES",
    "search_structures",
    "search_rna_structures",
    "download_structure",
    "download_structures",
    "download_rna_dataset",
    "download_cli",
]
