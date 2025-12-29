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
    >>>
    >>> # Use preset datasets (CASP15, CASP16)
    >>> from ciffy.datasets import CASP15, CASP16
    >>> result = download_structures(CASP15.pdb_ids, "data/casp15/")
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
from .presets import (
    DatasetPreset,
    PRESETS,
    CASP15,
    CASP16,
    get_preset,
    list_presets,
)

__all__ = [
    # Download functions
    "DownloadResult",
    "EXPERIMENTAL_METHODS",
    "POLYMER_TYPES",
    "search_structures",
    "search_rna_structures",
    "download_structure",
    "download_structures",
    "download_rna_dataset",
    "download_cli",
    # Presets
    "DatasetPreset",
    "PRESETS",
    "CASP15",
    "CASP16",
    "get_preset",
    "list_presets",
]
