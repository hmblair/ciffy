"""
RCSB PDB structure search and download utilities.

Provides functions for querying the RCSB PDB database and downloading
mmCIF files for RNA-containing structures.

Example:
    >>> from ciffy.datasets import search_rna_structures, download_structures
    >>>
    >>> # Find all RNA structures with resolution < 3.0 Å
    >>> pdb_ids = search_rna_structures(max_resolution=3.0)
    >>> print(f"Found {len(pdb_ids)} structures")
    >>>
    >>> # Download first 100
    >>> result = download_structures(pdb_ids[:100], output_dir="data/rna/")
    >>> print(f"Downloaded {len(result.downloaded)} structures")
"""

from __future__ import annotations

import json
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Callable  # Literal used for experimental_method
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

from ciffy.utils.formatting import Colors

logger = logging.getLogger(__name__)


def _colored_count(count: int) -> str:
    """Format a count with color (green if > 0, red if 0)."""
    if count > 0:
        return f"{Colors.GREEN}{count}{Colors.RESET}"
    else:
        return f"{Colors.RED}{count}{Colors.RESET}"

# RCSB API endpoints
RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_DOWNLOAD_URL = "https://files.rcsb.org/download"

# Experimental method mapping (short name -> RCSB value)
EXPERIMENTAL_METHODS = {
    "xray": "X-RAY DIFFRACTION",
    "em": "ELECTRON MICROSCOPY",
    "nmr": "SOLUTION NMR",
    "neutron": "NEUTRON DIFFRACTION",
}

# Polymer type mapping (short name -> RCSB value)
POLYMER_TYPES = {
    "rna": "RNA",
    "dna": "DNA",
    "protein": "Protein",
    "hybrid": "NA-hybrid",  # DNA/RNA hybrid
    "other": "Other",
}


@dataclass
class DownloadResult:
    """Result of a batch download operation.

    Attributes:
        downloaded: List of PDB IDs that were successfully downloaded.
        skipped: List of PDB IDs that were skipped (already exist).
        failed: List of (pdb_id, error_message) tuples for failed downloads.
        total_bytes: Total bytes downloaded.
    """

    downloaded: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    total_bytes: int = 0

    @property
    def total_mb(self) -> float:
        """Total megabytes downloaded."""
        return self.total_bytes / (1024 * 1024)

    def summary(self) -> str:
        """Return a summary string."""
        parts = [f"Downloaded: {len(self.downloaded)}"]
        if self.skipped:
            parts.append(f"Skipped: {len(self.skipped)}")
        if self.failed:
            parts.append(f"Failed: {len(self.failed)}")
        parts.append(f"Size: {self.total_mb:.1f} MB")
        return ", ".join(parts)

    def __repr__(self) -> str:
        return f"DownloadResult({self.summary()})"


def _build_search_query(
    polymer_types: list[str] | None = None,
    min_resolution: float | None = None,
    max_resolution: float | None = None,
    min_length: int | None = None,
    max_length: int | None = None,
    experimental_method: str | None = None,
    released_after: str | None = None,
    released_before: str | None = None,
) -> dict:
    """Build RCSB search query for structures containing specified polymer types.

    Args:
        polymer_types: List of polymer types (e.g., ["RNA", "DNA", "Protein"]).
            If None, no polymer type filter is applied.
        released_after: Only include structures released after this date (YYYY-MM-DD).
        released_before: Only include structures released before this date (YYYY-MM-DD).
    """
    nodes = []

    # Polymer type filter
    if polymer_types:
        if len(polymer_types) == 1:
            # Single type: exact match
            nodes.append({
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "entity_poly.rcsb_entity_polymer_type",
                    "operator": "exact_match",
                    "value": polymer_types[0],
                },
            })
        else:
            # Multiple types: OR group
            type_nodes = [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "entity_poly.rcsb_entity_polymer_type",
                        "operator": "exact_match",
                        "value": ptype,
                    },
                }
                for ptype in polymer_types
            ]
            nodes.append({
                "type": "group",
                "logical_operator": "or",
                "nodes": type_nodes,
            })

    # Resolution filter
    if min_resolution is not None or max_resolution is not None:
        resolution_node = {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_entry_info.resolution_combined",
                "operator": "range",
                "value": {
                    "from": min_resolution or 0,
                    "to": max_resolution or 100,
                    "include_lower": True,
                    "include_upper": True,
                },
            },
        }
        nodes.append(resolution_node)

    # Polymer length filter
    if min_length is not None or max_length is not None:
        length_node = {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "entity_poly.rcsb_sample_sequence_length",
                "operator": "range",
                "value": {
                    "from": min_length or 1,
                    "to": max_length or 100000,
                    "include_lower": True,
                    "include_upper": True,
                },
            },
        }
        nodes.append(length_node)

    # Experimental method filter
    if experimental_method is not None:
        method_node = {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "exptl.method",
                "operator": "exact_match",
                "value": experimental_method,
            },
        }
        nodes.append(method_node)

    # Release date filters
    if released_after is not None:
        nodes.append({
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_accession_info.initial_release_date",
                "operator": "greater",
                "value": released_after,
            },
        })

    if released_before is not None:
        nodes.append({
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_accession_info.initial_release_date",
                "operator": "less",
                "value": released_before,
            },
        })

    # Combine with AND
    if len(nodes) == 1:
        query = nodes[0]
    else:
        query = {
            "type": "group",
            "logical_operator": "and",
            "nodes": nodes,
        }

    return {
        "query": query,
        "return_type": "entry",
        "request_options": {
            "return_all_hits": True,
            "sort": [
                {
                    "sort_by": "rcsb_accession_info.deposit_date",
                    "direction": "desc",
                }
            ],
        },
    }


def search_structures(
    polymer_types: list[str] | str | None = None,
    min_resolution: float | None = None,
    max_resolution: float | None = None,
    min_length: int | None = None,
    max_length: int | None = None,
    experimental_method: Literal[
        "X-RAY DIFFRACTION",
        "ELECTRON MICROSCOPY",
        "SOLUTION NMR",
        "NEUTRON DIFFRACTION",
    ]
    | None = None,
    released_after: str | None = None,
    released_before: str | None = None,
    timeout: float = 60.0,
) -> list[str]:
    """
    Search RCSB PDB for structures containing specified polymer types.

    Queries the RCSB search API for structures containing the specified
    polymer entities. Returns PDB IDs sorted by deposit date (newest first).

    Args:
        polymer_types: Polymer type(s) to search for. Can be a single type
            or a list of types. Valid types: "RNA", "DNA", "Protein",
            "NA-hybrid", "Other". If None, no polymer type filter is applied.
        min_resolution: Minimum resolution in Ångströms (inclusive).
        max_resolution: Maximum resolution in Ångströms (inclusive).
        min_length: Minimum polymer length.
        max_length: Maximum polymer length.
        experimental_method: Filter by experimental method.
        released_after: Only include structures released after this date (YYYY-MM-DD).
        released_before: Only include structures released before this date (YYYY-MM-DD).
        timeout: Request timeout in seconds.

    Returns:
        List of PDB IDs matching the query.

    Raises:
        URLError: If the search request fails.
        ValueError: If the response cannot be parsed.

    Example:
        >>> # Find high-resolution RNA structures
        >>> pdb_ids = search_structures(
        ...     polymer_types="RNA",
        ...     max_resolution=2.5,
        ... )
        >>> print(f"Found {len(pdb_ids)} structures")

        >>> # Find structures released in 2024
        >>> pdb_ids = search_structures(
        ...     released_after="2024-01-01",
        ...     released_before="2025-01-01",
        ... )
    """
    # Normalize polymer_types to list
    if polymer_types is None:
        types_list = None
    elif isinstance(polymer_types, str):
        types_list = [polymer_types]
    else:
        types_list = list(polymer_types)

    query = _build_search_query(
        polymer_types=types_list,
        min_resolution=min_resolution,
        max_resolution=max_resolution,
        min_length=min_length,
        max_length=max_length,
        experimental_method=experimental_method,
        released_after=released_after,
        released_before=released_before,
    )

    query_json = json.dumps(query).encode("utf-8")

    request = Request(
        RCSB_SEARCH_URL,
        data=query_json,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    type_desc = ", ".join(types_list) if types_list else "all"
    logger.info(f"Searching RCSB PDB for {type_desc} structures...")

    try:
        with urlopen(request, timeout=timeout) as response:
            response_body = response.read().decode("utf-8")
            # Handle empty response (no results)
            if not response_body.strip():
                logger.info("Found 0 structures")
                return []
            data = json.loads(response_body)
    except HTTPError as e:
        # 204 No Content means no results found
        if e.code == 204:
            logger.info("Found 0 structures")
            return []
        raise URLError(f"RCSB search failed with status {e.code}: {e.reason}") from e

    # Extract PDB IDs from results
    pdb_ids = []
    for result in data.get("result_set", []):
        pdb_id = result.get("identifier")
        if pdb_id:
            pdb_ids.append(pdb_id.upper())

    logger.info(f"Found {len(pdb_ids)} structures")
    return pdb_ids


def search_rna_structures(
    min_resolution: float | None = None,
    max_resolution: float | None = None,
    min_length: int | None = None,
    max_length: int | None = None,
    experimental_method: Literal[
        "X-RAY DIFFRACTION",
        "ELECTRON MICROSCOPY",
        "SOLUTION NMR",
        "NEUTRON DIFFRACTION",
    ]
    | None = None,
    timeout: float = 60.0,
) -> list[str]:
    """
    Search RCSB PDB for RNA-containing structures.

    Convenience wrapper around search_structures() for RNA.

    Args:
        min_resolution: Minimum resolution in Ångströms (inclusive).
        max_resolution: Maximum resolution in Ångströms (inclusive).
        min_length: Minimum RNA polymer length in nucleotides.
        max_length: Maximum RNA polymer length in nucleotides.
        experimental_method: Filter by experimental method.
        timeout: Request timeout in seconds.

    Returns:
        List of PDB IDs matching the query.
    """
    return search_structures(
        polymer_types="RNA",
        min_resolution=min_resolution,
        max_resolution=max_resolution,
        min_length=min_length,
        max_length=max_length,
        experimental_method=experimental_method,
        timeout=timeout,
    )


def download_structure(
    pdb_id: str,
    output_dir: Path | str,
    overwrite: bool = False,
    timeout: float = 30.0,
) -> tuple[Path | None, int]:
    """
    Download a single structure from RCSB PDB in mmCIF format.

    Args:
        pdb_id: The 4-character PDB ID.
        output_dir: Directory to save the file.
        overwrite: If True, overwrite existing files.
        timeout: Request timeout in seconds.

    Returns:
        Tuple of (output_path, bytes_downloaded). Path is None if skipped.

    Raises:
        HTTPError: If download fails.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdb_id = pdb_id.upper()
    filename = f"{pdb_id}.cif"
    output_path = output_dir / filename

    # Skip if exists
    if output_path.exists() and not overwrite:
        return None, 0

    # Download
    url = f"{RCSB_DOWNLOAD_URL}/{filename}"
    request = Request(url, headers={"User-Agent": "ciffy/1.0"})

    with urlopen(request, timeout=timeout) as response:
        content = response.read()

    output_path.write_bytes(content)
    return output_path, len(content)


def download_structures(
    pdb_ids: list[str],
    output_dir: Path | str,
    overwrite: bool = False,
    max_workers: int = 4,
    delay: float = 0.01,
    timeout: float = 30.0,
    progress_callback: Callable[[int, int, str], None] | None = None,
    queue_callback: Callable[[int, int, str], None] | None = None,
) -> DownloadResult:
    """
    Download multiple structures from RCSB PDB in mmCIF format.

    Downloads structures in parallel with rate limiting. Existing files are
    skipped unless overwrite=True.

    Args:
        pdb_ids: List of PDB IDs to download.
        output_dir: Directory to save files.
        overwrite: If True, overwrite existing files.
        max_workers: Maximum concurrent downloads.
        delay: Delay between starting downloads (rate limiting).
        timeout: Per-request timeout in seconds.
        progress_callback: Optional callback(current, total, pdb_id) for completions.
        queue_callback: Optional callback(current, total, pdb_id) for queue progress.

    Returns:
        DownloadResult with download statistics.

    Example:
        >>> pdb_ids = ["1EHZ", "1EVV", "1F7Y"]
        >>> result = download_structures(pdb_ids, "data/rna/")
        >>> print(result)
        DownloadResult(Downloaded: 3, Size: 1.2 MB)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = DownloadResult()
    total = len(pdb_ids)

    def download_one(pdb_id: str) -> tuple[str, str, int]:
        """Download single structure, return (pdb_id, status, bytes)."""
        try:
            path, nbytes = download_structure(
                pdb_id,
                output_dir,
                overwrite=overwrite,
                timeout=timeout,
            )
            if path is None:
                return pdb_id, "skipped", 0
            return pdb_id, "downloaded", nbytes
        except Exception as e:
            return pdb_id, f"failed: {e}", 0

    # Download with thread pool
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for i, pdb_id in enumerate(pdb_ids):
            future = executor.submit(download_one, pdb_id)
            futures[future] = pdb_id
            if queue_callback:
                queue_callback(i + 1, total, pdb_id)
            # Rate limiting
            if delay > 0 and i < len(pdb_ids) - 1:
                time.sleep(delay)

        # Collect results
        completed = 0
        for future in as_completed(futures):
            pdb_id, status, nbytes = future.result()
            completed += 1

            if status == "downloaded":
                result.downloaded.append(pdb_id)
                result.total_bytes += nbytes
            elif status == "skipped":
                result.skipped.append(pdb_id)
            else:
                result.failed.append((pdb_id, status))

            if progress_callback:
                progress_callback(completed, total, pdb_id)

    return result


def download_rna_dataset(
    output_dir: Path | str,
    max_resolution: float | None = None,
    min_length: int | None = None,
    max_count: int | None = None,
    overwrite: bool = False,
    max_workers: int = 4,
    progress: bool = True,
) -> DownloadResult:
    """
    Download RNA structures from RCSB PDB in mmCIF format.

    Convenience function that searches for RNA structures and downloads them.

    Args:
        output_dir: Directory to save files.
        max_resolution: Maximum resolution filter in Ångströms.
        min_length: Minimum RNA polymer length filter.
        max_count: Maximum number of structures to download.
        overwrite: If True, overwrite existing files.
        max_workers: Maximum concurrent downloads.
        progress: If True, print progress to stdout.

    Returns:
        DownloadResult with download statistics.

    Example:
        >>> # Download high-resolution RNA structures
        >>> result = download_rna_dataset(
        ...     "data/rna/",
        ...     max_resolution=3.0,
        ...     max_count=100,
        ... )
    """
    # Search for structures
    if progress:
        print("Searching RCSB PDB for RNA structures...")

    pdb_ids = search_rna_structures(
        max_resolution=max_resolution,
        min_length=min_length,
    )

    if progress:
        print(f"Found {_colored_count(len(pdb_ids))} structures")

    # Limit count
    if max_count is not None and len(pdb_ids) > max_count:
        pdb_ids = pdb_ids[:max_count]
        if progress:
            print(f"Limiting to first {max_count} structures")

    # Progress callback
    callback = None
    if progress:
        def callback(current: int, total: int, pdb_id: str):
            pct = 100 * current / total
            print(f"\r[{current}/{total}] {pct:.1f}% - {pdb_id}", end="", flush=True)

    # Download
    result = download_structures(
        pdb_ids,
        output_dir,
        overwrite=overwrite,
        max_workers=max_workers,
        progress_callback=callback,
    )

    if progress:
        print()  # Newline after progress
        print(result.summary())

    return result


def download_cli(
    pdb_ids: list[str] | None = None,
    preset: str | None = None,
    polymer_types: list[str] | None = None,
    output_dir: Path | str = ".",
    max_count: int | None = None,
    max_resolution: float | None = None,
    min_resolution: float | None = None,
    min_length: int | None = None,
    max_length: int | None = None,
    method: str | None = None,
    released_after: str | None = None,
    released_before: str | None = None,
    overwrite: bool = False,
    max_workers: int = 4,
    search_only: bool = False,
    list_ids: bool = False,
    list_presets: bool = False,
    quiet: bool = False,
) -> DownloadResult | None:
    """
    CLI-friendly PDB download with progress output.

    This is the main entry point for the `ciffy download` command.

    Args:
        pdb_ids: Specific PDB IDs to download. If None, searches for structures.
        preset: Name of a premade dataset (e.g., "casp15", "casp16-rna").
        polymer_types: Polymer type(s) to search for (e.g., ["rna", "dna"]).
            Valid types: rna, dna, protein, hybrid, other.
            If None, defaults to ["rna"] for backwards compatibility.
        output_dir: Directory to save mmCIF files.
        max_count: Maximum number of structures to download.
        max_resolution: Maximum resolution in Ångströms.
        min_resolution: Minimum resolution in Ångströms.
        min_length: Minimum polymer length.
        max_length: Maximum polymer length.
        method: Experimental method shorthand (xray, em, nmr, neutron).
        released_after: Only include structures released after this date (YYYY-MM-DD).
        released_before: Only include structures released before this date (YYYY-MM-DD).
        overwrite: Overwrite existing files.
        max_workers: Maximum concurrent downloads.
        search_only: Only search, don't download.
        list_ids: Print PDB IDs (with search_only).
        list_presets: List available preset datasets.
        quiet: Suppress progress output.

    Returns:
        DownloadResult if download occurred, None if search_only.
    """
    from .presets import PRESETS, get_preset

    # List presets mode
    if list_presets:
        print("Available dataset presets:")
        for name, ds in PRESETS.items():
            print(f"  {name:<15} {ds.description} ({len(ds.pdb_ids)} structures)")
        return None

    # If preset specified, use those PDB IDs
    if preset is not None:
        ds = get_preset(preset)
        pdb_ids = ds.pdb_ids
        print(f"Using preset: {ds.name}")
        print(f"  {ds.description}")
        print(f"  {len(pdb_ids)} structures")
        print(f"  Source: {ds.url}")
        print()

    # If specific PDB IDs provided, skip search
    if pdb_ids is not None:
        pdb_ids = [pid.upper() for pid in pdb_ids]
        print(f"Downloading {len(pdb_ids)} specified structure(s): {', '.join(pdb_ids)}")
    else:
        # Convert method shorthand to full name
        experimental_method = None
        if method:
            experimental_method = EXPERIMENTAL_METHODS.get(method)
            if experimental_method is None:
                raise ValueError(
                    f"Unknown method '{method}'. "
                    f"Choose from: {list(EXPERIMENTAL_METHODS.keys())}"
                )

        # Convert polymer type shorthands to full names
        # Default to RNA, DNA, and protein
        if polymer_types is None:
            polymer_types = ["rna", "dna", "protein"]

        rcsb_types = []
        for ptype in polymer_types:
            rcsb_type = POLYMER_TYPES.get(ptype.lower())
            if rcsb_type is None:
                raise ValueError(
                    f"Unknown polymer type '{ptype}'. "
                    f"Choose from: {list(POLYMER_TYPES.keys())}"
                )
            rcsb_types.append(rcsb_type)

        type_desc = ", ".join(rcsb_types)

        # Search for structures
        print(f"Searching RCSB PDB for {type_desc} structures...")
        if max_resolution:
            print(f"  Max resolution: {max_resolution} Å")
        if min_resolution:
            print(f"  Min resolution: {min_resolution} Å")
        if min_length:
            print(f"  Min length: {min_length}")
        if max_length:
            print(f"  Max length: {max_length}")
        if experimental_method:
            print(f"  Method: {experimental_method}")
        if released_after:
            print(f"  Released after: {released_after}")
        if released_before:
            print(f"  Released before: {released_before}")

        pdb_ids = search_structures(
            polymer_types=rcsb_types,
            min_resolution=min_resolution,
            max_resolution=max_resolution,
            min_length=min_length,
            max_length=max_length,
            experimental_method=experimental_method,
            released_after=released_after,
            released_before=released_before,
        )

        print(f"\nFound {_colored_count(len(pdb_ids))} structures")

    # Search only mode
    if search_only:
        if list_ids:
            for pdb_id in pdb_ids:
                print(pdb_id)
        return None

    # Limit count
    if max_count is not None and len(pdb_ids) > max_count:
        pdb_ids = pdb_ids[:max_count]
        print(f"Limiting to first {max_count} structures")

    if not pdb_ids:
        print("No structures to download")
        return DownloadResult()

    # Download
    output_dir = Path(output_dir)
    print(f"\nDownloading to {output_dir}/")

    if not quiet:
        from rich.progress import (
            Progress,
            SpinnerColumn,
            TextColumn,
            BarColumn,
            TaskProgressColumn,
            MofNCompleteColumn,
            TimeRemainingColumn,
        )

        progress = Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            TextColumn("[bold blue]{task.fields[current_id]}"),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
        )
        queue_task = progress.add_task(
            "Queueing", total=len(pdb_ids), current_id="..."
        )
        download_task = progress.add_task(
            "Downloading", total=len(pdb_ids), current_id="...", visible=False
        )

        progress.start()

        def queue_callback(current: int, total: int, pdb_id: str):
            progress.update(queue_task, completed=current, current_id=pdb_id)
            # Show download bar once queueing is done
            if current == total:
                progress.update(queue_task, visible=False)
                progress.update(download_task, visible=True)

        def progress_callback(current: int, total: int, pdb_id: str):
            progress.update(download_task, advance=1, current_id=pdb_id)

        try:
            result = download_structures(
                pdb_ids,
                output_dir,
                overwrite=overwrite,
                max_workers=max_workers,
                progress_callback=progress_callback,
                queue_callback=queue_callback,
            )
        finally:
            progress.stop()
    else:
        result = download_structures(
            pdb_ids,
            output_dir,
            overwrite=overwrite,
            max_workers=max_workers,
            progress_callback=None,
        )

    # Summary
    print(f"\n{result.summary()}")

    if result.failed:
        print("\nFailed downloads:")
        for pdb_id, error in result.failed[:10]:
            print(f"  {pdb_id}: {error}")
        if len(result.failed) > 10:
            print(f"  ... and {len(result.failed) - 10} more")

    return result


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
