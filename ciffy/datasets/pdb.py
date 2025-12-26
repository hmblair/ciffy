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

logger = logging.getLogger(__name__)

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


def _build_rna_query(
    min_resolution: float | None = None,
    max_resolution: float | None = None,
    min_length: int | None = None,
    max_length: int | None = None,
    experimental_method: str | None = None,
) -> dict:
    """Build RCSB search query for RNA structures."""
    # Base query: must contain RNA polymer
    nodes = [
        {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "entity_poly.rcsb_entity_polymer_type",
                "operator": "exact_match",
                "value": "RNA",
            },
        }
    ]

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

    Queries the RCSB search API for structures containing RNA polymer entities.
    Returns PDB IDs sorted by deposit date (newest first).

    Args:
        min_resolution: Minimum resolution in Ångströms (inclusive).
        max_resolution: Maximum resolution in Ångströms (inclusive).
        min_length: Minimum RNA polymer length in nucleotides.
        max_length: Maximum RNA polymer length in nucleotides.
        experimental_method: Filter by experimental method.
        timeout: Request timeout in seconds.

    Returns:
        List of PDB IDs matching the query.

    Raises:
        URLError: If the search request fails.
        ValueError: If the response cannot be parsed.

    Example:
        >>> # Find high-resolution X-ray structures
        >>> pdb_ids = search_rna_structures(
        ...     max_resolution=2.5,
        ...     experimental_method="X-RAY DIFFRACTION",
        ... )
        >>> print(f"Found {len(pdb_ids)} structures")
    """
    query = _build_rna_query(
        min_resolution=min_resolution,
        max_resolution=max_resolution,
        min_length=min_length,
        max_length=max_length,
        experimental_method=experimental_method,
    )

    query_json = json.dumps(query).encode("utf-8")

    request = Request(
        RCSB_SEARCH_URL,
        data=query_json,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    logger.info("Searching RCSB PDB for RNA structures...")

    try:
        with urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except HTTPError as e:
        raise URLError(f"RCSB search failed with status {e.code}: {e.reason}") from e

    # Extract PDB IDs from results
    pdb_ids = []
    for result in data.get("result_set", []):
        pdb_id = result.get("identifier")
        if pdb_id:
            pdb_ids.append(pdb_id.upper())

    logger.info(f"Found {len(pdb_ids)} RNA structures")
    return pdb_ids


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
        print(f"Found {len(pdb_ids)} structures")

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


def download_rna_cli(
    pdb_ids: list[str] | None = None,
    output_dir: Path | str = ".",
    max_count: int | None = None,
    max_resolution: float | None = None,
    min_resolution: float | None = None,
    min_length: int | None = None,
    max_length: int | None = None,
    method: str | None = None,
    overwrite: bool = False,
    max_workers: int = 4,
    search_only: bool = False,
    list_ids: bool = False,
    quiet: bool = False,
) -> DownloadResult | None:
    """
    CLI-friendly PDB download with progress output.

    This is the main entry point for the `ciffy download` command.

    Args:
        pdb_ids: Specific PDB IDs to download. If None, searches for RNA structures.
        output_dir: Directory to save mmCIF files.
        max_count: Maximum number of structures to download.
        max_resolution: Maximum resolution in Ångströms.
        min_resolution: Minimum resolution in Ångströms.
        min_length: Minimum RNA polymer length.
        max_length: Maximum RNA polymer length.
        method: Experimental method shorthand (xray, em, nmr, neutron).
        overwrite: Overwrite existing files.
        max_workers: Maximum concurrent downloads.
        search_only: Only search, don't download.
        list_ids: Print PDB IDs (with search_only).
        quiet: Suppress progress output.

    Returns:
        DownloadResult if download occurred, None if search_only.
    """
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

        # Search for structures
        print("Searching RCSB PDB for RNA structures...")
        if max_resolution:
            print(f"  Max resolution: {max_resolution} Å")
        if min_resolution:
            print(f"  Min resolution: {min_resolution} Å")
        if min_length:
            print(f"  Min length: {min_length} nt")
        if max_length:
            print(f"  Max length: {max_length} nt")
        if experimental_method:
            print(f"  Method: {experimental_method}")

        pdb_ids = search_rna_structures(
            min_resolution=min_resolution,
            max_resolution=max_resolution,
            min_length=min_length,
            max_length=max_length,
            experimental_method=experimental_method,
        )

        print(f"\nFound {len(pdb_ids)} structures")

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
    "search_rna_structures",
    "download_structure",
    "download_structures",
    "download_rna_dataset",
    "download_rna_cli",
]
