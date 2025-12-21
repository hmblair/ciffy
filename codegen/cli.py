"""
CLI and download utilities for code generation.

Handles CCD download, caching, and command-line interface.
"""

from __future__ import annotations

import argparse
import gzip
import os
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .config import CCD_URL

# Additional data sources
PUBCHEM_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/periodictable/CSV"
MONLIB_URL = "https://raw.githubusercontent.com/MonomerLibrary/monomers/master/links_and_mods.cif"


# =============================================================================
# RESOURCE CACHE - Generic download and caching infrastructure
# =============================================================================


@dataclass
class Resource:
    """Definition of a downloadable resource."""
    name: str  # Human-readable name
    filename: str  # Cache filename
    url: str  # Download URL
    env_var: str | None = None  # Optional environment variable override
    post_process: Callable[[Path], None] | None = None  # Optional post-download processing


def _decompress_gzip(path: Path) -> None:
    """Decompress a .gz file in place."""
    gz_path = path.with_suffix(path.suffix + ".gz")
    # The file was downloaded as .gz, decompress it
    with gzip.open(gz_path, 'rb') as f_in:
        with open(path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    gz_path.unlink()


class ResourceCache:
    """
    Generic cache for downloadable resources.

    Provides a unified interface for downloading and caching external data files.
    Resources are stored in ~/.cache/ciffy/ by default.
    """

    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = cache_dir or (Path.home() / ".cache" / "ciffy")

    def download(self, resource: Resource) -> bool:
        """
        Download a resource to the cache.

        Args:
            resource: Resource definition to download.

        Returns:
            True if download succeeded, False otherwise.
        """
        dest_path = self.cache_dir / resource.filename
        print(f"Downloading {resource.name} from {resource.url}...")

        try:
            # For gzipped files, download to .gz first
            if resource.post_process == _decompress_gzip:
                gz_path = dest_path.with_suffix(dest_path.suffix + ".gz")
                urllib.request.urlretrieve(resource.url, gz_path)
                print(f"Decompressing {resource.name}...")
                resource.post_process(dest_path)
            else:
                urllib.request.urlretrieve(resource.url, dest_path)

            print(f"{resource.name} downloaded to {dest_path}")
            return True
        except Exception as e:
            print(f"Failed to download {resource.name}: {e}")
            # Clean up partial downloads
            if dest_path.exists():
                dest_path.unlink()
            gz_path = dest_path.with_suffix(dest_path.suffix + ".gz")
            if gz_path.exists():
                gz_path.unlink()
            return False

    def get(self, resource: Resource) -> Path:
        """
        Get path to a resource, downloading if necessary.

        Args:
            resource: Resource definition to get.

        Returns:
            Path to the cached resource file.

        Raises:
            FileNotFoundError: If resource not found and download failed.
        """
        # Check environment variable override first
        if resource.env_var:
            env_path = os.environ.get(resource.env_var)
            if env_path:
                path = Path(env_path)
                if path.exists():
                    return path

        # Check cache
        cached_path = self.cache_dir / resource.filename
        if cached_path.exists():
            return cached_path

        # Download to cache
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if self.download(resource):
            return cached_path

        raise FileNotFoundError(
            f"{resource.name} not found and download failed. "
            f"Download manually from {resource.url}"
        )


# Singleton cache instance
_cache = ResourceCache()

# Resource definitions
CCD_RESOURCE = Resource(
    name="CCD",
    filename="components.cif",
    url=CCD_URL,
    env_var="CIFFY_CCD_PATH",
    post_process=_decompress_gzip,
)

ELEMENTS_RESOURCE = Resource(
    name="PubChem periodic table",
    filename="elements.csv",
    url=PUBCHEM_URL,
)

MONLIB_RESOURCE = Resource(
    name="MonomerLibrary",
    filename="links_and_mods.cif",
    url=MONLIB_URL,
)


# =============================================================================
# PUBLIC API - Convenience functions using the cache
# =============================================================================


def get_ccd_path() -> Path:
    """Get path to CCD file, downloading if necessary."""
    return _cache.get(CCD_RESOURCE)


def get_elements_path() -> Path:
    """Get path to PubChem elements CSV, downloading if necessary."""
    return _cache.get(ELEMENTS_RESOURCE)


def get_monlib_path() -> Path:
    """Get path to MonomerLibrary links_and_mods.cif, downloading if necessary."""
    return _cache.get(MONLIB_RESOURCE)


# =============================================================================
# CLI ENTRY POINT
# =============================================================================


def main() -> None:
    """CLI entry point for code generation."""
    from .c_codegen import find_gperf, run_gperf
    from . import generate_all

    parser = argparse.ArgumentParser(
        description="Generate hash tables from PDB Chemical Component Dictionary"
    )
    parser.add_argument(
        "ccd_path",
        nargs="?",
        help="Path to components.cif file (auto-downloaded if not provided)"
    )
    parser.add_argument("--gperf-path", help="Path to gperf executable")
    parser.add_argument("--skip-gperf", action="store_true", help="Skip running gperf")
    args = parser.parse_args()

    # Get CCD path (auto-download if not provided)
    ccd_path = Path(args.ccd_path) if args.ccd_path else get_ccd_path()

    hash_dir, _ = generate_all(str(ccd_path))

    if not args.skip_gperf:
        gperf_path = args.gperf_path or find_gperf()
        run_gperf(gperf_path, hash_dir)

    print("Generation complete!")
