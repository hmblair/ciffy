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
from pathlib import Path

from .config import CCD_URL

# Additional data sources
PUBCHEM_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/periodictable/CSV"
MONLIB_URL = "https://raw.githubusercontent.com/MonomerLibrary/monomers/master/links_and_mods.cif"


def download_ccd(dest_path: Path) -> bool:
    """Download and decompress the CCD file."""
    print(f"Downloading CCD from {CCD_URL}...")
    gz_path = dest_path.with_suffix(".cif.gz")

    try:
        urllib.request.urlretrieve(CCD_URL, gz_path)
        print("Decompressing CCD...")
        with gzip.open(gz_path, 'rb') as f_in:
            with open(dest_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        gz_path.unlink()
        print(f"CCD downloaded to {dest_path}")
        return True
    except Exception as e:
        print(f"Failed to download CCD: {e}")
        if gz_path.exists():
            gz_path.unlink()
        return False


def download_elements(dest_path: Path) -> bool:
    """Download PubChem periodic table CSV."""
    print(f"Downloading periodic table from {PUBCHEM_URL}...")
    try:
        urllib.request.urlretrieve(PUBCHEM_URL, dest_path)
        print(f"Elements downloaded to {dest_path}")
        return True
    except Exception as e:
        print(f"Failed to download elements: {e}")
        return False


def download_monlib(dest_path: Path) -> bool:
    """Download MonomerLibrary links_and_mods.cif."""
    print(f"Downloading MonomerLibrary from {MONLIB_URL}...")
    try:
        urllib.request.urlretrieve(MONLIB_URL, dest_path)
        print(f"MonomerLibrary downloaded to {dest_path}")
        return True
    except Exception as e:
        print(f"Failed to download MonomerLibrary: {e}")
        return False


def get_ccd_path() -> Path:
    """Get path to CCD file, downloading if necessary."""
    # Check environment variable first
    env_path = os.environ.get("CIFFY_CCD_PATH")
    if env_path:
        path = Path(env_path)
        if path.exists():
            return path

    # Use centralized cache location
    cache_dir = Path.home() / ".cache" / "ciffy"
    ccd_path = cache_dir / "components.cif"

    if ccd_path.exists():
        return ccd_path

    # Download to cache directory
    cache_dir.mkdir(parents=True, exist_ok=True)
    if download_ccd(ccd_path):
        return ccd_path

    raise FileNotFoundError(
        f"CCD file not found and download failed. "
        f"Set CIFFY_CCD_PATH or download manually from {CCD_URL}"
    )


def get_elements_path() -> Path:
    """Get path to PubChem elements CSV, downloading if necessary."""
    cache_dir = Path.home() / ".cache" / "ciffy"
    elements_path = cache_dir / "elements.csv"

    if elements_path.exists():
        return elements_path

    cache_dir.mkdir(parents=True, exist_ok=True)
    if download_elements(elements_path):
        return elements_path

    raise FileNotFoundError(
        f"Elements file not found and download failed. "
        f"Download manually from {PUBCHEM_URL}"
    )


def get_monlib_path() -> Path:
    """Get path to MonomerLibrary links_and_mods.cif, downloading if necessary."""
    cache_dir = Path.home() / ".cache" / "ciffy"
    monlib_path = cache_dir / "links_and_mods.cif"

    if monlib_path.exists():
        return monlib_path

    cache_dir.mkdir(parents=True, exist_ok=True)
    if download_monlib(monlib_path):
        return monlib_path

    raise FileNotFoundError(
        f"MonomerLibrary file not found and download failed. "
        f"Download manually from {MONLIB_URL}"
    )


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
