#!/usr/bin/env python3
"""
Auto-generate hash lookup tables and Python enums from the PDB Chemical Component Dictionary.

Reads the CCD file directly and generates:
  - ciffy/src/hash/*.gperf (forward lookups)
  - ciffy/src/hash/*.c (gperf output)
  - ciffy/src/hash/reverse.h (reverse lookups for CIF writing)
  - ciffy/biochemistry/_generated_*.py (Python enums)

Usage:
  python -m codegen.generate [ccd_path] [--gperf-path /path/to/gperf] [--skip-gperf]

If ccd_path is not provided, the CCD will be auto-downloaded to ~/.cache/ciffy/.
This script is called automatically during build via setup.py.

Note: The actual implementation is in the codegen submodules:
  - config.py: Constants and molecule type definitions
  - names.py: Name conversion utilities
  - residue.py: ResidueDefinition class
  - ccd.py: CCD parsing
  - c_codegen.py: C code generation (gperf, reverse.h, bond_patterns.h)
  - python_codegen.py: Python code generation
  - cli.py: CLI and download utilities
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Codegen is now at project root - no sys.path manipulation needed
from . import generate_all
from .c_codegen import find_gperf, run_gperf
from .cli import get_ccd_path


def main() -> None:
    """CLI entry point for code generation."""
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


if __name__ == "__main__":
    main()
