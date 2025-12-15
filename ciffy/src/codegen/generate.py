#!/usr/bin/env python3
"""
Auto-generate hash lookup tables and Python enums from the PDB Chemical Component Dictionary.

Reads the CCD file directly and generates:
  - hash/*.gperf (forward lookups)
  - hash/*.c (gperf output)
  - hash/reverse.h (reverse lookups for CIF writing)
  - biochemistry/_generated_atoms.py (Python atom enums)
  - biochemistry/_generated_residues.py (Python Residue enum + mappings)

Usage:
  python generate.py [ccd_path] [--gperf-path /path/to/gperf] [--skip-gperf]

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
import sys
from pathlib import Path

# Add ciffy directory to path to allow importing src.codegen without triggering ciffy/__init__.py
_CODEGEN_DIR = Path(__file__).parent
_CIFFY_DIR = _CODEGEN_DIR.parent.parent  # ciffy/src/codegen -> ciffy
if str(_CIFFY_DIR) not in sys.path:
    sys.path.insert(0, str(_CIFFY_DIR))


def main() -> None:
    """CLI entry point for code generation."""
    # Import src.codegen directly (not ciffy.src.codegen) to avoid triggering
    # ciffy/__init__.py which requires the C extension that hasn't been built yet
    from src.codegen import generate_all
    from src.codegen.c_codegen import find_gperf, run_gperf
    from src.codegen.cli import get_ccd_path

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
