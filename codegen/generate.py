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

# Delegate to cli.main() - single source of truth for CLI
from .cli import main

if __name__ == "__main__":
    main()
