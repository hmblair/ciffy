#!/usr/bin/env python3
"""
Check that all public API exports are documented in api.md.

Run as: python scripts/check_api_docs.py
Returns exit code 1 if undocumented exports found.

This script compares ciffy.__all__ against the mkdocstrings directives
in docs/api.md to ensure all public API is documented.
"""

import re
import sys
from pathlib import Path


def get_all_exports():
    """Get all items from ciffy.__all__."""
    import ciffy

    return set(ciffy.__all__)


def get_documented_items(api_md_path: Path) -> set[str]:
    """Extract items documented with ::: directives in api.md."""
    content = api_md_path.read_text()

    # Match ::: ciffy.ItemName or ::: ciffy.submodule.ItemName
    pattern = r":::\s*ciffy\.(\w+(?:\.\w+)*)"
    matches = re.findall(pattern, content)

    # Extract top-level names (for ciffy.nn.X, we get "nn")
    documented = set()
    for match in matches:
        parts = match.split(".")
        documented.add(parts[0])

    return documented


def main():
    # Find project root (parent of scripts/)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    api_md_path = project_root / "docs" / "api.md"

    if not api_md_path.exists():
        print(f"ERROR: {api_md_path} not found")
        return 1

    exports = get_all_exports()
    documented = get_documented_items(api_md_path)

    # Items that are submodules or implementation details
    # These are handled separately or don't need individual docs
    skip = {
        "__version__",  # Version string, not a function
        "nn",  # Submodule - items within are documented
        "visualize",  # Submodule - items within are documented
    }

    # Check for undocumented exports
    missing = exports - documented - skip
    missing = {m for m in missing if not m.isupper() or m not in documented}

    # Filter out convenience aliases that point to documented types
    # (e.g., ATOM -> Scale.ATOM is documented under Scale)
    alias_targets = {
        "ATOM",
        "RESIDUE",
        "CHAIN",
        "MOLECULE",  # Scale aliases
        "PROTEIN",
        "RNA",
        "DNA",
        "LIGAND",
        "ION",
        "WATER",  # Molecule aliases
    }
    missing -= alias_targets

    if missing:
        print("ERROR: Undocumented exports in ciffy.__all__:")
        for item in sorted(missing):
            print(f"  - {item}")
        print("\nAdd these to docs/api.md with ::: ciffy.{name} directives")
        return 1

    # Check for items documented but not in __all__
    extra = documented - exports - {"Polymer"}  # Polymer is special-cased
    # Filter out submodule items (nn.*, visualize.*)
    extra = {e for e in extra if "." not in e}

    if extra:
        print("WARNING: Items documented but not in __all__:")
        for item in sorted(extra):
            print(f"  - {item}")
        print("\nConsider adding these to ciffy.__all__ or removing from docs")

    print(f"OK: {len(exports - skip - alias_targets)} exports documented")
    return 0


if __name__ == "__main__":
    sys.exit(main())
