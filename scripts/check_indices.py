#!/usr/bin/env python3
"""
Check CIF files for out-of-bounds embedding indices.

Usage:
    python scripts/check_indices.py /path/to/cifs/
    python scripts/check_indices.py /path/to/cifs/ --verbose
"""

import argparse
import sys
from pathlib import Path

import ciffy
from ciffy import NUM_ATOMS, NUM_RESIDUES, NUM_ELEMENTS


def check_file(filepath: str, verbose: bool = False) -> dict | None:
    """
    Check a single CIF file for out-of-bounds indices.

    Note: -1 values are expected for unknown atom/residue types and are
    handled by PolymerEmbedding (clamped to 0). Only values >= vocab size
    are flagged as errors.

    Returns dict with issues if any found, None otherwise.
    """
    try:
        p = ciffy.load(filepath, backend="numpy")
    except Exception as e:
        return {"file": filepath, "error": f"Load failed: {e}"}

    issues = []

    # Check atom indices (only flag >= vocab size, -1 is expected for unknown)
    atom_max = int(p.atoms.max())
    if atom_max >= NUM_ATOMS:
        count = int((p.atoms >= NUM_ATOMS).sum())
        issues.append(f"atoms: {count} values >= {NUM_ATOMS} (max={atom_max})")

    # Check element indices
    elem_max = int(p.elements.max())
    if elem_max >= NUM_ELEMENTS:
        count = int((p.elements >= NUM_ELEMENTS).sum())
        issues.append(f"elements: {count} values >= {NUM_ELEMENTS} (max={elem_max})")

    # Check residue indices (sequence)
    res_max = int(p.sequence.max())
    if res_max >= NUM_RESIDUES:
        count = int((p.sequence >= NUM_RESIDUES).sum())
        issues.append(f"residues: {count} values >= {NUM_RESIDUES} (max={res_max})")

    if issues:
        return {
            "file": filepath,
            "pdb_id": Path(filepath).stem.upper(),
            "atoms": p.size(),
            "issues": issues,
        }

    if verbose:
        print(f"  ✓ {Path(filepath).name}: OK ({p.size()} atoms)")

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Check CIF files for out-of-bounds embedding indices"
    )
    parser.add_argument("directory", type=str, help="Directory containing .cif files")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show progress for each file")
    parser.add_argument("--stop-on-error", action="store_true",
                        help="Stop after first problematic file")
    args = parser.parse_args()

    directory = Path(args.directory)
    if not directory.is_dir():
        print(f"Error: {directory} is not a directory")
        sys.exit(1)

    cif_files = sorted(directory.glob("*.cif"))
    if not cif_files:
        print(f"No .cif files found in {directory}")
        sys.exit(1)

    print(f"Checking {len(cif_files)} CIF files in {directory}")
    print(f"Vocabulary sizes: atoms={NUM_ATOMS}, elements={NUM_ELEMENTS}, residues={NUM_RESIDUES}")
    print()

    problems = []
    for filepath in cif_files:
        result = check_file(str(filepath), verbose=args.verbose)
        if result:
            problems.append(result)
            print(f"  ✗ {filepath.name}: {result.get('pdb_id', 'N/A')}")
            if "error" in result:
                print(f"      {result['error']}")
            else:
                for issue in result["issues"]:
                    print(f"      - {issue}")

            if args.stop_on_error:
                break

    print()
    print(f"Summary: {len(problems)}/{len(cif_files)} files with issues")

    if problems:
        sys.exit(1)


if __name__ == "__main__":
    main()
