"""
Command-line interface for ciffy.

Usage:
    ciffy <file.cif>          # Load and print polymer summary
    ciffy <file.cif> --atoms  # Also show atom counts per residue
"""

import argparse
import sys


def main():
    """Main entry point for the ciffy CLI."""
    parser = argparse.ArgumentParser(
        prog="ciffy",
        description="Load and inspect CIF files.",
    )
    parser.add_argument(
        "file",
        help="Path to CIF file",
    )
    parser.add_argument(
        "--atoms", "-a",
        action="store_true",
        help="Show detailed atom information",
    )
    parser.add_argument(
        "--sequence", "-s",
        action="store_true",
        help="Show sequence string",
    )

    args = parser.parse_args()

    try:
        from ciffy import load
        polymer = load(args.file)
    except FileNotFoundError:
        print(f"Error: File not found: {args.file}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading {args.file}: {e}", file=sys.stderr)
        sys.exit(1)

    # Print polymer summary
    print(polymer)

    # Optional: show sequence
    if args.sequence:
        print(f"\nSequence: {polymer.str()}")

    # Optional: show atom details
    if args.atoms:
        from ciffy import Scale
        print(f"\nAtoms per residue: {list(polymer.per(Scale.ATOM, Scale.RESIDUE))}")


if __name__ == "__main__":
    main()
