"""
Command-line interface for ciffy.

Usage:
    ciffy <file.cif>              # Load and print polymer summary
    ciffy <file1> <file2> ...     # Load and print multiple files
    ciffy <file.cif> --atoms      # Also show atom counts per residue
    ciffy <file.cif> --desc       # Show entity descriptions per chain
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
        "files",
        nargs="+",
        help="Path(s) to CIF file(s)",
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
    parser.add_argument(
        "--desc", "-d",
        action="store_true",
        help="Show entity descriptions for each chain",
    )

    args = parser.parse_args()

    from ciffy import load

    for i, filepath in enumerate(args.files):
        # Add separator between multiple files
        if i > 0:
            print("\n" + "=" * 40 + "\n")

        try:
            polymer = load(filepath, load_descriptions=args.desc)
        except FileNotFoundError:
            print(f"Error: File not found: {filepath}", file=sys.stderr)
            continue
        except Exception as e:
            print(f"Error loading {filepath}: {e}", file=sys.stderr)
            continue

        # Print polymer summary
        print(polymer)

        # Optional: show sequence per chain
        if args.sequence:
            print("\nSequence:")
            for chain in polymer.chains():
                seq = chain.str()
                if seq:
                    print(f"  {chain.names[0]}: {seq}")

        # Optional: show atom details
        if args.atoms:
            from ciffy import Scale
            atoms_per_res = polymer.per(Scale.ATOM, Scale.RESIDUE).tolist()
            print(f"\nAtoms per residue: {atoms_per_res}")

        # Optional: show entity descriptions
        if args.desc and polymer.descriptions:
            print("\nDescriptions:")
            for name, desc in zip(polymer.names, polymer.descriptions):
                # Strip CIF quoting (single/double quotes)
                if len(desc) >= 2 and desc[0] == desc[-1] and desc[0] in "'\"":
                    desc = desc[1:-1]
                print(f"  {name}: {desc}")


if __name__ == "__main__":
    main()
