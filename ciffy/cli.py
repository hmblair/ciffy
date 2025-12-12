"""
Command-line interface for ciffy.

Usage:
    ciffy <file.cif>              # Load and print polymer summary
    ciffy <file1> <file2> ...     # Load and print multiple files
    ciffy <file.cif> --atoms      # Also show atom counts per residue
    ciffy <file.cif> --desc       # Show entity descriptions per chain
    ciffy map <file.cif>          # Display contact map
"""

import argparse
import sys


def _info_command(args):
    """Handle the info/default command."""
    from ciffy import load

    for i, filepath in enumerate(args.files):
        # Add blank line between multiple files
        if i > 0:
            print()

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


def _split_command(args):
    """Handle the split subcommand."""
    import os
    from ciffy import load

    try:
        polymer = load(args.file)
    except FileNotFoundError:
        print(f"Error: File not found: {args.file}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading {args.file}: {e}", file=sys.stderr)
        sys.exit(1)

    # Filter to polymer chains only (unless --all specified)
    if not args.all:
        polymer = polymer.poly()
        if polymer.size() == 0:
            print("No polymer chains found.", file=sys.stderr)
            sys.exit(1)

    # Determine output directory
    if args.output:
        out_dir = args.output
        os.makedirs(out_dir, exist_ok=True)
    else:
        out_dir = "."

    # Get base name from input file
    base = os.path.splitext(os.path.basename(args.file))[0]

    # Split and write each chain
    written = 0
    for chain in polymer.chains():
        chain_name = chain.names[0]
        out_path = os.path.join(out_dir, f"{base}_{chain_name}.cif")
        chain.write(out_path)
        print(f"Wrote {out_path} ({chain.size()} atoms)")
        written += 1

    print(f"Split into {written} files.")


def _map_command(args):
    """Handle the map subcommand."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "Error: matplotlib is required for contact maps.\n"
            "Install with: pip install matplotlib",
            file=sys.stderr
        )
        sys.exit(1)

    from ciffy import load, Scale
    from ciffy.visualize import contact_map

    # Parse scale
    scale_map = {
        "residue": Scale.RESIDUE,
        "atom": Scale.ATOM,
    }
    scale = scale_map.get(args.scale.lower(), Scale.RESIDUE)

    try:
        polymer = load(args.file)
    except FileNotFoundError:
        print(f"Error: File not found: {args.file}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading {args.file}: {e}", file=sys.stderr)
        sys.exit(1)

    # Filter by chain if specified
    if args.chain is not None:
        try:
            chain_idx = int(args.chain)
            polymer = polymer.by_index(chain_idx)
        except ValueError:
            # Try to find by name
            chain_names = polymer.names
            if args.chain in chain_names:
                chain_idx = chain_names.index(args.chain)
                polymer = polymer.by_index(chain_idx)
            else:
                print(f"Error: Chain '{args.chain}' not found. "
                      f"Available: {chain_names}", file=sys.stderr)
                sys.exit(1)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))

    # Generate contact map
    contact_map(
        polymer,
        scale=scale,
        power=args.power,
        ax=ax,
        cmap=args.cmap,
    )

    # Set title
    title = f"{polymer.id()} Contact Map"
    if args.chain is not None:
        title = f"{polymer.id()} Chain {args.chain}"
    ax.set_title(title)

    # Save or show
    if args.output:
        plt.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved to {args.output}")
    else:
        plt.show()


def main():
    """Main entry point for the ciffy CLI."""
    # Check if first argument is a subcommand
    subcommands = {"map", "info", "split"}

    # If no args or first arg starts with - or is not a subcommand,
    # treat as the info command
    if len(sys.argv) > 1 and sys.argv[1] not in subcommands and not sys.argv[1].startswith('-'):
        # Insert 'info' as the subcommand
        sys.argv.insert(1, "info")

    parser = argparse.ArgumentParser(
        prog="ciffy",
        description="Load and inspect CIF files.",
    )

    subparsers = parser.add_subparsers(dest="command")

    # Info subcommand (default)
    info_parser = subparsers.add_parser(
        "info",
        help="Display structure information (default)",
        description="Load and display information about CIF files.",
    )
    info_parser.add_argument(
        "files",
        nargs="+",
        help="Path(s) to CIF file(s)",
    )
    info_parser.add_argument(
        "--atoms", "-a",
        action="store_true",
        help="Show detailed atom information",
    )
    info_parser.add_argument(
        "--sequence", "-s",
        action="store_true",
        help="Show sequence string",
    )
    info_parser.add_argument(
        "--desc", "-d",
        action="store_true",
        help="Show entity descriptions for each chain",
    )

    # Map subcommand
    map_parser = subparsers.add_parser(
        "map",
        help="Display contact map for a structure",
        description="Generate and display a contact map (1/r^n heatmap) for a CIF file.",
    )
    map_parser.add_argument(
        "file",
        help="Path to CIF file",
    )
    map_parser.add_argument(
        "--scale", "-s",
        default="residue",
        choices=["residue", "atom"],
        help="Scale for distance computation (default: residue)",
    )
    map_parser.add_argument(
        "--power", "-p",
        type=float,
        default=2.0,
        help="Exponent for 1/r^n transformation (default: 2.0)",
    )
    map_parser.add_argument(
        "--chain", "-c",
        help="Chain to display (name or index)",
    )
    map_parser.add_argument(
        "--cmap",
        default="RdPu",
        help="Matplotlib colormap (default: RdPu)",
    )
    map_parser.add_argument(
        "--output", "-o",
        help="Save to file instead of displaying",
    )
    map_parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for saved image (default: 150)",
    )

    # Split subcommand
    split_parser = subparsers.add_parser(
        "split",
        help="Split structure into separate files per chain",
        description="Split a CIF file into multiple files, one per chain.",
    )
    split_parser.add_argument(
        "file",
        help="Path to CIF file",
    )
    split_parser.add_argument(
        "--output", "-o",
        help="Output directory (default: current directory)",
    )
    split_parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Include all chains (default: polymer chains only)",
    )

    args = parser.parse_args()

    # Route to appropriate handler
    if args.command == "map":
        _map_command(args)
    elif args.command == "split":
        _split_command(args)
    elif args.command == "info":
        _info_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
