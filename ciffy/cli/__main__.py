"""
Command-line interface for ciffy.

Usage:
    ciffy info <file.cif>            # Load and print polymer summary
    ciffy info <file1> <file2> ...   # Load and print multiple files
    ciffy info <file.cif> --desc     # Show entity descriptions per chain
    ciffy geometry <file.cif>        # Compute geometric properties (Rg, etc.)
    ciffy map <file.cif>             # Display contact map
    ciffy split <file.cif>           # Split into per-chain files
    ciffy cluster data/*.cif         # Cluster structures by similarity, return representatives

    ciffy download 1EHZ 4V9F         # Download specific PDB IDs
    ciffy download --max_count 100   # Search and download structures from RCSB PDB
    ciffy download --preset casp15   # Download CASP15 benchmark targets
"""

import argparse
import sys


def _info_command(args):
    """Handle the info/default command."""
    from ciffy import load

    # For multiple files, collect all info and display as unified table
    if len(args.files) > 1:
        from ciffy.utils.formatting import format_multi_polymer_table

        polymers = []
        backend = "numpy"  # default
        for filepath in args.files:
            try:
                skip = ["connections"] if args.desc else ["descriptions", "connections"]
                polymer = load(filepath, skip=skip)
                if not polymers:  # First successful load
                    backend = polymer.backend
                rows = polymer.chain_info()
                polymers.append((polymer.pdb_id, polymer.date, filepath, rows))
            except FileNotFoundError:
                print(f"Error: File not found: {filepath}", file=sys.stderr)
                continue
            except Exception as e:
                print(f"Error loading {filepath}: {e}", file=sys.stderr)
                continue

        if polymers:
            print(format_multi_polymer_table(polymers, backend))
        return

    # Single file: use original behavior
    for i, filepath in enumerate(args.files):
        # Add blank line between multiple files
        if i > 0:
            print()

        try:
            skip = ["connections"] if args.desc else ["descriptions", "connections"]
            polymer = load(filepath, skip=skip)
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
                seq = chain.sequence_str()
                if seq:
                    print(f"  {chain.names[0]}: {seq}")

        # Optional: show atom details
        if args.atoms:
            from ciffy import Scale
            atoms_per_res = polymer.counts(Scale.RESIDUE).tolist()
            print(f"\nAtoms per residue: {atoms_per_res}")

        # Optional: show entity descriptions
        if args.desc and polymer.descriptions:
            print("\nDescriptions:")
            for name, desc in zip(polymer.names, polymer.descriptions):
                # Strip CIF quoting (single/double quotes)
                if len(desc) >= 2 and desc[0] == desc[-1] and desc[0] in "'\"":
                    desc = desc[1:-1]
                print(f"  {name}: {desc}")


def _geometry_command(args):
    """Handle the geometry subcommand."""
    from ciffy import rg, clashes, Scale
    from .helpers import load_structure

    for i, filepath in enumerate(args.files):
        # Add blank line between multiple files
        if i > 0:
            print()

        polymer = load_structure(filepath)
        if polymer is None:
            continue

        # Header
        print(f"{polymer.pdb_id}")
        print("-" * 40)

        # Radius of gyration
        rg_value = rg(polymer, scale=Scale.MOLECULE)
        print(f"Radius of gyration: {float(rg_value[0]):.2f} Å")

        # Clash detection
        clash_pairs = clashes(polymer)
        print(f"Clashes: {len(clash_pairs)}")

        # Per-chain Rg if requested
        if args.per_chain and polymer.size(Scale.CHAIN) > 1:
            rg_per_chain = rg(polymer, scale=Scale.CHAIN)
            print("\nPer-chain Rg:")
            for j, (name, r) in enumerate(zip(polymer.names, rg_per_chain)):
                print(f"  {name}: {float(r):.2f} Å")


def _split_command(args):
    """Handle the split subcommand."""
    import os
    from .helpers import load_structure_or_exit

    polymer = load_structure_or_exit(args.file)

    # Check for polymer chains
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
    from .helpers import require_matplotlib, load_structure_or_exit

    if not require_matplotlib():
        sys.exit(1)

    import matplotlib.pyplot as plt
    from ciffy import Scale
    from ciffy.visualize import contact_map

    # Parse scale
    scale_map = {
        "residue": Scale.RESIDUE,
        "atom": Scale.ATOM,
    }
    scale = scale_map.get(args.scale.lower(), Scale.RESIDUE)

    polymer = load_structure_or_exit(args.file)

    # Filter by chain if specified
    if args.chain is not None:
        try:
            chain_idx = int(args.chain)
            polymer = polymer.chain(chain_idx)
        except ValueError:
            # Try to find by name
            chain_names = polymer.names
            if args.chain in chain_names:
                chain_idx = chain_names.index(args.chain)
                polymer = polymer.chain(chain_idx)
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
    title = f"{polymer.pdb_id} Contact Map"
    if args.chain is not None:
        title = f"{polymer.pdb_id} Chain {args.chain}"
    ax.set_title(title)

    # Save or show
    if args.output:
        plt.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved to {args.output}")
    else:
        plt.show()


def _download_command(args):
    """Handle the download subcommand."""
    from ciffy.datasets import download_cli

    # Convert empty list to None (triggers search mode)
    pdb_ids = args.pdb_ids if args.pdb_ids else None

    download_cli(
        pdb_ids=pdb_ids,
        preset=args.preset,
        polymer_types=args.type,
        output_dir=args.output_dir,
        max_count=args.max_count,
        max_resolution=args.max_resolution,
        min_resolution=args.min_resolution,
        min_length=args.min_length,
        max_length=args.max_length,
        method=args.method,
        released_after=args.released_after,
        released_before=args.released_before,
        overwrite=args.overwrite,
        max_workers=args.max_workers,
        search_only=args.search_only,
        list_ids=args.list_ids,
        list_presets=args.list_presets,
        quiet=args.quiet,
    )


def _cluster_command(args):
    """Handle the cluster subcommand."""
    from glob import glob
    from pathlib import Path

    from ciffy.operations.cluster import cluster

    # Validate split arguments
    if args.split and not args.output:
        print("Error: --split requires --output to specify output directory", file=sys.stderr)
        sys.exit(1)

    # Parse split ratios if provided
    split_ratios = None
    if args.split:
        try:
            parts = [float(x.strip()) for x in args.split.split(",")]
            if len(parts) == 2:
                train, test = parts
                val = 0.0
            elif len(parts) == 3:
                train, val, test = parts
            else:
                raise ValueError("Expected 2 or 3 values")

            if not (0.99 <= train + val + test <= 1.01):
                raise ValueError(f"Ratios must sum to 1.0, got {train + val + test}")

            split_ratios = (train, val, test)
        except ValueError as e:
            print(f"Error: Invalid --split format: {e}", file=sys.stderr)
            print("Expected format: '0.8,0.1,0.1' (train,val,test)", file=sys.stderr)
            sys.exit(1)

    # Expand glob patterns
    paths = []
    for pattern in args.files:
        expanded = glob(pattern)
        if not expanded:
            # Try as literal path
            if Path(pattern).exists():
                paths.append(Path(pattern))
            else:
                print(f"Warning: No files match: {pattern}", file=sys.stderr)
        else:
            paths.extend(Path(p) for p in sorted(expanded))

    if not paths:
        print("Error: No structure files found.", file=sys.stderr)
        sys.exit(1)

    if not args.quiet:
        print(f"Clustering {len(paths)} structures at {args.threshold:.0%} sequence identity...")

    try:
        result = cluster(
            paths,
            threshold=args.threshold,
            threads=args.threads,
            coverage=args.coverage,
        )
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if not args.quiet:
        print(f"Found {result.n_clusters} clusters from {result.n_structures} structures")
        print()

    # Handle split mode
    if split_ratios:
        from ciffy.utils.split import split_by_clusters, split_to_directories

        train, val, test = split_ratios
        split = split_by_clusters(
            result.paths,
            result.labels.tolist(),
            train=train,
            val=val,
            test=test,
            seed=args.seed,
        )

        if not args.quiet:
            print(f"Split: train={len(split.train)}, val={len(split.val)}, test={len(split.test)}")

        # Create directories
        try:
            dirs = split_to_directories(
                split,
                args.output,
                symlink=not args.copy,
                exist_ok=False,
            )
        except FileExistsError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

        if not args.quiet:
            link_type = "copied" if args.copy else "symlinked"
            for name, dir_path in dirs.items():
                n_files = len(list(dir_path.glob("*.cif")))
                print(f"  {dir_path}/ ({n_files} files {link_type})")

    # Output representatives (non-split mode)
    elif args.output:
        # Write representative paths to file
        with open(args.output, "w") as f:
            for rep in result.representatives:
                f.write(f"{rep}\n")
        if not args.quiet:
            print(f"Wrote {len(result.representatives)} representatives to {args.output}")
    else:
        # Print to stdout
        if args.verbose:
            # Show all clusters with members
            for label in range(result.n_clusters):
                members = result.get_cluster(label)
                rep = result.representatives[label]
                print(f"Cluster {label} ({len(members)} members):")
                print(f"  Representative: {rep.name}")
                if len(members) > 1:
                    for m in members:
                        marker = " *" if m == rep else ""
                        print(f"    - {m.name}{marker}")
                print()
        else:
            # Just print representative paths
            for rep in result.representatives:
                print(rep)


def main():
    """Main entry point for the ciffy CLI."""
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
    # Geometry subcommand
    geometry_parser = subparsers.add_parser(
        "geometry",
        help="Compute geometric properties of a structure",
        description="Compute geometric properties such as radius of gyration.",
    )
    geometry_parser.add_argument(
        "files",
        nargs="+",
        help="Path(s) to CIF file(s)",
    )
    geometry_parser.add_argument(
        "--per-chain", "-c",
        action="store_true",
        help="Show per-chain values in addition to overall",
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

    # Download subcommand
    download_parser = subparsers.add_parser(
        "download",
        help="Download structures from RCSB PDB",
        description=(
            "Download mmCIF files from RCSB PDB.\n"
            "Supports filtering by polymer type, resolution, length, and method."
        ),
    )
    download_parser.add_argument(
        "pdb_ids",
        nargs="*",
        help="PDB ID(s) to download (e.g., 1EHZ 4V9F). If omitted, searches for structures.",
    )
    download_parser.add_argument(
        "--preset",
        type=str,
        default=None,
        help="Download a preset dataset (e.g., casp15, casp16). Use --list-presets to see all.",
    )
    download_parser.add_argument(
        "--list-presets",
        action="store_true",
        help="List available preset datasets and exit",
    )
    download_parser.add_argument(
        "--type", "-t",
        type=str.lower,
        nargs="+",
        default=None,
        choices=["rna", "dna", "protein", "hybrid", "other"],
        help="Polymer type(s) to search for (default: rna, dna, protein).",
    )
    download_parser.add_argument(
        "--output-dir", "-o",
        default=".",
        help="Directory to save mmCIF files (default: current directory)",
    )
    download_parser.add_argument(
        "--max-count", "-n",
        type=int,
        default=None,
        help="Maximum number of structures to download (default: all)",
    )
    download_parser.add_argument(
        "--max-resolution",
        type=float,
        default=None,
        help="Maximum resolution in Ångströms (e.g., 3.0)",
    )
    download_parser.add_argument(
        "--min-resolution",
        type=float,
        default=None,
        help="Minimum resolution in Ångströms",
    )
    download_parser.add_argument(
        "--min-length",
        type=int,
        default=None,
        help="Minimum polymer length (nucleotides for RNA/DNA, residues for protein)",
    )
    download_parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Maximum polymer length (nucleotides for RNA/DNA, residues for protein)",
    )
    download_parser.add_argument(
        "--method", "-m",
        choices=["xray", "em", "nmr", "neutron"],
        default=None,
        help="Filter by experimental method",
    )
    download_parser.add_argument(
        "--released-after",
        type=str,
        default=None,
        help="Only include structures released after this date (YYYY-MM-DD)",
    )
    download_parser.add_argument(
        "--released-before",
        type=str,
        default=None,
        help="Only include structures released before this date (YYYY-MM-DD)",
    )
    download_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files (default: skip)",
    )
    download_parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum concurrent downloads (default: 4)",
    )
    download_parser.add_argument(
        "--search-only",
        action="store_true",
        help="Only search and print count, don't download",
    )
    download_parser.add_argument(
        "--list-ids",
        action="store_true",
        help="Print all PDB IDs (use with --search-only)",
    )
    download_parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output",
    )

    # Cluster subcommand
    cluster_parser = subparsers.add_parser(
        "cluster",
        help="Cluster structures by sequence identity",
        description=(
            "Cluster structures using MMseqs2 sequence identity.\n"
            "Returns one representative structure per cluster.\n"
            "Requires mmseqs2: mamba install -c bioconda mmseqs2"
        ),
    )
    cluster_parser.add_argument(
        "files",
        nargs="+",
        help="Structure files or glob patterns (e.g., data/*.cif)",
    )
    cluster_parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.5,
        help="Sequence identity threshold for clustering (default: 0.5). "
             "0.3 = remote homologs, 0.5 = same family, 0.9 = near-identical",
    )
    cluster_parser.add_argument(
        "--coverage", "-c",
        type=float,
        default=0.8,
        help="Minimum alignment coverage (default: 0.8)",
    )
    cluster_parser.add_argument(
        "--output", "-o",
        help="Without --split: write representative paths to file. "
             "With --split: create train/val/test directories here.",
    )
    cluster_parser.add_argument(
        "--split", "-s",
        help="Split into train/val/test directories. Format: 'train,val,test' "
             "e.g., '0.8,0.1,0.1'. Requires --output for directory path.",
    )
    cluster_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for split reproducibility (default: 42)",
    )
    cluster_parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of symlinking (with --split)",
    )
    cluster_parser.add_argument(
        "--threads", "-j",
        type=int,
        default=4,
        help="Number of threads for mmseqs (default: 4)",
    )
    cluster_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show all clusters with their members",
    )
    cluster_parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress messages",
    )

    args = parser.parse_args()

    # Route to appropriate handler
    if args.command == "download":
        _download_command(args)
    elif args.command == "map":
        _map_command(args)
    elif args.command == "split":
        _split_command(args)
    elif args.command == "info":
        _info_command(args)
    elif args.command == "geometry":
        _geometry_command(args)
    elif args.command == "cluster":
        _cluster_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
