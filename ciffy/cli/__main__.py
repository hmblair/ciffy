"""
Command-line interface for ciffy.

Usage:
    ciffy <file.cif>              # Load and print polymer summary
    ciffy <file1> <file2> ...     # Load and print multiple files
    ciffy <file.cif> --atoms      # Also show atom counts per residue
    ciffy <file.cif> --desc       # Show entity descriptions per chain
    ciffy map <file.cif>          # Display contact map
    ciffy split <file.cif>        # Split into per-chain files
    ciffy template <sequence>     # Create template from sequence with sampled dihedrals
    ciffy train configs/*.yaml    # Run training from config files
    ciffy experiment configs/*.yaml  # Run multiple training experiments
    ciffy download --max_count 100   # Download RNA structures from RCSB PDB
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
                seq = chain.sequence_str()
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


def _template_command(args):
    """Handle the template subcommand."""
    from ciffy import from_sequence

    try:
        # Create polymer from sequence with sampled dihedrals
        # By default, use clash-free sampling unless --no-clash-free is specified
        clash_free = not args.no_clash_free

        polymer = from_sequence(
            args.sequence,
            sample_dihedrals=True,
            clash_free=clash_free,
            seed=args.seed,
        )

        # Write output
        if args.output:
            polymer.write(args.output)
            print(f"Wrote template to {args.output}")
        else:
            # Print to stdout
            print(polymer)

    except Exception as e:
        print(f"Error creating template: {e}", file=sys.stderr)
        sys.exit(1)


def _experiment_command(args):
    """Handle the experiment subcommand."""
    try:
        import torch
    except ImportError:
        print(
            "Error: PyTorch is required for experiment runner.\n"
            "Install with: pip install torch",
            file=sys.stderr,
        )
        sys.exit(1)

    from glob import glob

    try:
        from ciffy.nn.runners import format_results_table, run_experiments
    except ImportError:
        print(
            "Error: Neural network modules not available.\n"
            "Install from source: pip install git+https://github.com/hmblair/ciffy.git",
            file=sys.stderr,
        )
        sys.exit(1)

    # Expand glob patterns in config paths
    config_paths = []
    for pattern in args.configs:
        expanded = glob(pattern)
        if not expanded:
            print(f"Warning: No files match pattern: {pattern}", file=sys.stderr)
        config_paths.extend(sorted(expanded))

    if not config_paths:
        print("Error: No config files found.", file=sys.stderr)
        sys.exit(1)

    # Display experiment plan
    print()
    print("=" * 60)
    print("Ciffy Experiment Runner")
    print("=" * 60)
    print(f"Configs: {len(config_paths)}")
    print(f"Parallel: {not args.sequential}")
    print(f"Device: {args.device}")
    print()

    for i, path in enumerate(config_paths, 1):
        print(f"  {i}. {path}")
    print()

    # Run experiments
    print("Running experiments...")
    print("-" * 60)

    try:
        results = run_experiments(
            config_paths=config_paths,
            parallel=not args.sequential,
            device=args.device,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error running experiments: {e}", file=sys.stderr)
        sys.exit(1)

    # Print results table
    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print(format_results_table(results))
    print()

    # Exit with error code if any experiments failed
    failed = sum(1 for r in results if r.status != "success")
    if failed > 0:
        sys.exit(1)


def _download_command(args):
    """Handle the download subcommand."""
    from ciffy.datasets import download_cli

    download_cli(
        pdb_ids=args.id,
        polymer_types=args.type,
        output_dir=args.output_dir,
        max_count=args.max_count,
        max_resolution=args.max_resolution,
        min_resolution=args.min_resolution,
        min_length=args.min_length,
        max_length=args.max_length,
        method=args.method,
        overwrite=args.overwrite,
        max_workers=args.max_workers,
        search_only=args.search_only,
        list_ids=args.list_ids,
        quiet=args.quiet,
    )


def _inference_command(args):
    """Handle the inference subcommand."""
    try:
        import torch
    except ImportError:
        print(
            "Error: PyTorch is required for inference runner.\n"
            "Install with: pip install torch",
            file=sys.stderr,
        )
        sys.exit(1)

    from glob import glob

    try:
        from ciffy.nn.runners import format_inference_results_table, run_inference_jobs
    except ImportError:
        print(
            "Error: Neural network modules not available.\n"
            "Install from source: pip install git+https://github.com/hmblair/ciffy.git",
            file=sys.stderr,
        )
        sys.exit(1)

    # Expand glob patterns in config paths
    config_paths = []
    for pattern in args.configs:
        expanded = glob(pattern)
        if not expanded:
            print(f"Warning: No files match pattern: {pattern}", file=sys.stderr)
        config_paths.extend(sorted(expanded))

    if not config_paths:
        print("Error: No config files found.", file=sys.stderr)
        sys.exit(1)

    # Display inference plan
    print()
    print("=" * 60)
    print("Ciffy Inference Runner")
    print("=" * 60)
    print(f"Configs: {len(config_paths)}")
    print(f"Parallel: {not args.sequential}")
    print(f"Device: {args.device}")
    print()

    for i, path in enumerate(config_paths, 1):
        print(f"  {i}. {path}")
    print()

    # Run inference jobs
    print("Running inference...")
    print("-" * 60)

    try:
        results = run_inference_jobs(
            config_paths=config_paths,
            parallel=not args.sequential,
            device=args.device,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error running inference: {e}", file=sys.stderr)
        sys.exit(1)

    # Print results table
    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print(format_inference_results_table(results))
    print()

    # Exit with error code if any jobs failed
    failed = sum(1 for r in results if r.status != "success")
    if failed > 0:
        sys.exit(1)


def _train_command(args):
    """Handle the train subcommand."""
    try:
        import torch
    except ImportError:
        print(
            "Error: PyTorch is required for training.\n"
            "Install with: pip install torch",
            file=sys.stderr,
        )
        sys.exit(1)

    from glob import glob

    try:
        from ciffy.nn.runners import format_training_results_table, run_training_jobs
    except ImportError:
        print(
            "Error: Neural network modules not available.\n"
            "Install from source: pip install git+https://github.com/hmblair/ciffy.git",
            file=sys.stderr,
        )
        sys.exit(1)

    # Expand glob patterns in config paths
    config_paths = []
    for pattern in args.configs:
        expanded = glob(pattern)
        if not expanded:
            print(f"Warning: No files match pattern: {pattern}", file=sys.stderr)
        config_paths.extend(sorted(expanded))

    if not config_paths:
        print("Error: No config files found.", file=sys.stderr)
        sys.exit(1)

    # Display training plan
    print()
    print("=" * 60)
    print("Ciffy Training Runner")
    print("=" * 60)
    print(f"Configs: {len(config_paths)}")
    print(f"Parallel: {not args.sequential}")
    print(f"Device: {args.device}")
    print()

    for i, path in enumerate(config_paths, 1):
        print(f"  {i}. {path}")
    print()

    # Run training jobs
    print("Running training...")
    print("-" * 60)

    try:
        results = run_training_jobs(
            config_paths=config_paths,
            parallel=not args.sequential,
            device=args.device,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error running training: {e}", file=sys.stderr)
        sys.exit(1)

    # Print results table
    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print(format_training_results_table(results))
    print()

    # Exit with error code if any jobs failed
    failed = sum(1 for r in results if r.status != "success")
    if failed > 0:
        sys.exit(1)


def main():
    """Main entry point for the ciffy CLI."""
    # Check if first argument is a subcommand
    subcommands = {"map", "info", "split", "template", "train", "experiment", "inference", "download"}

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

    # Template subcommand
    template_parser = subparsers.add_parser(
        "template",
        help="Create a template structure from a sequence",
        description="Generate a polymer template from a sequence string with sampled backbone dihedrals.",
    )
    template_parser.add_argument(
        "sequence",
        help="Sequence string (e.g., 'MGKLF' for protein, 'acgu' for RNA)",
    )
    template_parser.add_argument(
        "--output", "-o",
        help="Output file path (.cif format)",
    )
    template_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible sampling (default: None)",
    )
    template_parser.add_argument(
        "--no-clash-free",
        action="store_true",
        help="Disable clash-free sampling. By default, backbone sampling uses "
             "autoregressive sampling with clash detection to avoid steric "
             "overlaps. Use this flag for faster (but potentially overlapping) sampling.",
    )

    # Train subcommand
    train_parser = subparsers.add_parser(
        "train",
        help="Run training from config files",
        description=(
            "Run model training from YAML config files.\n"
            "Supports multiple trainer types (flow, latent_diffusion, diffusion).\n"
            "Multiple configs run in parallel by default."
        ),
    )
    train_parser.add_argument(
        "configs",
        nargs="+",
        help="Config file paths or glob patterns (e.g., configs/*.yaml)",
    )
    train_parser.add_argument(
        "--sequential", "-s",
        action="store_true",
        help="Run training jobs sequentially (default: parallel)",
    )
    train_parser.add_argument(
        "--device", "-d",
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device strategy (default: auto)",
    )

    # Experiment subcommand
    experiment_parser = subparsers.add_parser(
        "experiment",
        help="Run multiple training experiments",
        description=(
            "Run multiple VAE training experiments from config files.\n"
            "Supports parallel execution across GPUs."
        ),
    )
    experiment_parser.add_argument(
        "configs",
        nargs="+",
        help="Config file paths or glob patterns (e.g., configs/*.yaml)",
    )
    experiment_parser.add_argument(
        "--sequential", "-s",
        action="store_true",
        help="Run experiments sequentially (default: parallel)",
    )
    experiment_parser.add_argument(
        "--device", "-d",
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device strategy (default: auto)",
    )

    # Inference subcommand
    inference_parser = subparsers.add_parser(
        "inference",
        help="Run inference to generate structures",
        description=(
            "Generate polymer structures from sequences using trained VAE models.\n"
            "Supports parallel execution across GPUs."
        ),
    )
    inference_parser.add_argument(
        "configs",
        nargs="+",
        help="Config file paths or glob patterns (e.g., configs/*.yaml)",
    )
    inference_parser.add_argument(
        "--sequential", "-s",
        action="store_true",
        help="Run inference jobs sequentially (default: parallel)",
    )
    inference_parser.add_argument(
        "--device", "-d",
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device strategy (default: auto)",
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
        "--id",
        type=str,
        nargs="+",
        default=None,
        help="Download specific PDB ID(s) instead of searching (e.g., --id 1EHZ 4V9F)",
    )
    download_parser.add_argument(
        "--type", "-t",
        type=str,
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

    args = parser.parse_args()

    # Route to appropriate handler
    if args.command == "train":
        _train_command(args)
    elif args.command == "experiment":
        _experiment_command(args)
    elif args.command == "template":
        _template_command(args)
    elif args.command == "inference":
        _inference_command(args)
    elif args.command == "download":
        _download_command(args)
    elif args.command == "map":
        _map_command(args)
    elif args.command == "split":
        _split_command(args)
    elif args.command == "info":
        _info_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
