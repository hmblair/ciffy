"""
Command-line interface for ciffy.

Usage:
    ciffy info <file.cif>            # Load and print polymer summary
    ciffy info <file1> <file2> ...   # Load and print multiple files
    ciffy info <file.cif> --poly     # Show only polymer atoms
    ciffy info <file.cif> --desc     # Show entity descriptions per chain
    ciffy map <file.cif>             # Display contact map
    ciffy split <file.cif>           # Split into per-chain files
    ciffy template <sequence>        # Create template from sequence with sampled dihedrals
    ciffy cluster data/*.cif         # Cluster structures by similarity, return representatives
    ciffy train configs/*.yaml       # Run training from config files
    ciffy experiment configs/*.yaml  # Run multiple training experiments
    ciffy predict model.safetensors --sequence ACGU -o out.cif  # Generate structure
    ciffy predict --config inference.yaml  # Batch prediction from config
    ciffy download --max_count 100   # Download structures from RCSB PDB
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
                polymer = load(filepath, load_descriptions=args.desc)
                if args.poly:
                    polymer = polymer.poly()
                if not polymers:  # First successful load
                    backend = polymer.backend
                rows = polymer.chain_info()
                polymers.append((polymer.pdb_id, filepath, rows))
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
            polymer = load(filepath, load_descriptions=args.desc)
            if args.poly:
                polymer = polymer.poly()
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
        from ciffy.nn.split import DataSplit

        train, val, test = split_ratios
        split = DataSplit.from_clusters(
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
            dirs = split.to_directories(
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


def _predict_command(args):
    """Handle the predict subcommand (unified inference)."""
    from glob import glob
    from pathlib import Path

    try:
        import torch
    except ImportError:
        print(
            "Error: PyTorch is required for prediction.\n"
            "Install with: pip install torch",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        from ciffy.nn import load_model, get_model_info
    except ImportError:
        print(
            "Error: Neural network modules not available.\n"
            "Install with: pip install ciffy[nn]",
            file=sys.stderr,
        )
        sys.exit(1)

    # Config file mode - use batch runner
    if args.config:
        from ciffy.nn.runners import run_inference_jobs, format_inference_results_table

        # Expand glob patterns
        config_paths = []
        for pattern in args.config:
            expanded = glob(pattern)
            if not expanded:
                print(f"Warning: No files match pattern: {pattern}", file=sys.stderr)
            config_paths.extend(sorted(expanded))

        if not config_paths:
            print("Error: No config files found.", file=sys.stderr)
            sys.exit(1)

        if not args.quiet:
            print()
            print("=" * 60)
            print("Ciffy Predict (Batch Mode)")
            print("=" * 60)
            print(f"Configs: {len(config_paths)}")
            print(f"Parallel: {not args.sequential}")
            print(f"Device: {args.device}")
            print()
            for i, path in enumerate(config_paths, 1):
                print(f"  {i}. {path}")
            print()
            print("Running predictions...")
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
            print(f"Error running predictions: {e}", file=sys.stderr)
            sys.exit(1)

        if not args.quiet:
            print()
            print("=" * 60)
            print("Results")
            print("=" * 60)
            print(format_inference_results_table(results))
            print()

        failed = sum(1 for r in results if r.status != "success")
        if failed > 0:
            sys.exit(1)
        return

    # Direct CLI mode - model is required
    if not args.model:
        print("Error: Model path required for direct mode.", file=sys.stderr)
        print("Usage: ciffy predict model.safetensors --sequence ACGU", file=sys.stderr)
        sys.exit(1)

    from ciffy import from_sequence

    # Determine device
    device = args.device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    # Collect sequences
    sequences = []  # List of (id, sequence) tuples

    if args.sequence:
        for i, seq in enumerate(args.sequence):
            sequences.append((f"seq_{i}", seq))
    elif args.fasta:
        # Read sequences from FASTA
        try:
            with open(args.fasta) as f:
                content = f.read()

            if content.startswith(">"):
                # FASTA format
                current_id = None
                current_seq = []
                for line in content.splitlines():
                    line = line.strip()
                    if line.startswith(">"):
                        if current_id is not None:
                            sequences.append((current_id, "".join(current_seq)))
                        current_id = line[1:].split()[0]
                        current_seq = []
                    elif line:
                        current_seq.append(line)
                if current_id is not None:
                    sequences.append((current_id, "".join(current_seq)))
            else:
                # Plain text, one per line
                for i, line in enumerate(content.splitlines()):
                    line = line.strip()
                    if line:
                        sequences.append((f"seq_{i}", line))

            if not sequences:
                print(f"Error: No sequences found in {args.fasta}", file=sys.stderr)
                sys.exit(1)
        except FileNotFoundError:
            print(f"Error: FASTA file not found: {args.fasta}", file=sys.stderr)
            sys.exit(1)
    else:
        print("Error: Must provide --sequence, --fasta, or --config", file=sys.stderr)
        sys.exit(1)

    # Check model file exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    # Show model info
    if not args.quiet:
        info = get_model_info(model_path)
        print(f"Loading {info['model_type']} model...")

    # Load model
    try:
        model = load_model(model_path, device=device)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)

    # Get atom filter from model
    atom_filter = getattr(model, "atom_filter", None)
    if atom_filter is None and hasattr(model, "flow_model"):
        atom_filter = model.flow_model.atom_filter

    # Set seed if specified
    if args.seed is not None:
        torch.manual_seed(args.seed)

    # Determine output handling
    output = Path(args.output)
    single_sequence = len(sequences) == 1
    single_sample = args.n_samples == 1

    # Generate structures for each sequence
    total_structures = 0
    for seq_id, sequence in sequences:
        if not args.quiet:
            seq_display = sequence[:50] + ('...' if len(sequence) > 50 else '')
            print(f"Generating {args.n_samples} sample(s) for {seq_id}: {seq_display} ({len(sequence)} residues)")

        # Create template
        try:
            template = from_sequence(sequence, atoms=atom_filter)
        except Exception as e:
            print(f"Error creating template for {seq_id}: {e}", file=sys.stderr)
            continue

        # Generate samples
        try:
            with torch.no_grad():
                samples = model.sample(
                    template,
                    n_samples=args.n_samples,
                    temperature=args.temperature,
                )
        except Exception as e:
            print(f"Error generating samples for {seq_id}: {e}", file=sys.stderr)
            continue

        # Write outputs
        if single_sequence and single_sample:
            # Single file output
            out_path = output if output.suffix == ".cif" else output.with_suffix(".cif")
            samples[0].write(str(out_path))
            if not args.quiet:
                print(f"Wrote {out_path} ({samples[0].size()} atoms)")
        elif single_sequence:
            # Multiple samples, single sequence -> numbered files
            if output.suffix == ".cif":
                base = output.stem
                parent = output.parent
            else:
                base = "sample"
                parent = output
            parent.mkdir(parents=True, exist_ok=True)
            for i, sample in enumerate(samples):
                out_path = parent / f"{base}_{i:03d}.cif"
                sample.write(str(out_path))
                if not args.quiet:
                    print(f"Wrote {out_path}")
        else:
            # Multiple sequences -> directory structure
            output.mkdir(parents=True, exist_ok=True)
            for i, sample in enumerate(samples):
                out_path = output / f"{seq_id}_{i:03d}.cif"
                sample.write(str(out_path))
                if not args.quiet:
                    print(f"Wrote {out_path}")

        total_structures += len(samples)

    if not args.quiet:
        print(f"Generated {total_structures} structure(s)")


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
    subcommands = {"map", "info", "split", "template", "train", "experiment", "predict", "download", "cluster"}

    # If no args or first arg starts with - or is not a subcommand,
    # treat as the info command (deprecated)
    if len(sys.argv) > 1 and sys.argv[1] not in subcommands and not sys.argv[1].startswith('-'):
        # Show deprecation warning
        print(
            f"Warning: 'ciffy <file>' is deprecated. Use 'ciffy info <file>' instead.",
            file=sys.stderr,
        )
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
    info_parser.add_argument(
        "--poly", "-p",
        action="store_true",
        help="Show only polymer atoms (exclude water, ions, ligands)",
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

    # Predict subcommand (unified inference)
    predict_parser = subparsers.add_parser(
        "predict",
        help="Generate structures from a trained model",
        description=(
            "Generate polymer structures from sequences using trained models.\n\n"
            "Two modes:\n"
            "  Direct:  ciffy predict model.safetensors --sequence ACGU -o out.cif\n"
            "  Batch:   ciffy predict --config configs/*.yaml"
        ),
    )
    predict_parser.add_argument(
        "model",
        nargs="?",
        help="Path to model file (.safetensors). Required for direct mode.",
    )
    predict_parser.add_argument(
        "--sequence", "-s",
        nargs="+",
        help="Sequence(s) to generate structures for (e.g., ACGU or MGKLF)",
    )
    predict_parser.add_argument(
        "--fasta", "-f",
        help="Path to FASTA or plain text file with sequences",
    )
    predict_parser.add_argument(
        "--config", "-c",
        nargs="+",
        help="Config file(s) for batch mode (glob patterns supported)",
    )
    predict_parser.add_argument(
        "--output", "-o",
        default="output.cif",
        help="Output path: file.cif for single, directory for multiple (default: output.cif)",
    )
    predict_parser.add_argument(
        "--n-samples", "-n",
        type=int,
        default=1,
        help="Number of samples per sequence (default: 1)",
    )
    predict_parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0)",
    )
    predict_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    predict_parser.add_argument(
        "--device", "-d",
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use (default: auto)",
    )
    predict_parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run batch jobs sequentially (default: parallel)",
    )
    predict_parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output",
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
        "--id",
        type=str,
        nargs="+",
        default=None,
        help="Download specific PDB ID(s) instead of searching (e.g., --id 1EHZ 4V9F)",
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
    if args.command == "train":
        _train_command(args)
    elif args.command == "experiment":
        _experiment_command(args)
    elif args.command == "template":
        _template_command(args)
    elif args.command == "predict":
        _predict_command(args)
    elif args.command == "download":
        _download_command(args)
    elif args.command == "map":
        _map_command(args)
    elif args.command == "split":
        _split_command(args)
    elif args.command == "info":
        _info_command(args)
    elif args.command == "cluster":
        _cluster_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
