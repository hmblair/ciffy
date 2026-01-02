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

    # Training
    ciffy train flow --data /path --output /path              # Train flow model
    ciffy train latent-diffusion --data /path --output /path  # Train latent diffusion
    ciffy train coord-diffusion --data /path --output /path   # Train coordinate diffusion

    # Prediction/Sampling
    ciffy predict flow model_dir --sequence acgu -o out.cif           # Sample from flow
    ciffy predict latent-diffusion model.safetensors --sequence acgu  # Generate from diffusion
    ciffy predict coord-diffusion model.safetensors --sequence acgu   # Generate from diffusion

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
                skip = ["connections"] if args.desc else ["descriptions", "connections"]
                polymer = load(filepath, skip=skip)
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
            skip = ["connections"] if args.desc else ["descriptions", "connections"]
            polymer = load(filepath, skip=skip)
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


def _train_flow_command(args):
    """Handle the train flow subcommand."""
    try:
        import lightning as L
        import torch
    except ImportError:
        print(
            "Error: PyTorch and Lightning are required for training.\n"
            "Install with: pip install torch lightning",
            file=sys.stderr,
        )
        sys.exit(1)

    from glob import glob
    from pathlib import Path

    from ciffy import Residue
    from ciffy.nn.lightning import FlowDataModule, ResidueFlowModule
    from ciffy.nn.lightning.modules.residue_flow import (
        ResidueFlowFullConfig,
        ResidueFlowModelConfig,
        ResidueFlowDataConfig,
    )
    from ciffy.nn.config import TrainingConfig
    from ciffy.nn.flow import PolymerFlowModel

    # Expand glob patterns in data paths
    cif_paths = []
    for pattern in glob(str(Path(args.data) / "*.cif")):
        cif_paths.append(pattern)

    if not cif_paths:
        # Try as glob pattern directly
        cif_paths = glob(args.data)

    if not cif_paths:
        print(f"Error: No CIF files found in {args.data}", file=sys.stderr)
        sys.exit(1)

    # Parse residue types (convert to uppercase for Residue lookup)
    residue_chars = args.residues.upper()
    residues = [getattr(Residue, c) for c in residue_chars]

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create config
    config = ResidueFlowFullConfig(
        model=ResidueFlowModelConfig(
            latent_dim=args.latent_dim,
            n_layers=args.n_layers,
            hidden_dim=args.hidden_dim,
            noise_std=args.noise_std,
        ),
        data=ResidueFlowDataConfig(
            batch_size=args.batch_size,
            min_coverage=args.min_coverage,
        ),
        training=TrainingConfig(
            lr=args.lr,
            epochs=args.epochs,
            grad_clip=args.grad_clip,
        ),
    )

    # Determine accelerator
    accelerator = args.accelerator
    if accelerator == "auto":
        if torch.cuda.is_available():
            accelerator = "gpu"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            accelerator = "mps"
        else:
            accelerator = "cpu"

    if not args.quiet:
        print()
        print("=" * 60)
        print("Ciffy Flow Model Training")
        print("=" * 60)
        print(f"Data: {len(cif_paths)} CIF files")
        print(f"Residues: {residue_chars}")
        print(f"Output: {output_dir}")
        print(f"Epochs: {args.epochs}")
        print(f"Accelerator: {accelerator}")
        print()

    # Set up W&B logging if requested
    logger = None
    if args.wandb:
        from lightning.pytorch.loggers import WandbLogger
        logger = WandbLogger(
            project=args.wandb_project or "ciffy-flow",
            name=args.wandb_name,
        )

    # Train each residue type
    models = {}
    for residue in residues:
        if not args.quiet:
            print(f"\nTraining model for {residue.name}...")
            print("-" * 40)

        try:
            # Create data module
            dm = FlowDataModule(
                cif_paths=cif_paths,
                residue=residue,
                batch_size=args.batch_size,
                min_coverage=args.min_coverage,
            )

            # Create Lightning module
            module = ResidueFlowModule(config, residue)

            # Create trainer
            trainer = L.Trainer(
                max_epochs=args.epochs,
                accelerator=accelerator,
                enable_progress_bar=not args.quiet,
                enable_model_summary=not args.quiet,
                logger=logger,
                default_root_dir=str(output_dir),
            )

            # Train
            trainer.fit(module, dm)

            # Get trained model
            model = module.get_model()
            models[residue] = model

            if not args.quiet:
                print(f"  {residue.name}: {model.n_atoms} atoms, latent_dim={model.latent_dim}")

        except Exception as e:
            print(f"  {residue.name}: Failed - {e}", file=sys.stderr)
            if args.verbose:
                import traceback
                traceback.print_exc()

    if not models:
        print("\nNo models were trained successfully.", file=sys.stderr)
        sys.exit(1)

    # Create and save PolymerFlowModel
    polymer_model = PolymerFlowModel(models)
    save_path = output_dir / "model"
    polymer_model.save(save_path)

    if not args.quiet:
        print()
        print("=" * 60)
        print("Training Complete")
        print("=" * 60)
        print(f"Trained {len(models)} residue models: {[r.name for r in models.keys()]}")
        print(f"Saved to: {save_path}")


def _train_latent_diffusion_command(args):
    """Handle the train latent-diffusion subcommand."""
    try:
        import lightning as L
        import torch
    except ImportError:
        print(
            "Error: PyTorch and Lightning are required for training.\n"
            "Install with: pip install torch lightning",
            file=sys.stderr,
        )
        sys.exit(1)

    from pathlib import Path

    from ciffy.nn.flow import load_pretrained
    from ciffy.nn.lightning.data.diffusion import LatentDiffusionDataModule
    from ciffy.nn.lightning.modules.latent_diffusion import (
        LatentDiffusionFullConfig,
        LatentDiffusionDataConfig,
        LatentDiffusionModule,
    )
    from ciffy.nn.diffusion.latent_diffusion import LatentDiffusionConfig
    from ciffy.nn.config import TrainingConfig

    # Check data directory
    data_dir = Path(args.data)
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine accelerator
    accelerator = args.accelerator
    if accelerator == "auto":
        if torch.cuda.is_available():
            accelerator = "gpu"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            accelerator = "mps"
        else:
            accelerator = "cpu"

    # Load flow model
    if not args.quiet:
        print("Loading flow model...")

    if args.flow_model:
        from ciffy.nn.flow import PolymerFlowModel
        flow_model = PolymerFlowModel.load(args.flow_model, device="cpu")
    else:
        flow_model = load_pretrained("rna", device="cpu")

    # Create config
    config = LatentDiffusionFullConfig(
        model=LatentDiffusionConfig(
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            num_timesteps=args.num_timesteps,
        ),
        data=LatentDiffusionDataConfig(
            batch_size=args.batch_size,
            min_residues=args.min_residues,
            max_residues=args.max_residues,
        ),
        training=TrainingConfig(
            lr=args.lr,
            epochs=args.epochs,
        ),
    )

    if not args.quiet:
        print()
        print("=" * 60)
        print("Ciffy Latent Diffusion Training")
        print("=" * 60)
        print(f"Data: {data_dir}")
        print(f"Output: {output_dir}")
        print(f"Epochs: {args.epochs}")
        print(f"Accelerator: {accelerator}")
        print(f"Flow model: {args.flow_model or 'pretrained RNA'}")
        print()

    # Set up W&B logging if requested
    logger = None
    if args.wandb:
        from lightning.pytorch.loggers import WandbLogger
        logger = WandbLogger(
            project=args.wandb_project or "ciffy-latent-diffusion",
            name=args.wandb_name,
        )

    # Create data module
    dm = LatentDiffusionDataModule(
        data_dir=data_dir,
        flow_model=flow_model,
        batch_size=args.batch_size,
        min_residues=args.min_residues,
        max_residues=args.max_residues,
    )

    # Create Lightning module
    module = LatentDiffusionModule(config, flow_model=flow_model)

    # Create trainer
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator=accelerator,
        enable_progress_bar=not args.quiet,
        enable_model_summary=not args.quiet,
        logger=logger,
        default_root_dir=str(output_dir),
    )

    # Train
    trainer.fit(module, dm)

    # Save model
    save_path = output_dir / "model.safetensors"
    module.model.save(save_path)

    if not args.quiet:
        print()
        print("=" * 60)
        print("Training Complete")
        print("=" * 60)
        print(f"Saved to: {save_path}")


def _train_coord_diffusion_command(args):
    """Handle the train coord-diffusion subcommand."""
    try:
        import lightning as L
        import torch
    except ImportError:
        print(
            "Error: PyTorch and Lightning are required for training.\n"
            "Install with: pip install torch lightning",
            file=sys.stderr,
        )
        sys.exit(1)

    from pathlib import Path

    from ciffy.nn.lightning.data.diffusion import CoordinateDiffusionDataModule
    from ciffy.nn.lightning.modules.coordinate_diffusion import (
        CoordinateDiffusionFullConfig,
        CoordinateDiffusionDataConfig,
        CoordinateDiffusionModule,
    )
    from ciffy.nn.diffusion.coordinate_diffusion import CoordinateDiffusionConfig
    from ciffy.nn.config import TrainingConfig

    # Check data directory
    data_dir = Path(args.data)
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine accelerator
    accelerator = args.accelerator
    if accelerator == "auto":
        if torch.cuda.is_available():
            accelerator = "gpu"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            accelerator = "mps"
        else:
            accelerator = "cpu"

    # Create config
    config = CoordinateDiffusionFullConfig(
        model=CoordinateDiffusionConfig(
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            num_timesteps=args.num_timesteps,
        ),
        data=CoordinateDiffusionDataConfig(
            batch_size=args.batch_size,
            min_atoms=args.min_atoms,
            max_atoms=args.max_atoms,
        ),
        training=TrainingConfig(
            lr=args.lr,
            epochs=args.epochs,
        ),
    )

    if not args.quiet:
        print()
        print("=" * 60)
        print("Ciffy Coordinate Diffusion Training")
        print("=" * 60)
        print(f"Data: {data_dir}")
        print(f"Output: {output_dir}")
        print(f"Epochs: {args.epochs}")
        print(f"Accelerator: {accelerator}")
        print()

    # Set up W&B logging if requested
    logger = None
    if args.wandb:
        from lightning.pytorch.loggers import WandbLogger
        logger = WandbLogger(
            project=args.wandb_project or "ciffy-coord-diffusion",
            name=args.wandb_name,
        )

    # Create data module
    dm = CoordinateDiffusionDataModule(
        data_dir=data_dir,
        batch_size=args.batch_size,
        min_atoms=args.min_atoms,
        max_atoms=args.max_atoms,
    )

    # Create Lightning module
    module = CoordinateDiffusionModule(config)

    # Create trainer
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator=accelerator,
        enable_progress_bar=not args.quiet,
        enable_model_summary=not args.quiet,
        logger=logger,
        default_root_dir=str(output_dir),
    )

    # Train
    trainer.fit(module, dm)

    # Save model
    save_path = output_dir / "model.safetensors"
    module.model.save(save_path)

    if not args.quiet:
        print()
        print("=" * 60)
        print("Training Complete")
        print("=" * 60)
        print(f"Saved to: {save_path}")


def _predict_flow_command(args):
    """Handle the predict flow subcommand."""
    from pathlib import Path

    from ciffy import from_sequence
    from ciffy.nn.flow import PolymerFlowModel

    # Load model
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    if not args.quiet:
        print(f"Loading flow model from {model_path}...")

    model = PolymerFlowModel.load(model_path)

    if not args.quiet:
        print(f"Residues: {[r.name for r in model.residue_types]}")
        print(f"Latent dim: {model.latent_dim}")

    # Create template
    template = from_sequence(args.sequence, atoms=model.atom_filter)

    if not args.quiet:
        print(f"\nSampling {args.n_samples} conformation(s) for '{args.sequence}'...")

    # Sample
    import torch
    if args.seed is not None:
        torch.manual_seed(args.seed)

    samples = model.sample(template, n_samples=args.n_samples)

    # Save outputs
    output = Path(args.output)
    if args.n_samples == 1:
        # Single file
        out_path = output if output.suffix == ".cif" else output.with_suffix(".cif")
        samples[0].write(str(out_path))
        if not args.quiet:
            print(f"Saved to {out_path}")
    else:
        # Multiple files
        output.mkdir(parents=True, exist_ok=True)
        for i, polymer in enumerate(samples):
            out_path = output / f"sample_{i:03d}.cif"
            polymer.write(str(out_path))
            if not args.quiet:
                print(f"Saved {out_path}")


def _predict_latent_diffusion_command(args):
    """Handle the predict latent-diffusion subcommand."""
    from pathlib import Path

    import torch

    from ciffy import from_sequence
    from ciffy.nn.flow import PolymerFlowModel, load_pretrained
    from ciffy.nn.diffusion.latent_diffusion import LatentDiffusionModel

    # Determine device
    device = args.device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    # Load flow model (for decoding)
    if not args.quiet:
        print(f"Loading flow model...")

    if args.flow_model:
        flow_model = PolymerFlowModel.load(args.flow_model, device=device)
    else:
        flow_model = load_pretrained("rna", device=device)

    # Load diffusion model
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    if not args.quiet:
        print(f"Loading diffusion model from {model_path}...")

    model = LatentDiffusionModel.load(model_path, flow_model=flow_model)
    model = model.to(device)
    model.eval()

    # Create template
    template = from_sequence(args.sequence, atoms=flow_model.atom_filter)

    if not args.quiet:
        print(f"\nGenerating {args.n_samples} structure(s) for '{args.sequence}'...")

    # Set seed
    if args.seed is not None:
        torch.manual_seed(args.seed)

    # Generate
    with torch.no_grad():
        samples = model.sample(template, n_samples=args.n_samples, num_steps=args.steps)

    # Save outputs
    output = Path(args.output)
    if args.n_samples == 1:
        out_path = output if output.suffix == ".cif" else output.with_suffix(".cif")
        samples[0].write(str(out_path))
        if not args.quiet:
            print(f"Saved to {out_path}")
    else:
        output.mkdir(parents=True, exist_ok=True)
        for i, polymer in enumerate(samples):
            out_path = output / f"sample_{i:03d}.cif"
            polymer.write(str(out_path))
            if not args.quiet:
                print(f"Saved {out_path}")


def _predict_coord_diffusion_command(args):
    """Handle the predict coord-diffusion subcommand."""
    from pathlib import Path

    import torch

    from ciffy import from_sequence
    from ciffy.nn.diffusion.coordinate_diffusion import CoordinateDiffusionModel

    # Determine device
    device = args.device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    # Load model
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    if not args.quiet:
        print(f"Loading diffusion model from {model_path}...")

    model = CoordinateDiffusionModel.load(model_path)
    model = model.to(device)
    model.eval()

    # Create template
    template = from_sequence(args.sequence)

    if not args.quiet:
        print(f"\nGenerating {args.n_samples} structure(s) for '{args.sequence}'...")

    # Set seed
    if args.seed is not None:
        torch.manual_seed(args.seed)

    # Generate
    with torch.no_grad():
        samples = model.sample(template, n_samples=args.n_samples, num_steps=args.steps)

    # Save outputs
    output = Path(args.output)
    if args.n_samples == 1:
        out_path = output if output.suffix == ".cif" else output.with_suffix(".cif")
        samples[0].write(str(out_path))
        if not args.quiet:
            print(f"Saved to {out_path}")
    else:
        output.mkdir(parents=True, exist_ok=True)
        for i, polymer in enumerate(samples):
            out_path = output / f"sample_{i:03d}.cif"
            polymer.write(str(out_path))
            if not args.quiet:
                print(f"Saved {out_path}")


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


def main():
    """Main entry point for the ciffy CLI."""
    # Check if first argument is a subcommand
    subcommands = {"map", "info", "split", "template", "train", "predict", "download", "cluster"}

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

    # Train subcommand with subparsers for model types
    train_parser = subparsers.add_parser(
        "train",
        help="Train models (flow, latent-diffusion, coord-diffusion)",
        description=(
            "Train ciffy models using PyTorch Lightning.\n\n"
            "Subcommands:\n"
            "  flow              Train residue flow models (ACGU)\n"
            "  latent-diffusion  Train latent diffusion model\n"
            "  coord-diffusion   Train coordinate diffusion model"
        ),
    )
    train_subparsers = train_parser.add_subparsers(dest="train_type")

    # Common arguments helper
    def add_common_train_args(parser):
        parser.add_argument(
            "--data", "-d",
            required=True,
            help="Path to training data (directory of CIF files or glob pattern)",
        )
        parser.add_argument(
            "--output", "-o",
            required=True,
            help="Output directory for trained model",
        )
        parser.add_argument(
            "--epochs", "-e",
            type=int,
            default=200,
            help="Number of training epochs (default: 200)",
        )
        parser.add_argument(
            "--lr",
            type=float,
            default=1e-3,
            help="Learning rate (default: 1e-3)",
        )
        parser.add_argument(
            "--batch-size", "-b",
            type=int,
            default=256,
            help="Batch size (default: 256)",
        )
        parser.add_argument(
            "--accelerator",
            default="auto",
            choices=["auto", "cpu", "gpu", "mps"],
            help="Accelerator (default: auto)",
        )
        parser.add_argument(
            "--wandb",
            action="store_true",
            help="Enable Weights & Biases logging",
        )
        parser.add_argument(
            "--wandb-project",
            help="W&B project name",
        )
        parser.add_argument(
            "--wandb-name",
            help="W&B run name",
        )
        parser.add_argument(
            "--grad-clip",
            type=float,
            default=None,
            help="Gradient clipping norm (default: None)",
        )
        parser.add_argument(
            "--quiet", "-q",
            action="store_true",
            help="Suppress progress output",
        )
        parser.add_argument(
            "--verbose", "-v",
            action="store_true",
            help="Show detailed error messages",
        )

    # Flow subcommand
    flow_parser = train_subparsers.add_parser(
        "flow",
        help="Train residue flow models",
        description="Train normalizing flow models for RNA residues (A, C, G, U).",
    )
    add_common_train_args(flow_parser)
    flow_parser.add_argument(
        "--residues",
        default="ACGU",
        help="Residue types to train (default: ACGU)",
    )
    flow_parser.add_argument(
        "--latent-dim",
        type=int,
        default=12,
        help="Latent space dimension (default: 12)",
    )
    flow_parser.add_argument(
        "--n-layers",
        type=int,
        default=6,
        help="Number of flow layers (default: 6)",
    )
    flow_parser.add_argument(
        "--hidden-dim",
        type=int,
        default=64,
        help="Hidden layer size (default: 64)",
    )
    flow_parser.add_argument(
        "--noise-std",
        type=float,
        default=0.05,
        help="Training noise standard deviation (default: 0.05)",
    )
    flow_parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.9,
        help="Minimum atom coverage for training data (default: 0.9)",
    )

    # Latent diffusion subcommand
    latent_parser = train_subparsers.add_parser(
        "latent-diffusion",
        help="Train latent diffusion model",
        description="Train diffusion model in latent space (requires pre-trained flow model).",
    )
    add_common_train_args(latent_parser)
    latent_parser.set_defaults(batch_size=32)  # Override default for diffusion
    latent_parser.add_argument(
        "--flow-model",
        help="Path to pre-trained flow model (default: built-in RNA model)",
    )
    latent_parser.add_argument(
        "--d-model",
        type=int,
        default=256,
        help="Transformer dimension (default: 256)",
    )
    latent_parser.add_argument(
        "--num-layers",
        type=int,
        default=6,
        help="Number of transformer layers (default: 6)",
    )
    latent_parser.add_argument(
        "--num-heads",
        type=int,
        default=8,
        help="Number of attention heads (default: 8)",
    )
    latent_parser.add_argument(
        "--num-timesteps",
        type=int,
        default=1000,
        help="Number of diffusion timesteps (default: 1000)",
    )
    latent_parser.add_argument(
        "--min-residues",
        type=int,
        default=10,
        help="Minimum residues per chain (default: 10)",
    )
    latent_parser.add_argument(
        "--max-residues",
        type=int,
        default=500,
        help="Maximum residues per chain (default: 500)",
    )

    # Coordinate diffusion subcommand
    coord_parser = train_subparsers.add_parser(
        "coord-diffusion",
        help="Train coordinate diffusion model",
        description="Train diffusion model directly on coordinates.",
    )
    add_common_train_args(coord_parser)
    coord_parser.set_defaults(batch_size=8)  # Override default for coord diffusion
    coord_parser.add_argument(
        "--d-model",
        type=int,
        default=256,
        help="Transformer dimension (default: 256)",
    )
    coord_parser.add_argument(
        "--num-layers",
        type=int,
        default=6,
        help="Number of transformer layers (default: 6)",
    )
    coord_parser.add_argument(
        "--num-heads",
        type=int,
        default=8,
        help="Number of attention heads (default: 8)",
    )
    coord_parser.add_argument(
        "--num-timesteps",
        type=int,
        default=1000,
        help="Number of diffusion timesteps (default: 1000)",
    )
    coord_parser.add_argument(
        "--min-atoms",
        type=int,
        default=50,
        help="Minimum atoms per chain (default: 50)",
    )
    coord_parser.add_argument(
        "--max-atoms",
        type=int,
        default=2000,
        help="Maximum atoms per chain (default: 2000)",
    )

    # Predict subcommand with subparsers for model types
    predict_parser = subparsers.add_parser(
        "predict",
        help="Generate structures (flow, latent-diffusion, coord-diffusion)",
        description=(
            "Generate polymer structures using trained models.\n\n"
            "Subcommands:\n"
            "  flow              Sample from flow model\n"
            "  latent-diffusion  Generate from latent diffusion model\n"
            "  coord-diffusion   Generate from coordinate diffusion model"
        ),
    )
    predict_subparsers = predict_parser.add_subparsers(dest="predict_type")

    # Common arguments helper for predict
    def add_common_predict_args(parser):
        parser.add_argument(
            "model",
            help="Path to model (directory for flow, .safetensors for diffusion)",
        )
        parser.add_argument(
            "--sequence", "-s",
            required=True,
            help="Sequence to generate (e.g., 'acguacgu' for RNA)",
        )
        parser.add_argument(
            "--output", "-o",
            default="output.cif",
            help="Output path (file for single, directory for multiple)",
        )
        parser.add_argument(
            "--n-samples", "-n",
            type=int,
            default=1,
            help="Number of samples to generate (default: 1)",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=None,
            help="Random seed for reproducibility",
        )
        parser.add_argument(
            "--quiet", "-q",
            action="store_true",
            help="Suppress output messages",
        )

    # Predict flow subcommand
    predict_flow_parser = predict_subparsers.add_parser(
        "flow",
        help="Sample conformations from a flow model",
        description="Generate polymer conformations using a trained flow model.",
    )
    add_common_predict_args(predict_flow_parser)

    # Predict latent-diffusion subcommand
    predict_latent_parser = predict_subparsers.add_parser(
        "latent-diffusion",
        help="Generate from latent diffusion model",
        description="Generate structures using a trained latent diffusion model.",
    )
    add_common_predict_args(predict_latent_parser)
    predict_latent_parser.add_argument(
        "--device", "-d",
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use (default: auto)",
    )
    predict_latent_parser.add_argument(
        "--flow-model",
        help="Path to flow model for decoding (default: built-in RNA model)",
    )
    predict_latent_parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Number of diffusion steps (default: 100)",
    )

    # Predict coord-diffusion subcommand
    predict_coord_parser = predict_subparsers.add_parser(
        "coord-diffusion",
        help="Generate from coordinate diffusion model",
        description="Generate structures using a trained coordinate diffusion model.",
    )
    add_common_predict_args(predict_coord_parser)
    predict_coord_parser.add_argument(
        "--device", "-d",
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use (default: auto)",
    )
    predict_coord_parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Number of diffusion steps (default: 100)",
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
        if args.train_type == "flow":
            _train_flow_command(args)
        elif args.train_type == "latent-diffusion":
            _train_latent_diffusion_command(args)
        elif args.train_type == "coord-diffusion":
            _train_coord_diffusion_command(args)
        else:
            train_parser.print_help()
    elif args.command == "template":
        _template_command(args)
    elif args.command == "predict":
        if args.predict_type == "flow":
            _predict_flow_command(args)
        elif args.predict_type == "latent-diffusion":
            _predict_latent_diffusion_command(args)
        elif args.predict_type == "coord-diffusion":
            _predict_coord_diffusion_command(args)
        else:
            predict_parser.print_help()
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
