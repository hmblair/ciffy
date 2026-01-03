#!/usr/bin/env python
"""
Train a latent diffusion model for polymer structure generation.

Uses a pre-trained PolymerModel (flow, VAE, or consolidated) as the encoder/decoder,
and trains a transformer denoiser in the latent space.

Examples:
    # Train with default flow encoder
    python scripts/train_latent_diffusion.py --encoder-path outputs/models/flow

    # Train with VAE encoder
    python scripts/train_latent_diffusion.py --encoder-path outputs/models/vae

    # Quick test run
    python scripts/train_latent_diffusion.py --encoder-path outputs/models/flow --epochs 5 --max-files 50
"""

import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")


def main():
    import os
    os.environ["CIFFY_LOG_LEVEL"] = "WARNING"

    import lightning as L
    import torch

    from ciffy.nn.polymer import PolymerModel
    from ciffy.nn.diffusion import LatentDiffusionConfig, LatentDiffusionModel
    from ciffy.nn.diffusion.latent_denoiser import LatentDenoiserConfig
    from ciffy.nn.lightning.modules.latent_diffusion import (
        LatentDiffusionModule,
        LatentDiffusionFullConfig,
        LatentDiffusionDataConfig,
    )
    from ciffy.nn.lightning.data.diffusion import LatentDiffusionDataModule
    from ciffy.nn.config import TrainingConfig, SchedulerConfig

    parser = argparse.ArgumentParser(description="Train latent diffusion model")
    parser.add_argument("--encoder-path", required=True, help="Path to trained PolymerModel")
    parser.add_argument("--data-dir", default="/Users/hmblair/academic/data/structures/rna")
    parser.add_argument("--max-files", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-timesteps", type=int, default=1000)
    parser.add_argument("--noise-schedule", default="cosine", choices=["cosine", "linear"])
    parser.add_argument("--accelerator", default="cpu")
    parser.add_argument("--output-dir", default="outputs/diffusion")
    parser.add_argument("--sample", action="store_true", help="Sample after training")
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--sample-steps", type=int, default=50)
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Latent Diffusion Training")
    print("=" * 60)

    # Load encoder model
    print(f"\nLoading encoder from {args.encoder_path}...")
    encoder_model = PolymerModel.load(args.encoder_path)
    latent_dim = encoder_model.latent_dim
    print(f"  Latent dim: {latent_dim}")
    print(f"  Residue types: {list(encoder_model.residue_models.keys())}")

    # Build config
    denoiser_config = LatentDenoiserConfig(
        latent_dim=latent_dim,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_timesteps=args.num_timesteps,
    )

    model_config = LatentDiffusionConfig(
        denoiser=denoiser_config,
        num_timesteps=args.num_timesteps,
        noise_schedule=args.noise_schedule,
        encoder_path=args.encoder_path,
        freeze_encoder=True,
    )

    training_config = TrainingConfig(
        lr=args.lr,
        weight_decay=0.01,
        scheduler=SchedulerConfig(
            scheduler_type="cosine",
            warmup_epochs=10,
        ),
    )

    data_config = LatentDiffusionDataConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        molecule_types=("RNA",),
        min_residues=10,
        max_residues=200,
    )

    full_config = LatentDiffusionFullConfig(
        model=model_config,
        data=data_config,
        training=training_config,
    )

    # Create data module
    print(f"\nLoading data from {args.data_dir}...")
    cif_files = sorted(Path(args.data_dir).glob("*.cif"))[:args.max_files]
    print(f"  Found {len(cif_files)} CIF files")

    dm = LatentDiffusionDataModule(
        data_dir=args.data_dir,
        encoder_model=encoder_model,
        batch_size=args.batch_size,
        molecule_types=("RNA",),
        min_residues=10,
        max_residues=200,
    )

    # Create module
    module = LatentDiffusionModule(full_config, encoder_model=encoder_model)

    # Count parameters
    denoiser_params = sum(p.numel() for p in module.model.denoiser.parameters())
    print(f"\nDenoiser parameters: {denoiser_params:,}")

    # Train
    print(f"\nTraining for {args.epochs} epochs...")
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=1,
        enable_progress_bar=True,
        enable_checkpointing=False,
        logger=False,
    )

    trainer.fit(module, dm)

    # Save model
    print(f"\nSaving model to {output_path}...")
    tensors, config = module.model.get_save_state()

    import json
    from safetensors.torch import save_file

    save_file(tensors, output_path / "model.safetensors")
    with open(output_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Sample
    if args.sample:
        print("\n" + "=" * 60)
        print("Sampling...")
        print("=" * 60)

        import numpy as np
        np.random.seed(42)

        chains_dir = output_path / "chains"
        chains_dir.mkdir(exist_ok=True)

        sequence = "".join(np.random.choice(list("acgu"), 20))
        print(f"  Sequence: {sequence}")

        module.model.eval()
        with torch.no_grad():
            polymers = module.model.sample_from_sequence(
                sequence,
                n_samples=args.n_samples,
                num_steps=args.sample_steps,
            )

        for i, polymer in enumerate(polymers):
            output_file = chains_dir / f"sample_{i}.cif"
            polymer.write(str(output_file))
            print(f"  Saved: {output_file.name} ({polymer.size()} atoms)")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
