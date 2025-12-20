#!/usr/bin/env python3
"""
Inference script for generating polymer structures from trained VAE models.

Loads a trained checkpoint, creates template structures from sequences,
runs model sampling, and saves .cif files.

Usage:
    python scripts/run_inference.py config.yaml
    python scripts/run_inference.py config.yaml --device cuda:0

Example config (config.yaml):
    model:
      checkpoint_path: ./checkpoints/vae/checkpoint_best.pt
      model_type: vae
      device: auto

    input:
      sequences:
        - MGKLF
        - acgu
      # OR
      # sequence_file: sequences.fasta

    sampling:
      n_samples: 10
      temperature: 1.0
      seed: 42

    output:
      output_dir: ./inference_output
      id_prefix: gen_
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Optional

try:
    import torch
except ImportError:
    print("PyTorch is required. Install with: pip install torch")
    sys.exit(1)

import numpy as np

import ciffy
from ciffy.nn.inference_config import InferenceConfig
from ciffy.nn.training import get_device, set_seed


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_sequences_from_file(filepath: str) -> tuple[list[str], list[str]]:
    """
    Load sequences from FASTA or plain text file.

    Auto-detects format and extracts IDs from FASTA headers if present.

    Args:
        filepath: Path to input file (FASTA or plain text).

    Returns:
        Tuple of (sequences, ids).
        For FASTA: extracts IDs from headers.
        For plain text: generates seq_0, seq_1, etc.

    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If file is empty or invalid.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Sequence file not found: {filepath}")

    sequences = []
    ids = []

    with open(path) as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        raise ValueError(f"Sequence file is empty: {filepath}")

    # Detect FASTA format
    is_fasta = any(line.startswith(">") for line in lines)

    if is_fasta:
        current_id = None
        current_seq = []

        for line in lines:
            if line.startswith(">"):
                # Save previous sequence
                if current_id is not None and current_seq:
                    sequences.append("".join(current_seq))
                    ids.append(current_id)

                # Start new sequence
                current_id = line[1:].split()[0]  # First word after >
                current_seq = []
            else:
                current_seq.append(line)

        # Save last sequence
        if current_id is not None and current_seq:
            sequences.append("".join(current_seq))
            ids.append(current_id)
    else:
        # Plain text: one sequence per line
        sequences = lines
        ids = [f"seq_{i}" for i in range(len(sequences))]

    return sequences, ids


def load_checkpoint_and_model(checkpoint_path: str, device: torch.device):
    """
    Load trained model from checkpoint.

    Args:
        checkpoint_path: Path to .pt checkpoint file.
        device: Device to load model on.

    Returns:
        Loaded model in eval mode.

    Raises:
        FileNotFoundError: If checkpoint doesn't exist.
        RuntimeError: If model can't be loaded.
    """
    from ciffy.nn.vae import PolymerVAE

    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract model config from checkpoint
    config_dict = checkpoint.get("config", {})
    model_config = config_dict.get("model", {})

    # Create model
    model = PolymerVAE(
        latent_dim=model_config.get("latent_dim", 64),
        hidden_dim=model_config.get("hidden_dim", 256),
        num_layers=model_config.get("num_layers", 4),
        num_heads=model_config.get("num_heads", 8),
        dropout=model_config.get("dropout", 0.1),
        beta=model_config.get("beta", 1.0),
    )

    # Load state dict
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model


def run_inference(
    config_path: str,
    device_override: Optional[str] = None,
    job_name: Optional[str] = None,
    quiet: bool = False,
) -> dict[str, Any]:
    """
    Run inference to generate structures from sequences.

    Main inference function that:
    1. Loads configuration from YAML
    2. Loads sequences (from list or file)
    3. Loads trained model checkpoint
    4. Generates structures for each sequence
    5. Saves .cif files to output directory

    Args:
        config_path: Path to YAML configuration file.
        device_override: Override device from config (e.g., "cuda:0", "cpu").
        job_name: Name for this job (used for output subdirectory if provided).
        quiet: If True, suppress logging output.

    Returns:
        Dict containing:
        - 'n_structures': Total structures generated
        - 'n_sequences': Number of sequences processed
        - 'output_dir': Path to output directory
        - 'device': Device string used
        - 'error': None if successful, error message otherwise

    Example:
        >>> result = run_inference("config.yaml", device_override="cuda")
        >>> if result["error"] is None:
        ...     print(f"Generated {result['n_structures']} structures")
    """
    try:
        # Load config
        config = InferenceConfig.from_yaml(config_path)

        # Override device if specified
        if device_override:
            config.model.device = device_override

        device = get_device(config.model.device or "auto")

        if not quiet:
            logger.info(f"Loaded config from {config_path}")
            logger.info(f"Using device: {device}")

        # Set random seed
        if config.sampling.seed is not None:
            set_seed(config.sampling.seed)
            if not quiet:
                logger.info(f"Set random seed to {config.sampling.seed}")

        # Load sequences
        if config.input.sequence_file:
            sequences, seq_ids = load_sequences_from_file(config.input.sequence_file)
        else:
            sequences = config.input.sequences
            if config.input.sequence_ids:
                seq_ids = config.input.sequence_ids
            else:
                seq_ids = [f"seq_{i}" for i in range(len(sequences))]

        if not quiet:
            logger.info(f"Loaded {len(sequences)} sequences")

        # Load model
        if not quiet:
            logger.info(f"Loading checkpoint: {config.model.checkpoint_path}")

        model = load_checkpoint_and_model(config.model.checkpoint_path, device)

        # Setup output directory
        output_dir = Path(config.output.output_dir)
        if job_name:
            output_dir = output_dir / job_name
        output_dir.mkdir(parents=True, exist_ok=True)

        if not quiet:
            logger.info(f"Output directory: {output_dir}")

        # Generate structures
        n_structures = 0
        latents_dict = {}  # For optional latent saving

        for seq_idx, (sequence, seq_id) in enumerate(zip(sequences, seq_ids)):
            if not quiet:
                logger.info(
                    f"Processing sequence {seq_idx + 1}/{len(sequences)}: {seq_id}"
                )

            # Create template from sequence
            try:
                template = ciffy.from_sequence(sequence, backend="torch")
                template = template.to(device)
            except Exception as e:
                logger.error(f"Failed to create template for {seq_id}: {e}")
                continue

            # Generate samples
            with torch.no_grad():
                for sample_idx in range(config.sampling.n_samples):
                    try:
                        # Sample from prior and decode
                        sample = model.sample(
                            template,
                            n_samples=1,
                            temperature=config.sampling.temperature,
                        )[0]

                        # Save .cif file
                        sample_cpu = sample.numpy()
                        output_filename = (
                            f"{config.output.id_prefix}{seq_id}_sample{sample_idx}.cif"
                        )
                        output_path = output_dir / output_filename
                        sample_cpu.write(str(output_path))

                        n_structures += 1

                        # Save latent if requested
                        if config.output.save_latents:
                            z_mu, _ = model.encode(sample)
                            latents_dict[output_filename] = z_mu.cpu().numpy()

                    except Exception as e:
                        logger.error(
                            f"Failed to generate sample {sample_idx} for {seq_id}: {e}"
                        )
                        continue

        # Save latents if requested
        if config.output.save_latents and latents_dict:
            latents_path = output_dir / "latents.npz"
            np.savez(str(latents_path), **latents_dict)
            if not quiet:
                logger.info(f"Saved latent vectors to {latents_path}")

        if not quiet:
            logger.info(f"Generated {n_structures} structures")

        return {
            "n_structures": n_structures,
            "n_sequences": len(sequences),
            "output_dir": str(output_dir),
            "device": str(device),
            "error": None,
        }

    except Exception as e:
        import traceback

        traceback.print_exc()
        return {
            "error": str(e),
            "device": device_override or "unknown",
            "n_structures": 0,
            "n_sequences": 0,
        }


def main():
    """Command-line entry point for inference."""
    parser = argparse.ArgumentParser(
        description="Generate polymer structures from trained VAE model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device from config (e.g., cuda, cpu, mps)",
    )
    args = parser.parse_args()

    result = run_inference(
        config_path=args.config,
        device_override=args.device,
        quiet=False,
    )

    if result.get("error"):
        logger.error(f"Inference failed: {result['error']}")
        sys.exit(1)

    logger.info(f"Successfully generated {result['n_structures']} structures")


if __name__ == "__main__":
    main()
