"""
Generic inference utilities for loading trained models and generating polymers.

Provides high-level API for:
- Loading models from checkpoints
- Generating polymer structures from sequences
- Working with trained models that implement PolymerGenerativeModel protocol
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

try:
    import torch
except ImportError:
    torch = None

try:
    import numpy as np
except ImportError:
    np = None

from .training import get_device
from .protocols import PolymerGenerativeModel
from .model_registry import get_model_class, create_model_from_config

if TYPE_CHECKING:
    import torch.nn as nn
    from ..polymer import Polymer


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    device: str = "auto",
    weights_only: bool = True,
) -> tuple[nn.Module, dict]:
    """
    Load a trained model from checkpoint.

    Automatically reconstructs the model from saved configuration and loads weights.
    Supports both new checkpoints with model metadata and legacy checkpoints.

    Args:
        checkpoint_path: Path to checkpoint file (.pt).
        device: Device to load model on ("auto", "cuda", "cpu", "mps").
        weights_only: If False, also restore optimizer state (not recommended for inference).

    Returns:
        Tuple of (model, checkpoint_dict) where:
        - model: Loaded model in eval mode on specified device
        - checkpoint_dict: Full checkpoint containing config, epoch, metrics, etc.

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist.
        ValueError: If checkpoint is missing required fields.
        KeyError: If model cannot be reconstructed from checkpoint.

    Example:
        >>> model, ckpt = load_model_from_checkpoint("checkpoints/vae_best.pt")
        >>> print(f"Loaded from epoch {ckpt['epoch']}")
        >>> print(f"Model type: {model.__class__.__name__}")
    """
    if torch is None:
        raise ImportError("PyTorch is required. Install with: pip install torch")

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device_obj = get_device(device)

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device_obj, weights_only=False)

    # Validate checkpoint has required fields
    if "model_state_dict" not in ckpt:
        raise ValueError(
            f"Checkpoint {checkpoint_path} missing 'model_state_dict'. "
            "Not a valid ciffy checkpoint."
        )

    if "config" not in ckpt:
        raise ValueError(
            f"Checkpoint {checkpoint_path} missing 'config'. "
            "Cannot reconstruct model architecture."
        )

    config = ckpt["config"]

    # Try to create model from registry
    if "model_class" in ckpt:
        # New-style checkpoint with explicit model class
        model_class_name = ckpt["model_class"]
        try:
            model_cls = get_model_class(model_class_name)
        except ValueError:
            raise ValueError(
                f"Checkpoint references model '{model_class_name}' which is not registered. "
                f"Make sure the model module is imported before loading."
            )

        # Extract model config
        if "model" in config:
            model_config = config["model"]
            if hasattr(model_config, "__dict__"):
                model_kwargs = vars(model_config)
            else:
                model_kwargs = model_config if isinstance(model_config, dict) else {}
        else:
            model_kwargs = {}

        model = model_cls(**model_kwargs)
    else:
        # Legacy checkpoint or dict config - infer from config structure
        try:
            model = create_model_from_config(config)
        except Exception as e:
            raise ValueError(
                f"Could not reconstruct model from checkpoint. "
                f"Try manually specifying the model class. Error: {e}"
            )

    # Load weights
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device_obj)
    model.eval()

    return model, ckpt


def load_vae(
    checkpoint_path: str | Path,
    device: str = "auto",
) -> nn.Module:
    """
    Convenience function to load a PolymerVAE from checkpoint.

    Args:
        checkpoint_path: Path to VAE checkpoint.
        device: Device to load on ("auto", "cuda", "cpu", "mps").

    Returns:
        Loaded PolymerVAE model in eval mode.

    Raises:
        FileNotFoundError: If checkpoint doesn't exist.
        TypeError: If checkpoint contains non-VAE model.

    Example:
        >>> vae = load_vae("checkpoints/vae_best.pt", device="cuda")
        >>> print(f"Latent dimension: {vae.latent_dim}")
    """
    model, _ = load_model_from_checkpoint(checkpoint_path, device)

    from .vae import PolymerVAE

    if not isinstance(model, PolymerVAE):
        raise TypeError(
            f"Checkpoint contains {model.__class__.__name__}, not PolymerVAE. "
            f"Use load_model_from_checkpoint() for generic models."
        )

    return model


def generate_samples(
    model: PolymerGenerativeModel,
    sequence: str | list[str],
    n_samples: int = 1,
    temperature: float = 1.0,
    output_dir: Optional[str | Path] = None,
    prefix: str = "sample",
    backend: str = "torch",
    device: str = "auto",
    quiet: bool = False,
) -> list[Polymer]:
    """
    Generate polymer structures from sequences using a trained model.

    High-level convenience function that:
    1. Creates template polymers from sequences
    2. Generates samples using the model
    3. Optionally saves output to .cif files

    Args:
        model: Generative model implementing PolymerGenerativeModel protocol.
        sequence: Single sequence string or list of sequences.
            Examples: "MGKLF", ["MGKLF", "acgu"], etc.
        n_samples: Number of structures to generate per sequence.
        temperature: Sampling temperature (higher = more diverse).
        output_dir: If provided, save generated .cif files to this directory.
        prefix: Filename prefix for saved structures (e.g., "sample" → "sample_0.cif").
        backend: Backend for template creation ("torch" or "numpy").
        device: Device for inference ("auto", "cuda", "cpu", "mps").
        quiet: If True, suppress logging output.

    Returns:
        List of generated Polymer objects with predicted coordinates.

    Raises:
        TypeError: If model doesn't implement PolymerGenerativeModel protocol.
        ValueError: If sequences are invalid or model fails to generate.
        RuntimeError: If model is in training mode (should call .eval() first).

    Example:
        >>> from ciffy.nn import load_vae, generate_samples
        >>>
        >>> vae = load_vae("checkpoints/vae_best.pt")
        >>> samples = generate_samples(
        ...     vae,
        ...     sequence=["MGKLF", "acgu"],
        ...     n_samples=10,
        ...     temperature=1.0,
        ...     output_dir="./generated",
        ...     prefix="gen_",
        ... )
        >>> print(f"Generated {len(samples)} structures")
    """
    import ciffy

    # Validate model implements protocol
    if not isinstance(model, PolymerGenerativeModel):
        raise TypeError(
            f"Model {model.__class__.__name__} does not implement "
            "PolymerGenerativeModel protocol. Must have sample(template, n_samples, temperature) method."
        )

    if torch is None:
        raise ImportError("PyTorch is required. Install with: pip install torch")

    # Normalize sequences to list
    if isinstance(sequence, str):
        sequences = [sequence]
    else:
        sequences = list(sequence)

    device_obj = get_device(device)

    # Create templates
    templates = []
    for seq in sequences:
        try:
            template = ciffy.from_sequence(seq, backend=backend)
            template = template.to(device_obj)
            templates.append(template)
        except Exception as e:
            if not quiet:
                print(f"Warning: Failed to create template for sequence {seq}: {e}")
            continue

    if not templates:
        raise ValueError(
            f"Failed to create templates for any sequences. Check sequence validity."
        )

    # Generate samples
    samples = []
    model.eval()

    with torch.no_grad():
        for template in templates:
            for _ in range(n_samples):
                try:
                    generated = model.sample(
                        template, n_samples=1, temperature=temperature
                    )
                    if isinstance(generated, list):
                        samples.extend(generated)
                    else:
                        samples.append(generated)
                except Exception as e:
                    if not quiet:
                        print(f"Warning: Sampling failed: {e}")
                    continue

    if not samples:
        raise RuntimeError(
            "Model failed to generate any samples. Check model state and inputs."
        )

    # Save to disk if directory specified
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, polymer in enumerate(samples):
            filename = f"{prefix}_{i:06d}.cif" if prefix else f"sample_{i:06d}.cif"
            filepath = output_dir / filename

            try:
                # Convert to numpy if needed
                if hasattr(polymer, "numpy"):
                    polymer = polymer.numpy()
                polymer.write(str(filepath))
            except Exception as e:
                if not quiet:
                    print(f"Warning: Failed to save {filepath}: {e}")

    return samples


__all__ = [
    "load_model_from_checkpoint",
    "load_vae",
    "generate_samples",
]
