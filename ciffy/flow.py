"""
High-level Flow API for polymer conformation generation.

This module provides simplified functions for common flow model operations:
- sample(): Generate polymer conformations from sequences
- train(): Train flow models on CIF structures
- encode(): Encode polymers to latent space
- decode(): Decode latents to polymers
- load(): Load pre-trained models

Note: This module requires the neural network components which are not
included in the PyPI distribution. Install from source for full functionality.

Example usage:
    >>> from ciffy import flow
    >>>
    >>> # Sample from sequence
    >>> polymer = flow.sample("acgu")
    >>> polymer.write("output.cif")
    >>>
    >>> # Multiple samples
    >>> samples = flow.sample("acgu", n_samples=10)
    >>> for i, p in enumerate(samples):
    ...     p.write(f"sample_{i}.cif")
    >>>
    >>> # Train a model
    >>> model = flow.train(["data/*.cif"], residues="ACGU", n_epochs=200)
    >>>
    >>> # Encode/decode
    >>> latents = flow.encode(polymer)
    >>> new_polymer = flow.decode(latents, "acgu")
"""

from __future__ import annotations


def _check_nn_available():
    """Check if neural network modules are available."""
    try:
        from . import nn  # noqa: F401
        return True
    except ImportError:
        return False


if not _check_nn_available():
    raise ImportError(
        "ciffy.flow requires the neural network modules which are not included "
        "in the PyPI distribution. Install from source: "
        "pip install git+https://github.com/hmblair/ciffy.git"
    )

from pathlib import Path
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    import numpy as np
    import torch
    from .polymer import Polymer
    from .nn.flow.polymer import PolymerFlowModel


def load(name: str = "rna", device: str = "cpu") -> "PolymerFlowModel":
    """
    Load a pre-trained flow model.

    Args:
        name: Name of the pre-trained model ("rna" for RNA residues A, C, G, U).
        device: Device to load model to ("cpu" or "cuda").

    Returns:
        PolymerFlowModel ready for encoding, decoding, and sampling.

    Raises:
        ValueError: If the model name is not recognized.
        FileNotFoundError: If the model files are not found.

    Example:
        >>> from ciffy import flow
        >>> model = flow.load("rna", device="cuda")
    """
    from .nn.flow import load_pretrained
    return load_pretrained(name, device=device)


def sample(
    sequence: str,
    n_samples: int = 1,
    model: Union[str, "PolymerFlowModel"] = "rna",
    device: str = "cpu",
) -> Union["Polymer", list["Polymer"]]:
    """
    Sample polymer conformations from a sequence using flow models.

    Returns Polymer objects ready to save to CIF files.

    Args:
        sequence: Single-letter sequence (lowercase for RNA/DNA, uppercase for protein).
        n_samples: Number of conformations to generate.
        model: Pre-trained model name ("rna") or PolymerFlowModel instance.
        device: Device for computation ("cpu" or "cuda").

    Returns:
        If n_samples=1: Single Polymer with sampled coordinates.
        If n_samples>1: List of Polymers.

    Example:
        >>> from ciffy import flow
        >>>
        >>> # Single sample
        >>> polymer = flow.sample("acgu")
        >>> polymer.write("output.cif")
        >>>
        >>> # Multiple samples
        >>> samples = flow.sample("acgu", n_samples=10)
        >>> for i, p in enumerate(samples):
        ...     p.write(f"sample_{i}.cif")
    """
    from .polymer import from_sequence
    from .nn.flow import PolymerFlowModel

    # Load or use provided model
    if isinstance(model, str):
        flow_model = load(model, device=device)
    else:
        flow_model = model

    # Create template with correct atoms for this model
    template = from_sequence(sequence, atoms=flow_model.atom_filter)

    # Sample coordinates
    coords = flow_model.sample(template.sequence, n_samples=n_samples)

    # Convert to Polymer(s)
    if n_samples == 1:
        if hasattr(coords, 'cpu'):
            coords = coords.cpu().numpy()
        return template.with_coordinates(coords)
    else:
        results = []
        for c in coords:
            if hasattr(c, 'cpu'):
                c = c.cpu().numpy()
            results.append(template.with_coordinates(c))
        return results


def train(
    cif_paths: list[Union[str, Path]],
    residues: Union[list[str], str] = "ACGU",
    output_dir: Union[str, Path, None] = None,
    **config_kwargs,
) -> "PolymerFlowModel":
    """
    Train a flow model on CIF structures.

    Simplified interface for training custom models.

    Args:
        cif_paths: Paths to CIF files for training data. Supports:
            - List of file paths
            - Glob patterns (e.g., ["data/*.cif"])
            - Directories (will find all .cif files)
        residues: Residue types to train. Can be:
            - String like "ACGU" (each character is a residue)
            - List of residue names ["A", "C", "G", "U"]
        output_dir: Where to save trained model (optional).
        **config_kwargs: Passed to ResidueFlowTrainingConfig. Common options:
            - latent_dim: Latent space dimension (default: 12)
            - n_epochs: Number of training epochs (default: 200)
            - device: Training device ("cpu" or "cuda")
            - n_layers: Number of flow layers (default: 8)
            - hidden_dim: Hidden layer dimension (default: 64)

    Returns:
        Trained PolymerFlowModel.

    Example:
        >>> from ciffy import flow
        >>>
        >>> model = flow.train(
        ...     ["data/*.cif"],
        ...     residues="ACGU",
        ...     output_dir="models/my_rna",
        ...     n_epochs=200,
        ...     device="cuda",
        ... )
    """
    from .nn.flow import ResidueFlowTrainer, ResidueFlowTrainingConfig
    from .biochemistry import Residue

    # Parse residues
    if isinstance(residues, str):
        residue_list = [getattr(Residue, c.upper()) for c in residues]
    else:
        residue_list = [
            getattr(Residue, r.upper()) if isinstance(r, str) else r
            for r in residues
        ]

    # Resolve paths (handle globs and directories)
    resolved_paths = []
    for p in cif_paths:
        path = Path(p) if not isinstance(p, Path) else p
        if path.is_dir():
            resolved_paths.extend(path.glob("*.cif"))
        elif "*" in str(p):
            resolved_paths.extend(Path(".").glob(str(p)))
        else:
            resolved_paths.append(path)

    # Create config and train
    config = ResidueFlowTrainingConfig(**config_kwargs)
    trainer = ResidueFlowTrainer(config)
    results = trainer.train_all(resolved_paths, residue_list)

    # Save if requested
    if output_dir:
        trainer.save(results, output_dir)

    return trainer.to_polymer_model(results)


def encode(
    polymer: "Polymer",
    model: Union[str, "PolymerFlowModel"] = "rna",
    device: str = "cpu",
) -> "torch.Tensor":
    """
    Encode a polymer to latent space using a flow model.

    Args:
        polymer: Polymer to encode.
        model: Model name or PolymerFlowModel instance.
        device: Device for computation.

    Returns:
        (n_residues, latent_dim) latent vectors.

    Example:
        >>> import ciffy
        >>> from ciffy import flow
        >>>
        >>> polymer = ciffy.load("structure.cif").poly()
        >>> latents = flow.encode(polymer)
    """
    # Load or use provided model
    if isinstance(model, str):
        flow_model = load(model, device=device)
    else:
        flow_model = model

    return flow_model.encode_polymer(polymer)


def decode(
    latents: "torch.Tensor",
    template: Union["Polymer", str],
    model: Union[str, "PolymerFlowModel"] = "rna",
    device: str = "cpu",
) -> "Polymer":
    """
    Decode latents to a polymer using a flow model.

    Args:
        latents: (n_residues, latent_dim) latent vectors.
        template: Template Polymer or sequence string.
        model: Model name or PolymerFlowModel instance.
        device: Device for computation.

    Returns:
        Polymer with decoded coordinates.

    Example:
        >>> from ciffy import flow
        >>>
        >>> # Modify latents
        >>> modified_latents = latents + noise
        >>> new_polymer = flow.decode(modified_latents, "acgu")
    """
    from .polymer import from_sequence

    # Load or use provided model
    if isinstance(model, str):
        flow_model = load(model, device=device)
    else:
        flow_model = model

    # Handle string template
    if isinstance(template, str):
        template = from_sequence(template, atoms=flow_model.atom_filter)

    return flow_model.decode_to_polymer(latents, template)


__all__ = [
    "load",
    "sample",
    "train",
    "encode",
    "decode",
]
