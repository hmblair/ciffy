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
    >>> # Train a model (uses PyTorch Lightning)
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
    from .polymer import template

    # Load or use provided model
    if isinstance(model, str):
        flow_model = load(model, device=device)
    else:
        flow_model = model

    # Create template with correct atoms for this model
    template = template(sequence, atoms=flow_model.atom_filter)

    # Sample coordinates
    coords = flow_model.sample(template.sequence, n_samples=n_samples)

    # Convert to Polymer(s)
    if n_samples == 1:
        if hasattr(coords, 'cpu'):
            coords = coords.cpu().numpy()
        return template.copy(coordinates=coords)
    else:
        results = []
        for c in coords:
            if hasattr(c, 'cpu'):
                c = c.cpu().numpy()
            results.append(template.copy(coordinates=c))
        return results


def train(
    cif_paths: list[Union[str, Path]],
    residues: Union[list[str], str] = "ACGU",
    output_dir: Union[str, Path, None] = None,
    n_epochs: int = 200,
    latent_dim: int = 12,
    n_layers: int = 6,
    hidden_dim: int = 64,
    batch_size: int = 256,
    lr: float = 1e-3,
    accelerator: str = "auto",
    verbose: bool = True,
    **kwargs,
) -> "PolymerFlowModel":
    """
    Train a flow model on CIF structures.

    Uses PyTorch Lightning for training. This is a simplified interface
    for training custom models.

    Args:
        cif_paths: Paths to CIF files for training data. Supports:
            - List of file paths
            - Glob patterns (e.g., ["data/*.cif"])
            - Directories (will find all .cif files)
        residues: Residue types to train. Can be:
            - String like "ACGU" (each character is a residue)
            - List of residue names ["A", "C", "G", "U"]
        output_dir: Where to save trained model (optional).
        n_epochs: Number of training epochs (default: 200).
        latent_dim: Latent space dimension (default: 12).
        n_layers: Number of flow layers (default: 6).
        hidden_dim: Hidden layer dimension (default: 64).
        batch_size: Training batch size (default: 256).
        lr: Learning rate (default: 1e-3).
        accelerator: Device for training ("auto", "cpu", "gpu", "mps").
        verbose: Whether to show progress bars.
        **kwargs: Additional arguments passed to Lightning Trainer.

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
        ...     accelerator="gpu",
        ... )
    """
    import lightning as L
    from .biochemistry import Residue
    from .nn.flow import PolymerFlowModel
    from .nn.lightning import FlowDataModule, ResidueFlowModule
    from .nn.lightning.modules.residue_flow import (
        ResidueFlowFullConfig,
        ResidueFlowModelConfig,
        ResidueFlowDataConfig,
    )
    from .nn.config import TrainingConfig

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

    if not resolved_paths:
        raise ValueError(f"No CIF files found in {cif_paths}")

    if verbose:
        print(f"Training on {len(resolved_paths)} CIF files")

    # Create config
    config = ResidueFlowFullConfig(
        model=ResidueFlowModelConfig(
            latent_dim=latent_dim,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
        ),
        data=ResidueFlowDataConfig(
            batch_size=batch_size,
        ),
        training=TrainingConfig(
            lr=lr,
            epochs=n_epochs,
        ),
    )

    # Train each residue type
    residue_models = {}

    for residue in residue_list:
        if verbose:
            print(f"\nTraining model for {residue.name}...")

        try:
            # Create data module
            dm = FlowDataModule(
                cif_paths=list(resolved_paths),
                residue=residue,
                batch_size=batch_size,
            )

            # Create Lightning module
            module = ResidueFlowModule(config, residue)

            # Create trainer
            trainer = L.Trainer(
                max_epochs=n_epochs,
                accelerator=accelerator,
                enable_progress_bar=verbose,
                enable_model_summary=verbose,
                logger=False,  # Disable logging for simple API
                **kwargs,
            )

            # Train
            trainer.fit(module, dm)

            # Get trained model
            model = module.get_model()
            residue_models[residue] = model

            if verbose:
                print(f"  {residue.name}: {model.n_atoms} atoms, latent_dim={model.latent_dim}")

            # Save individual model if output_dir specified
            if output_dir:
                model_path = Path(output_dir) / residue.name
                model.save(model_path)

        except Exception as e:
            if verbose:
                print(f"  {residue.name}: Failed - {e}")

    if not residue_models:
        raise ValueError("No models were trained successfully")

    # Create PolymerFlowModel
    polymer_model = PolymerFlowModel(residue_models)

    # Save if requested
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        polymer_model.save(output_dir)
        if verbose:
            print(f"\nSaved PolymerFlowModel to {output_dir}")

    return polymer_model


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
    from .polymer import template

    # Load or use provided model
    if isinstance(model, str):
        flow_model = load(model, device=device)
    else:
        flow_model = model

    # Handle string template
    if isinstance(template, str):
        template = template(template, atoms=flow_model.atom_filter)

    return flow_model.decode_to_polymer(latents, template)


__all__ = [
    "load",
    "sample",
    "train",
    "encode",
    "decode",
]
