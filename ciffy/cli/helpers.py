"""
Shared utilities for CLI commands.

This module contains helper functions to reduce duplication across CLI commands.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ciffy import Polymer


def resolve_accelerator(accelerator: str) -> str:
    """
    Resolve 'auto' accelerator to actual device type.

    Args:
        accelerator: One of 'auto', 'cpu', 'gpu', 'mps'.

    Returns:
        Resolved accelerator string ('cpu', 'gpu', or 'mps').
    """
    if accelerator != "auto":
        return accelerator

    import torch

    if torch.cuda.is_available():
        return "gpu"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_device(device: str) -> str:
    """
    Resolve 'auto' device to actual device string for torch.

    Args:
        device: One of 'auto', 'cpu', 'cuda', 'mps'.

    Returns:
        Resolved device string ('cpu', 'cuda', or 'mps').
    """
    if device != "auto":
        return device

    import torch

    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_structure(filepath: str, **kwargs) -> "Polymer | None":
    """
    Load a structure with standardized error handling.

    Args:
        filepath: Path to CIF file.
        **kwargs: Additional arguments passed to ciffy.load().

    Returns:
        Polymer object, or None if loading failed.
    """
    from ciffy import load

    try:
        return load(filepath, **kwargs)
    except FileNotFoundError:
        print(f"Error: File not found: {filepath}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error loading {filepath}: {e}", file=sys.stderr)
        return None


def load_structure_or_exit(filepath: str, **kwargs) -> "Polymer":
    """
    Load a structure, exiting on failure.

    Args:
        filepath: Path to CIF file.
        **kwargs: Additional arguments passed to ciffy.load().

    Returns:
        Polymer object.

    Raises:
        SystemExit: If loading fails.
    """
    polymer = load_structure(filepath, **kwargs)
    if polymer is None:
        sys.exit(1)
    return polymer


def save_samples(
    samples: list["Polymer"],
    output: Path,
    quiet: bool = False,
) -> None:
    """
    Save polymer samples to file(s).

    For single samples, saves to a single file. For multiple samples,
    creates a directory with numbered files.

    Args:
        samples: List of Polymer objects to save.
        output: Output path (file for single, directory for multiple).
        quiet: Suppress output messages.
    """
    if len(samples) == 1:
        out_path = output if output.suffix == ".cif" else output.with_suffix(".cif")
        samples[0].write(str(out_path))
        if not quiet:
            print(f"Saved to {out_path}")
    else:
        output.mkdir(parents=True, exist_ok=True)
        for i, polymer in enumerate(samples):
            out_path = output / f"sample_{i:03d}.cif"
            polymer.write(str(out_path))
            if not quiet:
                print(f"Saved {out_path}")


def print_training_header(model_type: str, quiet: bool = False, **info) -> None:
    """
    Print standardized training header.

    Args:
        model_type: Type of model being trained (e.g., "Flow Model").
        quiet: If True, suppress output.
        **info: Key-value pairs to display (e.g., Data="100 files").
    """
    if quiet:
        return

    print()
    print("=" * 60)
    print(f"Ciffy {model_type} Training")
    print("=" * 60)
    for key, value in info.items():
        # Convert underscores to spaces and capitalize
        label = key.replace("_", " ")
        print(f"{label}: {value}")
    print()


def print_training_complete(save_path: str | Path, quiet: bool = False, **info) -> None:
    """
    Print standardized training completion message.

    Args:
        save_path: Path where model was saved.
        quiet: If True, suppress output.
        **info: Additional key-value pairs to display.
    """
    if quiet:
        return

    print()
    print("=" * 60)
    print("Training Complete")
    print("=" * 60)
    for key, value in info.items():
        label = key.replace("_", " ")
        print(f"{label}: {value}")
    print(f"Saved to: {save_path}")


def setup_wandb_logger(
    enable: bool,
    project: str,
    name: str | None = None,
):
    """
    Set up Weights & Biases logger if enabled.

    Args:
        enable: Whether to enable W&B logging.
        project: W&B project name.
        name: Optional run name.

    Returns:
        WandbLogger instance, or None if not enabled.
    """
    if not enable:
        return None

    from lightning.pytorch.loggers import WandbLogger

    return WandbLogger(project=project, name=name)


def require_torch_lightning() -> bool:
    """
    Check that PyTorch and Lightning are available, print error if not.

    Returns:
        True if available, False otherwise.
    """
    try:
        import lightning  # noqa: F401
        import torch  # noqa: F401
        return True
    except ImportError:
        print(
            "Error: PyTorch and Lightning are required for this command.\n"
            "Install with: pip install torch lightning",
            file=sys.stderr,
        )
        return False


def require_matplotlib() -> bool:
    """
    Check that matplotlib is available, print error if not.

    Returns:
        True if available, False otherwise.
    """
    try:
        import matplotlib  # noqa: F401
        return True
    except ImportError:
        print(
            "Error: matplotlib is required for this command.\n"
            "Install with: pip install matplotlib",
            file=sys.stderr,
        )
        return False
