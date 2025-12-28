"""
Unified model save/load using safetensors with metadata.

Provides a simple, extensible system for saving and loading models
in a single .safetensors file with embedded config.

Example:
    >>> from ciffy.nn import save_model, load_model
    >>>
    >>> # Save any registered model
    >>> save_model(model, "my_model.safetensors")
    >>>
    >>> # Load (auto-detects type)
    >>> model = load_model("my_model.safetensors")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

try:
    import torch
    from safetensors.torch import load_file, save_file
    from safetensors import safe_open

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

if TYPE_CHECKING:
    import torch.nn as nn

import ciffy
from .model_registry import get_model_class


@runtime_checkable
class SaveableModel(Protocol):
    """Protocol for models that support unified save/load.

    Models implementing this protocol can be saved to a single safetensors
    file with embedded metadata, and loaded back with automatic type detection.

    To implement:
        1. Add @register_model("your_model_name") decorator to your class
        2. Implement get_save_state() and from_save_state() methods
    """

    _model_type: str  # Set by @register_model decorator

    def get_save_state(self) -> tuple[dict[str, "torch.Tensor"], dict[str, Any]]:
        """Get state for saving.

        Returns:
            Tuple of (tensors_dict, config_dict).
            - tensors_dict: Maps names to tensors (will be saved to safetensors)
            - config_dict: JSON-serializable config (stored in metadata)
        """
        ...

    @classmethod
    def from_save_state(
        cls,
        tensors: dict[str, "torch.Tensor"],
        config: dict[str, Any],
        device: str = "cpu",
    ) -> "SaveableModel":
        """Reconstruct model from saved state.

        Args:
            tensors: Loaded tensors dict.
            config: Loaded config dict.
            device: Device to load model to.

        Returns:
            Reconstructed model instance.
        """
        ...


def save_model(model: SaveableModel, path: str | Path) -> None:
    """Save model to safetensors file with embedded config.

    The model type and config are stored in safetensors metadata,
    enabling automatic type detection on load.

    Args:
        model: Model implementing SaveableModel protocol.
        path: Output path (will add .safetensors extension if missing).

    Raises:
        TypeError: If model doesn't implement SaveableModel protocol.
        ValueError: If model is not registered.

    Example:
        >>> model = PolymerFlowModel(...)
        >>> save_model(model, "rna_flow.safetensors")
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for save_model")

    path = Path(path)
    if not path.suffix:
        path = path.with_suffix(".safetensors")

    # Check model has required attributes
    if not hasattr(model, "_model_type"):
        raise TypeError(
            f"Model {type(model).__name__} is not registered. "
            f"Use @register_model decorator."
        )

    if not hasattr(model, "get_save_state"):
        raise TypeError(
            f"Model {type(model).__name__} doesn't implement get_save_state(). "
            f"See SaveableModel protocol."
        )

    # Get state from model
    tensors, config = model.get_save_state()

    # Ensure tensors are on CPU and contiguous
    tensors = {k: v.cpu().contiguous() for k, v in tensors.items()}

    # Build metadata
    metadata = {
        "model_type": model._model_type,
        "config": json.dumps(config),
        "ciffy_version": ciffy.__version__,
    }

    # Save
    save_file(tensors, path, metadata=metadata)


def load_model(path: str | Path, device: str = "cpu") -> SaveableModel:
    """Load model from safetensors file (auto-detects type).

    Args:
        path: Path to .safetensors file.
        device: Device to load model to ('cpu', 'cuda', 'mps').

    Returns:
        Loaded model instance.

    Raises:
        ValueError: If model type is not registered.
        FileNotFoundError: If path doesn't exist.

    Example:
        >>> model = load_model("rna_flow.safetensors", device="cuda")
        >>> samples = model.sample(template, n_samples=10)
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for load_model")

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    # Load metadata first
    with safe_open(path, framework="pt") as f:
        metadata = f.metadata()

    if "model_type" not in metadata:
        raise ValueError(
            f"File {path} is not a valid ciffy model (missing model_type metadata)"
        )

    model_type = metadata["model_type"]
    config = json.loads(metadata.get("config", "{}"))

    # Get model class
    model_cls = get_model_class(model_type)

    if not hasattr(model_cls, "from_save_state"):
        raise TypeError(
            f"Model class {model_cls.__name__} doesn't implement from_save_state()"
        )

    # Load tensors
    tensors = load_file(path, device=device)

    # Reconstruct model
    return model_cls.from_save_state(tensors, config, device=device)


def get_model_info(path: str | Path) -> dict[str, Any]:
    """Get metadata from a saved model without loading weights.

    Useful for inspecting model type and config before loading.

    Args:
        path: Path to .safetensors file.

    Returns:
        Dict with 'model_type', 'config', and 'ciffy_version'.

    Example:
        >>> info = get_model_info("model.safetensors")
        >>> print(info["model_type"])  # "polymer_flow"
        >>> print(info["config"]["latent_dim"])  # 12
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for get_model_info")

    path = Path(path)

    with safe_open(path, framework="pt") as f:
        metadata = f.metadata()

    return {
        "model_type": metadata.get("model_type"),
        "config": json.loads(metadata.get("config", "{}")),
        "ciffy_version": metadata.get("ciffy_version"),
    }


__all__ = [
    "SaveableModel",
    "save_model",
    "load_model",
    "get_model_info",
]
