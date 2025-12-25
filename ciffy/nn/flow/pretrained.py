"""
Pre-trained model loading utilities.

Provides easy access to bundled pre-trained flow models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from .polymer import PolymerFlowModel


# Model registry: name -> (subpath, description)
_PRETRAINED_MODELS = {
    "rna": ("rna_v1", "RNA flow model for A, C, G, U residues"),
}


def get_models_dir() -> Path:
    """Get the directory containing bundled pre-trained models."""
    return Path(__file__).parent.parent.parent / "data" / "models"


def list_pretrained() -> dict[str, str]:
    """
    List available pre-trained models.

    Returns:
        Dict mapping model name to description.

    Example:
        >>> from ciffy.nn.flow import list_pretrained
        >>> for name, desc in list_pretrained().items():
        ...     print(f"{name}: {desc}")
        rna: RNA flow model for A, C, G, U residues
    """
    available = {}
    models_dir = get_models_dir()

    for name, (subpath, desc) in _PRETRAINED_MODELS.items():
        model_path = models_dir / subpath
        if model_path.exists():
            available[name] = desc

    return available


def load_pretrained(
    name: str = "rna",
    device: str = "cpu",
    jit: bool = False,
) -> PolymerFlowModel:
    """
    Load a pre-trained PolymerFlowModel.

    Args:
        name: Name of the pre-trained model. Use list_pretrained() to see available models.
        device: Device to load model to ('cpu' or 'cuda').
        jit: Whether to JIT-compile the decoders for faster inference.

    Returns:
        Loaded PolymerFlowModel ready for use.

    Raises:
        ValueError: If the model name is not recognized.
        FileNotFoundError: If the model files are not found.

    Example:
        >>> from ciffy.nn.flow import load_pretrained
        >>>
        >>> # Load RNA model
        >>> model = load_pretrained("rna", device="cuda")
        >>>
        >>> # Use for encoding/decoding
        >>> latents = model.encode(coords, sequence)
        >>> coords_new = model.decode(latents, sequence)
    """
    if name not in _PRETRAINED_MODELS:
        available = list(_PRETRAINED_MODELS.keys())
        raise ValueError(
            f"Unknown model '{name}'. Available models: {available}"
        )

    subpath, _ = _PRETRAINED_MODELS[name]
    model_path = get_models_dir() / subpath

    if not model_path.exists():
        raise FileNotFoundError(
            f"Pre-trained model '{name}' not found at {model_path}. "
            f"The model may not be installed. Try reinstalling ciffy or "
            f"training your own model with ResidueFlowTrainer."
        )

    return PolymerFlowModel.load(model_path, device=device, jit=jit)


def is_pretrained_available(name: str = "rna") -> bool:
    """
    Check if a pre-trained model is available.

    Args:
        name: Name of the pre-trained model.

    Returns:
        True if the model is installed and available.
    """
    if name not in _PRETRAINED_MODELS:
        return False

    subpath, _ = _PRETRAINED_MODELS[name]
    model_path = get_models_dir() / subpath
    return model_path.exists()
