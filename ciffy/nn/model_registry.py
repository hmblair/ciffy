"""
Model registry for automatic checkpoint-based model instantiation.

Provides a registration system for neural network models, enabling automatic
creation of model instances from checkpoints without manual imports.

Example:
    >>> from ciffy.nn.model_registry import register_model, get_model_class
    >>>
    >>> # Register a model (in model class definition)
    >>> @register_model('MyVAE')
    >>> class MyVAE(nn.Module):
    ...     pass
    >>>
    >>> # Later, retrieve and instantiate
    >>> VAEClass = get_model_class('MyVAE')
    >>> model = VAEClass(latent_dim=64, hidden_dim=256)
"""

from __future__ import annotations

from typing import Any, Callable

try:
    import torch.nn as nn
except ImportError:
    nn = None


# Global registry mapping model names to their classes
_MODEL_REGISTRY: dict[str, Callable[..., Any]] = {}


def register_model(name: str) -> Callable[[type], type]:
    """
    Decorator to register a model class in the global registry.

    Args:
        name: Name to register the model under (e.g., 'PolymerVAE').

    Returns:
        Decorator function that registers the class and returns it unchanged.

    Example:
        >>> @register_model('PolymerVAE')
        >>> class PolymerVAE(nn.Module):
        ...     def __init__(self, latent_dim=64):
        ...         super().__init__()
        ...         self.latent_dim = latent_dim
    """

    def decorator(cls: type) -> type:
        if name in _MODEL_REGISTRY:
            raise ValueError(
                f"Model '{name}' is already registered to {_MODEL_REGISTRY[name].__name__}"
            )
        _MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def get_model_class(name: str) -> Callable[..., Any]:
    """
    Retrieve a registered model class by name.

    Args:
        name: Name the model was registered under.

    Returns:
        The model class (callable that creates instances).

    Raises:
        ValueError: If model name is not registered.

    Example:
        >>> VAEClass = get_model_class('PolymerVAE')
        >>> model = VAEClass(latent_dim=64)
    """
    if name not in _MODEL_REGISTRY:
        available = list(_MODEL_REGISTRY.keys())
        raise ValueError(
            f"Model '{name}' not found in registry. "
            f"Available models: {available}"
        )
    return _MODEL_REGISTRY[name]


def list_registered_models() -> list[str]:
    """
    List all registered model names.

    Returns:
        List of registered model names.
    """
    return sorted(list(_MODEL_REGISTRY.keys()))


def create_model_from_config(config: Any) -> Any:
    """
    Create a model instance from a config object.

    Attempts to infer the model type from the config and instantiate it.
    Supports both explicit model_class specification and automatic inference.

    Args:
        config: Configuration object with 'model' section. Can be:
            - A dict with 'model_class' and model config
            - A dataclass with nested 'model' attribute
            - A dict with model config that allows type inference

    Returns:
        Instantiated model in CPU mode (not on any device).

    Raises:
        ValueError: If model type cannot be determined or model not found.
        TypeError: If config structure is invalid.

    Example:
        >>> # From VAEConfig dataclass
        >>> vae_config = VAEConfig(
        ...     model=VAEModelConfig(latent_dim=64, hidden_dim=256),
        ...     ...
        ... )
        >>> model = create_model_from_config(vae_config)
        >>>
        >>> # From dict with explicit class
        >>> config_dict = {
        ...     'model_class': 'PolymerVAE',
        ...     'model': {'latent_dim': 64, 'hidden_dim': 256},
        ... }
        >>> model = create_model_from_config(config_dict)
    """
    import dataclasses

    # Handle dict-like configs
    if isinstance(config, dict):
        if "model_class" in config:
            # Explicit model class specification
            model_class_name = config["model_class"]
            model_cls = get_model_class(model_class_name)
            model_config = config.get("model", {})

            # Convert to dict if needed
            if hasattr(model_config, "__dict__"):
                model_kwargs = vars(model_config)
            else:
                model_kwargs = model_config if isinstance(model_config, dict) else {}

            return model_cls(**model_kwargs)
        else:
            raise ValueError(
                "Config dict must contain 'model_class' or be a dataclass with 'model' attribute"
            )

    # Handle dataclass configs
    if dataclasses.is_dataclass(config) and not isinstance(config, type):
        # Get model config from nested 'model' attribute
        if not hasattr(config, "model"):
            raise TypeError(
                f"Config dataclass {type(config).__name__} missing 'model' attribute"
            )

        model_config = config.model
        model_config_name = type(model_config).__name__

        # Infer model class from model config name
        # E.g., VAEModelConfig -> PolymerVAE, DiffusionModelConfig -> DiffusionModel
        if "VAE" in model_config_name:
            model_class_name = "PolymerVAE"
        elif "Diffusion" in model_config_name:
            model_class_name = "DiffusionModel"
        else:
            raise ValueError(
                f"Cannot infer model type from config class: {model_config_name}. "
                f"Available models: {list_registered_models()}"
            )

        model_cls = get_model_class(model_class_name)
        model_kwargs = vars(model_config)

        return model_cls(**model_kwargs)

    raise TypeError(
        f"Config must be dict or dataclass, got {type(config).__name__}"
    )


__all__ = [
    "register_model",
    "get_model_class",
    "list_registered_models",
    "create_model_from_config",
]
