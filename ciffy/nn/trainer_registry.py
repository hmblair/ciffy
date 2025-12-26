"""
Trainer registry for unified training CLI.

Provides a registration system for trainer classes, enabling the `ciffy train`
command to dispatch to the correct trainer based on the config's `trainer` field.

Example:
    >>> from ciffy.nn.trainer_registry import register_trainer, get_trainer
    >>>
    >>> # Register a trainer with its config class
    >>> @register_trainer('my_trainer', MyTrainerConfig)
    ... class MyTrainer:
    ...     def __init__(self, config: MyTrainerConfig, quiet: bool = False):
    ...         ...
    ...     def train(self, **kwargs) -> dict:
    ...         ...
    >>>
    >>> # Later, instantiate from a config dict
    >>> trainer_cls, config_cls = get_trainer('my_trainer')
    >>> config = config_cls.from_dict(config_dict)
    >>> trainer = trainer_cls(config, quiet=True)
    >>> result = trainer.train()
"""

from __future__ import annotations

from dataclasses import dataclass, fields, field
from typing import Any, Callable, Protocol, TypeVar, runtime_checkable


# Type for trainer registration entries
@dataclass
class TrainerEntry:
    """Entry in the trainer registry."""

    trainer_cls: type
    config_cls: type


# Global registry mapping trainer names to their entries
_TRAINER_REGISTRY: dict[str, TrainerEntry] = {}


@runtime_checkable
class TrainerProtocol(Protocol):
    """Protocol defining the interface all trainers must satisfy.

    Trainers must implement a `train()` method that returns a result dict.
    """

    def train(
        self,
        resume_path: str | None = None,
        progress_callback: Callable[[int, int, dict], None] | None = None,
    ) -> dict[str, Any]:
        """Run training and return results.

        Args:
            resume_path: Optional path to checkpoint to resume from.
            progress_callback: Optional callback for progress updates.
                Signature: callback(epoch, total_epochs, metrics)

        Returns:
            Dict with at least:
                - status: 'success' or 'failed'
                - epochs_trained: Number of epochs completed
                - checkpoint_path: Path to saved checkpoint (if any)
        """
        ...


@runtime_checkable
class ConfigProtocol(Protocol):
    """Protocol defining the interface all trainer configs must satisfy.

    Configs must implement a `from_dict()` classmethod for instantiation
    from YAML config dictionaries.
    """

    @classmethod
    def from_dict(cls, config: dict[str, Any], **overrides: Any) -> "ConfigProtocol":
        """Create config from a dictionary.

        Args:
            config: Dictionary from YAML config file.
            **overrides: Override specific fields (e.g., device).

        Returns:
            Config instance.
        """
        ...


def register_trainer(
    name: str,
    config_cls: type | None = None,
) -> Callable[[type], type]:
    """
    Decorator to register a trainer class with its config class.

    Args:
        name: Name to register the trainer under (e.g., 'latent_diffusion', 'flow').
        config_cls: The config class for this trainer. Must have a `from_dict()` method.

    Returns:
        Decorator function that registers the class and returns it unchanged.

    Example:
        >>> @register_trainer('my_trainer', MyTrainerConfig)
        ... class MyTrainer:
        ...     def __init__(self, config: MyTrainerConfig, quiet: bool = False):
        ...         self.config = config
        ...
        ...     def train(self, **kwargs) -> dict:
        ...         return {"status": "success", "epochs_trained": 100}
    """

    def decorator(cls: type) -> type:
        if name in _TRAINER_REGISTRY:
            existing = _TRAINER_REGISTRY[name].trainer_cls
            raise ValueError(
                f"Trainer '{name}' is already registered to {existing.__name__}"
            )

        if config_cls is None:
            raise ValueError(
                f"config_cls is required for register_trainer('{name}'). "
                f"Pass the config dataclass as the second argument."
            )

        _TRAINER_REGISTRY[name] = TrainerEntry(
            trainer_cls=cls,
            config_cls=config_cls,
        )
        return cls

    return decorator


def get_trainer(name: str) -> tuple[type, type]:
    """
    Retrieve a registered trainer and config class by name.

    Args:
        name: Name the trainer was registered under.

    Returns:
        Tuple of (trainer_class, config_class).

    Raises:
        ValueError: If trainer name is not registered.

    Example:
        >>> trainer_cls, config_cls = get_trainer('latent_diffusion')
        >>> config = config_cls.from_dict(yaml_config, device='cuda')
        >>> trainer = trainer_cls(config, quiet=True)
        >>> result = trainer.train()
    """
    if name not in _TRAINER_REGISTRY:
        available = list(_TRAINER_REGISTRY.keys())
        raise ValueError(
            f"Trainer '{name}' not found in registry. "
            f"Available trainers: {available}"
        )
    entry = _TRAINER_REGISTRY[name]
    return entry.trainer_cls, entry.config_cls


def get_trainer_class(name: str) -> type:
    """
    Retrieve a registered trainer class by name (legacy API).

    Args:
        name: Name the trainer was registered under.

    Returns:
        The trainer class.

    Raises:
        ValueError: If trainer name is not registered.
    """
    trainer_cls, _ = get_trainer(name)
    return trainer_cls


def list_registered_trainers() -> list[str]:
    """
    List all registered trainer names.

    Returns:
        Sorted list of registered trainer names.
    """
    return sorted(_TRAINER_REGISTRY.keys())


def is_trainer_registered(name: str) -> bool:
    """
    Check if a trainer is registered.

    Args:
        name: Trainer name to check.

    Returns:
        True if trainer is registered, False otherwise.
    """
    return name in _TRAINER_REGISTRY


def dataclass_from_dict(cls: type, data: dict[str, Any], **overrides: Any) -> Any:
    """
    Helper to instantiate a dataclass from a nested dict.

    Handles nested dataclasses and applies overrides.

    Args:
        cls: The dataclass type.
        data: Dictionary with field values.
        **overrides: Override specific top-level fields.

    Returns:
        Instance of cls.
    """
    # Merge overrides into data
    merged = {**data, **overrides}

    # Get field info
    field_types = {f.name: f.type for f in fields(cls)}
    kwargs = {}

    for f in fields(cls):
        name = f.name
        if name in merged:
            value = merged[name]

            # Handle nested dataclasses
            field_type = field_types.get(name)
            if (
                isinstance(value, dict)
                and hasattr(field_type, "__dataclass_fields__")
            ):
                value = dataclass_from_dict(field_type, value)

            kwargs[name] = value

    return cls(**kwargs)


__all__ = [
    "TrainerProtocol",
    "ConfigProtocol",
    "TrainerEntry",
    "register_trainer",
    "get_trainer",
    "get_trainer_class",
    "list_registered_trainers",
    "is_trainer_registered",
    "dataclass_from_dict",
]
