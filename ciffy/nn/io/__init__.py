"""Model I/O and distribution utilities for ciffy.

Provides model saving, loading, registry, and Hub integration.
"""

from .hub import (
    HubMixin,
    get_cache_dir,
    set_cache_dir,
)
from .inference import (
    generate_samples,
    load_model_from_checkpoint,
)
from .registry import (
    get_model_class,
    list_registered_models,
    register_model,
)
from .save_load import (
    SaveableModel,
    get_model_info,
    load_model,
    save_model,
)

__all__ = [
    # Save/Load
    "save_model",
    "load_model",
    "get_model_info",
    "SaveableModel",
    # Registry
    "register_model",
    "get_model_class",
    "list_registered_models",
    # Hub
    "HubMixin",
    "get_cache_dir",
    "set_cache_dir",
    # Inference
    "load_model_from_checkpoint",
    "generate_samples",
]
