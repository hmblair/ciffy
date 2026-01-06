"""
Hugging Face Hub integration for ciffy models.

Provides a mixin class that adds `push_to_hub()` and `from_pretrained()` methods
to any model with `save()` and `load()` methods.

Example:
    >>> # Push a trained model
    >>> model.push_to_hub("username/rna-flow-v1", token="hf_xxx")

    >>> # Load from hub
    >>> model = PolymerFlowModel.from_pretrained("username/rna-flow-v1")

    >>> # Load specific version
    >>> model = PolymerFlowModel.from_pretrained("username/rna-flow-v1", revision="v1.2")
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    pass

T = TypeVar("T", bound="HubMixin")

# Default organization for ciffy models
CIFFY_HUB_ORG = "ciffy-models"

# Cache directory for downloaded models
_CACHE_DIR: Path | None = None


def get_cache_dir() -> Path:
    """Get the cache directory for downloaded models."""
    global _CACHE_DIR
    if _CACHE_DIR is None:
        # Use HF cache structure for consistency
        import os
        cache_root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
        _CACHE_DIR = cache_root / "hub" / "ciffy"
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR


def set_cache_dir(path: str | Path) -> None:
    """Set a custom cache directory for downloaded models."""
    global _CACHE_DIR
    _CACHE_DIR = Path(path)
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)


class HubMixin:
    """
    Mixin class that adds Hugging Face Hub integration to models.

    Models using this mixin must implement:
        - save(path: str | Path) -> None
        - load(path: str | Path, **kwargs) -> Self (classmethod)

    This mixin adds:
        - push_to_hub(repo_id, ...) -> str
        - from_pretrained(repo_id, ...) -> Self (classmethod)

    Example:
        >>> class MyModel(nn.Module, HubMixin):
        ...     def save(self, path):
        ...         # Save model to directory
        ...         ...
        ...
        ...     @classmethod
        ...     def load(cls, path, device="cpu"):
        ...         # Load model from directory
        ...         ...
        >>>
        >>> # Now the model has hub methods
        >>> model.push_to_hub("username/my-model")
        >>> model = MyModel.from_pretrained("username/my-model")
    """

    # Subclasses can override these
    _hub_model_type: str = "unknown"  # e.g., "polymer-flow", "latent-diffusion"
    _hub_library_name: str = "ciffy"
    _hub_library_version: str | None = None

    def push_to_hub(
        self,
        repo_id: str,
        *,
        commit_message: str | None = None,
        private: bool = False,
        token: str | None = None,
        create_pr: bool = False,
        revision: str | None = None,
        model_card: str | None = None,
        tags: list[str] | None = None,
    ) -> str:
        """
        Push model to the Hugging Face Hub.

        Args:
            repo_id: Repository ID (e.g., "username/model-name" or "org/model-name").
            commit_message: Commit message for the upload.
            private: Whether the repository should be private.
            token: Hugging Face API token. If not provided, uses cached token.
            create_pr: Create a pull request instead of pushing directly.
            revision: Branch or tag to push to. Defaults to "main".
            model_card: Custom model card content (README.md). If None, generates one.
            tags: Additional tags for the model card.

        Returns:
            URL of the repository.

        Raises:
            ImportError: If huggingface_hub is not installed.
            ValueError: If the model doesn't have a save() method.
        """
        try:
            from huggingface_hub import HfApi, create_repo
        except ImportError:
            raise ImportError(
                "huggingface_hub is required for push_to_hub. "
                "Install with: pip install huggingface_hub"
            )

        if not hasattr(self, "save"):
            raise ValueError(
                f"{self.__class__.__name__} must implement save() to use push_to_hub()"
            )

        api = HfApi(token=token)

        # Create repo if it doesn't exist
        repo_url = create_repo(
            repo_id=repo_id,
            token=token,
            private=private,
            exist_ok=True,
            repo_type="model",
        )

        # Save model to temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Save the model
            self.save(tmpdir)

            # Add hub metadata
            self._write_hub_metadata(tmpdir, repo_id, tags)

            # Generate model card if not provided
            readme_path = tmpdir / "README.md"
            if model_card:
                readme_path.write_text(model_card)
            elif not readme_path.exists():
                readme_path.write_text(self._generate_model_card(repo_id, tags))

            # Upload
            if commit_message is None:
                commit_message = f"Upload {self.__class__.__name__}"

            api.upload_folder(
                folder_path=str(tmpdir),
                repo_id=repo_id,
                repo_type="model",
                commit_message=commit_message,
                create_pr=create_pr,
                revision=revision,
            )

        return str(repo_url)

    @classmethod
    def from_pretrained(
        cls: type[T],
        repo_id: str,
        *,
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        force_download: bool = False,
        token: str | None = None,
        local_files_only: bool = False,
        device: str = "cpu",
        **kwargs: Any,
    ) -> T:
        """
        Load a model from the Hugging Face Hub.

        Args:
            repo_id: Repository ID (e.g., "username/model-name").
                Can also be a local path to a directory.
            revision: Specific version (branch, tag, or commit hash).
            cache_dir: Directory to cache downloaded models.
            force_download: Force re-download even if cached.
            token: Hugging Face API token for private repos.
            local_files_only: Only use local cache, don't download.
            device: Device to load model on (default: "cpu").
            **kwargs: Additional arguments passed to load().

        Returns:
            Loaded model instance.

        Raises:
            ImportError: If huggingface_hub is not installed.
            ValueError: If the model doesn't have a load() method.
        """
        # Check if it's a local path
        local_path = Path(repo_id)
        if local_path.exists() and local_path.is_dir():
            return cls.load(local_path, device=device, **kwargs)

        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise ImportError(
                "huggingface_hub is required for from_pretrained. "
                "Install with: pip install huggingface_hub"
            )

        if not hasattr(cls, "load"):
            raise ValueError(
                f"{cls.__name__} must implement load() classmethod to use from_pretrained()"
            )

        # Use default cache dir if not specified
        if cache_dir is None:
            cache_dir = get_cache_dir()

        # Download model files
        local_dir = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            cache_dir=str(cache_dir),
            force_download=force_download,
            token=token,
            local_files_only=local_files_only,
        )

        # Load the model
        return cls.load(local_dir, device=device, **kwargs)

    def _write_hub_metadata(
        self,
        path: Path,
        repo_id: str,
        tags: list[str] | None = None,
    ) -> None:
        """Write hub-specific metadata to the model directory."""
        import ciffy

        metadata = {
            "library_name": self._hub_library_name,
            "library_version": getattr(ciffy, "__version__", "unknown"),
            "model_type": self._hub_model_type,
            "model_class": self.__class__.__name__,
        }

        if tags:
            metadata["tags"] = tags

        # Write to hub_metadata.json (separate from model config)
        metadata_path = path / "hub_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def _generate_model_card(
        self,
        repo_id: str,
        tags: list[str] | None = None,
    ) -> str:
        """Generate a default model card."""
        import ciffy

        all_tags = ["ciffy", self._hub_model_type]
        if tags:
            all_tags.extend(tags)
        tags_str = "\n".join(f"- {tag}" for tag in all_tags)

        return f"""---
library_name: ciffy
tags:
{tags_str}
---

# {repo_id.split("/")[-1]}

This is a **{self.__class__.__name__}** model trained with [ciffy](https://github.com/hmblair/ciffy).

## Usage

```python
from ciffy.nn.flow import PolymerFlowModel

# Load the model
model = {self.__class__.__name__}.from_pretrained("{repo_id}")

# Use the model
# ... (see ciffy documentation for details)
```

## Model Details

- **Model Type**: {self._hub_model_type}
- **Library**: ciffy v{getattr(ciffy, "__version__", "unknown")}
- **Class**: `{self.__class__.__module__}.{self.__class__.__name__}`

## Citation

If you use this model, please cite:

```bibtex
@software{{ciffy,
  title = {{ciffy: A library for macromolecular structure prediction}},
  url = {{https://github.com/hmblair/ciffy}}
}}
```
"""


def register_hub_model(
    model_class: type,
    model_type: str,
) -> type:
    """
    Register a model class for hub integration.

    This is a decorator that adds HubMixin to a class and sets the model type.

    Example:
        >>> @register_hub_model("polymer-flow")
        ... class PolymerFlowModel(nn.Module):
        ...     def save(self, path): ...
        ...     def load(cls, path): ...

    Args:
        model_class: The model class to register.
        model_type: String identifier for this model type.

    Returns:
        The modified class with HubMixin added.
    """
    # Add HubMixin if not already present
    if not issubclass(model_class, HubMixin):
        # Create new class with HubMixin
        model_class = type(
            model_class.__name__,
            (HubMixin, model_class),
            dict(model_class.__dict__),
        )

    model_class._hub_model_type = model_type
    return model_class
