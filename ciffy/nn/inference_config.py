"""
Configuration dataclasses for model inference.

Provides InferenceConfig and related classes for controlling structure generation
from trained models via YAML configuration files.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Optional, get_type_hints

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


@dataclass
class InferenceModelConfig:
    """Model configuration for inference.

    Attributes:
        checkpoint_path: Path to trained model checkpoint (.pt file).
        model_type: Type of model ('vae' for now, extensible to 'diffusion' later).
        device: Device override (e.g., 'cuda:0', 'cpu', 'mps'). None uses auto-detection.
    """

    checkpoint_path: str
    model_type: str = "vae"
    device: Optional[str] = None


@dataclass
class InferenceInputConfig:
    """Input configuration for inference.

    Attributes:
        sequences: List of sequences to generate structures for.
            Examples: ["MGKLF", "acgu", "ACDEFGHIKLMNPQRSTVWY"]
            Mutually exclusive with sequence_file.
        sequence_file: Path to FASTA or plain text file with sequences.
            One sequence per line for plain text, FASTA format with headers for FASTA files.
            Mutually exclusive with sequences.
        sequence_ids: Optional list of custom IDs for sequences (for output naming).
            If None, uses seq_0, seq_1, etc.

    Raises:
        ValueError: If both sequences and sequence_file are specified, or neither.
    """

    sequences: Optional[list[str]] = None
    sequence_file: Optional[str] = None
    sequence_ids: Optional[list[str]] = None

    def __post_init__(self):
        """Validate that exactly one input method is specified."""
        has_sequences = self.sequences is not None
        has_file = self.sequence_file is not None

        if not has_sequences and not has_file:
            raise ValueError(
                "Must specify either 'sequences' or 'sequence_file' in input config"
            )

        if has_sequences and has_file:
            raise ValueError(
                "Cannot specify both 'sequences' and 'sequence_file'. "
                "Choose one input method."
            )


@dataclass
class InferenceSamplingConfig:
    """Sampling configuration for inference.

    Attributes:
        n_samples: Number of structures to generate per sequence.
        temperature: Sampling temperature (higher = more diverse).
            For VAE: scales the diagonal of the latent covariance.
            For diffusion: affects denoising step size.
            Default 1.0.
        seed: Random seed for reproducibility. None for random initialization.
    """

    n_samples: int = 10
    temperature: float = 1.0
    seed: Optional[int] = None


@dataclass
class InferenceOutputConfig:
    """Output configuration for inference.

    Attributes:
        output_dir: Directory to save generated .cif files.
        id_prefix: Prefix for output file names (e.g., "gen_" → gen_seq0_sample0.cif).
        save_latents: If True, save latent vectors as .npz files (not yet implemented).
    """

    output_dir: str = "./inference_output"
    id_prefix: str = "gen_"
    save_latents: bool = False


@dataclass
class InferenceConfig:
    """Full inference configuration.

    This is a standalone config (not extending BaseConfig) because inference
    doesn't need training/wandb settings. However, it follows the same
    nested structure pattern for consistency with VAEConfig.

    Attributes:
        model: Model configuration (checkpoint path, type, device).
        input: Input configuration (sequences or file path).
        sampling: Sampling configuration (n_samples, temperature, seed).
        output: Output configuration (directory, prefix, etc.).

    Example YAML:
        model:
          checkpoint_path: ./checkpoints/vae/checkpoint_best.pt
          model_type: vae
          device: cuda:0

        input:
          sequences:
            - MGKLF
            - acgu
            - ARNDCEQGHILKMFPSTWYV

        sampling:
          n_samples: 10
          temperature: 1.0
          seed: 42

        output:
          output_dir: ./inference_output
          id_prefix: gen_
          save_latents: false
    """

    model: InferenceModelConfig = field(default_factory=InferenceModelConfig)
    input: InferenceInputConfig = field(default_factory=InferenceInputConfig)
    sampling: InferenceSamplingConfig = field(
        default_factory=InferenceSamplingConfig
    )
    output: InferenceOutputConfig = field(default_factory=InferenceOutputConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> InferenceConfig:
        """Load configuration from a YAML file.

        Args:
            path: Path to YAML configuration file.

        Returns:
            InferenceConfig instance with all sections properly initialized.

        Raises:
            ImportError: If PyYAML is not installed.
            FileNotFoundError: If config file does not exist.
            ValueError: If YAML is invalid or config is incomplete.

        Example:
            >>> config = InferenceConfig.from_yaml("configs/inference.yaml")
            >>> print(f"Checkpoint: {config.model.checkpoint_path}")
        """
        if not YAML_AVAILABLE:
            raise ImportError(
                "PyYAML is required for loading configs. "
                "Install with: pip install pyyaml"
            )

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        return cls._from_dict(raw)

    @classmethod
    def _from_dict(cls, data: dict) -> InferenceConfig:
        """Create config from dictionary, handling nested dataclasses.

        Args:
            data: Dictionary with config sections (model, input, sampling, output).

        Returns:
            InferenceConfig instance.
        """
        kwargs = {}

        # Get actual type hints (resolves string annotations from __future__)
        type_hints = get_type_hints(cls)

        all_fields = {f.name: f for f in fields(cls)}

        for name, f in all_fields.items():
            if name in data:
                value = data[name]
                # Get the actual type from type_hints
                actual_type = type_hints.get(name, f.type)
                # Check if this field type is a dataclass
                try:
                    if is_dataclass(actual_type):
                        kwargs[name] = cls._dict_to_dataclass(actual_type, value)
                    else:
                        kwargs[name] = value
                except TypeError:
                    # If is_dataclass fails (e.g., for generic types), just use the value
                    kwargs[name] = value

        return cls(**kwargs)

    @staticmethod
    def _dict_to_dataclass(dc_class: type, data: dict | None) -> object:
        """Convert a dictionary to a dataclass instance.

        Args:
            dc_class: Target dataclass type.
            data: Dictionary with field values, or None for defaults.

        Returns:
            Instance of dc_class with values from data.
        """
        if data is None:
            return dc_class()

        if not isinstance(data, dict):
            raise TypeError(f"Expected dict for {dc_class.__name__}, got {type(data)}")

        # Only pass known fields
        valid_fields = {f.name for f in fields(dc_class)}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return dc_class(**filtered)

    def validate(self) -> None:
        """Validate configuration completeness and compatibility.

        Raises:
            ValueError: If configuration is invalid.
        """
        # Check checkpoint exists
        if not Path(self.model.checkpoint_path).exists():
            raise ValueError(
                f"Checkpoint not found: {self.model.checkpoint_path}"
            )

        # Check sequence file exists if specified
        if self.input.sequence_file:
            if not Path(self.input.sequence_file).exists():
                raise ValueError(
                    f"Sequence file not found: {self.input.sequence_file}"
                )


__all__ = [
    "InferenceModelConfig",
    "InferenceInputConfig",
    "InferenceSamplingConfig",
    "InferenceOutputConfig",
    "InferenceConfig",
]
