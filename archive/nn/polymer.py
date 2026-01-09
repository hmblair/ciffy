"""
PolymerModel: Orchestrates per-residue generative models for full polymer encoding/decoding.

This module provides a wrapper around residue-level models (ResidueFlowModel, ResidueVAE)
that handles:
- Encoding each residue with its appropriate model (after alignment)
- Decoding and positioning residues using SE(3) transforms via Polymer.append()

Works with any model implementing ResidueGenerativeCore protocol (Flow, VAE, etc.).

Example (with Polymer objects - recommended):
    >>> from ciffy.nn import PolymerModel
    >>> import ciffy
    >>>
    >>> # Load pre-trained polymer model
    >>> polymer_model = PolymerModel.load("models/rna")
    >>>
    >>> # Encode polymer to latents (handles alignment automatically)
    >>> polymer = ciffy.load("structure.cif").poly()
    >>> latents = polymer_model.encode_polymer(polymer)
    >>>
    >>> # Decode latents back to polymer
    >>> new_polymer = polymer_model.decode_to_polymer(latents, polymer)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, Union, runtime_checkable

import numpy as np
import torch
import torch.nn as nn

from ciffy.polymer import Polymer
from ciffy.nn.io.hub import HubMixin
from ciffy.nn.io.registry import register_model

if TYPE_CHECKING:
    from ciffy.biochemistry import AtomGroup, Residue
    from ciffy.nn.flow.residue import ResidueFlowModel


# =============================================================================
# Int-Key ModuleDict Wrapper
# =============================================================================


class _IntKeyModuleDict(nn.ModuleDict):
    """ModuleDict that accepts int keys (converts to str internally).

    PyTorch's ModuleDict requires string keys, but ciffy uses int indices
    for residue types. This wrapper accepts both int and str keys.
    """

    def __getitem__(self, key):
        return super().__getitem__(str(key))

    def __contains__(self, key):
        return super().__contains__(str(key))

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


# =============================================================================
# Protocol for Residue-Level Generative Models
# =============================================================================


@runtime_checkable
class ResidueGenerativeCore(Protocol):
    """
    Protocol for residue-level generative models.

    Any model implementing this protocol can be used with PolymerModel
    for full-polymer encoding/decoding. Both ResidueFlowModel and ResidueVAE
    implement this interface.

    Required attributes:
        latent_dim: Dimensionality of latent space.
        n_atoms: Number of atoms per residue.
        residue: The residue type this model handles.

    Required properties:
        atoms: AtomGroup subset containing the atoms used.

    Required methods:
        encode: Encode pre-aligned coordinates to latent space.
        decode: Decode latents to (coords, transforms).
        sample: Sample new conformations from prior.
    """

    latent_dim: int
    n_atoms: int
    residue: "Residue"

    @property
    def atoms(self) -> "AtomGroup":
        """AtomGroup subset containing the atoms used by this model."""
        ...

    def encode(
        self,
        coords: torch.Tensor,
        next_coords: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode coordinates to latent space."""
        ...

    def decode(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode latents to (coords, transforms)."""
        ...

    def sample(self, n_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample new (coords, transforms) from prior."""
        ...


# Type alias for sequence input (numpy array or torch tensor of int)
SequenceArray = Union[np.ndarray, torch.Tensor]


def _to_numpy_int64(sequence: SequenceArray) -> np.ndarray:
    """Convert sequence to numpy int64 array."""
    if isinstance(sequence, torch.Tensor):
        return sequence.cpu().numpy().astype(np.int64)
    return np.asarray(sequence, dtype=np.int64)


def _normalize_key(key) -> int:
    """Convert a residue key to int, accepting both Residue and int."""
    if isinstance(key, int):
        return key
    # Assume it's a Residue (AtomGroup) with .value attribute
    return key.value


@register_model("polymer")
class PolymerModel(nn.Module, HubMixin):
    """
    Orchestrates per-residue generative models for full polymer encoding/decoding.

    This class wraps multiple residue-level models (one per residue type)
    and provides encode/decode methods that work on entire polymers.
    Works with any model implementing ResidueGenerativeCore (Flow, VAE, etc.).

    Encoding aligns each residue to its canonical frame (glycosidic for RNA,
    backbone for protein) via Polymer.align(), then encodes to latent space.
    Decoding reconstructs coordinates and chains residues together using
    SE(3) transforms via Polymer.append().

    Attributes:
        latent_dim: Dimension of per-residue latent space.
        supported_residues: Set of supported residue type indices.
        atom_counts: Dict mapping residue type (int) to atom count.

    Example:
        >>> polymer_model = PolymerModel.load("models/rna")
        >>> polymer = ciffy.load("structure.cif").poly()
        >>> latents = polymer_model.encode_polymer(polymer)
        >>> new_polymer = polymer_model.decode_to_polymer(latents, polymer)
    """

    _hub_model_type = "polymer"

    def __init__(self, residue_models: dict["int | Residue", "ResidueGenerativeCore"]):
        """
        Initialize with pre-trained per-residue generative models.

        Args:
            residue_models: Dict mapping residue type to model implementing
                            ResidueGenerativeCore (ResidueFlowModel, ResidueVAE, etc.).
                            Keys can be either int values or Residue enums.
                            All models should have the same latent_dim.

        Example:
            >>> # Works with ResidueFlowModel:
            >>> PolymerModel({Residue.A: flow_a, Residue.G: flow_g})
            >>> # Also works with ResidueVAE:
            >>> PolymerModel({Residue.A: vae_a, Residue.G: vae_g})
        """
        super().__init__()

        if not residue_models:
            raise ValueError("residue_models cannot be empty")

        # Normalize keys to int, store as strings (required by ModuleDict)
        normalized = {}
        for k, v in residue_models.items():
            int_key = _normalize_key(k)
            normalized[str(int_key)] = v

        # Use custom ModuleDict that accepts int keys
        self.residue_models = _IntKeyModuleDict(normalized)

        # Validate all models have same latent dim
        latent_dims = [m.latent_dim for m in self.residue_models.values()]
        if len(set(latent_dims)) > 1:
            raise ValueError(
                f"All residue models must have same latent_dim, got {latent_dims}"
            )
        self.latent_dim = latent_dims[0]

        # Cache atom counts and supported types for fast validation
        self._atom_counts: dict[int, int] = {
            int(k): model.n_atoms for k, model in self.residue_models.items()
        }
        self._supported_types_set: set[int] = set(self._atom_counts.keys())

    def _get_model(self, res_type: int) -> "ResidueFlowModel":
        """Get residue model by int key."""
        return self.residue_models[res_type]

    @property
    def supported_residues(self) -> set[int]:
        """Set of supported residue type indices.

        Example:
            >>> model.supported_residues
            {0, 1, 2, 3}  # A, C, G, U
        """
        return self._supported_types_set

    @property
    def atom_counts(self) -> dict[int, int]:
        """Dict mapping residue type (int) to number of atoms.

        Example:
            >>> model.atom_counts[0]  # Residue.A
            22
        """
        return self._atom_counts

    def get_residue_model(self, residue_type: int) -> "ResidueGenerativeCore":
        """Get the residue model for a specific residue type.

        Args:
            residue_type: Integer residue type index (e.g., Residue.A.value).

        Returns:
            The residue model for that type.

        Raises:
            KeyError: If residue type is not supported.

        Example:
            >>> model_a = polymer_model.get_residue_model(Residue.A.value)
            >>> model_a.n_atoms
            22
        """
        try:
            return self.residue_models[str(residue_type)]
        except KeyError:
            raise KeyError(
                f"Residue type {residue_type} not supported. "
                f"Supported types: {sorted(self._supported_types_set)}"
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Encoding / Decoding
    # ─────────────────────────────────────────────────────────────────────────

    def _validate_sequence(self, sequence: np.ndarray) -> None:
        """Validate sequence contains only supported residue types."""
        unique_types = set(sequence.tolist())
        unsupported = unique_types - self._supported_types_set

        if unsupported:
            from ciffy.biochemistry import Residue
            names = [Residue.from_index(v).name for v in sorted(unsupported)]
            available = [Residue.from_index(v).name for v in sorted(self._supported_types_set)]
            raise ValueError(
                f"Unsupported residue types: {names}. "
                f"Available: {available}"
            )

    def _get_atom_counts(self, sequence: np.ndarray) -> list[int]:
        """Get atom counts for each residue in sequence."""
        return [self._atom_counts[int(res_type)] for res_type in sequence]

    def _validate_coords_shape(
        self,
        coords: torch.Tensor,
        sequence: np.ndarray,
    ) -> None:
        """Validate that coords shape matches sequence."""
        expected_atoms = sum(self._atom_counts[int(t)] for t in sequence)
        if coords.shape[0] != expected_atoms:
            raise ValueError(
                f"coords has {coords.shape[0]} atoms but sequence expects {expected_atoms}. "
                f"Sequence has {len(sequence)} residues."
            )
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"coords must be (N, 3), got shape {tuple(coords.shape)}")

    def encode(
        self,
        aligned_coords: list[torch.Tensor],
        sequence: SequenceArray,
    ) -> torch.Tensor:
        """
        Encode pre-aligned per-residue coordinates to latent vectors.

        This is the low-level encoding method that expects coordinates already
        aligned to canonical frames (e.g., glycosidic frame for RNA). Use
        encode_from_polymer() for automatic alignment.

        Args:
            aligned_coords: List of (n_atoms, 3) tensors, one per residue,
                already aligned to canonical frames via Polymer.align().
            sequence: Int array of residue types (e.g., polymer.sequence).

        Returns:
            (n_residues, latent_dim) latent vectors.

        See Also:
            encode_from_polymer: Handles alignment automatically.
        """
        sequence = _to_numpy_int64(sequence)
        self._validate_sequence(sequence)

        if len(aligned_coords) != len(sequence):
            raise ValueError(
                f"Got {len(aligned_coords)} coordinate tensors but sequence "
                f"has {len(sequence)} residues"
            )

        latents = []

        for i, res_type in enumerate(sequence):
            model = self._get_model(int(res_type))
            res_coords = aligned_coords[i]

            # Validate shape
            if res_coords.shape[0] != model.n_atoms:
                from ciffy.biochemistry import Residue
                res_name = Residue.from_index(int(res_type)).name
                raise ValueError(
                    f"Residue {i} ({res_name}): expected {model.n_atoms} atoms, "
                    f"got {res_coords.shape[0]}"
                )

            # Reshape to (1, n_atoms, 3) for model.encode()
            res_coords = res_coords.unsqueeze(0)

            # Encode (models expect aligned input)
            z = model.encode(res_coords)  # (1, k)
            latents.append(z.squeeze(0))  # (k,)

        return torch.stack(latents)  # (n_residues, k)

    def decode(
        self,
        latents: torch.Tensor,
        sequence: SequenceArray,
        latent_bound: float | None = None,
        project: bool = False,
    ) -> torch.Tensor:
        """
        Decode latent vectors to positioned polymer coordinates.

        Uses Polymer.append() for chain assembly, which handles frame computation
        internally.

        Returns a PyTorch tensor. Call .numpy() if NumPy array is needed.

        Args:
            latents: (n_residues, latent_dim) latent vectors.
            sequence: Int array of residue types (e.g., polymer.sequence).
            latent_bound: Optional soft bound for latent values using tanh.
                When set, values are bounded to [-bound, bound] range.
                Useful during gradient-based optimization to prevent explosion
                from out-of-distribution latents. Default None (no bounding).
            project: If True, project coordinates to satisfy bond constraints.

        Returns:
            (N, 3) flat coordinate tensor with all residues positioned.
        """
        sequence = _to_numpy_int64(sequence)

        if latents.shape[0] != len(sequence):
            raise ValueError(
                f"latents has {latents.shape[0]} rows but sequence has {len(sequence)} residues"
            )

        if len(sequence) == 0:
            return torch.empty(0, 3, device=latents.device, dtype=latents.dtype)

        # Apply soft bound to prevent gradient explosion from out-of-distribution latents
        if latent_bound is not None:
            latents = latent_bound * torch.tanh(latents / latent_bound)

        # Build polymer using _append() - handles frame computation internally
        poly = Polymer()

        for i, res_type in enumerate(sequence):
            model = self._get_model(int(res_type))

            # Decode this residue (keep gradients for optimization)
            # coords_i: local coordinates, transform_i: positions THIS residue
            coords_i, transform_i = model.decode(latents[i:i + 1], project=project)

            # coords_i is (1, n_atoms, 3), squeeze to (n_atoms, 3)
            # transform_i is (1, 6), squeeze to (6,)
            coords_i = coords_i.squeeze(0)
            transform_i = transform_i.squeeze(0)

            # Get atoms array for frame computation
            atoms_i = torch.tensor(model.atoms.index(), dtype=torch.int64, device=coords_i.device)

            if i == 0:
                # First residue - no transform needed
                poly = poly._append(model.residue, coords_i, atoms=atoms_i)
            else:
                # Position using THIS residue's transform
                poly = poly._append(model.residue, coords_i, transform_i, atoms=atoms_i)

        return poly.coordinates

    def _sample_coords(
        self,
        sequence: SequenceArray,
        n_samples: int = 1,
        temperature: float = 1.0,
        project: bool = False,
    ) -> list[torch.Tensor]:
        """
        Sample coordinate tensors from sequence (internal method).

        Args:
            sequence: Int array of residue types.
            n_samples: Number of samples to generate.
            temperature: Scales latent noise (higher = more diverse).
            project: If True, project coordinates to satisfy bond constraints.

        Returns:
            List of (N, 3) coordinate tensors.
        """
        sequence = _to_numpy_int64(sequence)

        if len(sequence) == 0:
            return [torch.empty(0, 3, device=self.device) for _ in range(n_samples)]

        samples = []
        for _ in range(n_samples):
            latents = torch.randn(len(sequence), self.latent_dim, device=self.device)
            latents = latents * temperature
            coords = self.decode(latents, sequence, project=project)
            samples.append(coords)

        return samples

    def sample_template(
        self,
        sequence: str,
        n_samples: int = 1,
        temperature: float = 1.0,
        id: str = "sampled",
        project: bool = False,
    ) -> "Polymer | list[Polymer]":
        """
        Sample polymer conformations directly from a sequence string.

        Generates a template Polymer from the sequence string and samples
        new conformations. This is the simplest way to generate structures
        from scratch.

        Args:
            sequence: Sequence string (e.g., "acgu" for RNA, "MGKLF" for protein).
            n_samples: Number of conformations to generate.
            temperature: Sampling temperature (higher = more diverse).
            id: PDB ID for the generated polymers.
            project: If True, project coordinates to satisfy bond constraints.

        Returns:
            If n_samples=1: Single Polymer with generated coordinates.
            If n_samples>1: List of Polymers.

        Example:
            >>> model = PolymerModel.load("path/to/model")
            >>> polymer = model.sample_template("acgu")
            >>> polymer.write("sampled.cif")
            >>>
            >>> # Generate multiple samples
            >>> samples = model.sample_template("acgu", n_samples=10)
            >>> for i, p in enumerate(samples):
            ...     p.write(f"sample_{i}.cif")
        """
        from ciffy import template

        # Create template with correct atoms for this model
        template = template(sequence, atoms=self.atom_filter, id=id)

        # Use protocol-compliant sample method
        samples = self.sample(
            template, n_samples=n_samples, temperature=temperature, project=project
        )

        if n_samples == 1:
            return samples[0]
        return samples

    def sample(
        self,
        template: "Polymer",
        n_samples: int = 1,
        temperature: float = 1.0,
        project: bool = False,
        **kwargs,
    ) -> list["Polymer"]:
        """
        Generate polymer conformations from a template.

        This method implements the PolymerGenerativeModel protocol, enabling
        this model to be used interchangeably with other generative models.

        Args:
            template: Template Polymer with sequence and topology information.
                Must have numpy backend (will be validated).
            n_samples: Number of independent conformations to generate.
            temperature: Sampling temperature. Scales the latent noise
                (higher = more diverse). Default 1.0.
            project: If True, project coordinates to satisfy bond constraints.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            List of n_samples Polymers with generated coordinates.

        Raises:
            ValueError: If template has incompatible backend or unsupported residues.

        Example:
            >>> model = PolymerModel.load("path/to/model")
            >>> template = ciffy.load("structure.cif").poly()
            >>> samples = model.sample(template, n_samples=10)
            >>> for i, p in enumerate(samples):
            ...     p.write(f"sample_{i}.cif")
        """
        # Validate template backend - output will be numpy
        if template.backend != "numpy":
            raise ValueError(
                f"Template must have numpy backend, got '{template.backend}'. "
                f"Call template.numpy() first."
            )

        # Validate sequence contains only supported residue types
        sequence = _to_numpy_int64(template.sequence)
        self._validate_sequence(sequence)

        if len(sequence) == 0:
            return [template.copy(coordinates=np.empty((0, 3)))]

        # Create output template with only atoms this model knows about
        # This ensures the output polymer matches the sampled coordinates
        from ciffy import template as _template
        output_template = _template(
            template.sequence_str(),
            atoms=self.atom_filter,
            id=template.pdb_id or "sampled",
        )

        # Sample coordinates
        coords_list = self._sample_coords(sequence, n_samples, temperature, project)

        # Convert to Polymers with correct atom structure
        return [
            output_template.copy(coordinates=coords.detach().cpu().numpy())
            for coords in coords_list
        ]

    @property
    def supported_residue_types(self) -> np.ndarray:
        """Array of supported residue type indices (int)."""
        return np.array(sorted(int(k) for k in self.residue_models.keys()), dtype=np.int64)

    @property
    def residue_types(self) -> list["AtomGroup"]:
        """List of residue types this model can handle (as AtomGroup/Residue).

        For the set of integer indices, use `supported_residues` instead.
        """
        from ciffy.biochemistry import Residue
        return [Residue.from_index(int(k)) for k in sorted(self.residue_models.keys(), key=int)]

    @property
    def atom_filter(self) -> dict[int, list[int]]:
        """
        Get atom filter dict for use with ciffy.template(atoms=...).

        Returns a dict mapping residue type (int) to the list of atom values
        that this model uses. Pass this to template() to create templates
        with only the atoms the model knows about.

        Example:
            >>> template = ciffy.template("acgu", atoms=polymer_model.atom_filter)
            >>> # template now has only the atoms used by the residue models
        """
        return {
            int(res_type): list(model._atom_indices)
            for res_type, model in self.residue_models.items()
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Device Management (inherited from nn.Module, but add convenience property)
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def device(self) -> "torch.device":
        """Device where model parameters reside."""
        return next(iter(self.residue_models.values())).device

    # ─────────────────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """
        Save model to directory.

        Each residue model is saved to a subdirectory named by residue.
        Supports any model implementing ResidueGenerativeCore with save().

        Args:
            path: Directory to save to.
        """
        import json
        from ciffy.biochemistry import Residue

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save each residue model (use residue name as subdirectory)
        residue_names = []
        for res_type_str, model in self.residue_models.items():
            res_name = Residue.from_index(int(res_type_str)).name
            model.save(path / res_name)
            residue_names.append(res_name)

        # Save metadata
        config = {
            "residue_types": residue_names,
            "latent_dim": self.latent_dim,
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(
        cls,
        path: str | Path,
        device: str = "cpu",
        jit: bool = False,
    ) -> "PolymerModel":
        """
        Load model from directory.

        Automatically detects model types (ResidueFlowModel, ResidueVAE,
        InvariantResidueVAE) from saved config.

        Args:
            path: Directory containing saved model.
            device: Device to load models to.
            jit: Whether to JIT-compile the decoders (Flow models only).

        Returns:
            Loaded PolymerModel.
        """
        import json
        from ciffy.biochemistry import Residue

        path = Path(path)

        # Check for consolidated model (saved in path/consolidated/)
        consolidated_path = path / "consolidated"
        if (consolidated_path / "config.json").exists():
            from ciffy.nn.vae.residue.consolidated import ConsolidatedResidueVAEModel
            consolidated_model = ConsolidatedResidueVAEModel.load(consolidated_path, device=device)
            residue_models = consolidated_model.as_residue_models()
            return cls(residue_models)

        with open(path / "config.json") as f:
            config = json.load(f)

        # Load models with int keys, detecting type from each model's config
        residue_models = {}
        for res_name in config["residue_types"]:
            res_type = getattr(Residue, res_name).value
            model_path = path / res_name

            # Read the model's config to determine type
            with open(model_path / "config.json") as f:
                model_config = json.load(f)

            model_type = model_config.get("model_type", "residue-flow")

            # Load the appropriate model type
            model = cls._load_residue_model(model_type, model_path, device, jit)
            residue_models[res_type] = model

        return cls(residue_models)

    @staticmethod
    def _load_residue_model(
        model_type: str,
        path: Path,
        device: str,
        jit: bool,
    ) -> "ResidueGenerativeCore":
        """Load a residue model based on its type."""
        if model_type == "residue-flow":
            from ciffy.nn.flow.residue import ResidueFlowModel
            return ResidueFlowModel.load(path, device=device, jit=jit)
        elif model_type in ("residue-vae", "residue-invariant-vae", "attention-residue-vae"):
            raise ValueError(
                f"Model type '{model_type}' is no longer supported. "
                "Please retrain using model_type='consolidated'."
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    # ─────────────────────────────────────────────────────────────────────────
    # Unified Save/Load (SaveableModel protocol)
    # ─────────────────────────────────────────────────────────────────────────

    def get_save_state(self) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        """Get state for unified save format.

        Returns:
            Tuple of (tensors_dict, config_dict) for safetensors serialization.
        """
        from ciffy.biochemistry import Residue

        tensors = {}
        residue_configs = {}

        for res_type_str, model in self.residue_models.items():
            res_name = Residue.from_index(int(res_type_str)).name

            # Prefix all tensors with residue name
            for k, v in model.flow.state_dict().items():
                tensors[f"{res_name}.{k}"] = v

            # Store per-residue config
            residue_configs[res_name] = {
                "atom_indices": [int(x) for x in model._atom_indices],
                "n_atoms": model.n_atoms,
                "n_layers": len(model.flow.layers) // 2,
                "hidden_dim": model.flow.layers[1].net.net[0].out_features,
                "bound": float(model.flow.bound) if model.flow.bound is not None else None,
            }

        config = {
            "latent_dim": self.latent_dim,
            "residues": residue_configs,
        }

        return tensors, config

    @classmethod
    def from_save_state(
        cls,
        tensors: dict[str, torch.Tensor],
        config: dict[str, Any],
        device: str = "cpu",
    ) -> "PolymerModel":
        """Reconstruct model from unified save format.

        Args:
            tensors: Loaded tensors dict with prefixed keys.
            config: Loaded config dict.
            device: Device to load model to.

        Returns:
            Reconstructed PolymerModel.
        """
        from ciffy.biochemistry import Residue
        from ciffy.nn.flow.residue import ResidueFlowModel, PCAFlow

        residue_models = {}

        for res_name, res_config in config["residues"].items():
            res_type = getattr(Residue, res_name)

            # Extract tensors for this residue
            prefix = f"{res_name}."
            res_tensors = {
                k[len(prefix):]: v
                for k, v in tensors.items()
                if k.startswith(prefix)
            }

            # Reconstruct PCAFlow
            V = res_tensors["V"].float()
            mean = res_tensors["mean"].float()

            flow = PCAFlow(
                V, mean,
                n_layers=res_config["n_layers"],
                hidden_dim=res_config["hidden_dim"],
                bound=res_config["bound"],
            ).to(device)
            flow.load_state_dict(res_tensors)

            # Create ResidueFlowModel
            residue_models[res_type.value] = ResidueFlowModel(
                flow=flow,
                residue=res_type,
                atom_indices=res_config["atom_indices"],
                n_atoms=res_config["n_atoms"],
            )

        return cls(residue_models)

    # ─────────────────────────────────────────────────────────────────────────
    # Convenience Methods for Polymer Objects
    # ─────────────────────────────────────────────────────────────────────────

    def encode_polymer(self, polymer: "Polymer") -> torch.Tensor:
        """
        Encode a Polymer object directly to latent vectors.

        This is the recommended way to encode polymers. It automatically
        aligns each residue to its canonical frame, sorts atoms to canonical
        order, and filters to only include atoms the model was trained on.

        Args:
            polymer: Polymer object to encode. Must have coordinates.

        Returns:
            (n_residues, latent_dim) latent vectors.

        Raises:
            ValueError: If the polymer contains unsupported residue types.
            AttributeError: If polymer has no coordinates.

        Example:
            >>> polymer = ciffy.load("structure.cif").poly()
            >>> latents = model.encode_polymer(polymer)
        """
        from ciffy.backend import to_numpy
        from ciffy.backend.ops import isin

        # Validate sequence
        sequence = polymer.sequence
        seq_np = to_numpy(sequence)
        self._validate_sequence(seq_np)

        # Align to canonical frames and sort atoms by enum value
        # This ensures consistent ordering matching training data
        aligned, _ = polymer.align()
        canonical = aligned.sort_atoms()

        # Filter each residue to only include model's atoms
        filtered_coords = []
        for i, res_type in enumerate(seq_np):
            res_type = int(res_type)

            # Get expected atoms for this residue type (already sorted)
            expected_atoms = np.array(self.atom_filter[res_type], dtype=np.int64)

            # Get residue (atoms are sorted by enum value)
            res = canonical.residue(i)
            atoms_i = to_numpy(res.atoms)

            # Filter to expected atoms using boolean mask
            mask = isin(atoms_i, expected_atoms)
            res_coords = res.coordinates[mask]

            # Validate we got all expected atoms
            if res_coords.shape[0] != len(expected_atoms):
                from ciffy.biochemistry import Residue
                res_name = Residue.from_index(res_type).name
                missing = set(expected_atoms) - set(atoms_i[mask])
                raise ValueError(
                    f"Residue {i} ({res_name}): missing atoms {missing}"
                )

            filtered_coords.append(torch.as_tensor(res_coords, dtype=torch.float32))

        return self.encode(filtered_coords, seq_np)

    def decode_to_polymer(
        self,
        latents: torch.Tensor,
        template: "Polymer",
    ) -> "Polymer":
        """
        Decode latents to a new Polymer with same metadata as template.

        Creates a new Polymer object with the decoded coordinates while
        preserving all metadata (sequence, chain info, etc.) from the template.

        Args:
            latents: (n_residues, latent_dim) latent vectors.
            template: Polymer to use as template for metadata.

        Returns:
            New Polymer with decoded coordinates.

        Example:
            >>> # Encode, modify latents, decode
            >>> latents = model.encode_polymer(polymer)
            >>> latents_modified = latents + torch.randn_like(latents) * 0.1
            >>> polymer_new = model.decode_to_polymer(latents_modified, polymer)
        """
        # Use template.sequence directly (already int array)
        coords = self.decode(latents, template.sequence)

        # Convert to numpy for Polymer
        import numpy as np
        if isinstance(coords, torch.Tensor):
            coords_np = coords.detach().cpu().numpy()
        else:
            coords_np = np.asarray(coords)

        return template.copy(coordinates=coords_np)

    def interpolate(
        self,
        polymer1: "Polymer",
        polymer2: "Polymer",
        n_steps: int = 10,
        include_endpoints: bool = True,
    ) -> list["Polymer"]:
        """
        Generate interpolated conformations between two polymers.

        Creates a smooth transition in latent space between two polymer
        conformations.

        Args:
            polymer1: Starting polymer conformation.
            polymer2: Ending polymer conformation.
            n_steps: Number of interpolation steps.
            include_endpoints: If True, include polymer1 and polymer2 in output.

        Returns:
            List of Polymer objects representing the interpolated conformations.

        Raises:
            ValueError: If polymers have different sequences.

        Example:
            >>> conformations = model.interpolate(polymer_open, polymer_closed, n_steps=20)
            >>> for i, conf in enumerate(conformations):
            ...     conf.write(f"frame_{i:03d}.cif")
        """
        import numpy as np

        # Validate sequences match
        seq1 = list(polymer1.sequence)
        seq2 = list(polymer2.sequence)
        if seq1 != seq2:
            raise ValueError(
                f"Polymers must have same sequence. "
                f"Got lengths {len(seq1)} and {len(seq2)}"
            )

        # Encode both
        z1 = self.encode_polymer(polymer1)
        z2 = self.encode_polymer(polymer2)

        # Generate interpolation weights
        if include_endpoints:
            weights = np.linspace(0, 1, n_steps)
        else:
            weights = np.linspace(0, 1, n_steps + 2)[1:-1]

        # Interpolate and decode
        results = []
        for w in weights:
            z_interp = (1 - w) * z1 + w * z2
            polymer_interp = self.decode_to_polymer(z_interp, polymer1)
            results.append(polymer_interp)

        return results

    def project_geometry(
        self,
        coords: torch.Tensor,
        sequence: "SequenceArray",
        n_steps: int = 2,
        implicit: bool = True,
    ) -> torch.Tensor:
        """
        Project coordinates onto ideal bond length constraints.

        Applies geometry projection to each residue using its corresponding
        residue model. This fixes local geometry errors while preserving
        overall conformation.

        Args:
            coords: (N, 3) flat coordinates for entire polymer.
            sequence: Int array of residue types.
            n_steps: Number of Newton steps per residue (default 2).
            implicit: If True, use implicit differentiation for clean gradients
                that stay on the constraint manifold. Recommended for optimization.

        Returns:
            Projected coordinates with same shape as input.

        Example:
            >>> coords = model.decode(latents, sequence)
            >>> coords_fixed = model.project_geometry(coords, sequence)
        """
        sequence = _to_numpy_int64(sequence)

        if len(sequence) == 0:
            return coords

        # Get atom counts per residue to split coordinates
        atom_counts = self._get_atom_counts(sequence)
        total_atoms = sum(atom_counts)

        if coords.shape[0] != total_atoms:
            raise ValueError(
                f"Coordinate count {coords.shape[0]} doesn't match "
                f"sequence atom count {total_atoms}"
            )

        # Split into per-residue coordinates
        coords_split = torch.split(coords, atom_counts, dim=0)

        # Project each residue
        projected = []
        for i, res_type in enumerate(sequence):
            model = self._get_model(int(res_type))
            coords_i = coords_split[i]  # (n_atoms_i, 3)

            # Project geometry for this residue
            coords_proj = model.project_geometry(
                coords_i.unsqueeze(0),  # (1, n_atoms, 3)
                n_steps=n_steps,
                implicit=implicit,
            ).squeeze(0)  # Back to (n_atoms, 3)

            projected.append(coords_proj)

        return torch.cat(projected, dim=0)

    def __repr__(self) -> str:
        from ciffy.biochemistry import Residue
        residues = [Residue.from_index(int(k)).name for k in self.residue_models.keys()]
        return f"PolymerModel(residues={residues}, latent_dim={self.latent_dim})"


# Backwards-compatible alias
PolymerFlowModel = PolymerModel

__all__ = [
    "ResidueGenerativeCore",
    "PolymerModel",
    "PolymerFlowModel",  # Deprecated alias
]
