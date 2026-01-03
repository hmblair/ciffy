"""
PolymerModel: Orchestrates per-residue generative models for full polymer encoding/decoding.

This module provides a wrapper around residue-level models (ResidueFlowModel, ResidueVAE)
that handles:
- Partitioning flat coordinate arrays by residue
- Encoding each residue with its appropriate model
- Decoding and positioning residues using SE(3) transforms
- Lazy computation with dirty-flag caching for efficient coordinate/latent access

Works with any model implementing ResidueGenerativeCore protocol (Flow, VAE, etc.).

Example (stateless API):
    >>> from ciffy.nn import PolymerModel
    >>> from ciffy.nn.flow.residue import ResidueFlowModel
    >>> from ciffy.biochemistry import Residue
    >>> import numpy as np
    >>>
    >>> # Load pre-trained per-residue models
    >>> models = {
    ...     Residue.A: ResidueFlowModel.load("models/A"),
    ...     Residue.G: ResidueFlowModel.load("models/G"),
    ... }
    >>> polymer_model = PolymerModel(models)
    >>>
    >>> # Encode polymer coordinates (sequence as int array)
    >>> sequence = np.array([Residue.A.value, Residue.G.value, Residue.A.value])
    >>> latents = polymer_model.encode(coords, sequence)  # (3, k)
    >>>
    >>> # Decode back to positioned coordinates
    >>> coords_recon = polymer_model.decode(latents, sequence)  # (N, 3)

Example (with Polymer objects - recommended):
    >>> polymer = ciffy.load("structure.cif").poly()
    >>> latents = polymer_model.encode_polymer(polymer)  # Uses polymer.sequence directly
    >>> new_polymer = polymer_model.decode_to_polymer(latents, polymer)

Example (stateful lazy API):
    >>> # Bind to a sequence for lazy computation
    >>> polymer_model.bind(polymer.sequence)
    >>>
    >>> # Set coordinates - latents computed lazily
    >>> polymer_model.coordinates = coords
    >>> z = polymer_model.latents  # Computed on first access
    >>>
    >>> # Modify latents - coordinates recomputed lazily
    >>> polymer_model.latents = modified_z
    >>> new_coords = polymer_model.coordinates  # Recomputed on access
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, Union, runtime_checkable

import numpy as np
import torch
import torch.nn as nn

# Frame-based positioning for chain assembly
from ciffy.geometry import FrameIndices, position_next_residue, align_to_frame
from ciffy.nn.hub import HubMixin
from ciffy.nn.model_registry import register_model

if TYPE_CHECKING:
    from ciffy.biochemistry import Residue, AtomGroup
    from ciffy.nn.flow.residue import ResidueFlowModel


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
        frame_indices: FrameIndices for positioning (or None).

    Required methods:
        encode: Encode coordinates to latent space.
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

    @property
    def frame_indices(self) -> FrameIndices | None:
        """FrameIndices for positioning, or None if not available."""
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

    The encode method partitions a flat (N, 3) coordinate array by residue
    and encodes each residue independently. The decode method reconstructs
    coordinates and chains them together using the SE(3) transforms that
    each residue model outputs.

    Supports two usage patterns:

    1. **Stateless API**: Pass coordinates/latents and sequence to encode()/decode().
       Each call is independent with no state. Sequence is an int array.

    2. **Stateful Lazy API**: Call bind(sequence), then use coordinates/latents
       properties. Values are cached and lazily recomputed only when needed.

    Attributes:
        latent_dim: Dimension of per-residue latent space.
        supported_residues: Set of supported residue type indices.
        atom_counts: Dict mapping residue type (int) to atom count.

    Example (with Polymer - recommended):
        >>> polymer_model = PolymerModel(models)
        >>> latents = polymer_model.encode_polymer(polymer)
        >>> new_polymer = polymer_model.decode_to_polymer(latents, polymer)

    Example (stateless with int array):
        >>> seq = np.array([Residue.A.value, Residue.G.value])
        >>> latents = polymer_model.encode(coords, seq)
        >>> coords_recon = polymer_model.decode(latents, seq)

    Example (stateful):
        >>> polymer_model.bind(polymer.sequence)
        >>> polymer_model.coordinates = coords
        >>> z = polymer_model.latents  # Lazily computed
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

        # Normalize keys to int and store mapping
        # nn.ModuleDict requires string keys, so we store int->str mapping
        self._key_to_str: dict[int, str] = {}
        normalized = {}
        for k, v in residue_models.items():
            int_key = _normalize_key(k)
            str_key = str(int_key)
            self._key_to_str[int_key] = str_key
            normalized[str_key] = v

        # Use nn.ModuleDict for proper parameter tracking
        self.residue_models = nn.ModuleDict(normalized)

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

        # Stateful lazy computation attributes (not parameters, just cache)
        self._sequence: np.ndarray | None = None
        self._cached_latents: torch.Tensor | None = None
        self._cached_coordinates: torch.Tensor | None = None
        self._latents_dirty: bool = True
        self._coords_dirty: bool = True

    def _get_model(self, res_type: int) -> "ResidueFlowModel":
        """Get residue model by int key."""
        return self.residue_models[str(res_type)]

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
    # Stateful Lazy API
    # ─────────────────────────────────────────────────────────────────────────

    def bind(self, sequence: SequenceArray) -> "PolymerModel":
        """
        Bind the model to a specific sequence for stateful lazy computation.

        After binding, you can use the `coordinates` and `latents` properties
        for lazy get/set access with automatic cache invalidation.

        Args:
            sequence: Int array of residue types (e.g., polymer.sequence).

        Returns:
            Self for method chaining.

        Example:
            >>> polymer_model.bind(polymer.sequence).coordinates = coords
            >>> z = polymer_model.latents  # Lazily computed
        """
        sequence = _to_numpy_int64(sequence)
        self._validate_sequence(sequence)

        self._sequence = sequence
        self._cached_latents = None
        self._cached_coordinates = None
        self._latents_dirty = True
        self._coords_dirty = True
        return self

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

    @property
    def sequence(self) -> np.ndarray | None:
        """The currently bound sequence (int array), or None if unbound."""
        return self._sequence

    @property
    def is_bound(self) -> bool:
        """Whether a sequence is currently bound."""
        return self._sequence is not None

    def _ensure_bound(self) -> None:
        """Raise if no sequence is bound."""
        if not self.is_bound:
            raise RuntimeError(
                "No sequence bound. Call bind(sequence) first, or use "
                "the stateless encode()/decode() methods."
            )

    @property
    def latents(self) -> torch.Tensor:
        """
        (n_residues, latent_dim) latent representation.

        Lazily recomputed from coordinates when dirty.

        Raises:
            RuntimeError: If no sequence is bound.
            RuntimeError: If latents are dirty and no coordinates are set.
        """
        self._ensure_bound()

        if self._latents_dirty:
            if self._cached_coordinates is None:
                raise RuntimeError(
                    "Cannot compute latents: no coordinates set. "
                    "Set coordinates first via the coordinates property."
                )
            self._cached_latents = self.encode(self._cached_coordinates, self._sequence)
            self._latents_dirty = False

        return self._cached_latents

    @latents.setter
    def latents(self, value: torch.Tensor) -> None:
        """
        Set latent representation, marks coordinates as dirty.

        Args:
            value: (n_residues, latent_dim) latent vectors.
        """
        self._ensure_bound()

        if value.shape[0] != len(self._sequence):
            raise ValueError(
                f"latents has {value.shape[0]} rows but sequence has "
                f"{len(self._sequence)} residues"
            )
        if value.shape[1] != self.latent_dim:
            raise ValueError(
                f"latents has dim {value.shape[1]} but model expects {self.latent_dim}"
            )

        self._cached_latents = value
        self._latents_dirty = False
        self._coords_dirty = True  # Coordinates need recomputation

    @property
    def coordinates(self) -> torch.Tensor:
        """
        (N, 3) Cartesian coordinates.

        Lazily recomputed from latents when dirty.

        Raises:
            RuntimeError: If no sequence is bound.
            RuntimeError: If coordinates are dirty and no latents are set.
        """
        self._ensure_bound()

        if self._coords_dirty:
            if self._cached_latents is None:
                raise RuntimeError(
                    "Cannot compute coordinates: no latents set. "
                    "Set latents first via the latents property."
                )
            self._cached_coordinates = self.decode(self._cached_latents, self._sequence)
            self._coords_dirty = False

        return self._cached_coordinates

    @coordinates.setter
    def coordinates(self, value: torch.Tensor) -> None:
        """
        Set Cartesian coordinates, marks latents as dirty.

        Args:
            value: (N, 3) coordinate array.
        """
        self._ensure_bound()
        self._validate_coords_shape(value, self._sequence)

        self._cached_coordinates = value
        self._coords_dirty = False
        self._latents_dirty = True  # Latents need recomputation

    def unbind(self) -> None:
        """
        Unbind from current sequence and clear cached state.

        After unbinding, the stateful properties will raise RuntimeError
        until bind() is called again.
        """
        self._sequence = None
        self._cached_latents = None
        self._cached_coordinates = None
        self._latents_dirty = True
        self._coords_dirty = True

    # ─────────────────────────────────────────────────────────────────────────
    # Stateless API
    # ─────────────────────────────────────────────────────────────────────────

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
        coords: "torch.Tensor | np.ndarray",
        sequence: SequenceArray,
    ) -> torch.Tensor:
        """
        Encode polymer coordinates to per-residue latent vectors.

        Handles alignment automatically for frame-dependent models (ResidueFlowModel,
        ResidueVAE). Invariant models (InvariantResidueVAE) don't require alignment.

        Accepts both NumPy arrays and PyTorch tensors. NumPy arrays are
        converted to tensors internally.

        Args:
            coords: (N, 3) flat coordinate array for all atoms (NumPy or Tensor).
            sequence: Int array of residue types (e.g., polymer.sequence).

        Returns:
            (n_residues, latent_dim) latent vectors (always Tensor).
        """
        # Convert sequence to numpy
        sequence = _to_numpy_int64(sequence)
        self._validate_sequence(sequence)

        # Convert numpy coords to tensor if needed
        if isinstance(coords, np.ndarray):
            coords = torch.from_numpy(coords).float()

        self._validate_coords_shape(coords, sequence)

        latents = []
        offset = 0

        for res_type in sequence:
            model = self._get_model(int(res_type))
            n_atoms = model.n_atoms

            # Extract this residue's coordinates
            res_coords = coords[offset:offset + n_atoms]

            # Reshape to (1, n_atoms, 3) for alignment and model.encode()
            res_coords = res_coords.unsqueeze(0)

            # Align if model has frame indices (frame-dependent model)
            frame_indices = model.frame_indices
            if frame_indices is not None:
                res_coords = align_to_frame(res_coords, frame_indices)

            # Encode (models expect aligned input)
            z = model.encode(res_coords)  # (1, k)
            latents.append(z.squeeze(0))  # (k,)

            offset += n_atoms

        return torch.stack(latents)  # (n_residues, k)

    def decode(
        self,
        latents: torch.Tensor,
        sequence: SequenceArray,
        latent_bound: float | None = 5.0,
    ) -> torch.Tensor:
        """
        Decode latent vectors to positioned polymer coordinates.

        Returns a PyTorch tensor. Call .numpy() if NumPy array is needed.

        Args:
            latents: (n_residues, latent_dim) latent vectors.
            sequence: Int array of residue types (e.g., polymer.sequence).
            latent_bound: Soft bound for latent values using tanh. Values are
                bounded to [-bound, bound] range. Set to None to disable.
                Default 5.0 prevents gradient explosion from out-of-distribution
                latents during optimization.

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

        # Decode and position residues iteratively using frame-based positioning
        # Each residue is positioned relative to the previous using the transform
        # from the previous residue (transform[i-1] positions residue[i])
        positioned_coords = []
        prev_coords = None
        prev_transform = None
        prev_indices = None

        for i, res_type in enumerate(sequence):
            model = self._get_model(int(res_type))

            # Decode this residue (keep gradients for optimization)
            coords_i, transform_i = model.decode(latents[i:i + 1])

            # coords_i is (1, n_atoms, 3), squeeze to (n_atoms, 3)
            # transform_i is (1, 6), squeeze to (6,)
            coords_i = coords_i.squeeze(0)
            transform_i = transform_i.squeeze(0)

            if i == 0:
                # First residue stays at origin
                positioned_coords.append(coords_i)
            elif prev_indices is None:
                # No frame indices available (e.g., test models with partial atoms)
                # Fall back to simple concatenation without positioning
                positioned_coords.append(coords_i)
            else:
                # Position this residue using previous residue's transform
                # This aligns current residue's P frame to target derived from
                # previous residue's O3' frame + transform
                positioned = position_next_residue(
                    prev_coords, coords_i, prev_transform, prev_indices
                )
                positioned_coords.append(positioned)

            # Store for next iteration
            prev_coords = positioned_coords[-1]
            prev_transform = transform_i
            prev_indices = model.frame_indices

        # Concatenate all positioned coordinates
        return torch.cat(positioned_coords, dim=0)

    def _sample_coords(
        self,
        sequence: SequenceArray,
        n_samples: int = 1,
        temperature: float = 1.0,
    ) -> list[torch.Tensor]:
        """
        Sample coordinate tensors from sequence (internal method).

        Args:
            sequence: Int array of residue types.
            n_samples: Number of samples to generate.
            temperature: Scales latent noise (higher = more diverse).

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
            coords = self.decode(latents, sequence)
            samples.append(coords)

        return samples

    def sample_from_sequence(
        self,
        sequence: str,
        n_samples: int = 1,
        temperature: float = 1.0,
        id: str = "sampled",
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

        Returns:
            If n_samples=1: Single Polymer with generated coordinates.
            If n_samples>1: List of Polymers.

        Example:
            >>> model = PolymerModel.load("path/to/model")
            >>> polymer = model.sample_from_sequence("acgu")
            >>> polymer.write("sampled.cif")
            >>>
            >>> # Generate multiple samples
            >>> samples = model.sample_from_sequence("acgu", n_samples=10)
            >>> for i, p in enumerate(samples):
            ...     p.write(f"sample_{i}.cif")
        """
        from ciffy import from_sequence

        # Create template with correct atoms for this model
        template = from_sequence(sequence, atoms=self.atom_filter, id=id)

        # Use protocol-compliant sample method
        samples = self.sample(template, n_samples=n_samples, temperature=temperature)

        if n_samples == 1:
            return samples[0]
        return samples

    def sample(
        self,
        template: "Polymer",
        n_samples: int = 1,
        temperature: float = 1.0,
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
        from ciffy import from_sequence as _from_sequence
        output_template = _from_sequence(
            template.sequence_str(),
            atoms=self.atom_filter,
            id=template.pdb_id or "sampled",
        )

        # Sample coordinates
        coords_list = self._sample_coords(sequence, n_samples, temperature)

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
        Get atom filter dict for use with ciffy.from_sequence(atoms=...).

        Returns a dict mapping residue type (int) to the list of atom values
        that this model uses. Pass this to from_sequence() to create templates
        with only the atoms the model knows about.

        Example:
            >>> template = ciffy.from_sequence("acgu", atoms=polymer_model.atom_filter)
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
        elif model_type == "residue-vae":
            from ciffy.nn.vae.residue import ResidueVAE
            return ResidueVAE.load(path, device=device)
        elif model_type == "residue-invariant-vae":
            from ciffy.nn.vae.residue import InvariantResidueVAE
            return InvariantResidueVAE.load(path, device=device)
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

        This is the recommended way to encode polymers, as it uses the
        polymer's sequence directly without conversion. Automatically filters
        to only include the atoms the model was trained on.

        Args:
            polymer: Polymer object to encode.

        Returns:
            (n_residues, latent_dim) latent vectors.

        Raises:
            ValueError: If the polymer contains unsupported residue types.

        Example:
            >>> polymer = ciffy.load("structure.cif").poly()
            >>> latents = model.encode_polymer(polymer)
        """
        from ciffy.biochemistry import Scale
        from ciffy.operations.reduction import Reduction
        from ciffy.backend import to_numpy

        # Get per-residue data
        sequence = polymer.sequence
        seq_np = to_numpy(sequence)
        n_residues = len(seq_np)

        per_res_atoms = polymer.reduce(polymer.atoms, Scale.RESIDUE, Reduction.COLLATE)
        per_res_coords = polymer.reduce(polymer.coordinates, Scale.RESIDUE, Reduction.COLLATE)

        # Extract and filter coordinates per residue
        filtered_coords = []
        for i in range(n_residues):
            res_type = int(seq_np[i])
            if res_type not in self._supported_types_set:
                from ciffy.biochemistry import Residue
                res_name = Residue.from_index(res_type).name
                raise ValueError(f"Unsupported residue type: {res_name}")

            # Get expected atoms for this residue type
            expected_atoms = self.atom_filter[res_type]
            atoms_i = to_numpy(per_res_atoms[i])
            coords_i = to_numpy(per_res_coords[i])

            # Build mapping from atom value to coordinate
            atom_to_coord = {int(a): c for a, c in zip(atoms_i, coords_i)}

            # Extract coordinates in expected order
            res_coords = []
            for atom_val in expected_atoms:
                if atom_val not in atom_to_coord:
                    from ciffy.biochemistry import Residue
                    res = Residue.from_index(res_type)
                    raise ValueError(f"Missing atom {atom_val} in residue {i} ({res.name})")
                res_coords.append(atom_to_coord[atom_val])

            filtered_coords.extend(res_coords)

        # Convert to tensor
        coords = torch.tensor(filtered_coords, dtype=torch.float32)

        return self.encode(coords, sequence)

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
