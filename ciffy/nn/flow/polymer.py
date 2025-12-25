"""
PolymerFlowModel: Orchestrates per-residue flows for full polymer encoding/decoding.

This module provides a wrapper around ResidueFlowModel that handles:
- Partitioning flat coordinate arrays by residue
- Encoding each residue with its appropriate model
- Decoding and positioning residues using SE(3) transforms
- Lazy computation with dirty-flag caching for efficient coordinate/latent access

Example (stateless API):
    >>> from ciffy.nn.flow import PolymerFlowModel, ResidueFlowModel
    >>> from ciffy.biochemistry import Residue
    >>> import numpy as np
    >>>
    >>> # Load pre-trained per-residue models
    >>> models = {
    ...     Residue.A: ResidueFlowModel.load("models/A"),
    ...     Residue.G: ResidueFlowModel.load("models/G"),
    ... }
    >>> polymer_flow = PolymerFlowModel.from_residue_models(models)
    >>>
    >>> # Encode polymer coordinates (sequence as int array)
    >>> sequence = np.array([Residue.A.value, Residue.G.value, Residue.A.value])
    >>> latents = polymer_flow.encode(coords, sequence)  # (3, k)
    >>>
    >>> # Decode back to positioned coordinates
    >>> coords_recon = polymer_flow.decode(latents, sequence)  # (N, 3)

Example (with Polymer objects - recommended):
    >>> polymer = ciffy.load("structure.cif").poly()
    >>> latents = polymer_flow.encode_polymer(polymer)  # Uses polymer.sequence directly
    >>> new_polymer = polymer_flow.decode_to_polymer(latents, polymer)

Example (stateful lazy API):
    >>> # Bind to a sequence for lazy computation
    >>> polymer_flow.bind(polymer.sequence)
    >>>
    >>> # Set coordinates - latents computed lazily
    >>> polymer_flow.coordinates = coords
    >>> z = polymer_flow.latents  # Computed on first access
    >>>
    >>> # Modify latents - coordinates recomputed lazily
    >>> polymer_flow.latents = modified_z
    >>> new_coords = polymer_flow.coordinates  # Recomputed on access
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Union

import numpy as np
import torch

# Import geometry functions directly (data.py wrappers still available for backward compat)
from ciffy.geometry import position_residue_fast

# Keep imports from data.py for functions that accept atoms array (ML-friendly signature)
from ciffy.nn.flow.residue.data import (
    position_next_residue,
    position_next_residue_torch,
)

if TYPE_CHECKING:
    from ciffy.biochemistry import Residue
    from ciffy.nn.flow.residue import ResidueFlowModel

# Type alias for sequence input (numpy array or torch tensor of int)
SequenceArray = Union[np.ndarray, torch.Tensor]


def _to_numpy_int64(sequence: SequenceArray) -> np.ndarray:
    """Convert sequence to numpy int64 array."""
    if isinstance(sequence, torch.Tensor):
        return sequence.cpu().numpy().astype(np.int64)
    return np.asarray(sequence, dtype=np.int64)


class PolymerFlowModel:
    """
    Orchestrates per-residue flow models for full polymer encoding/decoding.

    This class wraps multiple ResidueFlowModel instances (one per residue type)
    and provides encode/decode methods that work on entire polymers.

    The encode method partitions a flat (N, 3) coordinate array by residue
    and encodes each residue independently. The decode method reconstructs
    coordinates and chains them together using the SE(3) transforms that
    each ResidueFlowModel outputs.

    Supports two usage patterns:

    1. **Stateless API**: Pass coordinates/latents and sequence to encode()/decode().
       Each call is independent with no state. Sequence is an int array.

    2. **Stateful Lazy API**: Call bind(sequence), then use coordinates/latents
       properties. Values are cached and lazily recomputed only when needed.

    Attributes:
        latent_dim: Dimension of per-residue latent space.
        residue_models: Dict mapping residue type (int) to ResidueFlowModel.

    Example (with Polymer - recommended):
        >>> polymer_flow = PolymerFlowModel.from_residue_models(models)
        >>> latents = polymer_flow.encode_polymer(polymer)
        >>> new_polymer = polymer_flow.decode_to_polymer(latents, polymer)

    Example (stateless with int array):
        >>> seq = np.array([Residue.A.value, Residue.G.value])
        >>> latents = polymer_flow.encode(coords, seq)
        >>> coords_recon = polymer_flow.decode(latents, seq)

    Example (stateful):
        >>> polymer_flow.bind(polymer.sequence)
        >>> polymer_flow.coordinates = coords
        >>> z = polymer_flow.latents  # Lazily computed
    """

    def __init__(self, residue_models: dict[int, "ResidueFlowModel"]):
        """
        Initialize with pre-trained per-residue models.

        Args:
            residue_models: Dict mapping residue type (int) to ResidueFlowModel.
                            All models should have the same latent_dim.

        Note:
            For convenience, use from_residue_models() to create from a dict
            keyed by Residue enum instead of int.
        """
        if not residue_models:
            raise ValueError("residue_models cannot be empty")

        self.residue_models = residue_models

        # Validate all models have same latent dim
        latent_dims = [m.latent_dim for m in residue_models.values()]
        if len(set(latent_dims)) > 1:
            raise ValueError(
                f"All ResidueFlowModels must have same latent_dim, got {latent_dims}"
            )
        self.latent_dim = latent_dims[0]

        # Cache atom counts and supported types for fast validation
        self._atom_counts: dict[int, int] = {
            res_type: model.n_atoms for res_type, model in residue_models.items()
        }
        self._supported_types_set: set[int] = set(residue_models.keys())

        # Stateful lazy computation attributes
        self._sequence: np.ndarray | None = None
        self._cached_latents: torch.Tensor | None = None
        self._cached_coordinates: torch.Tensor | None = None
        self._latents_dirty: bool = True
        self._coords_dirty: bool = True

    @classmethod
    def from_residue_models(
        cls,
        models: dict["Residue", "ResidueFlowModel"],
    ) -> "PolymerFlowModel":
        """
        Create from a dict keyed by Residue enum (convenience constructor).

        This is the recommended way to create a PolymerFlowModel from
        individually trained ResidueFlowModels.

        Args:
            models: Dict mapping Residue enum to ResidueFlowModel.

        Returns:
            New PolymerFlowModel instance.

        Example:
            >>> models = {
            ...     Residue.A: ResidueFlowModel.load("models/A"),
            ...     Residue.G: ResidueFlowModel.load("models/G"),
            ... }
            >>> polymer_flow = PolymerFlowModel.from_residue_models(models)
        """
        return cls({r.value: m for r, m in models.items()})

    # ─────────────────────────────────────────────────────────────────────────
    # Stateful Lazy API
    # ─────────────────────────────────────────────────────────────────────────

    def bind(self, sequence: SequenceArray) -> "PolymerFlowModel":
        """
        Bind the model to a specific sequence for stateful lazy computation.

        After binding, you can use the `coordinates` and `latents` properties
        for lazy get/set access with automatic cache invalidation.

        Args:
            sequence: Int array of residue types (e.g., polymer.sequence).

        Returns:
            Self for method chaining.

        Example:
            >>> polymer_flow.bind(polymer.sequence).coordinates = coords
            >>> z = polymer_flow.latents  # Lazily computed
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
            model = self.residue_models[int(res_type)]
            n_atoms = model.n_atoms

            # Extract this residue's coordinates
            res_coords = coords[offset:offset + n_atoms]

            # Reshape to (1, n_atoms, 3) for model.encode()
            res_coords = res_coords.unsqueeze(0)

            # Encode (transforms=None uses zeros, which is fine for encoding)
            z = model.encode(res_coords)  # (1, k)
            latents.append(z.squeeze(0))  # (k,)

            offset += n_atoms

        return torch.stack(latents)  # (n_residues, k)

    def decode(
        self,
        latents: torch.Tensor,
        sequence: SequenceArray,
    ) -> torch.Tensor:
        """
        Decode latent vectors to positioned polymer coordinates.

        Returns a PyTorch tensor. Call .numpy() if NumPy array is needed.

        Args:
            latents: (n_residues, latent_dim) latent vectors.
            sequence: Int array of residue types (e.g., polymer.sequence).

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

        all_coords = []
        prev_coords = None
        prev_transform = None
        prev_model = None

        for i, res_type in enumerate(sequence):
            model = self.residue_models[int(res_type)]

            # Decode this residue
            with torch.no_grad():
                coords_i, transform_i = model.decode(latents[i:i + 1])

            # coords_i is (1, n_atoms, 3), squeeze to (n_atoms, 3)
            coords_i = coords_i.squeeze(0)
            # transform_i is (1, 6), squeeze to (6,)
            transform_i = transform_i.squeeze(0)

            if i == 0:
                # First residue: place at origin (already in canonical frame)
                positioned = coords_i
            else:
                # Position relative to previous residue using its transform
                # Use fast path with pre-resolved frame indices
                positioned = position_residue_fast(
                    prev_coords,
                    coords_i,
                    prev_transform,
                    prev_model.prev_frame_cols,
                    prev_model.prev_z_toward_origin,
                    model.next_frame_cols,
                    model.next_z_toward_origin,
                )

            all_coords.append(positioned)

            # Store for next iteration
            prev_coords = positioned
            prev_transform = transform_i
            prev_model = model

        return torch.cat(all_coords, dim=0)  # (N, 3)

    def sample(
        self,
        sequence: SequenceArray,
        n_samples: int = 1,
    ) -> torch.Tensor | list[torch.Tensor]:
        """
        Sample new polymer conformations.

        Args:
            sequence: Int array of residue types (e.g., polymer.sequence).
            n_samples: Number of samples to generate.

        Returns:
            If n_samples=1: (N, 3) coordinate array.
            If n_samples>1: List of (N, 3) coordinate arrays.
        """
        sequence = _to_numpy_int64(sequence)

        if len(sequence) == 0:
            if n_samples == 1:
                return torch.empty(0, 3)
            return [torch.empty(0, 3) for _ in range(n_samples)]

        # Get device from first model
        device = next(iter(self.residue_models.values())).device

        samples = []
        for _ in range(n_samples):
            # Sample random latents from standard normal
            latents = torch.randn(len(sequence), self.latent_dim, device=device)
            coords = self.decode(latents, sequence)
            samples.append(coords)

        if n_samples == 1:
            return samples[0]
        return samples

    @property
    def supported_residue_types(self) -> np.ndarray:
        """Array of supported residue type indices (int)."""
        return np.array(sorted(self.residue_models.keys()), dtype=np.int64)

    @property
    def supported_residues(self) -> list["AtomGroup"]:
        """List of residue types this model can handle (as AtomGroup)."""
        from ciffy.biochemistry import Residue
        return [Residue.from_index(v) for v in sorted(self.residue_models.keys())]

    @property
    def atom_filter(self) -> dict[int, list[int]]:
        """
        Get atom filter dict for use with ciffy.from_sequence(atoms=...).

        Returns a dict mapping residue type (int) to the list of atom values
        that this model uses. Pass this to from_sequence() to create templates
        with only the atoms the flow model knows about.

        Example:
            >>> template = ciffy.from_sequence("acgu", atoms=polymer_model.atom_filter)
            >>> # template now has only the atoms used by the flow models
        """
        return {
            res_type: list(model._atom_indices)
            for res_type, model in self.residue_models.items()
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Device Management
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def device(self) -> "torch.device":
        """Device where model parameters reside."""
        return next(iter(self.residue_models.values())).device

    def to(self, device: str | "torch.device") -> "PolymerFlowModel":
        """
        Move all residue models to specified device.

        Args:
            device: Target device (e.g., "cpu", "cuda", "cuda:0").

        Returns:
            Self for method chaining.
        """
        for model in self.residue_models.values():
            model.to(device)
        return self

    def cuda(self, device_id: int = 0) -> "PolymerFlowModel":
        """Move all residue models to CUDA device."""
        return self.to(f"cuda:{device_id}")

    def cpu(self) -> "PolymerFlowModel":
        """Move all residue models to CPU."""
        return self.to("cpu")

    # ─────────────────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """
        Save model to directory.

        Each ResidueFlowModel is saved to a subdirectory named by residue.

        Args:
            path: Directory to save to.
        """
        import json
        from ciffy.biochemistry import Residue

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save each residue model (use residue name as subdirectory)
        residue_names = []
        for res_type, model in self.residue_models.items():
            res_name = Residue.from_index(res_type).name
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
    ) -> "PolymerFlowModel":
        """
        Load model from directory.

        Args:
            path: Directory containing saved model.
            device: Device to load models to.
            jit: Whether to JIT-compile the decoders.

        Returns:
            Loaded PolymerFlowModel.
        """
        import json
        from ciffy.biochemistry import Residue
        from ciffy.nn.flow.residue import ResidueFlowModel

        path = Path(path)

        with open(path / "config.json") as f:
            config = json.load(f)

        # Load models with int keys
        residue_models = {}
        for res_name in config["residue_types"]:
            res_type = getattr(Residue, res_name).value
            residue_models[res_type] = ResidueFlowModel.load(
                path / res_name,
                device=device,
                jit=jit,
            )

        return cls(residue_models)

    # ─────────────────────────────────────────────────────────────────────────
    # Convenience Methods for Polymer Objects
    # ─────────────────────────────────────────────────────────────────────────

    def encode_polymer(self, polymer: "Polymer") -> torch.Tensor:
        """
        Encode a Polymer object directly to latent vectors.

        This is the recommended way to encode polymers, as it uses the
        polymer's sequence directly without conversion.

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
        # Convert coordinates to tensor
        coords = polymer.coordinates
        if not isinstance(coords, torch.Tensor):
            if isinstance(coords, np.ndarray):
                coords = torch.from_numpy(coords).float()
            else:
                coords = torch.tensor(coords, dtype=torch.float32)

        # Use polymer.sequence directly (already int array)
        return self.encode(coords, polymer.sequence)

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

        return template.with_coordinates(coords_np)

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

    def __repr__(self) -> str:
        from ciffy.biochemistry import Residue
        residues = [Residue.from_index(r).name for r in self.residue_models.keys()]
        return f"PolymerFlowModel(residues={residues}, latent_dim={self.latent_dim})"
