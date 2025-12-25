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
    >>>
    >>> # Load pre-trained per-residue models
    >>> models = {
    ...     Residue.A: ResidueFlowModel.load("models/A"),
    ...     Residue.G: ResidueFlowModel.load("models/G"),
    ... }
    >>> polymer_flow = PolymerFlowModel(models)
    >>>
    >>> # Encode polymer coordinates
    >>> sequence = [Residue.A, Residue.G, Residue.A]
    >>> latents = polymer_flow.encode(coords, sequence)  # (3, k)
    >>>
    >>> # Decode back to positioned coordinates
    >>> coords_recon = polymer_flow.decode(latents, sequence)  # (N, 3)

Example (stateful lazy API):
    >>> # Bind to a sequence for lazy computation
    >>> polymer_flow.bind(sequence)
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
from typing import TYPE_CHECKING

import torch

from ciffy.nn.flow.residue.data import (
    position_next_residue,
    position_next_residue_torch,
    position_next_residue_fast,
)

if TYPE_CHECKING:
    from ciffy.biochemistry import Residue
    from ciffy.nn.flow.residue import ResidueFlowModel


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
       Each call is independent with no state.

    2. **Stateful Lazy API**: Call bind(sequence), then use coordinates/latents
       properties. Values are cached and lazily recomputed only when needed.

    Attributes:
        residue_models: Dict mapping Residue type to trained ResidueFlowModel.
        latent_dim: Dimension of per-residue latent space (assumes all models match).

    Example (stateless):
        >>> polymer = PolymerFlowModel(models)
        >>> latents = polymer.encode(coords, sequence)
        >>> coords_recon = polymer.decode(latents, sequence)

    Example (stateful):
        >>> polymer = PolymerFlowModel(models)
        >>> polymer.bind(sequence)
        >>> polymer.coordinates = coords
        >>> z = polymer.latents  # Lazily computed
    """

    def __init__(self, residue_models: dict["Residue", "ResidueFlowModel"]):
        """
        Initialize with pre-trained per-residue models.

        Args:
            residue_models: Dict mapping Residue enum to ResidueFlowModel.
                            All models should have the same latent_dim.
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

        # Stateful lazy computation attributes
        self._sequence: list["Residue"] | None = None
        self._cached_latents: torch.Tensor | None = None
        self._cached_coordinates: torch.Tensor | None = None
        self._latents_dirty: bool = True
        self._coords_dirty: bool = True

    # ─────────────────────────────────────────────────────────────────────────
    # Stateful Lazy API
    # ─────────────────────────────────────────────────────────────────────────

    def bind(self, sequence: list["Residue"]) -> "PolymerFlowModel":
        """
        Bind the model to a specific sequence for stateful lazy computation.

        After binding, you can use the `coordinates` and `latents` properties
        for lazy get/set access with automatic cache invalidation.

        Args:
            sequence: List of Residue types defining the polymer.

        Returns:
            Self for method chaining.

        Example:
            >>> polymer_flow.bind(sequence).coordinates = coords
            >>> z = polymer_flow.latents  # Lazily computed
        """
        # Validate all residue types are supported
        for res_type in sequence:
            if res_type not in self.residue_models:
                available = list(self.residue_models.keys())
                raise ValueError(
                    f"No model for residue type {res_type}. "
                    f"Available: {[r.name for r in available]}"
                )

        self._sequence = sequence
        self._cached_latents = None
        self._cached_coordinates = None
        self._latents_dirty = True
        self._coords_dirty = True
        return self

    @property
    def sequence(self) -> list["Residue"] | None:
        """The currently bound sequence, or None if unbound."""
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
    # Stateless API (original methods)
    # ─────────────────────────────────────────────────────────────────────────

    def _get_atom_counts(self, sequence: list["Residue"]) -> list[int]:
        """Get atom counts for each residue in sequence."""
        counts = []
        for res_type in sequence:
            if res_type not in self.residue_models:
                available = list(self.residue_models.keys())
                raise ValueError(
                    f"No model for residue type {res_type}. "
                    f"Available: {[r.name for r in available]}"
                )
            counts.append(self.residue_models[res_type].n_atoms)
        return counts

    def _validate_coords_shape(
        self,
        coords: torch.Tensor,
        sequence: list["Residue"],
    ) -> None:
        """Validate that coords shape matches sequence."""
        expected_atoms = sum(self._get_atom_counts(sequence))
        if coords.shape[0] != expected_atoms:
            raise ValueError(
                f"coords has {coords.shape[0]} atoms but sequence expects {expected_atoms}"
            )
        if coords.shape[1] != 3:
            raise ValueError(f"coords must be (N, 3), got shape {coords.shape}")

    def encode(
        self,
        coords: "torch.Tensor | np.ndarray",
        sequence: list["Residue"],
    ) -> torch.Tensor:
        """
        Encode polymer coordinates to per-residue latent vectors.

        Accepts both NumPy arrays and PyTorch tensors. NumPy arrays are
        converted to tensors internally.

        Args:
            coords: (N, 3) flat coordinate array for all atoms (NumPy or Tensor).
            sequence: List of Residue types, one per residue.

        Returns:
            (n_residues, latent_dim) latent vectors (always Tensor).
        """
        import numpy as np

        # Convert numpy to tensor if needed
        if isinstance(coords, np.ndarray):
            coords = torch.from_numpy(coords).float()

        self._validate_coords_shape(coords, sequence)

        latents = []
        offset = 0

        for res_type in sequence:
            model = self.residue_models[res_type]
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
        sequence: list["Residue"],
    ) -> torch.Tensor:
        """
        Decode latent vectors to positioned polymer coordinates.

        Returns a PyTorch tensor. Call .numpy() if NumPy array is needed.

        Args:
            latents: (n_residues, latent_dim) latent vectors.
            sequence: List of Residue types, one per residue.

        Returns:
            (N, 3) flat coordinate tensor with all residues positioned.
        """
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
            model = self.residue_models[res_type]

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
                positioned = position_next_residue_fast(
                    prev_coords,
                    coords_i,
                    prev_transform,
                    prev_model._prev_frame_cols,
                    prev_model._prev_z_toward_origin,
                    model._next_frame_cols,
                    model._next_z_toward_origin,
                )

            all_coords.append(positioned)

            # Store for next iteration
            prev_coords = positioned
            prev_transform = transform_i
            prev_model = model

        return torch.cat(all_coords, dim=0)  # (N, 3)

    def sample(
        self,
        sequence: list["Residue"],
        n_samples: int = 1,
    ) -> torch.Tensor | list[torch.Tensor]:
        """
        Sample new polymer conformations.

        Args:
            sequence: List of Residue types defining the polymer.
            n_samples: Number of samples to generate.

        Returns:
            If n_samples=1: (N, 3) coordinate array.
            If n_samples>1: List of (N, 3) coordinate arrays.
        """
        if len(sequence) == 0:
            if n_samples == 1:
                return torch.empty(0, 3)
            return [torch.empty(0, 3) for _ in range(n_samples)]

        # Get device from first model
        device = next(iter(self.residue_models.values())).flow.V.device

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
    def supported_residues(self) -> list["Residue"]:
        """List of residue types this model can handle."""
        return list(self.residue_models.keys())

    def save(self, path: str | Path) -> None:
        """
        Save model to directory.

        Each ResidueFlowModel is saved to a subdirectory named by residue.

        Args:
            path: Directory to save to.
        """
        import json

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save each residue model
        for res_type, model in self.residue_models.items():
            model.save(path / res_type.name)

        # Save metadata
        config = {
            "residue_types": [r.name for r in self.residue_models.keys()],
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

        residue_models = {}
        for res_name in config["residue_types"]:
            res_type = getattr(Residue, res_name)
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

        This is a convenience method that extracts coordinates and sequence
        from a Polymer object and encodes them.

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
        from ciffy.biochemistry import Residue

        # Convert coordinates to tensor
        coords = polymer.coordinates
        if not isinstance(coords, torch.Tensor):
            import numpy as np
            if isinstance(coords, np.ndarray):
                coords = torch.from_numpy(coords).float()
            else:
                coords = torch.tensor(coords, dtype=torch.float32)

        # Extract sequence as Residue list
        sequence = [Residue(int(idx)) for idx in polymer.sequence]

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
        from ciffy.biochemistry import Residue

        sequence = [Residue(int(idx)) for idx in template.sequence]
        coords = self.decode(latents, sequence)

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
        residues = [r.name for r in self.residue_models.keys()]
        return f"PolymerFlowModel(residues={residues}, latent_dim={self.latent_dim})"
