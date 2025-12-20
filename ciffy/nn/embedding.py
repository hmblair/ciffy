"""
Learnable embeddings for polymer features.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..polymer import Polymer

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = None  # Placeholder

from ..types import Scale
from ..biochemistry import NUM_ATOMS, NUM_RESIDUES, NUM_ELEMENTS


class PolymerEmbedding(nn.Module if TORCH_AVAILABLE else object):
    """
    Learnable embeddings for polymer features.

    Creates embeddings for atom types, residue types, and/or element types,
    then concatenates them. The output scale determines which embedding
    types are valid.

    Example:
        >>> from ciffy.nn import PolymerEmbedding
        >>> from ciffy import Scale
        >>> embed = PolymerEmbedding(
        ...     scale=Scale.ATOM,
        ...     atom_dim=64,
        ...     residue_dim=32,
        ...     element_dim=16,
        ... )
        >>> features = embed(polymer)  # (num_atoms, 112)
    """

    def __init__(
        self,
        scale: Scale,
        atom_dim: int | None = None,
        residue_dim: int | None = None,
        element_dim: int | None = None,
    ):
        """
        Initialize embedding layers.

        Args:
            scale: Output scale. Determines valid embedding types:
                - Scale.ATOM: atom, residue (expanded), element all valid
                - Scale.RESIDUE: only residue valid
            atom_dim: Embedding dimension for atom types (requires scale=ATOM).
            residue_dim: Embedding dimension for residue types.
            element_dim: Embedding dimension for element types (requires scale=ATOM).

        Raises:
            ImportError: If PyTorch is not installed.
            ValueError: If scale is not ATOM or RESIDUE.
            ValueError: If atom_dim or element_dim specified with scale=RESIDUE.
            ValueError: If no embedding dimensions are specified.
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for PolymerEmbedding. "
                "Install with: pip install torch"
            )

        super().__init__()

        if scale not in (Scale.ATOM, Scale.RESIDUE):
            raise ValueError(
                f"scale must be ATOM or RESIDUE, got {scale.name}"
            )

        if scale == Scale.RESIDUE:
            if atom_dim is not None:
                raise ValueError(
                    "atom_dim cannot be used with scale=RESIDUE "
                    "(atom indices are per-atom, not per-residue)"
                )
            if element_dim is not None:
                raise ValueError(
                    "element_dim cannot be used with scale=RESIDUE "
                    "(element indices are per-atom, not per-residue)"
                )

        if atom_dim is None and residue_dim is None and element_dim is None:
            raise ValueError(
                "At least one embedding dimension must be specified"
            )

        self.scale = scale
        self.atom_dim = atom_dim
        self.residue_dim = residue_dim
        self.element_dim = element_dim

        # Create embedding layers
        if atom_dim is not None:
            self.atom_embedding = nn.Embedding(NUM_ATOMS, atom_dim)
        else:
            self.atom_embedding = None

        if residue_dim is not None:
            self.residue_embedding = nn.Embedding(NUM_RESIDUES, residue_dim)
        else:
            self.residue_embedding = None

        if element_dim is not None:
            self.element_embedding = nn.Embedding(NUM_ELEMENTS, element_dim)
        else:
            self.element_embedding = None

    @property
    def output_dim(self) -> int:
        """Total output dimension (sum of enabled embedding dimensions)."""
        dim = 0
        if self.atom_dim is not None:
            dim += self.atom_dim
        if self.residue_dim is not None:
            dim += self.residue_dim
        if self.element_dim is not None:
            dim += self.element_dim
        return dim

    def _validate_indices(
        self,
        indices: torch.Tensor,
        max_idx: int,
        name: str,
    ) -> torch.Tensor:
        """
        Validate and clamp embedding indices.

        Args:
            indices: Tensor of indices to validate.
            max_idx: Maximum valid index (vocabulary size).
            name: Name of the index type for error messages.

        Returns:
            Clamped indices tensor (invalid indices mapped to 0).

        Raises:
            IndexError: If any indices exceed the vocabulary size.
        """
        # Check for out-of-bounds indices
        invalid_mask = indices >= max_idx
        if invalid_mask.any():
            invalid_indices = indices[invalid_mask].unique().tolist()
            invalid_count = invalid_mask.sum().item()
            raise IndexError(
                f"PolymerEmbedding: {invalid_count} {name} indices out of bounds. "
                f"Valid range: [0, {max_idx}), got values: {invalid_indices[:10]}"
                f"{'...' if len(invalid_indices) > 10 else ''}. "
                f"This may indicate corrupted data or unsupported atom/residue types."
            )

        # Clamp -1 (unknown) to 0
        return indices.clamp(min=0)

    def forward(self, polymer: Polymer) -> torch.Tensor:
        """
        Embed polymer features and concatenate.

        Args:
            polymer: Polymer object (must be torch backend).

        Returns:
            Tensor of shape (N, total_dim) where:
            - N = num_atoms if scale=ATOM
            - N = num_residues if scale=RESIDUE

        Raises:
            IndexError: If any indices exceed the vocabulary size.

        Note:
            Unknown indices (-1) are mapped to index 0 (unknown/padding).
        """
        embeddings = []

        if self.atom_embedding is not None:
            atom_idx = self._validate_indices(polymer.atoms, NUM_ATOMS, "atom")
            embeddings.append(self.atom_embedding(atom_idx))

        if self.residue_embedding is not None:
            res_idx = self._validate_indices(polymer.sequence, NUM_RESIDUES, "residue")
            res_emb = self.residue_embedding(res_idx)
            if self.scale == Scale.ATOM:
                # Expand to atom level (only covers polymer atoms)
                res_emb = polymer.expand(res_emb, Scale.RESIDUE)
                # Pad with zeros for non-polymer atoms (water, ions, etc.)
                if polymer.nonpoly > 0:
                    padding = torch.zeros(
                        polymer.nonpoly, res_emb.shape[-1],
                        dtype=res_emb.dtype, device=res_emb.device
                    )
                    res_emb = torch.cat([res_emb, padding], dim=0)
            embeddings.append(res_emb)

        if self.element_embedding is not None:
            elem_idx = self._validate_indices(polymer.elements, NUM_ELEMENTS, "element")
            embeddings.append(self.element_embedding(elem_idx))

        return torch.cat(embeddings, dim=-1)
