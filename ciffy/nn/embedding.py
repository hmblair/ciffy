"""
Learnable embeddings for polymer features.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ciffy.polymer import Polymer

import torch
import torch.nn as nn

from ciffy.biochemistry import Scale, NUM_ATOMS, NUM_RESIDUES, NUM_ELEMENTS


class PolymerEmbedding(nn.Module):
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
        dropout: float = 0.0,
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
            dropout: Dropout probability applied to the concatenated embeddings.

        Raises:
            ImportError: If PyTorch is not installed.
            ValueError: If scale is not ATOM or RESIDUE.
            ValueError: If atom_dim or element_dim specified with scale=RESIDUE.
            ValueError: If no embedding dimensions are specified.
        """
        super().__init__()

        # Treat 0 as None (disabled)
        atom_dim = atom_dim or None
        residue_dim = residue_dim or None
        element_dim = element_dim or None

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

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

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

    def _validate_and_raise(
        self,
        indices: torch.Tensor,
        max_idx: int,
        name: str,
    ) -> None:
        """
        Validate indices and raise detailed error if invalid.

        Called only when an embedding error occurs, to provide a helpful
        error message. This avoids GPU sync on the happy path.

        Args:
            indices: Tensor of indices to validate.
            max_idx: Maximum valid index (vocabulary size).
            name: Name of the index type for error messages.

        Raises:
            IndexError: If any indices are out of bounds.
        """
        # Force sync to surface any pending CUDA errors
        if indices.is_cuda:
            torch.cuda.synchronize()

        # Check for negative indices (invalid - should never happen)
        neg_mask = indices < 0
        if neg_mask.any():
            neg_indices = indices[neg_mask].unique().tolist()
            neg_count = neg_mask.sum().item()
            raise IndexError(
                f"PolymerEmbedding: {neg_count} negative {name} indices found. "
                f"Got values: {neg_indices[:10]}{'...' if len(neg_indices) > 10 else ''}. "
                f"This indicates a bug in the data loading pipeline."
            )

        # Check for indices >= max_idx
        high_mask = indices >= max_idx
        if high_mask.any():
            high_indices = indices[high_mask].unique().tolist()
            high_count = high_mask.sum().item()
            raise IndexError(
                f"PolymerEmbedding: {high_count} {name} indices out of bounds. "
                f"Valid range: [0, {max_idx}), got values: {high_indices[:10]}"
                f"{'...' if len(high_indices) > 10 else ''}. "
                f"This may indicate unsupported atom/residue types."
            )

    def _embed(
        self,
        embedding: nn.Embedding,
        indices: torch.Tensor,
        max_idx: int,
        name: str,
    ) -> torch.Tensor:
        """
        Embed indices with lazy validation.

        Performs embedding lookup. If an error occurs, validates indices
        to provide a detailed error message.

        Args:
            embedding: The embedding layer.
            indices: Indices to embed. Index 0 = unknown/sentinel,
                indices 1+ = valid values. Negative indices are invalid.
            max_idx: Maximum valid index for validation.
            name: Name of the index type for error messages.

        Returns:
            Embedded tensor.

        Raises:
            IndexError: If any indices are out of bounds (negative or >= max_idx).
        """
        try:
            return embedding(indices)
        except (IndexError, RuntimeError):
            # Embedding failed - validate indices for detailed error
            self._validate_and_raise(indices, max_idx, name)
            raise  # Re-raise if validation didn't find the issue

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
            IndexError: If any indices are out of bounds.

        Note:
            Index 0 represents unknown/unrecognized values and maps to a
            dedicated "unknown" embedding. Valid indices start at 1.
            Validation is lazy - indices are only checked if an error occurs,
            avoiding GPU sync overhead on the happy path.
        """
        embeddings = []

        if self.atom_embedding is not None:
            embeddings.append(
                self._embed(self.atom_embedding, polymer.atoms, NUM_ATOMS, "atom")
            )

        if self.residue_embedding is not None:
            res_emb = self._embed(
                self.residue_embedding, polymer.sequence, NUM_RESIDUES, "residue"
            )
            if self.scale == Scale.ATOM:
                # Expand to atom level
                res_emb = polymer.expand(res_emb, Scale.RESIDUE)
            embeddings.append(res_emb)

        if self.element_embedding is not None:
            embeddings.append(
                self._embed(self.element_embedding, polymer.elements, NUM_ELEMENTS, "element")
            )

        out = torch.cat(embeddings, dim=-1)

        if self.dropout is not None:
            out = self.dropout(out)

        return out
