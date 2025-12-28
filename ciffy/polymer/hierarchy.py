"""
Hierarchical scale bookkeeping for molecular structures.

This module provides the _Hierarchy class which encapsulates all operations
that depend solely on size bookkeeping without requiring actual atomic data.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

from ..backend import Array, ops, size as arr_size
from ..biochemistry import Scale
from ..operations.reduction import Reduction, REDUCTIONS, ReductionResult, create_reduction_index

if TYPE_CHECKING:
    pass


class _Hierarchy:
    """
    Internal class for hierarchical scale bookkeeping.

    Tracks the count of units at each scale and provides operations for
    expanding, reducing, and masking across scales. This class is independent
    of actual molecular data - it only tracks counts and indices.

    The hierarchy is stored as a dict mapping (inner, outer) scale pairs to
    arrays of counts, representing the upper triangle of a 4x4 relationship:

        _per[(ATOM, RESIDUE)]   = atoms per residue (R elements)
        _per[(ATOM, CHAIN)]     = atoms per chain (C elements)
        _per[(ATOM, MOLECULE)]  = total atoms (1 element)
        _per[(RESIDUE, CHAIN)]  = residues per chain (C elements)
        _per[(RESIDUE, MOLECULE)] = total residues (1 element)
        _per[(CHAIN, MOLECULE)] = total chains (1 element)

    Attributes:
        _per: Dict mapping (inner_scale, outer_scale) to count arrays.
        _polymer_count: Number of polymer atoms (first _polymer_count atoms).
        _ref: Reference array for backend/device detection.
    """

    __slots__ = ('_per', '_polymer_count', '_ref')

    def __init__(
        self,
        per: dict[tuple[Scale, Scale], Array],
        polymer_count: int,
        ref: Array,
    ):
        """
        Initialize a hierarchy.

        Args:
            per: Dict mapping (inner_scale, outer_scale) to count arrays.
            polymer_count: Number of polymer atoms.
            ref: Reference array for backend/device detection.
        """
        self._per = per
        self._polymer_count = polymer_count
        self._ref = ref

    # ─────────────────────────────────────────────────────────────────────────
    # Factory Methods
    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def from_sizes_and_lengths(
        cls,
        sizes: dict[Scale, Array],
        lengths: Array,
        polymer_count: int,
        ref: Array,
    ) -> _Hierarchy:
        """
        Create a Hierarchy from the legacy sizes dict and lengths array.

        This factory method converts the old representation to the new
        unified _per dict.

        Args:
            sizes: Dict mapping Scale to atoms-per-unit arrays.
            lengths: Array of residues per chain.
            polymer_count: Number of polymer atoms.
            ref: Reference array for backend/device detection.

        Returns:
            New _Hierarchy instance.
        """
        # Build the unified _per dict
        per = {
            # Atoms per {residue, chain, molecule}
            (Scale.ATOM, Scale.RESIDUE): sizes[Scale.RESIDUE],
            (Scale.ATOM, Scale.CHAIN): sizes[Scale.CHAIN],
            (Scale.ATOM, Scale.MOLECULE): sizes[Scale.MOLECULE],
            # Residues per {chain, molecule}
            (Scale.RESIDUE, Scale.CHAIN): lengths,
            (Scale.RESIDUE, Scale.MOLECULE): ops.array([arr_size(lengths, 0)], like=ref),
            # Chains per molecule
            (Scale.CHAIN, Scale.MOLECULE): ops.array([arr_size(lengths, 0)], like=ref),
        }
        return cls(per, polymer_count, ref)

    # ─────────────────────────────────────────────────────────────────────────
    # Properties
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def polymer_count(self) -> int:
        """Number of polymer atoms."""
        return self._polymer_count

    @property
    def nonpoly(self) -> int:
        """Number of non-polymer atoms."""
        total = self._per[(Scale.ATOM, Scale.MOLECULE)][0].item()
        return total - self._polymer_count

    @property
    def lengths(self) -> Array:
        """Residues per chain (backward compatibility)."""
        return self._per[(Scale.RESIDUE, Scale.CHAIN)]

    # ─────────────────────────────────────────────────────────────────────────
    # Size Queries
    # ─────────────────────────────────────────────────────────────────────────

    def size(self, scale: Scale) -> int:
        """
        Get the count at a specific scale.

        Args:
            scale: Scale level (ATOM, RESIDUE, CHAIN, MOLECULE).

        Returns:
            Number of units at the specified scale.
        """
        if scale == Scale.ATOM:
            return self._per[(Scale.ATOM, Scale.MOLECULE)][0].item()
        if scale == Scale.RESIDUE:
            return arr_size(self._per[(Scale.ATOM, Scale.RESIDUE)], 0)
        if scale == Scale.CHAIN:
            return arr_size(self._per[(Scale.ATOM, Scale.CHAIN)], 0)
        if scale == Scale.MOLECULE:
            return 1
        raise ValueError(f"Unknown scale: {scale}")

    def sizes(self, scale: Scale) -> Array:
        """
        Get the sizes tensor for a scale.

        Args:
            scale: Scale level.

        Returns:
            Tensor of atom counts per unit at this scale.
        """
        return self._per[(Scale.ATOM, scale)]

    def per(self, inner: Scale, outer: Scale) -> Array:
        """
        Get the count of inner units per outer unit.

        Args:
            inner: Inner scale (e.g., RESIDUE).
            outer: Outer scale (e.g., CHAIN).

        Returns:
            Array with count of inner units per outer unit.

        Example:
            >>> hierarchy.per(Scale.RESIDUE, Scale.CHAIN)
            array([150, 200, 175])  # residues per chain
        """
        if inner == outer:
            return ops.ones(self.size(inner), like=self._ref)

        # Direct lookup in _per dict
        key = (inner, outer)
        if key in self._per:
            return self._per[key]

        raise ValueError(f"Cannot compute {inner.name} per {outer.name}")

    def empty(self) -> bool:
        """Check if the hierarchy has no atoms."""
        return self._per[(Scale.ATOM, Scale.MOLECULE)][0].item() == 0

    # ─────────────────────────────────────────────────────────────────────────
    # Reduction Operations
    # ─────────────────────────────────────────────────────────────────────────

    def reduce(
        self,
        features: Array,
        out_scale: Scale,
        rtype: Reduction = Reduction.MEAN,
        in_scale: Scale = Scale.ATOM,
    ) -> ReductionResult:
        """
        Reduce features from one scale to a coarser scale.

        Aggregates features within each unit at the output scale
        using the chosen reduction operation.

        Args:
            features: Feature tensor at in_scale.
            out_scale: Target scale to reduce to.
            rtype: Reduction type (MEAN, SUM, MIN, MAX, COLLATE).
            in_scale: Scale of input features (default: ATOM).

        Returns:
            Reduced features. For MIN/MAX, returns (values, indices).

        Examples:
            >>> # Atom -> residue (default)
            >>> res_feats = hierarchy.reduce(atom_feats, Scale.RESIDUE)
            >>> # Residue -> chain (with explicit in_scale)
            >>> chain_feats = hierarchy.reduce(res_feats, Scale.CHAIN, in_scale=Scale.RESIDUE)
            >>> # Chain -> molecule
            >>> mol_feats = hierarchy.reduce(chain_feats, Scale.MOLECULE, in_scale=Scale.CHAIN)

        Note:
            When reducing from ATOM to RESIDUE scale, non-polymer atoms are
            automatically excluded since they don't belong to any residue.
        """
        # Non-polymer atoms don't belong to residues, so slice them out
        if in_scale == Scale.ATOM and out_scale == Scale.RESIDUE and self.nonpoly > 0:
            features = features[:self._polymer_count]

        count = self.size(out_scale)
        sizes = self._per[(in_scale, out_scale)]
        device = getattr(features, 'device', None)
        ix = create_reduction_index(count, sizes, device=device)

        return REDUCTIONS[rtype](features, ix, dim=0, dim_size=count)

    def expand(
        self,
        features: Array,
        source: Scale,
        dest: Scale = Scale.ATOM,
    ) -> Array:
        """
        Expand per-scale features to a finer scale.

        Broadcasts values from a coarser scale to a finer scale by
        repeating each value for all units in the finer scale.

        Args:
            features: Per-source-scale feature tensor.
            source: Source scale.
            dest: Destination scale (default: ATOM).

        Returns:
            Expanded feature tensor.
        """
        if dest == Scale.ATOM:
            return ops.repeat_interleave(features, self._per[(Scale.ATOM, source)])
        if dest == Scale.RESIDUE:
            return ops.repeat_interleave(features, self._per[(Scale.RESIDUE, source)])
        raise ValueError(f"Cannot expand to {dest.name}")

    def count(self, mask: Array, scale: Scale) -> Array:
        """
        Count True values in mask per scale unit.

        Args:
            mask: Boolean mask tensor (at ATOM scale).
            scale: Scale at which to count.

        Returns:
            Count tensor with one value per scale unit.
        """
        return self.reduce(ops.to_int64(mask), scale, Reduction.SUM, in_scale=Scale.ATOM)

    def index(self, scale: Scale) -> Array:
        """
        Get the index of each atom within units at the specified scale.

        Creates an integer array where each atom is labeled with its
        containing unit's index at the given scale.

        Args:
            scale: Scale at which to compute indices.

        Returns:
            Integer array of shape (num_atoms,) with indices.
        """
        n = self.size(scale)
        idx = ops.arange(n, like=self._ref)
        return self.expand(idx, scale, Scale.ATOM)

    # ─────────────────────────────────────────────────────────────────────────
    # Mask Operations
    # ─────────────────────────────────────────────────────────────────────────

    def resolved(self, scale: Scale = Scale.RESIDUE) -> Array:
        """
        Get mask of resolved (non-empty) units.

        Args:
            scale: Scale to check.

        Returns:
            Boolean tensor where True indicates resolved units.
        """
        return self._per[(Scale.ATOM, scale)] != 0

    def derive_masks(
        self,
        input_mask: Array,
        input_scale: Scale,
        remove_empty_residues: bool = False,
    ) -> tuple[Array, Array, Array]:
        """
        Derive masks at all scales from an input mask at a specific scale.

        Args:
            input_mask: Boolean mask at input_scale.
            input_scale: Scale of the input mask (ATOM, RESIDUE, or CHAIN).
            remove_empty_residues: If True, remove residues with 0 atoms
                (ATOM scale behavior).

        Returns:
            Tuple of (atom_mask, res_mask, chn_mask).
        """
        if input_scale == Scale.ATOM:
            atom_mask = input_mask
            res_sizes = self.count(input_mask, Scale.RESIDUE)
            if remove_empty_residues:
                res_mask = res_sizes > 0
            else:
                res_mask = ops.ones(self.size(Scale.RESIDUE), like=self._ref, dtype='bool')
            new_lengths = self.reduce(ops.to_int64(res_mask), Scale.CHAIN, Reduction.SUM, in_scale=Scale.RESIDUE)
            chn_mask = new_lengths > 0

        elif input_scale == Scale.RESIDUE:
            res_mask = input_mask
            polymer_atom_mask = self.expand(res_mask, Scale.RESIDUE, Scale.ATOM)
            # Pad with False for non-polymer atoms
            if self.nonpoly > 0:
                atom_mask = ops.zeros(self.size(Scale.ATOM), like=self._ref, dtype='bool')
                atom_mask[:self._polymer_count] = polymer_atom_mask
            else:
                atom_mask = polymer_atom_mask
            new_lengths = self.reduce(ops.to_int64(res_mask), Scale.CHAIN, Reduction.SUM, in_scale=Scale.RESIDUE)
            chn_mask = new_lengths > 0

        elif input_scale == Scale.CHAIN:
            chn_mask = input_mask
            res_mask = self.expand(chn_mask, Scale.CHAIN, Scale.RESIDUE)
            polymer_atom_mask = self.expand(res_mask, Scale.RESIDUE, Scale.ATOM)
            # Pad with False for non-polymer atoms
            if self.nonpoly > 0:
                atom_mask = ops.zeros(self.size(Scale.ATOM), like=self._ref, dtype='bool')
                atom_mask[:self._polymer_count] = polymer_atom_mask
            else:
                atom_mask = polymer_atom_mask

        else:
            raise ValueError(f"Selection not supported at {input_scale.name} scale")

        return atom_mask, res_mask, chn_mask

    def compute_per(
        self,
        atom_mask: Array,
        res_mask: Array,
        chn_mask: Array,
        input_scale: Scale,
    ) -> dict[tuple[Scale, Scale], Array]:
        """
        Compute the _per dict for a new Hierarchy after selection.

        Args:
            atom_mask: Boolean mask for atoms.
            res_mask: Boolean mask for residues.
            chn_mask: Boolean mask for chains.
            input_scale: Scale of the original input mask.

        Returns:
            New _per dict.
        """
        if input_scale == Scale.ATOM:
            # Count atoms per unit after masking
            res_sizes_after = self.count(atom_mask, Scale.RESIDUE)
            chn_sizes_after = self.count(atom_mask, Scale.CHAIN)
            mol_sizes = self.count(atom_mask, Scale.MOLECULE)

            new_res_sizes = res_sizes_after[res_mask]
            new_chn_sizes = chn_sizes_after[chn_mask]

            # Compute new lengths (residues per chain)
            # Count how many residues remain in each chain
            res_per_chain = self.reduce(ops.to_int64(res_mask), Scale.CHAIN, Reduction.SUM, in_scale=Scale.RESIDUE)
            new_lengths = res_per_chain[chn_mask]

        else:
            # For RESIDUE/CHAIN: use original sizes filtered by masks
            orig_res_sizes = self._per[(Scale.ATOM, Scale.RESIDUE)]

            if input_scale == Scale.RESIDUE:
                masked_res_sizes = orig_res_sizes * ops.to_int64(res_mask)
                chn_sizes = self.reduce(masked_res_sizes, Scale.CHAIN, Reduction.SUM, in_scale=Scale.RESIDUE)
            else:  # CHAIN
                chn_sizes = self._per[(Scale.ATOM, Scale.CHAIN)]

            new_res_sizes = orig_res_sizes[res_mask]
            new_chn_sizes = chn_sizes[chn_mask]

            # Compute new lengths
            orig_lengths = self._per[(Scale.RESIDUE, Scale.CHAIN)]
            if input_scale == Scale.RESIDUE:
                res_per_chain = self.reduce(ops.to_int64(res_mask), Scale.CHAIN, Reduction.SUM, in_scale=Scale.RESIDUE)
                new_lengths = res_per_chain[chn_mask]
            else:  # CHAIN
                new_lengths = orig_lengths[chn_mask]

            # Total atoms
            total_atoms = new_res_sizes.sum().item()
            mol_sizes = ops.array([total_atoms], like=self._ref)

        # Count new units
        n_res = arr_size(new_res_sizes, 0)
        n_chn = arr_size(new_chn_sizes, 0)

        return {
            (Scale.ATOM, Scale.RESIDUE): new_res_sizes,
            (Scale.ATOM, Scale.CHAIN): new_chn_sizes,
            (Scale.ATOM, Scale.MOLECULE): mol_sizes,
            (Scale.RESIDUE, Scale.CHAIN): new_lengths,
            (Scale.RESIDUE, Scale.MOLECULE): ops.array([n_res], like=self._ref),
            (Scale.CHAIN, Scale.MOLECULE): ops.array([n_chn], like=self._ref),
        }

    def compute_polymer_count(
        self,
        atom_mask: Array,
        res_mask: Array,
        input_scale: Scale,
    ) -> int:
        """
        Compute new polymer_count based on selection scale.

        Args:
            atom_mask: Boolean mask for atoms.
            res_mask: Boolean mask for residues.
            input_scale: Scale of the original input mask.

        Returns:
            New polymer_count value.
        """
        if input_scale == Scale.ATOM:
            return atom_mask[:self._polymer_count].sum().item()
        else:
            res_sizes = self._per[(Scale.ATOM, Scale.RESIDUE)][res_mask]
            return res_sizes.sum().item()

    def select(
        self,
        mask: Array,
        scale: Scale,
    ) -> _Hierarchy:
        """
        Create a new Hierarchy for a selection.

        Args:
            mask: Boolean mask at the specified scale.
            scale: Scale of the input mask.

        Returns:
            New _Hierarchy for the selected subset.
        """
        remove_empty = (scale == Scale.ATOM)
        atom_mask, res_mask, chn_mask = self.derive_masks(mask, scale, remove_empty)
        new_per = self.compute_per(atom_mask, res_mask, chn_mask, scale)
        new_polymer_count = self.compute_polymer_count(atom_mask, res_mask, scale)

        return _Hierarchy(new_per, new_polymer_count, self._ref)

    # ─────────────────────────────────────────────────────────────────────────
    # Backend Conversion
    # ─────────────────────────────────────────────────────────────────────────

    def torch(self) -> _Hierarchy:
        """
        Convert all arrays to PyTorch tensors.

        Returns:
            New _Hierarchy with PyTorch tensors.
        """
        from ..backend import to_torch, is_torch, Dtype
        if is_torch(self._ref):
            return self

        new_per = {k: to_torch(v, dtype=Dtype.INT64) for k, v in self._per.items()}
        new_ref = to_torch(self._ref)
        return _Hierarchy(new_per, self._polymer_count, new_ref)

    def numpy(self) -> _Hierarchy:
        """
        Convert all arrays to NumPy arrays.

        Returns:
            New _Hierarchy with NumPy arrays.
        """
        from ..backend import to_numpy, is_torch
        if not is_torch(self._ref):
            return self

        new_per = {k: to_numpy(v) for k, v in self._per.items()}
        new_ref = to_numpy(self._ref)
        return _Hierarchy(new_per, self._polymer_count, new_ref)

    def to(self, device) -> _Hierarchy:
        """
        Move tensors to specified device (torch backend only).

        Args:
            device: Target device (e.g., 'cuda', 'cpu', torch.device).

        Returns:
            New _Hierarchy with tensors on the specified device.
        """
        from ..backend import is_torch
        if not is_torch(self._ref):
            return self  # NumPy arrays don't have devices

        new_per = {k: v.to(device) for k, v in self._per.items()}
        new_ref = self._ref.to(device) if hasattr(self._ref, 'to') else self._ref
        return _Hierarchy(new_per, self._polymer_count, new_ref)
