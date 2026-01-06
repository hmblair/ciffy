"""
Hierarchical scale bookkeeping for molecular structures.

This module provides the _Hierarchy class which encapsulates all operations
that depend solely on size bookkeeping without requiring actual atomic data.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from ..backend import Array, ops, size as arr_size
from ..biochemistry import Scale
from ..operations.reduction import Reduction, REDUCTIONS, ReductionResult, create_reduction_index

if TYPE_CHECKING:
    pass


def _empty_per(with_residues: bool = True) -> dict[tuple[Scale, Scale], np.ndarray | None]:
    """Create the _per dict for an empty hierarchy.

    Args:
        with_residues: If True, include residue-level arrays.
            If False, residue arrays are None (for HeteroAtoms).
    """
    if with_residues:
        return {
            (Scale.ATOM, Scale.RESIDUE): np.array([], dtype=np.int64),
            (Scale.ATOM, Scale.CHAIN): np.array([], dtype=np.int64),
            (Scale.ATOM, Scale.MOLECULE): np.array([0], dtype=np.int64),
            (Scale.RESIDUE, Scale.CHAIN): np.array([], dtype=np.int64),
            (Scale.RESIDUE, Scale.MOLECULE): np.array([0], dtype=np.int64),
            (Scale.CHAIN, Scale.MOLECULE): np.array([0], dtype=np.int64),
        }
    else:
        return {
            (Scale.ATOM, Scale.RESIDUE): None,
            (Scale.ATOM, Scale.CHAIN): np.array([], dtype=np.int64),
            (Scale.ATOM, Scale.MOLECULE): np.array([0], dtype=np.int64),
            (Scale.RESIDUE, Scale.CHAIN): None,
            (Scale.RESIDUE, Scale.MOLECULE): None,
            (Scale.CHAIN, Scale.MOLECULE): np.array([0], dtype=np.int64),
        }


def backend_marker(arr: Array) -> Array:
    """Create a minimal backend/device marker from an array.

    Returns a zero-element array with the same backend and device as the input.
    Used as a lightweight reference for backend/device detection without
    holding actual data.

    Args:
        arr: Source array to match backend/device from.

    Returns:
        Empty (0,) array on same backend/device.
    """
    return ops.zeros(0, like=arr)


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

    Some arrays may be None if fields were skipped during loading. The scalar
    counts (_n_atoms, _n_residues, _n_chains) are always available as they're
    computed from whichever arrays are present.

    Attributes:
        _per: Dict mapping (inner_scale, outer_scale) to count arrays (may be None).
        _ref: Minimal (0,) array used solely for backend/device detection.
            Does not hold actual data - just matches the target backend/device.
        _n_atoms: Total atom count (cached).
        _n_residues: Total residue count (cached).
        _n_chains: Total chain count (cached).
    """

    __slots__ = ('_per', '_ref', '_n_atoms', '_n_residues', '_n_chains', '_has_residues')

    def __init__(
        self,
        per: dict[tuple[Scale, Scale], Array] | None = None,
        ref: Array | None = None,
        *,
        with_residues: bool = True,
    ):
        """
        Initialize a hierarchy.

        When called with no arguments, creates an empty hierarchy with 0 atoms,
        0 residues, and 0 chains.

        Args:
            per: Dict mapping (inner_scale, outer_scale) to count arrays.
                If None, creates an empty hierarchy.
            ref: Minimal array for backend/device detection (shape doesn't matter).
                If None, uses an empty numpy array.
            with_residues: If True (default), hierarchy has residue scale.
                If False, residue arrays are None (for HeteroAtoms).
        """
        if per is None:
            per = _empty_per(with_residues)
        if ref is None:
            ref = np.zeros(0, dtype=np.float32)

        self._per = per
        self._ref = ref
        self._has_residues = with_residues

        # Precompute scalar counts from available arrays
        self._n_atoms, self._n_residues, self._n_chains = self._compute_counts()

    @property
    def has_residues(self) -> bool:
        """Whether this hierarchy has residue-level information."""
        return self._has_residues

    def _compute_counts(self) -> tuple[int, int, int]:
        """Compute scalar counts from available arrays.

        Uses a priority order of data sources for each count:
        - atoms: mol_sizes[0] > sum(atoms_per_chain) > 0
        - residues: len(atoms_per_res) > sum(res_per_chain) > 0
        - chains: len(atoms_per_chain) > len(res_per_chain) > 0
        """
        # Atoms: prefer mol_sizes, fallback to sum of atoms_per_chain
        mol_sizes = self._per.get((Scale.ATOM, Scale.MOLECULE))
        atoms_per_chain = self._per.get((Scale.ATOM, Scale.CHAIN))
        if mol_sizes is not None:
            n_atoms = int(mol_sizes[0])
        elif atoms_per_chain is not None:
            n_atoms = int(atoms_per_chain.sum())
        else:
            n_atoms = 0

        # Residues: prefer len(atoms_per_res), fallback to sum of res_per_chain
        atoms_per_res = self._per.get((Scale.ATOM, Scale.RESIDUE))
        res_per_chain = self._per.get((Scale.RESIDUE, Scale.CHAIN))
        if atoms_per_res is not None:
            n_residues = arr_size(atoms_per_res, 0)
        elif res_per_chain is not None:
            n_residues = int(res_per_chain.sum())
        else:
            n_residues = 0

        # Chains: prefer len(atoms_per_chain), fallback to len(res_per_chain)
        if atoms_per_chain is not None:
            n_chains = arr_size(atoms_per_chain, 0)
        elif res_per_chain is not None:
            n_chains = arr_size(res_per_chain, 0)
        else:
            n_chains = 0

        return n_atoms, n_residues, n_chains

    # ─────────────────────────────────────────────────────────────────────────
    # Factory Methods
    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def from_sizes_and_lengths(
        cls,
        sizes: dict[Scale, Array],
        lengths: Array,
        ref: Array,
    ) -> _Hierarchy:
        """
        Create a Hierarchy from the legacy sizes dict and lengths array.

        This factory method converts the old representation to the new
        unified _per dict.

        Args:
            sizes: Dict mapping Scale to atoms-per-unit arrays.
            lengths: Array of residues per chain.
            ref: Array to derive backend/device from (will be converted to marker).

        Returns:
            New _Hierarchy instance.
        """
        # Use lengths as fallback ref for array creation when coordinates is None
        arr_ref = ref if ref is not None else lengths

        # Build the unified _per dict (values may be None if fields were skipped)
        per = {
            # Atoms per {residue, chain, molecule}
            (Scale.ATOM, Scale.RESIDUE): sizes.get(Scale.RESIDUE),
            (Scale.ATOM, Scale.CHAIN): sizes.get(Scale.CHAIN),
            (Scale.ATOM, Scale.MOLECULE): sizes.get(Scale.MOLECULE),
            # Residues per {chain, molecule}
            (Scale.RESIDUE, Scale.CHAIN): lengths,
            (Scale.RESIDUE, Scale.MOLECULE): ops.array([arr_size(lengths, 0)], like=arr_ref),
            # Chains per molecule
            (Scale.CHAIN, Scale.MOLECULE): ops.array([arr_size(lengths, 0)], like=arr_ref),
        }
        # Convert ref to minimal marker
        marker = backend_marker(arr_ref)
        return cls(per, marker, with_residues=True)

    @classmethod
    def from_atoms_per_chain(
        cls,
        atoms_per_chain: Array,
        ref: Array | None = None,
    ) -> "_Hierarchy":
        """
        Create a hierarchy without residue scale (for HeteroAtoms).

        This creates a simplified hierarchy with only ATOM, CHAIN, and MOLECULE
        scales. Residue-level arrays are set to None.

        Args:
            atoms_per_chain: Array of atom counts per chain.
            ref: Array to derive backend/device from. If None, uses atoms_per_chain.

        Returns:
            New _Hierarchy without residue scale.
        """
        arr_ref = ref if ref is not None else atoms_per_chain
        n_atoms = int(atoms_per_chain.sum()) if arr_size(atoms_per_chain, 0) > 0 else 0
        n_chains = arr_size(atoms_per_chain, 0)

        per = {
            (Scale.ATOM, Scale.RESIDUE): None,
            (Scale.ATOM, Scale.CHAIN): atoms_per_chain,
            (Scale.ATOM, Scale.MOLECULE): ops.array([n_atoms], like=arr_ref, dtype='int64'),
            (Scale.RESIDUE, Scale.CHAIN): None,
            (Scale.RESIDUE, Scale.MOLECULE): None,
            (Scale.CHAIN, Scale.MOLECULE): ops.array([n_chains], like=arr_ref, dtype='int64'),
        }
        marker = backend_marker(arr_ref)
        return cls(per, marker, with_residues=False)

    # ─────────────────────────────────────────────────────────────────────────
    # Properties
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def lengths(self) -> Array | None:
        """Residues per chain. Returns None if hierarchy has no residues."""
        return self._per.get((Scale.RESIDUE, Scale.CHAIN))

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
            return self._n_atoms
        if scale == Scale.RESIDUE:
            return self._n_residues
        if scale == Scale.CHAIN:
            return self._n_chains
        if scale == Scale.MOLECULE:
            return 1
        raise ValueError(f"Unknown scale: {scale}")

    def counts(self, scale: Scale) -> Array:
        """
        Get atom counts per unit at the specified scale.

        Args:
            scale: Scale level.

        Returns:
            Tensor of atom counts per unit at this scale.

        Raises:
            ValueError: If the counts array is not available (field was skipped).
        """
        arr = self._per.get((Scale.ATOM, scale))
        if arr is None:
            raise ValueError(
                f"Counts at {scale.name} scale not available. "
                f"This field may have been skipped during loading."
            )
        return arr

    # Alias for backwards compatibility
    sizes = counts

    def per(self, inner: Scale, outer: Scale) -> Array:
        """
        Get the count of inner units per outer unit.

        Args:
            inner: Inner scale (e.g., RESIDUE).
            outer: Outer scale (e.g., CHAIN).

        Returns:
            Array with count of inner units per outer unit.

        Raises:
            ValueError: If the requested array is not available (field was skipped).

        Example:
            >>> hierarchy.per(Scale.RESIDUE, Scale.CHAIN)
            array([150, 200, 175])  # residues per chain
        """
        if inner == outer:
            return ops.ones(self.size(inner), like=self._ref)

        # Direct lookup in _per dict
        key = (inner, outer)
        arr = self._per.get(key)
        if arr is not None:
            return arr

        # Check if key exists but value is None (skipped field)
        if key in self._per:
            raise ValueError(
                f"{inner.name} per {outer.name} not available. "
                f"This field may have been skipped during loading."
            )

        raise ValueError(f"Cannot compute {inner.name} per {outer.name}")

    def empty(self) -> bool:
        """Check if the hierarchy has no atoms."""
        return self._n_atoms == 0

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

    def membership(self, scale: Scale) -> Array:
        """
        Get which unit each atom belongs to at the specified scale.

        Creates an integer array where each atom is labeled with its
        containing unit's index at the given scale.

        Args:
            scale: Scale at which to compute membership.

        Returns:
            Integer array of shape (num_atoms,) with unit indices.
        """
        n = self.size(scale)
        idx = ops.arange(n, like=self._ref)
        return self.expand(idx, scale, Scale.ATOM)

    # Alias for backwards compatibility
    index = membership

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

        Raises:
            ValueError: If RESIDUE scale requested but hierarchy has no residues.
        """
        if scale == Scale.RESIDUE and not self._has_residues:
            raise ValueError("RESIDUE scale not supported for hierarchies without residues")
        arr = self._per.get((Scale.ATOM, scale))
        if arr is None:
            raise ValueError(f"Scale {scale.name} not available in this hierarchy")
        return arr != 0

    def derive_masks(
        self,
        input_mask: Array,
        input_scale: Scale,
        remove_empty_residues: bool = False,
    ) -> tuple[Array, Array | None, Array]:
        """
        Derive masks at all scales from an input mask at a specific scale.

        Args:
            input_mask: Boolean mask at input_scale.
            input_scale: Scale of the input mask (ATOM, RESIDUE, or CHAIN).
            remove_empty_residues: If True, remove residues with 0 atoms
                (ATOM scale behavior).

        Returns:
            Tuple of (atom_mask, res_mask, chn_mask).
            res_mask is None if hierarchy has no residues.
        """
        # No-residue path (HeteroAtoms)
        if not self._has_residues:
            if input_scale == Scale.RESIDUE:
                raise ValueError("RESIDUE scale not supported for hierarchies without residues")
            if input_scale == Scale.ATOM:
                atom_mask = input_mask
                chn_sizes_after = self.count(input_mask, Scale.CHAIN)
                chn_mask = chn_sizes_after > 0
            else:  # CHAIN
                chn_mask = input_mask
                atom_mask = self.expand(chn_mask, Scale.CHAIN, Scale.ATOM)
            return atom_mask, None, chn_mask

        # Standard path with residues
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
            atom_mask = self.expand(res_mask, Scale.RESIDUE, Scale.ATOM)
            new_lengths = self.reduce(ops.to_int64(res_mask), Scale.CHAIN, Reduction.SUM, in_scale=Scale.RESIDUE)
            chn_mask = new_lengths > 0

        elif input_scale == Scale.CHAIN:
            chn_mask = input_mask
            res_mask = self.expand(chn_mask, Scale.CHAIN, Scale.RESIDUE)
            atom_mask = self.expand(res_mask, Scale.RESIDUE, Scale.ATOM)

        else:
            raise ValueError(f"Selection not supported at {input_scale.name} scale")

        return atom_mask, res_mask, chn_mask

    def compute_per(
        self,
        atom_mask: Array,
        res_mask: Array | None,
        chn_mask: Array,
        input_scale: Scale,
    ) -> dict[tuple[Scale, Scale], Array | None]:
        """
        Compute the _per dict for a new Hierarchy after selection.

        Args:
            atom_mask: Boolean mask for atoms.
            res_mask: Boolean mask for residues (None if no residues).
            chn_mask: Boolean mask for chains.
            input_scale: Scale of the original input mask.

        Returns:
            New _per dict.
        """
        # No-residue path (HeteroAtoms)
        if not self._has_residues:
            chn_sizes_after = self.count(atom_mask, Scale.CHAIN)
            mol_sizes = self.count(atom_mask, Scale.MOLECULE)
            new_chn_sizes = chn_sizes_after[chn_mask]
            n_chn = arr_size(new_chn_sizes, 0)

            return {
                (Scale.ATOM, Scale.RESIDUE): None,
                (Scale.ATOM, Scale.CHAIN): new_chn_sizes,
                (Scale.ATOM, Scale.MOLECULE): mol_sizes,
                (Scale.RESIDUE, Scale.CHAIN): None,
                (Scale.RESIDUE, Scale.MOLECULE): None,
                (Scale.CHAIN, Scale.MOLECULE): ops.array([n_chn], like=self._ref),
            }

        # Standard path with residues
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
        remove_empty = (scale == Scale.ATOM) and self._has_residues
        atom_mask, res_mask, chn_mask = self.derive_masks(mask, scale, remove_empty)
        new_per = self.compute_per(atom_mask, res_mask, chn_mask, scale)

        return _Hierarchy(new_per, self._ref, with_residues=self._has_residues)

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

        new_per = {k: to_torch(v, dtype=Dtype.INT64) if v is not None else None
                   for k, v in self._per.items()}
        new_ref = to_torch(self._ref)
        return _Hierarchy(new_per, new_ref, with_residues=self._has_residues)

    def numpy(self) -> _Hierarchy:
        """
        Convert all arrays to NumPy arrays.

        Returns:
            New _Hierarchy with NumPy arrays.
        """
        from ..backend import to_numpy, is_torch
        if not is_torch(self._ref):
            return self

        new_per = {k: to_numpy(v) if v is not None else None
                   for k, v in self._per.items()}
        new_ref = to_numpy(self._ref)
        return _Hierarchy(new_per, new_ref, with_residues=self._has_residues)

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

        new_per = {k: v.to(device) if v is not None else None
                   for k, v in self._per.items()}
        new_ref = self._ref.to(device) if hasattr(self._ref, 'to') else self._ref
        return _Hierarchy(new_per, new_ref, with_residues=self._has_residues)

    # ─────────────────────────────────────────────────────────────────────────
    # Chain Extension
    # ─────────────────────────────────────────────────────────────────────────

    def extend_residue(self, n_atoms: int) -> "_Hierarchy":
        """
        Create a new Hierarchy with one additional residue appended to the last chain.

        For use by Polymer.extend() when adding a residue to the current chain.
        All atoms in the new residue are considered polymer atoms.

        Args:
            n_atoms: Number of atoms in the new residue.

        Returns:
            New _Hierarchy with updated counts.
        """
        old_total = self._n_atoms

        # Append new residue's atom count
        new_res_sizes = ops.cat([
            self._per[(Scale.ATOM, Scale.RESIDUE)],
            ops.array([n_atoms], like=self._ref, dtype='int64')
        ])

        # Update last chain's size (add new atoms to last chain)
        old_chn_sizes = self._per[(Scale.ATOM, Scale.CHAIN)]
        if self._n_chains == 1:
            new_chn_sizes = ops.array([old_total + n_atoms], like=self._ref, dtype='int64')
        else:
            # Update only the last chain's atom count
            last_chain_size = int(old_chn_sizes[-1]) + n_atoms
            new_chn_sizes = ops.cat([
                old_chn_sizes[:-1],
                ops.array([last_chain_size], like=self._ref, dtype='int64')
            ])

        # Update molecule size
        new_mol_sizes = ops.array([old_total + n_atoms], like=self._ref, dtype='int64')

        # Update residues in last chain
        old_lengths = self._per[(Scale.RESIDUE, Scale.CHAIN)]
        if self._n_chains == 1:
            new_lengths = ops.array([self._n_residues + 1], like=self._ref, dtype='int64')
        else:
            last_chain_residues = int(old_lengths[-1]) + 1
            new_lengths = ops.cat([
                old_lengths[:-1],
                ops.array([last_chain_residues], like=self._ref, dtype='int64')
            ])

        new_per = {
            (Scale.ATOM, Scale.RESIDUE): new_res_sizes,
            (Scale.ATOM, Scale.CHAIN): new_chn_sizes,
            (Scale.ATOM, Scale.MOLECULE): new_mol_sizes,
            (Scale.RESIDUE, Scale.CHAIN): new_lengths,
            (Scale.RESIDUE, Scale.MOLECULE): ops.array([self._n_residues + 1], like=self._ref),
            (Scale.CHAIN, Scale.MOLECULE): ops.array([self._n_chains], like=self._ref),
        }

        return _Hierarchy(new_per, self._ref)

    def extend_new_chain(self, n_atoms: int) -> "_Hierarchy":
        """
        Create a new Hierarchy with a new chain containing one residue.

        For use by Polymer.extend() when starting a new chain.
        All atoms in the new residue are considered polymer atoms.

        Args:
            n_atoms: Number of atoms in the new residue.

        Returns:
            New _Hierarchy with an additional chain.
        """
        old_total = self._n_atoms

        # Append new residue's atom count
        new_res_sizes = ops.cat([
            self._per[(Scale.ATOM, Scale.RESIDUE)],
            ops.array([n_atoms], like=self._ref, dtype='int64')
        ])

        # Append new chain size (just the new residue's atoms)
        new_chn_sizes = ops.cat([
            self._per[(Scale.ATOM, Scale.CHAIN)],
            ops.array([n_atoms], like=self._ref, dtype='int64')
        ])

        # Update molecule size (all chains combined)
        new_mol_sizes = ops.array([old_total + n_atoms], like=self._ref, dtype='int64')

        # Append residues in new chain (1 residue)
        new_lengths = ops.cat([
            self._per[(Scale.RESIDUE, Scale.CHAIN)],
            ops.array([1], like=self._ref, dtype='int64')
        ])

        new_per = {
            (Scale.ATOM, Scale.RESIDUE): new_res_sizes,
            (Scale.ATOM, Scale.CHAIN): new_chn_sizes,
            (Scale.ATOM, Scale.MOLECULE): new_mol_sizes,
            (Scale.RESIDUE, Scale.CHAIN): new_lengths,
            (Scale.RESIDUE, Scale.MOLECULE): ops.array([self._n_residues + 1], like=self._ref),
            (Scale.CHAIN, Scale.MOLECULE): ops.array([self._n_chains + 1], like=self._ref),
        }

        return _Hierarchy(new_per, self._ref)

    # ─────────────────────────────────────────────────────────────────────────
    # Contiguous Selection Support
    # ─────────────────────────────────────────────────────────────────────────

    def bounds(self, ix: int, scale: Scale) -> tuple[slice, slice | None, slice]:
        """
        Get slice bounds for a contiguous selection at a given scale.

        For chain/residue selections, units are stored contiguously, so we can
        use slice indexing (much faster than boolean masking).

        Args:
            ix: Index of the unit to select.
            scale: Scale of the selection (CHAIN or RESIDUE).

        Returns:
            Tuple of (atom_slice, res_slice, chain_slice).
            res_slice is None if hierarchy has no residues.

        Raises:
            IndexError: If index is out of range.
            ValueError: If RESIDUE scale requested but hierarchy has no residues.
        """
        if scale == Scale.CHAIN:
            if ix < 0 or ix >= self._n_chains:
                raise IndexError(f"Chain index {ix} out of range [0, {self._n_chains})")

            # Chain slice
            chain_slice = slice(ix, ix + 1)

            # Atom slice: cumsum of atoms_per_chain
            atoms_per_chain = self._per[(Scale.ATOM, Scale.CHAIN)]
            atom_cumsum = np.cumsum(np.asarray(atoms_per_chain))
            atom_start = 0 if ix == 0 else int(atom_cumsum[ix - 1])
            atom_end = int(atom_cumsum[ix])
            atom_slice = slice(atom_start, atom_end)

            # Residue slice: cumsum of residues_per_chain (None if no residues)
            if not self._has_residues:
                return atom_slice, None, chain_slice

            res_per_chain = self._per[(Scale.RESIDUE, Scale.CHAIN)]
            res_cumsum = np.cumsum(np.asarray(res_per_chain))
            res_start = 0 if ix == 0 else int(res_cumsum[ix - 1])
            res_end = int(res_cumsum[ix])
            res_slice = slice(res_start, res_end)

            return atom_slice, res_slice, chain_slice

        elif scale == Scale.RESIDUE:
            if not self._has_residues:
                raise ValueError("RESIDUE scale not supported for hierarchies without residues")
            if ix < 0 or ix >= self._n_residues:
                raise IndexError(f"Residue index {ix} out of range [0, {self._n_residues})")

            # Residue slice
            res_slice = slice(ix, ix + 1)

            # Atom slice: cumsum of atoms_per_residue
            atoms_per_res = self._per[(Scale.ATOM, Scale.RESIDUE)]
            atom_cumsum = np.cumsum(np.asarray(atoms_per_res))
            atom_start = 0 if ix == 0 else int(atom_cumsum[ix - 1])
            atom_end = int(atom_cumsum[ix])
            atom_slice = slice(atom_start, atom_end)

            # Chain slice: find which chain this residue belongs to
            res_per_chain = self._per[(Scale.RESIDUE, Scale.CHAIN)]
            res_cumsum = np.cumsum(np.asarray(res_per_chain))
            chain_ix = int(np.searchsorted(res_cumsum, ix + 1))
            chain_slice = slice(chain_ix, chain_ix + 1)

            return atom_slice, res_slice, chain_slice

        else:
            raise ValueError(f"bounds() not supported for {scale.name} scale")
