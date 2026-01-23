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


def _empty_per(scales: frozenset[Scale]) -> dict[tuple[Scale, Scale], np.ndarray]:
    """Create the _per dict for an empty hierarchy.

    Args:
        scales: Set of scales this hierarchy supports.

    Returns:
        Dict containing only the scale pairs that are supported.
    """
    per = {}

    # Always have ATOM -> MOLECULE
    per[(Scale.ATOM, Scale.MOLECULE)] = np.array([0], dtype=np.int64)

    if Scale.RESIDUE in scales:
        per[(Scale.ATOM, Scale.RESIDUE)] = np.array([], dtype=np.int64)
        per[(Scale.RESIDUE, Scale.MOLECULE)] = np.array([0], dtype=np.int64)

    if Scale.CHAIN in scales:
        per[(Scale.ATOM, Scale.CHAIN)] = np.array([], dtype=np.int64)
        per[(Scale.CHAIN, Scale.MOLECULE)] = np.array([0], dtype=np.int64)

    if Scale.RESIDUE in scales and Scale.CHAIN in scales:
        per[(Scale.RESIDUE, Scale.CHAIN)] = np.array([], dtype=np.int64)

    return per


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
    arrays of counts. Only supported scale pairs are present in the dict:

        _per[(ATOM, RESIDUE)]   = atoms per residue (R elements)
        _per[(ATOM, CHAIN)]     = atoms per chain (C elements)
        _per[(ATOM, MOLECULE)]  = total atoms (1 element)
        _per[(RESIDUE, CHAIN)]  = residues per chain (C elements)
        _per[(RESIDUE, MOLECULE)] = total residues (1 element)
        _per[(CHAIN, MOLECULE)] = total chains (1 element)

    Attributes:
        _per: Dict mapping (inner_scale, outer_scale) to count arrays.
            Only contains entries for supported scale pairs.
        _ref: Minimal (0,) array used solely for backend/device detection.
        _n_atoms: Total atom count (cached).
        _n_residues: Total residue count (cached).
        _n_chains: Total chain count (cached).
    """

    # Default scales for a full polymer hierarchy
    ALL_SCALES = frozenset({Scale.ATOM, Scale.RESIDUE, Scale.CHAIN, Scale.MOLECULE})

    __slots__ = ('_per', '_ref', '_n_atoms', '_n_residues', '_n_chains')

    def __init__(
        self,
        per: dict[tuple[Scale, Scale], Array] | None = None,
        ref: Array | None = None,
        *,
        scales: frozenset[Scale] | None = None,
    ):
        """
        Initialize a hierarchy.

        When called with no arguments, creates an empty hierarchy with 0 atoms,
        0 residues, and 0 chains.

        Args:
            per: Dict mapping (inner_scale, outer_scale) to count arrays.
                If None, creates an empty hierarchy with the given scales.
            ref: Minimal array for backend/device detection (shape doesn't matter).
                If None, uses an empty numpy array.
            scales: Frozenset of scales this hierarchy supports. Only used when
                per is None to create an empty hierarchy. When per is provided,
                scales are inferred from per.keys().
        """
        if per is None:
            if scales is None:
                scales = self.ALL_SCALES
            per = _empty_per(scales)
        if ref is None:
            ref = np.zeros(0, dtype=np.float32)

        self._per = per
        self._ref = ref

        # Precompute scalar counts from available arrays
        self._n_atoms, self._n_residues, self._n_chains = self._compute_counts()

        # Validate internal consistency
        self._validate()

    @property
    def scales(self) -> frozenset[Scale]:
        """The scales this hierarchy supports (inferred from _per keys)."""
        result = {Scale.ATOM, Scale.MOLECULE}  # Always present
        for inner, outer in self._per.keys():
            result.add(inner)
            result.add(outer)
        return frozenset(result)

    def has_scale(self, scale: Scale) -> bool:
        """Check if this hierarchy supports a given scale."""
        if scale in (Scale.ATOM, Scale.MOLECULE):
            return True
        # Check if any key contains this scale
        for inner, outer in self._per.keys():
            if inner == scale or outer == scale:
                return True
        return False


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

    def _validate(self) -> None:
        """Validate internal consistency of the hierarchy.

        Checks that counts at different scales are consistent with each other.
        This catches issues like atoms_per_chain being computed from one data
        source while atoms_per_residue is computed from another.

        Raises:
            ValueError: If any consistency check fails.
        """
        atoms_per_res = self._per.get((Scale.ATOM, Scale.RESIDUE))
        atoms_per_chain = self._per.get((Scale.ATOM, Scale.CHAIN))
        atoms_per_mol = self._per.get((Scale.ATOM, Scale.MOLECULE))
        res_per_chain = self._per.get((Scale.RESIDUE, Scale.CHAIN))
        res_per_mol = self._per.get((Scale.RESIDUE, Scale.MOLECULE))

        # Check 1: Global atom count consistency
        # sum(atoms_per_res) == sum(atoms_per_chain) == atoms_per_mol[0]
        if atoms_per_mol is not None:
            expected_atoms = int(atoms_per_mol[0])

            if atoms_per_res is not None:
                sum_res = int(atoms_per_res.sum())
                if sum_res != expected_atoms:
                    raise ValueError(
                        f"Hierarchy inconsistency: sum(atoms_per_residue)={sum_res} "
                        f"!= atoms_per_molecule={expected_atoms}"
                    )

            if atoms_per_chain is not None:
                sum_chain = int(atoms_per_chain.sum())
                if sum_chain != expected_atoms:
                    raise ValueError(
                        f"Hierarchy inconsistency: sum(atoms_per_chain)={sum_chain} "
                        f"!= atoms_per_molecule={expected_atoms}"
                    )

        # Check 2: Global residue count consistency
        # len(atoms_per_res) == sum(res_per_chain) == res_per_mol[0]
        if atoms_per_res is not None:
            n_residues = arr_size(atoms_per_res, 0)

            if res_per_chain is not None:
                sum_res_chain = int(res_per_chain.sum())
                if sum_res_chain != n_residues:
                    raise ValueError(
                        f"Hierarchy inconsistency: sum(residues_per_chain)={sum_res_chain} "
                        f"!= len(atoms_per_residue)={n_residues}"
                    )

            if res_per_mol is not None:
                res_mol = int(res_per_mol[0])
                if res_mol != n_residues:
                    raise ValueError(
                        f"Hierarchy inconsistency: residues_per_molecule={res_mol} "
                        f"!= len(atoms_per_residue)={n_residues}"
                    )

        # Check 3: Per-chain atom consistency
        # For each chain i: sum(atoms_per_res[start:end]) == atoms_per_chain[i]
        if atoms_per_res is not None and atoms_per_chain is not None and res_per_chain is not None:
            res_cumsum = ops.cumsum(res_per_chain)
            res_start = 0
            n_chains = arr_size(atoms_per_chain, 0)
            for chain_idx in range(n_chains):
                res_end = int(res_cumsum[chain_idx])
                atoms_from_residues = int(atoms_per_res[res_start:res_end].sum())
                atoms_from_chain = int(atoms_per_chain[chain_idx])

                if atoms_from_residues != atoms_from_chain:
                    raise ValueError(
                        f"Hierarchy inconsistency in chain {chain_idx}: "
                        f"sum(atoms_per_residue[{res_start}:{res_end}])={atoms_from_residues} "
                        f"!= atoms_per_chain[{chain_idx}]={atoms_from_chain}. "
                        f"This typically indicates the CIF file has unresolved residues "
                        f"in the sequence table that have no atoms in the coordinate section."
                    )
                res_start = res_end

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
        Create a Hierarchy from sizes dict and lengths array.

        Args:
            sizes: Dict mapping Scale to atoms-per-unit arrays.
            lengths: Array of residues per chain.
            ref: Array to derive backend/device from (will be converted to marker).

        Returns:
            New _Hierarchy instance with all scales.
        """
        # Use lengths as fallback ref for array creation when coordinates is None
        arr_ref = ref if ref is not None else lengths

        # Build the _per dict - only include entries that exist
        n_residues = int(lengths.sum()) if arr_size(lengths, 0) > 0 else 0
        n_chains = arr_size(lengths, 0)
        per = {
            (Scale.ATOM, Scale.RESIDUE): sizes[Scale.RESIDUE],
            (Scale.ATOM, Scale.CHAIN): sizes[Scale.CHAIN],
            (Scale.ATOM, Scale.MOLECULE): sizes[Scale.MOLECULE],
            (Scale.RESIDUE, Scale.CHAIN): lengths,
            (Scale.RESIDUE, Scale.MOLECULE): ops.array([n_residues], like=arr_ref),
            (Scale.CHAIN, Scale.MOLECULE): ops.array([n_chains], like=arr_ref),
        }
        marker = backend_marker(arr_ref)
        return cls(per, marker)

    @classmethod
    def from_atoms_per_chain(
        cls,
        atoms_per_chain: Array,
        ref: Array | None = None,
    ) -> "_Hierarchy":
        """
        Create a hierarchy without residue scale (for HeteroAtoms).

        This creates a simplified hierarchy with only ATOM, CHAIN, and MOLECULE
        scales.

        Args:
            atoms_per_chain: Array of atom counts per chain.
            ref: Array to derive backend/device from. If None, uses atoms_per_chain.

        Returns:
            New _Hierarchy without residue scale.
        """
        arr_ref = ref if ref is not None else atoms_per_chain
        n_atoms = int(atoms_per_chain.sum()) if arr_size(atoms_per_chain, 0) > 0 else 0
        n_chains = arr_size(atoms_per_chain, 0)

        # Only include ATOM, CHAIN, MOLECULE scale pairs
        per = {
            (Scale.ATOM, Scale.CHAIN): atoms_per_chain,
            (Scale.ATOM, Scale.MOLECULE): ops.array([n_atoms], like=arr_ref, dtype='int64'),
            (Scale.CHAIN, Scale.MOLECULE): ops.array([n_chains], like=arr_ref, dtype='int64'),
        }
        marker = backend_marker(arr_ref)
        return cls(per, marker)

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
            ValueError: If the requested scale pair is not supported.

        Example:
            >>> hierarchy.per(Scale.RESIDUE, Scale.CHAIN)
            array([150, 200, 175])  # residues per chain
        """
        if inner == outer:
            return ops.ones(self.size(inner), like=self._ref)

        key = (inner, outer)
        if key in self._per:
            return self._per[key]

        raise ValueError(
            f"{inner.name} per {outer.name} not supported by this hierarchy"
        )

    def empty(self) -> bool:
        """Check if the hierarchy has no atoms."""
        return self._n_atoms == 0

    # ─────────────────────────────────────────────────────────────────────────
    # Reduction Operations
    # ─────────────────────────────────────────────────────────────────────────

    def reduce(
        self,
        features: Array,
        to_scale: Scale,
        reduction: Reduction = Reduction.MEAN,
        from_scale: Scale = Scale.ATOM,
    ) -> ReductionResult:
        """
        Reduce features from one scale to a coarser scale.

        Aggregates features within each unit at the output scale
        using the chosen reduction operation.

        Args:
            features: Feature tensor at from_scale.
            to_scale: Target scale to reduce to.
            reduction: Reduction type (MEAN, SUM, MIN, MAX, COLLATE).
            from_scale: Scale of input features (default: ATOM).

        Returns:
            Reduced features. For MIN/MAX, returns (values, indices).

        Examples:
            >>> # Atom -> residue (default)
            >>> res_feats = hierarchy.reduce(atom_feats, Scale.RESIDUE)
            >>> # Residue -> chain (with explicit from_scale)
            >>> chain_feats = hierarchy.reduce(res_feats, Scale.CHAIN, from_scale=Scale.RESIDUE)
            >>> # Chain -> molecule
            >>> mol_feats = hierarchy.reduce(chain_feats, Scale.MOLECULE, from_scale=Scale.CHAIN)

        Note:
            When reducing from ATOM to RESIDUE scale, non-polymer atoms are
            automatically excluded since they don't belong to any residue.
        """
        count = self.size(to_scale)
        sizes = self._per[(from_scale, to_scale)]
        device = getattr(features, 'device', None)
        ix = create_reduction_index(count, sizes, device=device)

        return REDUCTIONS[reduction](features, ix, dim=0, dim_size=count)

    def expand(
        self,
        features: Array,
        from_scale: Scale,
        to_scale: Scale = Scale.ATOM,
    ) -> Array:
        """
        Expand per-scale features to a finer scale.

        Broadcasts values from a coarser scale to a finer scale by
        repeating each value for all units in the finer scale.

        Args:
            features: Per-from_scale feature tensor.
            from_scale: Source scale (coarser).
            to_scale: Destination scale (default: ATOM, finer).

        Returns:
            Expanded feature tensor.
        """
        if to_scale == Scale.ATOM:
            return ops.repeat_interleave(features, self._per[(Scale.ATOM, from_scale)])
        if to_scale == Scale.RESIDUE:
            return ops.repeat_interleave(features, self._per[(Scale.RESIDUE, from_scale)])
        raise ValueError(f"Cannot expand to {to_scale.name}")

    def count(self, mask: Array, scale: Scale) -> Array:
        """
        Count True values in mask per scale unit.

        Args:
            mask: Boolean mask tensor (at ATOM scale).
            scale: Scale at which to count.

        Returns:
            Count tensor with one value per scale unit.
        """
        return self.reduce(ops.to_int64(mask), scale, Reduction.SUM, from_scale=Scale.ATOM)

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
            ValueError: If requested scale is not supported by this hierarchy.
        """
        key = (Scale.ATOM, scale)
        if key not in self._per:
            raise ValueError(f"{scale.name} scale not supported by this hierarchy")
        return self._per[key] != 0

    def derive_masks(
        self,
        input_mask: Array,
        input_scale: Scale,
        remove_empty_residues: bool = False,
    ) -> dict[Scale, Array]:
        """
        Derive masks at all scales from an input mask at a specific scale.

        Args:
            input_mask: Boolean mask at input_scale.
            input_scale: Scale of the input mask (ATOM, RESIDUE, or CHAIN).
            remove_empty_residues: If True, remove residues with 0 atoms
                (ATOM scale behavior).

        Returns:
            Dict mapping Scale to boolean mask. Only contains entries for
            scales that this hierarchy supports.
        """
        if not self.has_scale(input_scale):
            raise ValueError(f"{input_scale.name} scale not supported by this hierarchy")

        masks: dict[Scale, Array] = {}

        # Get ordered list of scales from finest to coarsest (excluding MOLECULE)
        # by checking which keys exist in _per
        scales_present = []
        for scale in [Scale.ATOM, Scale.RESIDUE, Scale.CHAIN]:
            if self.has_scale(scale):
                scales_present.append(scale)

        # Find position of input scale
        input_idx = scales_present.index(input_scale)

        # Set the input mask
        masks[input_scale] = input_mask

        # Expand to finer scales (go backwards from input)
        for i in range(input_idx - 1, -1, -1):
            finer = scales_present[i]
            coarser = scales_present[i + 1]
            masks[finer] = self.expand(masks[coarser], coarser, finer)

        # Derive coarser scales from atom mask
        atom_mask = masks[Scale.ATOM]
        for i in range(1, len(scales_present)):
            scale = scales_present[i]
            if scale in masks:
                continue  # Already set (was input or coarser than input)

            # Count atoms per this scale, check if > 0
            counts = self.count(atom_mask, scale)

            # For residues at ATOM input scale, optionally remove empty
            if scale == Scale.RESIDUE and input_scale == Scale.ATOM and remove_empty_residues:
                masks[scale] = counts > 0
            elif scale == Scale.RESIDUE and input_scale == Scale.ATOM:
                masks[scale] = ops.ones(self.size(scale), like=self._ref, dtype='bool')
            else:
                # For chains: check if any atoms/residues remain
                prev_scale = scales_present[i - 1]
                prev_counts = self.reduce(
                    ops.to_int64(masks[prev_scale]), scale, Reduction.SUM, from_scale=prev_scale
                )
                masks[scale] = prev_counts > 0

        return masks

    def compute_per(
        self,
        masks: dict[Scale, Array],
    ) -> dict[tuple[Scale, Scale], Array]:
        """
        Compute the _per dict for a new Hierarchy after selection.

        Args:
            masks: Dict mapping Scale to boolean mask for that scale.
                Must contain ATOM mask. Other masks are optional based on
                which scales this hierarchy supports.

        Returns:
            New _per dict (only contains supported scale pairs).
        """
        per = {}

        for (inner, outer), arr in self._per.items():
            if outer == Scale.MOLECULE:
                # X per MOLECULE = [count of X remaining]
                if inner == Scale.ATOM:
                    # Total atoms = sum of atom mask
                    per[(inner, outer)] = self.count(masks[Scale.ATOM], Scale.MOLECULE)
                else:
                    # Total of inner scale = size of filtered array
                    n = arr_size(masks[inner], 0) if masks[inner].all() else int(masks[inner].sum())
                    per[(inner, outer)] = ops.array([n], like=self._ref)

            elif inner == Scale.ATOM:
                # ATOM per X = count atoms in each X, then filter by X mask
                counts = self.count(masks[Scale.ATOM], outer)
                per[(inner, outer)] = counts[masks[outer]]

            else:
                # inner per outer (e.g., RESIDUE per CHAIN)
                # = reduce inner mask over outer, then filter by outer mask
                inner_per_outer = self.reduce(
                    ops.to_int64(masks[inner]), outer, Reduction.SUM, from_scale=inner
                )
                per[(inner, outer)] = inner_per_outer[masks[outer]]

        return per

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
        remove_empty = (scale == Scale.ATOM) and self.has_scale(Scale.RESIDUE)
        masks = self.derive_masks(mask, scale, remove_empty)
        new_per = self.compute_per(masks)

        return _Hierarchy(new_per, self._ref)

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
        return _Hierarchy(new_per, new_ref)

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
        return _Hierarchy(new_per, new_ref)

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
        return _Hierarchy(new_per, new_ref)

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
            ValueError: If requested scale is not supported by this hierarchy.
        """
        has_residue = self.has_scale(Scale.RESIDUE)

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
            if not has_residue:
                return atom_slice, None, chain_slice

            res_per_chain = self._per[(Scale.RESIDUE, Scale.CHAIN)]
            res_cumsum = np.cumsum(np.asarray(res_per_chain))
            res_start = 0 if ix == 0 else int(res_cumsum[ix - 1])
            res_end = int(res_cumsum[ix])
            res_slice = slice(res_start, res_end)

            return atom_slice, res_slice, chain_slice

        elif scale == Scale.RESIDUE:
            if not has_residue:
                raise ValueError("RESIDUE scale not supported by this hierarchy")
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
