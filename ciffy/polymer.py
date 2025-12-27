"""
Polymer class representing molecular structures.

The Polymer class provides a unified interface for working with molecular
structures loaded from CIF files. It supports RNA, DNA, proteins, and
other molecular types.
"""

from __future__ import annotations
from typing import Generator, TYPE_CHECKING
from copy import copy

import numpy as np

from .backend import Array, is_torch, get_backend, size as arr_size, check_compatible, to_numpy, Dtype
from .backend import ops
from .biochemistry import Scale, Molecule
from .biochemistry._generated_molecule import molecule_type

if TYPE_CHECKING:
    import torch
from .operations.reduction import Reduction, REDUCTIONS, ReductionResult, create_reduction_index
from .biochemistry import (
    Residue,
    ATOM_NAMES,
    ELEMENT_NAMES,
)
from .utils import all_equal, filter_by_mask
from .utils.formatting import format_chain_table


UNKNOWN = "UNKNOWN"


class _BaseDescriptor:
    """Base class for Polymer field/metadata descriptors."""

    __slots__ = ('scale', 'required', 'is_list', 'name', 'private_name')

    def __init__(
        self,
        scale: Scale,
        required: bool = True,
        is_list: bool = False,
    ):
        self.scale = scale
        self.required = required
        self.is_list = is_list
        self.name = ""
        self.private_name = ""

    def __set_name__(self, owner, name):
        self.name = name
        self.private_name = f"_{name}"

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self  # Class-level access returns descriptor
        value = getattr(obj, self.private_name, None)
        if value is None and self.required:
            raise AttributeError(f"'{self.name}' is not available on this Polymer")
        return value

    def __set__(self, obj, value):
        setattr(obj, self.private_name, value)


class Field(_BaseDescriptor):
    """
    Descriptor for Polymer array fields with automatic slicing and backend conversion.

    Fields describe arrays at different scales (atom, residue, chain) with dtype
    information for backend conversion between NumPy and PyTorch.

    Args:
        scale: Scale at which the field is defined (Scale.ATOM, RESIDUE, CHAIN).
        dtype: Target dtype for backend conversion (Dtype.FLOAT32, INT64, etc.).
        required: Whether the field must have a value (raises AttributeError if None).
        validate: Whether to validate backend/device compatibility on set.

    Example:
        >>> coordinates = Field(Scale.ATOM, dtype=Dtype.FLOAT32)
        >>> bfactors = Field(Scale.ATOM, dtype=Dtype.FLOAT32, required=False)
    """

    __slots__ = ('dtype', 'validate')

    def __init__(
        self,
        scale: Scale,
        dtype: Dtype | None = None,
        required: bool = True,
        validate: bool = True,
    ):
        super().__init__(scale, required, is_list=False)
        self.dtype = dtype
        self.validate = validate

    def __set__(self, obj, value):
        # Validate backend/device compatibility if enabled and reference exists
        if self.validate and value is not None:
            ref = getattr(obj, '_coordinates', None)
            if ref is not None:
                check_compatible(ref, value, self.name)
        setattr(obj, self.private_name, value)

    def __repr__(self):
        return f"Field({self.scale.name}, dtype={self.dtype}, required={self.required})"


class Metadata(_BaseDescriptor):
    """
    Descriptor for Polymer metadata (non-array values passed through without conversion).

    Metadata describes simple values like scalars, strings, or lists that don't need
    backend conversion but still need proper slicing behavior.

    Args:
        scale: Scale at which the metadata is defined (Scale.CHAIN, MOLECULE).
        required: Whether the metadata must have a value.
        is_list: True for Python list fields (names, strands, descriptions).

    Example:
        >>> pdb_id = Metadata(Scale.MOLECULE)
        >>> names = Metadata(Scale.CHAIN, is_list=True)
        >>> descriptions = Metadata(Scale.CHAIN, is_list=True, required=False)
    """

    def __repr__(self):
        return f"Metadata({self.scale.name}, required={self.required}, is_list={self.is_list})"


class Polymer:
    """
    A molecular structure with coordinates, atom types, and hierarchy.

    Represents a complete molecular assembly with multiple scales of
    organization: atoms, residues, chains, and molecules. Provides
    methods for geometric operations, selection, and analysis.

    Atoms are ordered with polymer atoms first [0, polymer_count),
    followed by non-polymer atoms [polymer_count, total). This enables
    efficient slicing instead of boolean masking.

    Attributes:
        coordinates: (N, 3) tensor of atom positions.
        atoms: (N,) tensor of atom type indices.
        elements: (N,) tensor of element indices.
        sequence: (R,) tensor of residue type indices.
        names: List of chain names.
        strands: List of strand identifiers.
        lengths: (C,) tensor of residues per chain.
        polymer_count: Number of polymer atoms (first polymer_count atoms).
        nonpoly: Count of non-polymer atoms (last nonpoly atoms).
    """

    # ─────────────────────────────────────────────────────────────────────────
    # Field Descriptors - arrays with dtype conversion
    # ─────────────────────────────────────────────────────────────────────────

    # Per-atom arrays
    coordinates = Field(Scale.ATOM, dtype=Dtype.FLOAT32)
    atoms = Field(Scale.ATOM, dtype=Dtype.INT64)
    elements = Field(Scale.ATOM, dtype=Dtype.INT64)
    bfactors = Field(Scale.ATOM, dtype=Dtype.FLOAT32, required=False)

    # Per-residue arrays
    sequence = Field(Scale.RESIDUE, dtype=Dtype.INT64)

    # Per-chain arrays
    lengths = Field(Scale.CHAIN, dtype=Dtype.INT64)
    molecule_types = Field(Scale.CHAIN, dtype=Dtype.INT64, required=False, validate=False)

    # ─────────────────────────────────────────────────────────────────────────
    # Metadata Descriptors - values passed through without conversion
    # ─────────────────────────────────────────────────────────────────────────

    # Molecule-level
    pdb_id = Metadata(Scale.MOLECULE)
    polymer_count = Metadata(Scale.MOLECULE)
    resolution = Metadata(Scale.MOLECULE, required=False)

    # Per-chain lists
    names = Metadata(Scale.CHAIN, is_list=True)
    strands = Metadata(Scale.CHAIN, is_list=True)
    descriptions = Metadata(Scale.CHAIN, is_list=True, required=False)

    def __init__(
        self: Polymer,
        coordinates: Array,
        atoms: Array,
        elements: Array,
        sequence: Array,
        sizes: dict[Scale, Array],
        id: str,
        names: list[str],
        strands: list[str],
        lengths: Array,
        polymer_count: int | None = None,
        molecule_types: Array | None = None,
        descriptions: list[str] | None = None,
        bfactors: Array | None = None,
        resolution: float | None = None,
    ) -> None:
        """
        Initialize a Polymer structure.

        Args:
            coordinates: (N, 3) tensor of atom positions.
            atoms: (N,) tensor of atom type indices.
            elements: (N,) tensor of element indices.
            sequence: (R,) tensor of residue type indices.
            sizes: Dict mapping Scale to atom counts per unit.
            id: PDB identifier.
            names: List of chain names.
            strands: List of strand identifiers.
            lengths: (C,) tensor of residues per chain.
            polymer_count: Number of polymer atoms. If None, all atoms
                are assumed to be polymer atoms.
            molecule_types: (C,) array of molecule types per chain from CIF.
                If None, molecule types will be inferred from residue indices.
            descriptions: List of entity descriptions per chain, or None.
            bfactors: (N,) array of B-factors (temperature factors) per atom.
                Higher values indicate greater atomic mobility/disorder.
            resolution: Structure resolution in Angstroms (from _refine.ls_d_res_high).
                None if not available (e.g., NMR structures).

        Raises:
            ValueError: If tensor sizes are inconsistent.
        """
        self.pdb_id = id or UNKNOWN
        self.names = names
        self.strands = strands

        # Store polymer/nonpoly counts
        # If polymer_count is None, assume all atoms are polymer (backward compat)
        total_atoms = arr_size(coordinates, 0)
        if polymer_count is not None:
            self.polymer_count = polymer_count
            self.nonpoly = total_atoms - polymer_count
        else:
            self.polymer_count = total_atoms
            self.nonpoly = 0

        if not all_equal(
            arr_size(coordinates, 0),
            arr_size(atoms, 0),
            arr_size(elements, 0),
        ):
            raise ValueError(
                f"Coordinate, atom, and element tensors must have equal size "
                f"for PDB {self.pdb_id}."
            )

        res_count = sizes[Scale.RESIDUE].sum().item()
        chn_count = sizes[Scale.CHAIN].sum().item()
        mol_count = sizes[Scale.MOLECULE].sum().item()

        if not all_equal(res_count + self.nonpoly, chn_count, mol_count):
            raise ValueError(
                f"Atom counts do not match: residues ({res_count} + {self.nonpoly}), "
                f"chains ({chn_count}), molecule ({mol_count}) for PDB {self.pdb_id}."
            )

        # Store atomic properties
        self._atoms = atoms
        self._elements = elements
        self._sequence = sequence
        self._sizes = sizes
        self._lengths = lengths
        self.molecule_types = molecule_types
        self.descriptions = descriptions
        self._bfactors = bfactors
        self._resolution = resolution

        # Create topology info
        from .backend.graph import TopologyInfo
        self._topology = TopologyInfo.from_polymer(self)

        # Store coordinates directly (no internal coordinate system)
        self._coordinates = coordinates
        self._bonds: np.ndarray | None = None

    # ─────────────────────────────────────────────────────────────────────────
    # Descriptor Helpers - for automatic slicing and backend conversion
    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def _get_fields(cls) -> dict[str, Field]:
        """Return all Field descriptors (arrays needing conversion)."""
        return {
            name: attr for name, attr in vars(cls).items()
            if isinstance(attr, Field)
        }

    @classmethod
    def _get_metadata(cls) -> dict[str, Metadata]:
        """Return all Metadata descriptors (values passed through)."""
        return {
            name: attr for name, attr in vars(cls).items()
            if isinstance(attr, Metadata)
        }

    @classmethod
    def _get_descriptors(cls) -> dict[str, _BaseDescriptor]:
        """Return all descriptors (Field and Metadata)."""
        return {
            name: attr for name, attr in vars(cls).items()
            if isinstance(attr, _BaseDescriptor)
        }

    def _slice_all(
        self,
        atom_mask: Array,
        res_mask: Array,
        chain_mask: Array,
    ) -> dict:
        """
        Slice all descriptor-based attributes according to their scale.

        Args:
            atom_mask: Boolean mask for atoms.
            res_mask: Boolean mask for residues.
            chain_mask: Boolean mask for chains.

        Returns:
            Dict mapping descriptor names to sliced values.
        """
        result = {}
        for name, desc in self._get_descriptors().items():
            value = getattr(self, desc.private_name, None)
            if value is None:
                result[name] = None
            elif desc.scale == Scale.ATOM:
                result[name] = value[atom_mask]
            elif desc.scale == Scale.RESIDUE:
                result[name] = value[res_mask]
            elif desc.scale == Scale.CHAIN:
                if desc.is_list:
                    result[name] = filter_by_mask(value, chain_mask)
                else:
                    result[name] = value[chain_mask]
            else:  # Scale.MOLECULE - scalars, no slicing
                result[name] = value
        return result

    def _derive_masks(
        self: Polymer,
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
            # Count atoms per residue after masking
            res_sizes = self.count(input_mask, Scale.RESIDUE)
            if remove_empty_residues:
                res_mask = res_sizes > 0
            else:
                res_mask = ops.ones(self.size(Scale.RESIDUE), like=self.coordinates, dtype='bool')
            # Derive chain mask from residue mask
            new_lengths = self.rreduce(ops.to_int64(res_mask), Scale.CHAIN, Reduction.SUM)
            chn_mask = new_lengths > 0

        elif input_scale == Scale.RESIDUE:
            res_mask = input_mask
            # Expand residue mask to atoms
            atom_mask = self.expand(res_mask, Scale.RESIDUE, Scale.ATOM)
            # Derive chain mask
            new_lengths = self.rreduce(ops.to_int64(res_mask), Scale.CHAIN, Reduction.SUM)
            chn_mask = new_lengths > 0

        elif input_scale == Scale.CHAIN:
            chn_mask = input_mask
            # Expand to residues and atoms
            res_mask = self.expand(chn_mask, Scale.CHAIN, Scale.RESIDUE)
            atom_mask = self.expand(res_mask, Scale.RESIDUE, Scale.ATOM)

        else:
            raise ValueError(f"Selection not supported at {input_scale.name} scale")

        return atom_mask, res_mask, chn_mask

    def _compute_sizes(
        self: Polymer,
        atom_mask: Array,
        res_mask: Array,
        chn_mask: Array,
        input_scale: Scale,
    ) -> dict[Scale, Array]:
        """
        Compute the sizes dict for the new Polymer.

        Semantics differ by input_scale:
        - ATOM: Count atoms per unit after masking, filter empty residues
        - RESIDUE/CHAIN: Use original sizes, just filter by masks

        Args:
            atom_mask: Boolean mask for atoms.
            res_mask: Boolean mask for residues.
            chn_mask: Boolean mask for chains.
            input_scale: Scale of the original input mask.

        Returns:
            Dict mapping Scale to atom counts per unit.
        """
        if input_scale == Scale.ATOM:
            # Count atoms per unit after masking
            res_sizes_after = self.count(atom_mask, Scale.RESIDUE)
            chn_sizes_after = self.count(atom_mask, Scale.CHAIN)
            mol_sizes = self.count(atom_mask, Scale.MOLECULE)

            return {
                Scale.RESIDUE: res_sizes_after[res_mask],
                Scale.CHAIN: chn_sizes_after[chn_mask],
                Scale.MOLECULE: mol_sizes,
            }
        else:
            # For RESIDUE/CHAIN: use original sizes filtered by masks
            orig_res_sizes = self._sizes[Scale.RESIDUE]

            if input_scale == Scale.RESIDUE:
                # Compute chain sizes by summing masked residue sizes
                masked_res_sizes = orig_res_sizes * ops.to_int64(res_mask)
                chn_sizes = self.rreduce(masked_res_sizes, Scale.CHAIN, Reduction.SUM)
            else:  # CHAIN
                chn_sizes = self._sizes[Scale.CHAIN]

            # Filter residue sizes by mask
            filtered_res_sizes = orig_res_sizes[res_mask]

            # Total atom count for molecule
            total_atoms = filtered_res_sizes.sum().item()

            return {
                Scale.RESIDUE: filtered_res_sizes,
                Scale.CHAIN: chn_sizes[chn_mask],
                Scale.MOLECULE: ops.array([total_atoms], like=self.coordinates),
            }

    def _compute_polymer_count(
        self: Polymer,
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
            return atom_mask[:self.polymer_count].sum().item()
        else:
            res_sizes = self._sizes[Scale.RESIDUE][res_mask]
            return res_sizes.sum().item()

    def _compute_lengths(
        self: Polymer,
        res_mask: Array,
        chn_mask: Array,
        input_scale: Scale,
    ) -> Array:
        """
        Compute new chain lengths.

        Args:
            res_mask: Boolean mask for residues.
            chn_mask: Boolean mask for chains.
            input_scale: Scale of the original input mask.

        Returns:
            Array of chain lengths.
        """
        if input_scale == Scale.CHAIN:
            return self.lengths[chn_mask]
        else:
            new_lengths = self.rreduce(ops.to_int64(res_mask), Scale.CHAIN, Reduction.SUM)
            return new_lengths[chn_mask]

    def _convert_backend(self, to_func) -> dict:
        """
        Convert all Field arrays to a target backend, pass through Metadata.

        Args:
            to_func: Function to convert arrays (e.g., to_numpy, to_torch).

        Returns:
            Dict mapping all descriptor names to converted/passed values.
        """
        result = {}

        # Convert Field arrays with dtype
        for name, field in self._get_fields().items():
            value = getattr(self, field.private_name, None)
            if value is None:
                result[name] = None
            elif field.dtype is not None:
                result[name] = to_func(value, dtype=field.dtype)
            else:
                result[name] = to_func(value)

        # Pass through Metadata unchanged (copy lists)
        for name, meta in self._get_metadata().items():
            value = getattr(self, meta.private_name, None)
            if meta.is_list and value is not None:
                result[name] = value.copy()
            else:
                result[name] = value

        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Factory Methods
    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def create_empty(cls, id: str = "empty", backend: str = "numpy") -> "Polymer":
        """
        Create an empty Polymer with 0 atoms and 0 chains.

        Useful as a base case for operations that may produce empty results,
        or for testing edge cases.

        Args:
            id: PDB identifier for the empty polymer.
            backend: Array backend, either "numpy" or "torch".

        Returns:
            An empty Polymer with no atoms, residues, or chains.

        Example:
            >>> empty = Polymer.create_empty()
            >>> empty.size()
            0
            >>> empty.size(Scale.CHAIN)
            0
        """
        polymer = cls(
            coordinates=np.zeros((0, 3), dtype=np.float32),
            atoms=np.array([], dtype=np.int64),
            elements=np.array([], dtype=np.int64),
            sequence=np.array([], dtype=np.int64),
            sizes={
                Scale.RESIDUE: np.array([], dtype=np.int64),
                Scale.CHAIN: np.array([], dtype=np.int64),
                Scale.MOLECULE: np.array([0], dtype=np.int64),
            },
            id=id,
            names=[],
            strands=[],
            lengths=np.array([], dtype=np.int64),
            polymer_count=0,
        )
        return polymer.torch() if backend == "torch" else polymer

    # ─────────────────────────────────────────────────────────────────────────
    # Computed Properties
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def bonds(self) -> np.ndarray:
        """
        Covalent bonds as atom index pairs.

        Returns:
            (B, 2) int64 array where each row [i, j] represents a bond
            between atoms i and j (with i < j).

        Note:
            Computed lazily from topology and cached. Includes both
            intra-residue bonds and inter-residue linkages.
        """
        if self._bonds is None:
            from .backend.graph import build_bond_graph_from_topology
            edges, _ = build_bond_graph_from_topology(self._topology)
            # Filter to i < j to avoid duplicates
            self._bonds = edges[edges[:, 0] < edges[:, 1]]
        return self._bonds

    # ─────────────────────────────────────────────────────────────────────────
    # Identification
    # ─────────────────────────────────────────────────────────────────────────

    def chain_id(self: Polymer, ix: int) -> str:
        """
        Get a unique identifier for a specific chain.

        Args:
            ix: Chain index.

        Returns:
            String combining PDB ID and chain name (e.g., "1ABC_A").
        """
        return f"{self.pdb_id}_{self.names[ix]}"

    def strand_id(self: Polymer, ix: int) -> str:
        """
        Get the strand identifier for a specific chain.

        Args:
            ix: Chain index.

        Returns:
            String combining PDB ID and strand name.
        """
        return f"{self.pdb_id}_{self.strands[ix]}"

    # ─────────────────────────────────────────────────────────────────────────
    # Size and Structure
    # ─────────────────────────────────────────────────────────────────────────

    def empty(self: Polymer) -> bool:
        """Check if the polymer has no atoms."""
        return arr_size(self.coordinates, 0) == 0

    def size(self: Polymer, scale: Scale | None = None) -> int:
        """
        Get the count at a specific scale.

        Args:
            scale: Scale level (ATOM, RESIDUE, CHAIN, MOLECULE).
                   If None, returns atom count.

        Returns:
            Number of units at the specified scale.
        """
        if scale is None:
            return arr_size(self.coordinates, 0)
        return arr_size(self._sizes[scale], 0)

    def sizes(self: Polymer, scale: Scale) -> Array:
        """
        Get the sizes tensor for a scale.

        Args:
            scale: Scale level.

        Returns:
            Tensor of atom counts per unit at this scale.
        """
        return self._sizes[scale]

    def per(self: Polymer, inner: Scale, outer: Scale) -> Array:
        """
        Get the count of inner units per outer unit.

        Args:
            inner: Inner scale (e.g., RESIDUE).
            outer: Outer scale (e.g., CHAIN).

        Returns:
            Array with count of inner units per outer unit.

        Example:
            >>> polymer.per(Scale.RESIDUE, Scale.CHAIN)
            array([150, 200, 175])  # residues per chain
        """
        if inner == outer:
            return ops.ones(self.size(inner), like=self.coordinates)

        # Atoms per {residue, chain, molecule} are stored in _sizes
        if inner == Scale.ATOM:
            return self._sizes[outer]

        # Residues per chain are stored in lengths
        if inner == Scale.RESIDUE and outer == Scale.CHAIN:
            return self.lengths

        # Single-value cases: total count as 1-element array
        if outer == Scale.MOLECULE:
            return ops.array([self.size(inner)], like=self.coordinates)

        raise ValueError(f"Cannot compute {inner.name} per {outer.name}")

    def istype(self: Polymer, mol: Molecule) -> bool:
        """
        Check if this is a single chain of the specified type.

        Args:
            mol: Molecule type to check.

        Returns:
            True if single chain matches type, False otherwise.

        Raises:
            ValueError: If molecule_types is not available on this polymer.
        """
        types = self.molecule_types
        if types is None:
            raise ValueError("Cannot check type: molecule_types not available on this polymer")
        if arr_size(types, 0) != 1:
            return False
        return types[0].item() == mol.value

    # ─────────────────────────────────────────────────────────────────────────
    # Reduction Operations
    # ─────────────────────────────────────────────────────────────────────────

    def reduce(
        self: Polymer,
        features: Array,
        scale: Scale,
        rtype: Reduction = Reduction.MEAN,
    ) -> ReductionResult:
        """
        Reduce per-atom features to per-scale values.

        Aggregates atom-level features within each unit at the specified
        scale using the chosen reduction operation.

        Args:
            features: Per-atom feature tensor.
            scale: Scale at which to aggregate.
            rtype: Reduction type (MEAN, SUM, MIN, MAX, COLLATE).

        Returns:
            Reduced features. For MIN/MAX, returns (values, indices).

        Note:
            When reducing to RESIDUE scale, non-polymer atoms are excluded
            since they don't belong to any residue.
        """
        # Non-polymer atoms don't belong to residues, so slice them out
        # when reducing to RESIDUE scale. With reordered atoms, polymer
        # atoms are always first [0, polymer_count), so we can use simple slicing.
        if scale == Scale.RESIDUE and self.nonpoly > 0:
            features = features[:self.polymer_count]

        count = self.size(scale)
        sizes = self._sizes[scale]
        # Pass device to ensure index is on same device as features
        device = getattr(features, 'device', None)
        ix = create_reduction_index(count, sizes, device=device)

        return REDUCTIONS[rtype](features, ix, dim=0, dim_size=count)

    def rreduce(
        self: Polymer,
        features: Array,
        scale: Scale,
        rtype: Reduction = Reduction.MEAN,
    ) -> ReductionResult:
        """
        Reduce per-residue features to per-scale values.

        Like reduce(), but for features with one value per residue
        instead of per atom.

        Args:
            features: Per-residue feature tensor.
            scale: Scale at which to aggregate.
            rtype: Reduction type.

        Returns:
            Reduced features.
        """
        count = self.size(scale)
        # Pass device to ensure index is on same device as features
        device = getattr(features, 'device', None)
        ix = create_reduction_index(count, self.lengths, device=device)

        return REDUCTIONS[rtype](features, ix, dim=0, dim_size=count)

    def expand(
        self: Polymer,
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
        # Device mismatch is handled by ops.repeat_interleave
        if dest == Scale.ATOM:
            return ops.repeat_interleave(features, self._sizes[source])
        if dest == Scale.RESIDUE:
            return ops.repeat_interleave(features, self.lengths)
        raise ValueError(f"Cannot expand to {dest.name}")

    def count(
        self: Polymer,
        mask: Array,
        scale: Scale,
    ) -> Array:
        """
        Count True values in mask per scale unit.

        Args:
            mask: Boolean mask tensor.
            scale: Scale at which to count.

        Returns:
            Count tensor with one value per scale unit.
        """
        return self.reduce(ops.to_int64(mask), scale, Reduction.SUM)

    def index(self: Polymer, scale: Scale) -> Array:
        """
        Get the index of each atom within units at the specified scale.

        Creates an integer array where each atom is labeled with its
        containing unit's index at the given scale. Useful for positional
        encodings, attention masking, and grouping operations.

        Args:
            scale: Scale at which to compute indices.
                - RESIDUE: atom -> residue index (0 to num_residues-1)
                - CHAIN: atom -> chain index (0 to num_chains-1)
                - MOLECULE: all atoms get index 0

        Returns:
            Integer array of shape (num_atoms,) with indices.

        Examples:
            >>> polymer = ciffy.load("structure.cif")
            >>> res_idx = polymer.index(Scale.RESIDUE)  # atom -> residue
            >>> chain_idx = polymer.index(Scale.CHAIN)  # atom -> chain

            # Use for attention masking (same-residue attention)
            >>> mask = res_idx[:, None] == res_idx[None, :]
        """
        n = self.size(scale)
        idx = ops.arange(n, like=self.coordinates)
        return self.expand(idx, scale, Scale.ATOM)

    # ─────────────────────────────────────────────────────────────────────────
    # Geometry Operations
    # ─────────────────────────────────────────────────────────────────────────

    def center(
        self: Polymer,
        scale: Scale = Scale.MOLECULE,
    ) -> tuple[Polymer, Array]:
        """
        Center coordinates at the specified scale.

        Subtracts the centroid of each unit at the specified scale
        from all atoms in that unit.

        Args:
            scale: Scale at which to center.

        Returns:
            Tuple of (centered polymer, centroid positions).
        """
        means = self.reduce(self.coordinates, scale)
        expanded = self.expand(means, scale)
        coordinates = self.coordinates - expanded

        centered = copy(self)
        centered._coordinates = coordinates
        centered._bonds = None  # Clear cached bonds

        return centered, means

    def scale(
        self: Polymer,
        scale: Scale = Scale.MOLECULE,
        size: float = 1.0,
    ) -> tuple[Polymer, Array]:
        """
        Center and scale coordinates at the specified scale.

        Centers each unit at the specified scale, then scales coordinates
        so that each unit has standard deviation equal to `size`.

        This is useful for normalizing coordinates before statistical
        learning, ensuring consistent scale across different residues
        or molecules.

        Args:
            scale: Scale at which to center and scale.
            size: Target standard deviation for each unit. Default 1.0
                gives unit variance.

        Returns:
            Tuple of (scaled polymer, standard deviations before scaling).
        """
        # Center first
        centered, _ = self.center(scale)

        # Compute std per unit: sqrt(mean(x^2)) since already centered
        sq = centered.coordinates ** 2
        var = self.reduce(sq, scale).mean(axis=-1, keepdims=True)  # (n_units, 1)

        # Backend-agnostic sqrt and clamp
        std = ops.sqrt(var)
        std = ops.clamp(std, min_val=1e-8)

        # Scale coordinates
        std_expanded = self.expand(std, scale)
        coordinates = centered.coordinates / std_expanded * size

        scaled = copy(centered)
        scaled.coordinates = coordinates

        return scaled, std

    def pairwise_distances(self: Polymer, scale: Scale | None = None) -> Array:
        """
        Compute pairwise distances.

        If scale is provided, computes distances between centroids
        at that scale. Otherwise, computes atom-atom distances.

        Args:
            scale: Optional scale for centroid distances.

        Returns:
            Pairwise distance matrix.
        """
        if scale is None or scale == Scale.ATOM:
            coords = self.coordinates
        else:
            coords = self.reduce(self.coordinates, scale)

        return ops.cdist(coords, coords)

    def knn(self: Polymer, k: int, scale: Scale = Scale.ATOM) -> Array:
        """
        Find k-nearest neighbors at the specified scale.

        Args:
            k: Number of neighbors per point (excluding self).
            scale: Scale at which to compute (ATOM, RESIDUE, CHAIN).

        Returns:
            Tensor of shape (k, N) where N = size at scale.
            Entry [i, j] is the index of j's i-th nearest neighbor.

        Example:
            >>> p = ciffy.load("structure.cif", backend="torch")
            >>> neighbors = p.knn(k=16, scale=Scale.ATOM)  # (16, num_atoms)
            >>> # Convert to edge_index for PyG:
            >>> src = torch.arange(p.size()).repeat_interleave(16)
            >>> dst = neighbors.flatten()
            >>> edge_index = torch.stack([src, dst])
        """
        # Compute pairwise distances at the given scale
        if scale == Scale.ATOM:
            dists = self.pairwise_distances()
        else:
            dists = self.pairwise_distances(scale)

        n = dists.shape[0]
        if k >= n:
            raise ValueError(f"k={k} must be less than number of points ({n})")

        # Use topk to find k+1 smallest (includes self at distance 0)
        _, indices = ops.topk(dists, k + 1, dim=1, largest=False)
        # Exclude self (first column) and transpose to (k, N)
        return indices[:, 1:].T

    def _pc(
        self: Polymer,
        scale: Scale,
    ) -> tuple[Array, Array]:
        """
        Compute principal components at the specified scale.

        Args:
            scale: Scale at which to compute.

        Returns:
            Tuple of (eigenvalues, eigenvectors).

        Note:
            Principal components are only defined up to sign.
            Use align() for stable, unique orientations.
        """
        cov = self.coordinates[:, None, :] * self.coordinates[:, :, None]
        cov = self.reduce(cov, scale)
        return ops.eigh(cov)

    def align(
        self: Polymer,
        scale: Scale,
    ) -> tuple[Polymer, Array]:
        """
        Align structure to principal axes at the specified scale.

        Centers the structure and rotates it so that the covariance
        matrix is diagonal. Signs are chosen so that the largest
        two third moments are positive.

        Args:
            scale: Scale at which to align.

        Returns:
            Tuple of (aligned polymer, rotation matrices Q).
        """
        aligned, _ = self.center(scale)
        _, Q = aligned._pc(scale)

        Q_exp = aligned.expand(Q, scale)
        aligned.coordinates = (
            Q_exp @ aligned.coordinates[..., None]
        ).squeeze()

        # Ensure stability by fixing signs based on third moments
        signs = ops.sign(aligned.moment(3, scale))
        signs[:, 0] = signs[:, 1] * signs[:, 2] * ops.det(Q)
        signs_exp = aligned.expand(signs, scale)

        aligned.coordinates = aligned.coordinates * signs_exp
        Q = Q * signs[..., None]

        return aligned, Q

    def moment(
        self: Polymer,
        n: int,
        scale: Scale,
    ) -> Array:
        """
        Compute the n-th moment of coordinates at a scale.

        Args:
            n: Moment order (1=mean, 2=variance, 3=skewness).
            scale: Scale at which to compute.

        Returns:
            Moment tensor with one value per scale unit per dimension.
        """
        return self.reduce(self.coordinates ** n, scale)

    # ─────────────────────────────────────────────────────────────────────────
    # Selection Operations
    # ─────────────────────────────────────────────────────────────────────────

    def mask(
        self: Polymer,
        indices: Array | int,
        source: Scale,
        dest: Scale = Scale.ATOM,
    ) -> Array:
        """
        Create a boolean mask selecting specific units.

        Args:
            indices: Indices of units to select.
            source: Scale of the indices.
            dest: Scale of the output mask.

        Returns:
            Boolean array at dest scale.
        """
        from .selection import mask
        return mask(self, indices, source, dest)

    def _select(self: Polymer, mask: Array, scale: Scale) -> Polymer:
        """
        Unified selection implementation for all scales.

        Uses helper methods to derive masks, slice fields, and compute
        scale-specific metadata (sizes, polymer_count, lengths).

        Args:
            mask: Boolean mask at the specified scale.
            scale: Scale of the input mask (ATOM, RESIDUE, or CHAIN).

        Returns:
            New Polymer with selected units.
        """
        # Step 1: Derive masks at all scales
        remove_empty = (scale == Scale.ATOM)
        atom_mask, res_mask, chn_mask = self._derive_masks(mask, scale, remove_empty)

        # Step 2: Slice all fields using existing _slice_all
        sliced = self._slice_all(atom_mask, res_mask, chn_mask)

        # Step 3: Compute sizes dict
        sizes = self._compute_sizes(atom_mask, res_mask, chn_mask, scale)

        # Step 4: Compute polymer_count
        new_polymer_count = self._compute_polymer_count(atom_mask, res_mask, scale)

        # Step 5: Compute lengths
        lengths = self._compute_lengths(res_mask, chn_mask, scale)

        # Step 6: Construct Polymer
        return Polymer(
            coordinates=sliced['coordinates'],
            atoms=sliced['atoms'],
            elements=sliced['elements'],
            sequence=sliced['sequence'],
            sizes=sizes,
            id=sliced['pdb_id'],
            names=sliced['names'],
            strands=sliced['strands'],
            lengths=lengths,
            polymer_count=new_polymer_count,
            molecule_types=sliced['molecule_types'],
            descriptions=sliced['descriptions'],
            bfactors=sliced['bfactors'],
            resolution=sliced['resolution'],
        )

    def select(self: Polymer, mask: Array, scale: Scale) -> Polymer:
        """
        Select units at the specified scale.

        This is the unified selection method that handles different scales
        with appropriate semantics for unresolved (0-atom) residues.

        Args:
            mask: Boolean mask of units to keep at the specified scale.
            scale: Scale of the mask (ATOM, RESIDUE, or CHAIN).

        Returns:
            New Polymer with selected units.

        Semantics by scale:
            - ATOM: Residues with 0 atoms after masking are REMOVED.
            - RESIDUE: Selected residues are KEPT even if they have 0 atoms.
            - CHAIN: All residues in selected chains are KEPT.

        Example:
            >>> # Select atoms (removes empty residues)
            >>> backbone = polymer.select(backbone_mask, Scale.ATOM)
            >>>
            >>> # Select residues (keeps unresolved)
            >>> adenines = polymer.select(polymer.sequence == Residue.A, Scale.RESIDUE)
            >>>
            >>> # Select chains (keeps all residues)
            >>> chain_a = polymer.select(chain_mask, Scale.CHAIN)
        """
        if scale not in (Scale.ATOM, Scale.RESIDUE, Scale.CHAIN):
            raise ValueError(f"Selection not supported at {scale.name} scale")
        return self._select(mask, scale)

    def __getitem__(self: Polymer, key: Array | slice) -> Polymer:
        """
        Select atoms by boolean mask or slice.

        Residues with 0 atoms after masking are removed. For selection
        that preserves unresolved residues, use select() with the
        appropriate scale.

        Args:
            key: Boolean mask of atoms to keep, or slice for contiguous range.

        Returns:
            New Polymer with selected atoms.
        """
        # Handle slice by converting to boolean mask
        if isinstance(key, slice):
            mask = ops.zeros(self.size(), like=self.coordinates, dtype='bool')
            mask[key] = True
            return self.select(mask, Scale.ATOM)

        return self.select(key, Scale.ATOM)

    def by_index(self: Polymer, ix: Array | int) -> Polymer:
        """
        Select chains by index.

        Args:
            ix: Chain index or indices to select.

        Returns:
            New Polymer with selected chains.

        Raises:
            IndexError: If any index is out of range.
        """
        from .selection import by_index
        return by_index(self, ix)

    def by_atom(self: Polymer, name: Array | int) -> Polymer:
        """
        Select atoms by atom type index.

        Args:
            name: Atom type index or indices.

        Returns:
            New Polymer with matching atoms.
        """
        from .selection import by_atom
        return by_atom(self, name)

    def by_residue(self: Polymer, res: Array | int) -> Polymer:
        """
        Select residues by residue type index.

        Args:
            res: Residue type index or indices (from Residue enum).

        Returns:
            New Polymer with matching residues.

        Example:
            >>> from ciffy.biochemistry import Residue
            >>> adenosines = polymer.by_residue(Residue.A)
            >>> purines = polymer.by_residue([Residue.A, Residue.G])
        """
        from .selection import by_residue
        return by_residue(self, res)

    def by_residue_index(self: Polymer, ix: Array | int) -> Polymer:
        """
        Select residues by positional index.

        Unlike by_residue() which selects by residue TYPE (e.g., all adenines),
        this method selects by positional INDEX (e.g., residue 0, 1, 2...).

        Args:
            ix: Residue index or indices (0-indexed position in polymer).

        Returns:
            New Polymer with selected residues.

        Raises:
            IndexError: If any index is out of range.

        Example:
            >>> # Select first residue
            >>> first = polymer.by_residue_index(0)
            >>> # Select residues 0, 2, 4
            >>> subset = polymer.by_residue_index([0, 2, 4])
            >>> # Combine with by_atom to get specific atoms
            >>> from ciffy.biochemistry import Sugar
            >>> first_c5 = polymer.by_residue_index(0).by_atom(Sugar.C5p.index())
        """
        from .selection import by_residue_index
        return by_residue_index(self, ix)

    def canonical(self: Polymer) -> Polymer:
        """
        Filter to canonical residue types only.

        Returns a new Polymer containing only standard residues:
        - 4 RNA nucleotides (A, C, G, U)
        - 4 DNA nucleotides (DA, DC, DG, DT)
        - 20 amino acids

        This removes modified residues, unknown residues, and ligands,
        making the polymer compatible with standard models.

        Returns:
            New Polymer with only canonical residue types.

        Example:
            >>> polymer = ciffy.load("structure.cif").poly().canonical()
            >>> # Now safe to use with flow models
            >>> latents = model.encode(polymer)
        """
        from .biochemistry import CANONICAL_ALL
        return self.by_residue(CANONICAL_ALL)

    def by_type(self: Polymer, mol: Molecule) -> Polymer:
        """
        Select chains by molecule type.

        Args:
            mol: Molecule type to select.

        Returns:
            New Polymer with chains of that type.
        """
        from .selection import by_type
        return by_type(self, mol)

    def poly(self: Polymer) -> Polymer:
        """
        Return polymer portion only (excludes HETATM/non-polymer atoms).

        The returned Polymer has valid residue information and can be used
        with residue-scale operations like reduce(scale=Scale.RESIDUE).

        This is more permissive than `polymer_only()` as it keeps atoms
        with unknown types (useful for modified residues).

        Returns:
            New Polymer with only polymer atoms, or self if no HETATM atoms.

        Example:
            >>> p = load("file.cif")
            >>> rna = p.poly()  # Get polymer only
            >>> rna.reduce(features, Scale.RESIDUE)  # Works correctly
        """
        from .selection import poly
        return poly(self)

    def hetero(self: Polymer) -> Polymer:
        """
        Return non-polymer atoms only (HETATM: water, ions, ligands).

        Warning:
            The returned Polymer has no valid residue information.
            Residue-scale operations like reduce(scale=Scale.RESIDUE)
            will return empty results.

        Returns:
            New Polymer with only HETATM atoms. If there are no HETATM atoms,
            returns a Polymer with 0 atoms.

        Example:
            >>> p = load("file.cif")
            >>> ligands = p.hetero()  # Get waters/ions/ligands
            >>> if not ligands.empty():
            ...     ligands.center(Scale.ATOM)  # Works on atom scale
        """
        from .selection import hetero
        return hetero(self)

    def chains(
        self: Polymer,
        mol: Molecule | None = None,
    ) -> Generator[Polymer, None, None]:
        """
        Iterate over chains, optionally filtered by type.

        Args:
            mol: Optional molecule type filter.

        Yields:
            Individual chain Polymers.
        """
        from .selection import chains
        return chains(self, mol)

    def resolved(self: Polymer, scale: Scale = Scale.RESIDUE) -> Array:
        """
        Get mask of resolved (non-empty) units.

        Args:
            scale: Scale to check.

        Returns:
            Boolean tensor where True indicates resolved units.
        """
        from .selection import resolved
        return resolved(self, scale)

    def strip(self: Polymer, scale: Scale = Scale.RESIDUE) -> Polymer:
        """
        Remove unresolved units at a scale.

        Args:
            scale: Scale at which to strip.

        Returns:
            New Polymer without empty units.
        """
        from .selection import strip
        return strip(self, scale)

    # ─────────────────────────────────────────────────────────────────────────
    # Specialized Selections
    # ─────────────────────────────────────────────────────────────────────────

    def backbone(self: Polymer) -> Polymer:
        """Select backbone atoms (sugar-phosphate for RNA/DNA, N-CA-C-O for protein)."""
        from .selection import backbone
        return backbone(self)

    def nucleobase(self: Polymer) -> Polymer:
        """Select RNA nucleobase atoms."""
        from .selection import nucleobase
        return nucleobase(self)

    def phosphate(self: Polymer) -> Polymer:
        """Select RNA/DNA phosphate atoms."""
        from .selection import phosphate
        return phosphate(self)

    def sidechain(self: Polymer) -> Polymer:
        """Select protein sidechain atoms."""
        from .selection import sidechain
        return sidechain(self)

    # ─────────────────────────────────────────────────────────────────────────
    # Chain Operations
    # ─────────────────────────────────────────────────────────────────────────

    def extend(
        self: Polymer,
        residue: Residue,
        coords: Array | None = None,
        transform: Array | None = None,
    ) -> Polymer:
        """
        Append a residue to the end of a single-chain polymer.

        Creates a new Polymer with an additional residue positioned at the
        C-terminus (proteins) or 3' end (nucleic acids).

        Args:
            residue: Residue type being added (e.g., Residue.ALA, Residue.A).
            coords: Optional (n_atoms, 3) coordinates of the residue to append.
                If None (default), uses the residue's ideal coordinates.
                For custom conformations (e.g., from a flow model), pass
                explicit coordinates.
            transform: Optional (6,) SE(3) transform [axis-angle, translation]
                for positioning. If None, uses linear extension along the
                Z-axis with appropriate spacing.

        Returns:
            New Polymer with the residue appended to the end.

        Raises:
            ValueError: If polymer has multiple chains, has HETATM atoms,
                or lacks required linking atoms.

        Example:
            >>> from ciffy import Residue
            >>> from ciffy.template import from_sequence
            >>>
            >>> # Create initial polymer and extend
            >>> p = from_sequence("ac")
            >>> p = p.extend(Residue.G)
            >>> p = p.extend(Residue.U)
            >>> p.sequence_str()
            'acgu'
            >>>
            >>> # With custom coordinates (e.g., from a flow model)
            >>> custom_coords = model.predict_residue()
            >>> p = p.extend(Residue.A, coords=custom_coords)
        """
        # Use ideal coordinates if none provided
        if coords is None:
            coords = residue.ideal
            # Convert to backend if needed
            if self.backend == "torch":
                import torch
                coords = torch.from_numpy(coords).to(
                    dtype=self.coordinates.dtype,
                    device=self.coordinates.device
                )
        from .geometry import position_residue
        from .biochemistry.linking import LINKING_BY_TYPE
        from .utils import atoms_to_col_map

        # Validate single chain and poly-only
        if self.size(Scale.CHAIN) != 1:
            raise ValueError(
                f"extend() requires a single-chain polymer. "
                f"Got {self.size(Scale.CHAIN)} chains."
            )
        if self.nonpoly > 0:
            raise ValueError(
                "extend() requires a poly-only polymer (no HETATM atoms). "
                "Use polymer.poly() first."
            )

        # Validate backend compatibility
        check_compatible(self.coordinates, coords, "coords")

        # Get last residue's state
        n_residues = self.size(Scale.RESIDUE)
        last_res_idx = n_residues - 1
        last_res_type = Residue.from_index(self.sequence[last_res_idx].item())

        # Compute atom offset for last residue
        res_sizes = self._sizes[Scale.RESIDUE]
        atom_offset = res_sizes[:last_res_idx].sum().item()
        last_res_n_atoms = res_sizes[last_res_idx].item()

        # Extract last residue's coordinates and build atom_to_col
        last_res_coords = self.coordinates[atom_offset:atom_offset + last_res_n_atoms]
        last_res_atoms = self.atoms[atom_offset:atom_offset + last_res_n_atoms]
        last_res_atom_to_col = atoms_to_col_map(last_res_atoms)

        # Build atom_to_col for new residue
        # The new residue's atoms must correspond to the coords columns
        new_res_atoms = [a.value for a in residue.atoms] if residue.atoms else []
        if len(new_res_atoms) != coords.shape[0]:
            raise ValueError(
                f"Coordinate shape {coords.shape} doesn't match residue {residue.name} "
                f"which has {len(new_res_atoms)} atoms."
            )
        new_res_atom_to_col = atoms_to_col_map(new_res_atoms)

        # Get linking definition
        link_def = LINKING_BY_TYPE.get(last_res_type.molecule_type)
        if link_def is None:
            raise ValueError(
                f"No linking definition for molecule type {last_res_type.molecule_type}. "
                f"Cannot extend chains of this type."
            )

        # Position the new residue
        positioned_coords = position_residue(
            prev_coords=last_res_coords,
            next_coords=coords,
            prev_atom_to_col=last_res_atom_to_col,
            next_atom_to_col=new_res_atom_to_col,
            prev_residue=last_res_type,
            next_residue=residue,
            transform=transform,
        )

        # Build element indices for new residue
        def atom_name_to_element(name: str) -> int:
            element_map = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'P': 15, 'S': 16}
            return element_map.get(name[0].upper(), 0)

        new_elements = [atom_name_to_element(a.name) for a in residue.atoms]

        # Concatenate arrays
        new_coords = ops.cat([self.coordinates, positioned_coords], axis=0)
        new_atoms_arr = ops.cat([
            self.atoms,
            ops.to_backend(np.array(new_res_atoms, dtype=np.int64), self.atoms)
        ], axis=0)
        new_elements_arr = ops.cat([
            self.elements,
            ops.to_backend(np.array(new_elements, dtype=np.int64), self.elements)
        ], axis=0)
        new_sequence = ops.cat([
            self.sequence,
            ops.to_backend(np.array([residue.value], dtype=np.int64), self.sequence)
        ], axis=0)

        # Update sizes
        n_new_atoms = len(new_res_atoms)
        new_res_sizes = ops.cat([
            self._sizes[Scale.RESIDUE],
            ops.to_backend(np.array([n_new_atoms], dtype=np.int64), self._sizes[Scale.RESIDUE])
        ], axis=0)
        # Chain size increases by new atoms
        new_chn_sizes = ops.to_backend(
            np.array([self._sizes[Scale.CHAIN][0].item() + n_new_atoms], dtype=np.int64),
            self._sizes[Scale.CHAIN]
        )
        new_mol_sizes = ops.to_backend(
            np.array([self.size() + n_new_atoms], dtype=np.int64),
            self._sizes[Scale.MOLECULE]
        )

        sizes = {
            Scale.RESIDUE: new_res_sizes,
            Scale.CHAIN: new_chn_sizes,
            Scale.MOLECULE: new_mol_sizes,
        }

        # Update lengths
        new_lengths = ops.to_backend(
            np.array([self.lengths[0].item() + 1], dtype=np.int64),
            self.lengths
        )

        return Polymer(
            coordinates=new_coords,
            atoms=new_atoms_arr,
            elements=new_elements_arr,
            sequence=new_sequence,
            sizes=sizes,
            id=self.pdb_id,
            names=list(self.names),
            strands=list(self.strands),
            lengths=new_lengths,
            polymer_count=self.polymer_count + n_new_atoms,
            molecule_types=ops.clone(self._molecule_types) if self._molecule_types is not None else None,
            descriptions=list(self.descriptions) if self.descriptions else None,
            # bfactors not preserved - new atoms don't have experimental B-factors
            resolution=self._resolution,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # String Representations
    # ─────────────────────────────────────────────────────────────────────────

    def sequence_str(self: Polymer) -> str:
        """
        Get the sequence as a single-letter string.

        Returns:
            Single-letter sequence string (e.g., "ACGU" for RNA,
            "MGKLV" for protein).
        """
        def abbrev(x: int) -> str:
            try:
                return Residue.from_index(x).abbrev
            except (ValueError, KeyError):
                return 'n'
        return "".join(abbrev(ix.item()) for ix in self.sequence)

    def atom_names(self: Polymer) -> list[str]:
        """
        Get atom names as a list of strings.

        Returns:
            List of atom name strings.
        """
        return [ATOM_NAMES.get(ix.item(), '?') for ix in self.atoms]

    def chain_info(self: Polymer) -> list[dict]:
        """
        Get information about each chain.

        Returns:
            List of dicts with keys: 'chain', 'type', 'res', 'atoms'.
        """
        types_np = to_numpy(self.molecule_types) if self.molecule_types is not None else None
        lengths_np = to_numpy(self.lengths)
        atoms_np = to_numpy(self._sizes[Scale.CHAIN])
        elements_np = to_numpy(self.elements)

        rows = []
        atom_offset = 0
        for ix in range(self.size(Scale.CHAIN)):
            mol = molecule_type(int(types_np[ix])) if types_np is not None else Molecule.UNKNOWN
            res = int(lengths_np[ix])
            atoms = int(atoms_np[ix])

            # For ION chains, show element name (e.g., "MG ION")
            type_str = mol.name
            if mol == Molecule.ION and atoms > 0:
                elem_idx = int(elements_np[atom_offset])
                elem_name = ELEMENT_NAMES.get(elem_idx, "")
                if elem_name:
                    type_str = f"{elem_name} {mol.name}"

            rows.append({
                'chain': self.names[ix],
                'type': type_str,
                'res': res,
                'atoms': atoms,
            })
            atom_offset += atoms

        return rows

    def __repr__(self: Polymer) -> str:
        """String representation with structure summary."""
        rows = self.chain_info()
        return format_chain_table(self.pdb_id, self.backend, rows)

    # ─────────────────────────────────────────────────────────────────────────
    # Backend Conversion
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def backend(self: Polymer) -> str:
        """
        Get the array backend type.

        Returns:
            'numpy' if arrays are NumPy, 'torch' if PyTorch tensors.
        """
        from .backend import get_backend
        return get_backend(self.coordinates).value

    @property
    def device(self: Polymer) -> str | None:
        """
        Get the device of the polymer's arrays.

        Returns:
            Device string (e.g., 'cpu', 'cuda:0', 'mps:0') for PyTorch tensors,
            None for NumPy arrays.
        """
        from .backend import get_device
        return get_device(self.coordinates)

    def numpy(self: Polymer) -> Polymer:
        """
        Convert all arrays to NumPy.

        Returns:
            New Polymer with NumPy arrays. If already NumPy, returns self.
        """
        from .backend import is_numpy
        if is_numpy(self.coordinates):
            return self

        # Create new polymer with converted arrays
        return Polymer(
            coordinates=to_numpy(self.coordinates),
            atoms=to_numpy(self.atoms),
            elements=to_numpy(self.elements),
            sequence=to_numpy(self.sequence),
            sizes={k: to_numpy(v) for k, v in self._sizes.items()},
            id=self.pdb_id,
            names=self.names.copy(),
            strands=self.strands.copy(),
            lengths=to_numpy(self.lengths),
            polymer_count=self.polymer_count,
            molecule_types=to_numpy(self._molecule_types) if self._molecule_types is not None else None,
            bfactors=to_numpy(self._bfactors) if self._bfactors is not None else None,
            resolution=self._resolution,
        )

    def torch(self: Polymer) -> Polymer:
        """
        Convert all arrays to PyTorch tensors.

        Returns:
            New Polymer with PyTorch tensors. If already PyTorch, returns self.

        Raises:
            ImportError: If PyTorch is not installed.
        """
        from .backend import to_torch, is_torch
        if is_torch(self.coordinates):
            return self

        # Create new polymer with converted arrays
        return Polymer(
            coordinates=to_torch(self.coordinates).float(),
            atoms=to_torch(self.atoms).long(),
            elements=to_torch(self.elements).long(),
            sequence=to_torch(self.sequence).long(),
            sizes={k: to_torch(v).long() for k, v in self._sizes.items()},
            id=self.pdb_id,
            names=self.names.copy(),
            strands=self.strands.copy(),
            lengths=to_torch(self.lengths).long(),
            polymer_count=self.polymer_count,
            molecule_types=to_torch(self._molecule_types).long() if self._molecule_types is not None else None,
            bfactors=to_torch(self._bfactors).float() if self._bfactors is not None else None,
            resolution=self._resolution,
        )

    def to(self: Polymer, device=None, dtype=None) -> Polymer:
        """
        Move tensors to device and/or convert dtype (torch backend only).

        Args:
            device: Target device (e.g., 'cuda', 'cpu', torch.device).
            dtype: Target dtype for float tensors only (e.g., torch.float16).
                   Integer tensors (atoms, elements, sequence, etc.) remain long.

        Returns:
            New Polymer with tensors on the specified device/dtype.
            Returns self if no changes needed.

        Raises:
            ValueError: If called on NumPy backend.

        Example:
            >>> p = load("file.cif", backend="torch")
            >>> p_gpu = p.to("cuda")
            >>> p_fp16 = p.to(dtype=torch.float16)
            >>> p_gpu_fp16 = p.to("cuda", torch.float16)
        """
        from .backend import is_torch
        if not is_torch(self.coordinates):
            raise ValueError("to() is only supported for torch backend. "
                           "Use polymer.torch().to(...) to convert first.")

        if device is None and dtype is None:
            return self

        # For coordinates (float), apply both device and dtype
        coords = self.coordinates
        if device is not None:
            coords = coords.to(device)
        if dtype is not None:
            coords = coords.to(dtype)

        # For integer tensors, only apply device (keep as long)
        def move_int(t):
            return t.to(device) if device is not None else t

        # For float tensors, apply device and dtype
        def move_float(t):
            if t is None:
                return None
            result = t
            if device is not None:
                result = result.to(device)
            if dtype is not None:
                result = result.to(dtype)
            return result

        # Create new polymer with moved arrays
        return Polymer(
            coordinates=coords,
            atoms=move_int(self.atoms),
            elements=move_int(self.elements),
            sequence=move_int(self.sequence),
            sizes={k: move_int(v) for k, v in self._sizes.items()},
            id=self.pdb_id,
            names=self.names.copy(),
            strands=self.strands.copy(),
            lengths=move_int(self.lengths),
            polymer_count=self.polymer_count,
            bfactors=move_float(self._bfactors),
            resolution=self._resolution,
        )

    def cuda(self: Polymer) -> Polymer:
        """
        Move tensors to CUDA device (torch backend only).

        Shorthand for `polymer.to("cuda")`.

        Returns:
            New Polymer with tensors on CUDA device.

        Raises:
            ValueError: If called on NumPy backend.
            RuntimeError: If CUDA is not available.

        Example:
            >>> p = load("file.cif", backend="torch")
            >>> p_gpu = p.cuda()
        """
        return self.to("cuda")

    def cpu(self: Polymer) -> Polymer:
        """
        Move tensors to CPU (torch backend only).

        Shorthand for `polymer.to("cpu")`.

        Returns:
            New Polymer with tensors on CPU.

        Raises:
            ValueError: If called on NumPy backend.

        Example:
            >>> p_gpu = load("file.cif", backend="torch").cuda()
            >>> p_cpu = p_gpu.cpu()
        """
        return self.to("cpu")

    def detach(self: Polymer) -> Polymer:
        """
        Detach all tensors from their computation graphs (torch backend only).

        Detaches the coordinate tensor from its computation graph. For NumPy
        arrays, this is a no-op since NumPy doesn't have computation graphs.

        Returns:
            Self, for method chaining.

        Example:
            >>> coords = polymer.coordinates.clone().requires_grad_(True)
            >>> polymer.coordinates = coords
            >>> loss = polymer.coordinates.sum()
            >>> loss.backward()
            >>> polymer.detach()
        """
        if is_torch(self._coordinates) and self._coordinates is not None:
            self._coordinates = self._coordinates.detach()
        return self

    # ─────────────────────────────────────────────────────────────────────────
    # I/O
    # ─────────────────────────────────────────────────────────────────────────

    def write(self: Polymer, filename: str) -> None:
        """
        Write structure to an mmCIF file.

        Supports all molecule types (protein, RNA, DNA) and includes
        both polymer and non-polymer atoms.

        Args:
            filename: Output file path (must have .cif extension).

        Raises:
            ValueError: If filename does not end with .cif extension,
                or if the polymer is empty.

        Example:
            >>> polymer = ciffy.load("structure.cif", backend="numpy")
            >>> polymer.write("output.cif")
        """
        if self.empty():
            raise ValueError("Cannot write empty polymer to CIF file")
        if not filename.lower().endswith('.cif'):
            raise ValueError(
                f"Output file must have .cif extension, got: {filename!r}"
            )
        from .io.writer import write_cif
        write_cif(self, filename)

    # ─────────────────────────────────────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────────────────────────────────────

    def with_coordinates(self: Polymer, coordinates: Array) -> Polymer:
        """
        Create a copy with new coordinates.

        Args:
            coordinates: New coordinate tensor. Must match the polymer's
                backend and device.

        Returns:
            New Polymer with updated coordinates.

        Raises:
            TypeError: If backend doesn't match.
            ValueError: If device doesn't match (for PyTorch tensors).
        """
        # Validate backend and device compatibility
        check_compatible(self.coordinates, coordinates, "coordinates")

        result = copy(self)
        result._coordinates = coordinates
        result._bonds = None  # Clear cached bonds
        return result
