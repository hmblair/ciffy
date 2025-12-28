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

from ..backend import Array, is_torch, get_backend, size as arr_size, check_compatible, to_numpy, Dtype
from ..backend import ops
from ..biochemistry import Scale, Molecule
from ..biochemistry._generated_molecule import molecule_type

if TYPE_CHECKING:
    import torch
    from ..hetero import HeteroAtoms
from ..operations.reduction import Reduction, REDUCTIONS, ReductionResult, create_reduction_index
from .hierarchy import _Hierarchy
from ..biochemistry import (
    Residue,
    Atom,
    ELEMENT_NAMES,
)
from ..utils import all_equal, filter_by_mask
from ..utils.formatting import format_chain_table


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
        dtype: Data type category (Dtype.FLOAT or Dtype.INT). Precision is
            preserved from the source array during backend conversion.
        required: Whether the field must have a value (raises AttributeError if None).
        validate: Whether to validate backend/device compatibility on set.

    Example:
        >>> coordinates = Field(Scale.ATOM, dtype=Dtype.FLOAT)
        >>> bfactors = Field(Scale.ATOM, dtype=Dtype.FLOAT, required=False)
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
        nonpoly: Count of non-polymer atoms (computed property).
    """

    # ─────────────────────────────────────────────────────────────────────────
    # Field Descriptors - arrays with dtype conversion
    # ─────────────────────────────────────────────────────────────────────────

    # Per-atom arrays
    coordinates = Field(Scale.ATOM, dtype=Dtype.FLOAT)
    atoms = Field(Scale.ATOM, dtype=Dtype.INT)
    elements = Field(Scale.ATOM, dtype=Dtype.INT)
    bfactors = Field(Scale.ATOM, dtype=Dtype.FLOAT, required=False)

    # Per-residue arrays
    sequence = Field(Scale.RESIDUE, dtype=Dtype.INT)

    # Per-chain arrays (lengths is handled by hierarchy, not a descriptor)
    molecule_types = Field(Scale.CHAIN, dtype=Dtype.INT, required=False, validate=False)

    # ─────────────────────────────────────────────────────────────────────────
    # Metadata Descriptors - values passed through without conversion
    # ─────────────────────────────────────────────────────────────────────────

    # Molecule-level (polymer_count is a property delegated to hierarchy)
    pdb_id = Metadata(Scale.MOLECULE, required=False)
    resolution = Metadata(Scale.MOLECULE, required=False)

    # Per-chain lists
    names = Metadata(Scale.CHAIN, is_list=True)
    strands = Metadata(Scale.CHAIN, is_list=True)
    descriptions = Metadata(Scale.CHAIN, is_list=True, required=False)

    def __init__(
        self: Polymer,
        hierarchy: _Hierarchy,
        **fields,
    ) -> None:
        """
        Initialize a Polymer structure.

        All Field and Metadata descriptors defined on the class can be passed
        as keyword arguments. Required fields (coordinates, atoms, elements,
        sequence, names, strands) must be provided.

        Args:
            hierarchy: _Hierarchy object containing scale bookkeeping.
            **fields: Field and Metadata values matching class descriptors:
                - coordinates: (N, 3) array of atom positions.
                - atoms: (N,) array of atom type indices.
                - elements: (N,) array of element indices.
                - sequence: (R,) array of residue type indices.
                - pdb_id: PDB identifier string (optional).
                - names: List of chain names.
                - strands: List of strand identifiers.
                - molecule_types: (C,) array of molecule types per chain.
                - descriptions: List of entity descriptions per chain.
                - bfactors: (N,) array of B-factors per atom.
                - resolution: Structure resolution in Angstroms.

        Raises:
            TypeError: If required fields are missing or unknown fields provided.
            ValueError: If field sizes are inconsistent.
        """
        # Assign all descriptor fields from kwargs
        missing = []
        for name, desc in self._get_descriptors().items():
            if name in fields:
                value = fields.pop(name)
                setattr(self, desc.private_name, value)
            elif desc.required:
                missing.append(name)
            else:
                setattr(self, desc.private_name, None)

        if missing:
            raise TypeError(
                f"__init__() missing required keyword arguments: {missing}"
            )
        if fields:
            raise TypeError(
                f"__init__() got unexpected keyword arguments: {list(fields.keys())}"
            )

        self._hierarchy = hierarchy
        self._bonds: np.ndarray | None = None

    @property
    def lengths(self) -> Array:
        """Residues per chain (C,) array. Delegated to hierarchy."""
        return self._hierarchy.lengths

    @property
    def polymer_count(self) -> int:
        """Number of polymer atoms. Delegated to hierarchy."""
        return self._hierarchy.polymer_count

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
        return self._hierarchy.derive_masks(input_mask, input_scale, remove_empty_residues)

    def _convert_backend(self, to_func) -> dict:
        """
        Convert all Field arrays to a target backend, pass through Metadata.

        Args:
            to_func: Function to convert arrays (e.g., to_numpy, to_torch).

        Returns:
            Dict mapping all descriptor names to converted/passed values.
        """
        result = {}

        # Convert Field arrays (precision preserved from source)
        for name, field in self._get_fields().items():
            value = getattr(self, field.private_name, None)
            if value is None:
                result[name] = None
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

    def _residue_slice(
        self: Polymer,
        idx: int,
    ) -> tuple[Array, Array, dict[int, int], Residue]:
        """
        Extract coordinates, atoms, atom_to_col, and residue type for a residue.

        This is a helper for methods that need to work with individual residue
        data, such as extend() and geometry operations.

        Args:
            idx: Residue index. Negative indices are supported (e.g., -1 for last).

        Returns:
            Tuple of:
            - coords: (n_atoms, 3) coordinates for this residue
            - atoms: (n_atoms,) atom type indices
            - atom_to_col: dict mapping atom type value to column index
            - residue: Residue enum for this residue type

        Raises:
            IndexError: If idx is out of range.

        Example:
            >>> coords, atoms, atom_to_col, res_type = polymer._residue_slice(-1)
            >>> # Get last residue's P atom position
            >>> p_col = atom_to_col[res_type.P.value]
            >>> p_pos = coords[p_col]
        """
        from ..utils import atoms_to_col_map

        n_residues = self.size(Scale.RESIDUE)

        # Handle negative indices
        if idx < 0:
            idx = n_residues + idx
        if idx < 0 or idx >= n_residues:
            raise IndexError(
                f"Residue index {idx} out of range for Polymer with {n_residues} residues"
            )

        # Compute atom offset and size for this residue
        res_sizes = self._sizes[Scale.RESIDUE]
        atom_offset = res_sizes[:idx].sum().item() if idx > 0 else 0
        n_atoms = res_sizes[idx].item()

        # Extract data
        coords = self.coordinates[atom_offset:atom_offset + n_atoms]
        atoms = self.atoms[atom_offset:atom_offset + n_atoms]
        atom_to_col = atoms_to_col_map(atoms)
        residue = Residue.from_index(self.sequence[idx].item())

        return coords, atoms, atom_to_col, residue

    @property
    def _sizes(self) -> dict[Scale, Array]:
        """
        Get sizes dict from hierarchy.

        Returns dict mapping Scale to atoms-per-unit arrays, for compatibility
        with code expecting the old _sizes storage.
        """
        return {
            Scale.RESIDUE: self._hierarchy.sizes(Scale.RESIDUE),
            Scale.CHAIN: self._hierarchy.sizes(Scale.CHAIN),
            Scale.MOLECULE: self._hierarchy.sizes(Scale.MOLECULE),
        }

    def _validate_consistency(self, sizes: dict[Scale, Array]) -> None:
        """
        Validate that field sizes are consistent at each scale and across scales.

        Checks:
        1. All Fields at each scale (ATOM, RESIDUE, CHAIN) have the same size.
        2. Atom counts are consistent across hierarchy (residue + nonpoly = chain = molecule).

        Args:
            sizes: Dict mapping Scale to atom counts per unit.

        Raises:
            ValueError: If field sizes are inconsistent or hierarchy doesn't match.
        """
        # Validate that all Fields at each scale have consistent sizes
        for scale in [Scale.ATOM, Scale.RESIDUE, Scale.CHAIN]:
            field_sizes = []
            field_names = []
            for name, field in self._get_fields().items():
                if field.scale == scale:
                    value = getattr(self, field.private_name, None)
                    if value is not None:
                        field_sizes.append(arr_size(value, 0))
                        field_names.append(name)
            if field_sizes and not all_equal(*field_sizes):
                id_str = f" for PDB {self.pdb_id}" if self.pdb_id else ""
                raise ValueError(
                    f"Fields at {scale.name} scale have inconsistent sizes: "
                    f"{dict(zip(field_names, field_sizes))}{id_str}."
                )

        # Validate hierarchy consistency (atom counts must match across scales)
        res_count = sizes[Scale.RESIDUE].sum().item()
        chn_count = sizes[Scale.CHAIN].sum().item()
        mol_count = sizes[Scale.MOLECULE].sum().item()
        # Compute nonpoly locally (can't use property - hierarchy not yet created)
        nonpoly = mol_count - self.polymer_count

        if not all_equal(res_count + nonpoly, chn_count, mol_count):
            id_str = f" for PDB {self.pdb_id}" if self.pdb_id else ""
            raise ValueError(
                f"Atom counts do not match: residues ({res_count} + {nonpoly}), "
                f"chains ({chn_count}), molecule ({mol_count}){id_str}."
            )

    def _clone(self, **overrides) -> Polymer:
        """
        Create a copy of this Polymer with optional field overrides.

        Collects all descriptor values and sizes, applies overrides, and
        constructs a new Polymer. This is the single place that maps
        descriptor names to constructor parameters.

        Args:
            **overrides: Field values to override. Can include any descriptor
                name (coordinates, atoms, pdb_id, etc.) or 'sizes' for the
                hierarchy sizes dict.

        Returns:
            New Polymer with the specified overrides applied.

        Example:
            >>> # Create copy with new coordinates
            >>> moved = polymer._clone(coordinates=new_coords)
            >>> # Create copy with converted arrays
            >>> converted = polymer._clone(**self._convert_backend(to_numpy))
        """
        # Collect all descriptor values
        data = {}
        for name, desc in self._get_descriptors().items():
            value = getattr(self, desc.private_name, None)
            # Copy lists to avoid mutation
            if desc.is_list and value is not None:
                value = list(value)
            data[name] = value

        # Apply overrides to field data
        data.update(overrides)

        # Extract hierarchy (passed positionally, not in kwargs)
        hierarchy = data.pop('hierarchy', self._hierarchy)

        return Polymer(hierarchy, **data)

    # ─────────────────────────────────────────────────────────────────────────
    # Factory Methods
    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def create_empty(cls, pdb_id: str = "empty", backend: str = "numpy") -> "Polymer":
        """
        Create an empty Polymer with 0 atoms and 0 chains.

        Useful as a base case for operations that may produce empty results,
        or for testing edge cases.

        Args:
            pdb_id: PDB identifier for the empty polymer.
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
        # Create empty arrays
        coordinates = np.zeros((0, 3), dtype=np.float32)
        sizes = {
            Scale.RESIDUE: np.array([], dtype=np.int64),
            Scale.CHAIN: np.array([], dtype=np.int64),
            Scale.MOLECULE: np.array([0], dtype=np.int64),
        }
        lengths = np.array([], dtype=np.int64)

        # Create hierarchy
        hierarchy = _Hierarchy.from_sizes_and_lengths(
            sizes=sizes,
            lengths=lengths,
            polymer_count=0,
            ref=coordinates,
        )

        polymer = cls(
            hierarchy,
            coordinates=coordinates,
            atoms=np.array([], dtype=np.int64),
            elements=np.array([], dtype=np.int64),
            sequence=np.array([], dtype=np.int64),
            pdb_id=pdb_id,
            names=[],
            strands=[],
        )
        return polymer.torch() if backend == "torch" else polymer

    # ─────────────────────────────────────────────────────────────────────────
    # Computed Properties
    # ─────────────────────────────────────────────────────────────────────────

    def nonpoly(self) -> int:
        """Return the number of non-polymer atoms (waters, ions, ligands)."""
        return self._hierarchy.nonpoly

    @property
    def bonds(self) -> np.ndarray:
        """
        Covalent bonds as atom index pairs.

        Returns:
            (B, 2) int64 array where each row [i, j] represents a bond
            between atoms i and j (with i < j).

        Note:
            Computed lazily and cached. Includes both intra-residue bonds
            and inter-residue linkages.
        """
        if self._bonds is None:
            from ..backend.graph import build_bond_graph
            edges, _ = build_bond_graph(self)
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
            String combining PDB ID and chain name (e.g., "1ABC_A"),
            or just the chain name if no PDB ID is set.
        """
        if self.pdb_id is not None:
            return f"{self.pdb_id}_{self.names[ix]}"
        return self.names[ix]

    def strand_id(self: Polymer, ix: int) -> str:
        """
        Get the strand identifier for a specific chain.

        Args:
            ix: Chain index.

        Returns:
            String combining PDB ID and strand name,
            or just the strand name if no PDB ID is set.
        """
        if self.pdb_id is not None:
            return f"{self.pdb_id}_{self.strands[ix]}"
        return self.strands[ix]

    # ─────────────────────────────────────────────────────────────────────────
    # Size and Structure
    # ─────────────────────────────────────────────────────────────────────────

    def empty(self: Polymer) -> bool:
        """Check if the polymer has no atoms."""
        return self._hierarchy.empty()

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
        return self._hierarchy.size(scale)

    def __len__(self: Polymer) -> int:
        """Return the number of atoms."""
        return self.size()

    def counts(self: Polymer, scale: Scale, per: Scale | None = None) -> Array:
        """
        Get counts at a scale, optionally per outer unit.

        Args:
            scale: Scale to count.
            per: Optional outer scale. If provided, returns count of `scale`
                 units per `per` unit.

        Returns:
            Array of counts.

        Examples:
            >>> polymer.counts(Scale.CHAIN)              # atoms per chain
            >>> polymer.counts(Scale.RESIDUE)            # atoms per residue
            >>> polymer.counts(Scale.RESIDUE, per=Scale.CHAIN)  # residues per chain
        """
        if per is None:
            return self._hierarchy.sizes(scale)
        return self._hierarchy.per(scale, per)

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
            >>> res_feats = polymer.reduce(coords, Scale.RESIDUE)
            >>> # Residue -> chain (with explicit in_scale)
            >>> chain_feats = polymer.reduce(res_feats, Scale.CHAIN, in_scale=Scale.RESIDUE)
            >>> # Chain -> molecule
            >>> mol_feats = polymer.reduce(chain_feats, Scale.MOLECULE, in_scale=Scale.CHAIN)

        Note:
            When reducing from ATOM to RESIDUE scale, non-polymer atoms are
            automatically excluded since they don't belong to any residue.
        """
        return self._hierarchy.reduce(features, out_scale, rtype, in_scale)

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
        return self._hierarchy.expand(features, source, dest)

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
        return self._hierarchy.count(mask, scale)

    def membership(self: Polymer, scale: Scale) -> Array:
        """
        Get which unit each atom belongs to at the specified scale.

        Creates an integer array where each atom is labeled with its
        containing unit's index at the given scale. Useful for positional
        encodings, attention masking, and grouping operations.

        Args:
            scale: Scale at which to compute membership.
                - RESIDUE: atom -> residue index (0 to num_residues-1)
                - CHAIN: atom -> chain index (0 to num_chains-1)
                - MOLECULE: all atoms get index 0

        Returns:
            Integer array of shape (num_atoms,) with unit indices.

        Examples:
            >>> polymer = ciffy.load("structure.cif")
            >>> res_idx = polymer.membership(Scale.RESIDUE)  # atom -> residue
            >>> chain_idx = polymer.membership(Scale.CHAIN)  # atom -> chain

            # Use for attention masking (same-residue attention)
            >>> mask = res_idx[:, None] == res_idx[None, :]
        """
        return self._hierarchy.index(scale)

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

    def _mask(
        self: Polymer,
        indices: Array | int,
        source: Scale,
        dest: Scale = Scale.ATOM,
    ) -> Array:
        """
        Create a boolean mask selecting specific units.

        Internal method used by selection operations.

        Args:
            indices: Indices of units to select.
            source: Scale of the indices.
            dest: Scale of the output mask.

        Returns:
            Boolean array at dest scale.
        """
        from .._selection import mask
        return mask(self, indices, source, dest)

    def _to_mask(self: Polymer, selector: Array | int | list | slice, scale: Scale) -> Array:
        """
        Convert a selector to a boolean mask.

        Args:
            selector: Boolean mask, int index, list of indices, or slice.
            scale: Scale at which the selector operates.

        Returns:
            Boolean mask array at the specified scale.

        Raises:
            IndexError: If any index is out of range.
        """
        max_size = self.size(scale)

        # Handle slice
        if isinstance(selector, slice):
            mask = ops.zeros(max_size, like=self.coordinates, dtype='bool')
            mask[selector] = True
            return mask

        # Already a boolean mask - return as-is
        if hasattr(selector, 'dtype'):
            dtype_str = str(selector.dtype)
            if 'bool' in dtype_str:
                return selector

        # Convert int to list
        if isinstance(selector, int):
            indices = [selector]
        elif isinstance(selector, list):
            indices = selector
        elif hasattr(selector, 'tolist'):
            # Array of indices
            indices = selector.tolist()
        else:
            # Assume it's already a mask
            return selector

        # Validate indices and create mask
        for ix in indices:
            if ix < 0 or ix >= max_size:
                raise IndexError(
                    f"{scale.name} index {ix} out of range for Polymer with {max_size} {scale.name.lower()}s"
                )

        mask = ops.zeros(max_size, like=self.coordinates, dtype='bool')
        for ix in indices:
            mask[ix] = True
        return mask

    def _select(self: Polymer, mask: Array, scale: Scale) -> Polymer:
        """
        Unified selection implementation for all scales.

        Uses the hierarchy to derive masks, compute new sizes/lengths,
        and slice fields accordingly.

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

        # Step 3: Use hierarchy to compute new hierarchy for selection
        new_per = self._hierarchy.compute_per(atom_mask, res_mask, chn_mask, scale)
        new_polymer_count = self._hierarchy.compute_polymer_count(atom_mask, res_mask, scale)

        # Create new hierarchy for the selection
        new_hierarchy = _Hierarchy(new_per, new_polymer_count, self._hierarchy._ref)
        sliced['hierarchy'] = new_hierarchy

        return self._clone(**sliced)

    def select(self: Polymer, selector: Array | int | list | slice, scale: Scale) -> Polymer:
        """
        Select units at the specified scale.

        This is the unified selection method that handles different scales
        with appropriate semantics for unresolved (0-atom) residues.

        Args:
            selector: Selection criteria. Can be:
                - Boolean mask array (True = keep)
                - Integer index (single unit)
                - List/array of integer indices
                - Slice for contiguous range
            scale: Scale of selection (ATOM, RESIDUE, or CHAIN).

        Returns:
            New Polymer with selected units.

        Raises:
            IndexError: If any index is out of range (when using indices).

        Semantics by scale:
            - ATOM: Residues with 0 atoms after masking are REMOVED.
            - RESIDUE: Selected residues are KEPT even if they have 0 atoms.
            - CHAIN: All residues in selected chains are KEPT.

        Example:
            >>> # Select by boolean mask
            >>> backbone = polymer.select(backbone_mask, Scale.ATOM)
            >>> adenines = polymer.select(polymer.sequence == Residue.A, Scale.RESIDUE)
            >>>
            >>> # Select by index
            >>> first_residue = polymer.select(0, Scale.RESIDUE)
            >>> first_chain = polymer.select(0, Scale.CHAIN)
            >>>
            >>> # Select by index list or slice
            >>> residues = polymer.select([0, 2, 4], Scale.RESIDUE)
            >>> first_100_atoms = polymer.select(slice(100), Scale.ATOM)
        """
        if scale not in (Scale.ATOM, Scale.RESIDUE, Scale.CHAIN):
            raise ValueError(f"Selection not supported at {scale.name} scale")

        # Convert indices to mask if needed
        mask = self._to_mask(selector, scale)
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
        return self.select(ix, Scale.CHAIN)

    def by_atom(self: Polymer, name: Array | int) -> Polymer:
        """
        Select atoms by atom type index.

        Args:
            name: Atom type index or indices.

        Returns:
            New Polymer with matching atoms.
        """
        from .._selection import by_atom
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
        from .._selection import by_residue
        return by_residue(self, res)

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
        from ..biochemistry import CANONICAL_ALL
        return self.by_residue(CANONICAL_ALL)

    def by_type(self: Polymer, mol: Molecule) -> Polymer:
        """
        Select chains by molecule type.

        Args:
            mol: Molecule type to select.

        Returns:
            New Polymer with chains of that type.
        """
        from .._selection import by_type
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
        from .._selection import poly
        return poly(self)

    def hetero(self: Polymer) -> "HeteroAtoms":
        """
        Return non-polymer atoms only (HETATM: water, ions, ligands).

        Returns a lightweight HeteroAtoms container with only atom-level data.
        Unlike Polymer, HeteroAtoms has no residue or chain hierarchy.

        Returns:
            HeteroAtoms container with HETATM atoms. If there are no HETATM atoms,
            returns an empty HeteroAtoms.

        Example:
            >>> p = load("file.cif")
            >>> hetero_atoms = p.hetero()  # Get waters/ions/ligands
            >>> if not hetero_atoms.empty():
            ...     waters = hetero_atoms.by_element(8)  # Oxygen atoms
        """
        from .._selection import hetero
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
        from .._selection import chains
        return chains(self, mol)

    def resolved(self: Polymer, scale: Scale = Scale.RESIDUE) -> Array:
        """
        Get mask of resolved (non-empty) units.

        Args:
            scale: Scale to check.

        Returns:
            Boolean tensor where True indicates resolved units.
        """
        from .._selection import resolved
        return resolved(self, scale)

    def strip(self: Polymer, scale: Scale = Scale.RESIDUE) -> Polymer:
        """
        Remove unresolved units at a scale.

        Args:
            scale: Scale at which to strip.

        Returns:
            New Polymer without empty units.
        """
        from .._selection import strip
        return strip(self, scale)

    # ─────────────────────────────────────────────────────────────────────────
    # Specialized Selections
    # ─────────────────────────────────────────────────────────────────────────

    def backbone(self: Polymer) -> Polymer:
        """Select backbone atoms (sugar-phosphate for RNA/DNA, N-CA-C-O for protein)."""
        from .._selection import backbone
        return backbone(self)

    def nucleobase(self: Polymer) -> Polymer:
        """Select RNA nucleobase atoms."""
        from .._selection import nucleobase
        return nucleobase(self)

    def phosphate(self: Polymer) -> Polymer:
        """Select RNA/DNA phosphate atoms."""
        from .._selection import phosphate
        return phosphate(self)

    def sidechain(self: Polymer) -> Polymer:
        """Select protein sidechain atoms."""
        from .._selection import sidechain
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
            >>> from ciffy import from_sequence
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
        from ..geometry import position_residue
        from ..biochemistry.linking import LINKING_BY_TYPE
        from ..utils import atoms_to_col_map
        from .builder import expand_residue

        # Validate single chain and poly-only
        if self.size(Scale.CHAIN) != 1:
            raise ValueError(
                f"extend() requires a single-chain polymer. "
                f"Got {self.size(Scale.CHAIN)} chains."
            )
        if self.nonpoly() > 0:
            raise ValueError(
                "extend() requires a poly-only polymer (no HETATM atoms). "
                "Use polymer.poly() first."
            )

        # Get atom data for the new residue (uses cached expansion)
        new_res_atoms, new_elements, _, ideal_coords = expand_residue(residue)

        # Use ideal coordinates if none provided
        if coords is None:
            coords = ideal_coords
            # Convert to backend if needed
            if self.backend == "torch":
                import torch
                coords = torch.from_numpy(coords).to(
                    dtype=self.coordinates.dtype,
                    device=self.coordinates.device
                )

        # Validate coordinate shape
        if len(new_res_atoms) != coords.shape[0]:
            raise ValueError(
                f"Coordinate shape {coords.shape} doesn't match residue {residue.name} "
                f"which has {len(new_res_atoms)} atoms."
            )

        # Validate backend compatibility
        check_compatible(self.coordinates, coords, "coords")

        # Get last residue's state
        last_res_coords, _, last_res_atom_to_col, last_res_type = self._residue_slice(-1)

        # Build atom_to_col for new residue
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
        new_lengths = ops.to_backend(
            np.array([self.lengths[0].item() + 1], dtype=np.int64),
            self.lengths
        )

        # Create new hierarchy
        sizes = {
            Scale.RESIDUE: new_res_sizes,
            Scale.CHAIN: new_chn_sizes,
            Scale.MOLECULE: new_mol_sizes,
        }
        new_hierarchy = _Hierarchy.from_sizes_and_lengths(
            sizes=sizes,
            lengths=new_lengths,
            polymer_count=self.polymer_count + n_new_atoms,
            ref=new_coords,
        )

        return self._clone(
            coordinates=new_coords,
            atoms=new_atoms_arr,
            elements=new_elements_arr,
            sequence=new_sequence,
            hierarchy=new_hierarchy,
            # bfactors not preserved - new atoms don't have experimental B-factors
            bfactors=None,
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
        def get_name(value: int) -> str:
            try:
                return Atom.from_value(value).name
            except KeyError:
                return '?'
        return [get_name(ix.item()) for ix in self.atoms]

    def chain_info(self: Polymer) -> list[dict]:
        """
        Get information about each chain.

        Returns:
            List of dicts with keys: 'chain', 'type', 'res', 'atoms'.
        """
        mol_types = to_numpy(self.molecule_types) if self.molecule_types is not None else None
        residue_counts = to_numpy(self.lengths)
        atom_counts = to_numpy(self._hierarchy.sizes(Scale.CHAIN))
        elements = to_numpy(self.elements)

        rows = []
        atom_offset = 0

        for i, name in enumerate(self.names):
            mol = molecule_type(int(mol_types[i])) if mol_types is not None else Molecule.UNKNOWN
            n_residues = int(residue_counts[i])
            n_atoms = int(atom_counts[i])

            # For ions, prefix with element symbol (e.g., "MG ION")
            if mol == Molecule.ION and n_atoms > 0:
                element_name = ELEMENT_NAMES.get(int(elements[atom_offset]), "")
                type_str = f"{element_name} {mol.name}" if element_name else mol.name
            else:
                type_str = mol.name

            rows.append({
                'chain': name,
                'type': type_str,
                'res': n_residues,
                'atoms': n_atoms,
            })
            atom_offset += n_atoms

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
        from ..backend import get_backend
        return get_backend(self.coordinates).value

    @property
    def device(self: Polymer) -> str | None:
        """
        Get the device of the polymer's arrays.

        Returns:
            Device string (e.g., 'cpu', 'cuda:0', 'mps:0') for PyTorch tensors,
            None for NumPy arrays.
        """
        from ..backend import get_device
        return get_device(self.coordinates)

    def numpy(self: Polymer) -> Polymer:
        """
        Convert all arrays to NumPy.

        Returns:
            New Polymer with NumPy arrays. If already NumPy, returns self.
        """
        from ..backend import is_numpy
        if is_numpy(self.coordinates):
            return self

        converted = self._convert_backend(to_numpy)
        converted['hierarchy'] = self._hierarchy.numpy()
        return self._clone(**converted)

    def torch(self: Polymer) -> Polymer:
        """
        Convert all arrays to PyTorch tensors.

        Returns:
            New Polymer with PyTorch tensors. If already PyTorch, returns self.

        Raises:
            ImportError: If PyTorch is not installed.
        """
        from ..backend import to_torch, is_torch
        if is_torch(self.coordinates):
            return self

        converted = self._convert_backend(to_torch)
        converted['hierarchy'] = self._hierarchy.torch()
        return self._clone(**converted)

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
        from ..backend import is_torch
        if not is_torch(self.coordinates):
            raise ValueError("to() is only supported for torch backend. "
                           "Use polymer.torch().to(...) to convert first.")

        if device is None and dtype is None:
            return self

        # Convert Fields based on their dtype (float vs int)
        converted = {}
        for name, field in self._get_fields().items():
            value = getattr(self, field.private_name, None)
            if value is None:
                converted[name] = None
            elif field.dtype == Dtype.FLOAT:
                # Float tensors: apply device and dtype
                result = value
                if device is not None:
                    result = result.to(device)
                if dtype is not None:
                    result = result.to(dtype)
                converted[name] = result
            else:
                # Int tensors: apply device only
                converted[name] = value.to(device) if device is not None else value

        # Move hierarchy to device (int tensors only, no dtype change)
        if device is not None:
            converted['hierarchy'] = self._hierarchy.to(device)

        return self._clone(**converted)

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
        Detach all float tensors from their computation graphs (torch backend only).

        Detaches all float Fields (coordinates, bfactors) from their computation
        graphs. For NumPy arrays, this is a no-op since NumPy doesn't have
        computation graphs.

        Returns:
            Self, for method chaining.

        Example:
            >>> coords = polymer.coordinates.clone().requires_grad_(True)
            >>> polymer.coordinates = coords
            >>> loss = polymer.coordinates.sum()
            >>> loss.backward()
            >>> polymer.detach()
        """
        for name, field in self._get_fields().items():
            if field.dtype == Dtype.FLOAT:
                value = getattr(self, field.private_name, None)
                if value is not None and is_torch(value):
                    setattr(self, field.private_name, value.detach())
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
        from ..io.writer import write_cif
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
