"""
Polymer class representing molecular structures.

The Polymer class provides a unified interface for working with molecular
structures loaded from CIF files. It supports RNA, DNA, proteins, and
other molecular types.
"""

from __future__ import annotations
from typing import Generator, TYPE_CHECKING

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

# Known field schemas for creating missing fields in copy()/_clone()
# Only used when adding fields that don't yet exist on the instance
_KNOWN_FIELDS: dict[str, tuple[Scale, Dtype]] = {
    'coordinates': (Scale.ATOM, Dtype.FLOAT),
    'atoms': (Scale.ATOM, Dtype.INT),
    'elements': (Scale.ATOM, Dtype.INT),
    'bfactors': (Scale.ATOM, Dtype.FLOAT),
    'sequence': (Scale.RESIDUE, Dtype.INT),
    'molecule_types': (Scale.CHAIN, Dtype.INT),
}


def _infer_dtype(data: Array) -> Dtype:
    """Infer Dtype from array dtype."""
    dtype_str = str(data.dtype)
    if 'float' in dtype_str:
        return Dtype.FLOAT
    elif 'int' in dtype_str:
        return Dtype.INT
    else:
        # Default to float for unknown types (e.g., complex)
        return Dtype.FLOAT


class Field:
    """
    Container for Polymer array fields with scale and dtype metadata.

    Fields are stored as instance attributes on Polymer objects. Accessing a Field
    via attribute access returns its data (unwrapped by Polymer.__getattribute__).
    Setting a Field validates backend/device compatibility and size.

    Attributes:
        data: The array data, or None if not available.
        scale: Scale at which the field is defined (ATOM, RESIDUE, CHAIN).
        dtype: Data type category (Dtype.FLOAT or Dtype.INT).

    Example:
        >>> field = Field(data=coords_array, scale=Scale.ATOM, dtype=Dtype.FLOAT)
        >>> field.data  # The raw array
        >>> field.scale  # Scale.ATOM
    """

    __slots__ = ('data', 'scale', 'dtype')

    def __init__(
        self,
        data: Array | None,
        scale: Scale,
        dtype: Dtype | None = None,
    ):
        self.data = data
        self.scale = scale
        self.dtype = dtype

    def __repr__(self):
        shape = self.data.shape if self.data is not None else None
        return f"Field({self.scale.name}, dtype={self.dtype}, shape={shape})"


class _MetadataDescriptor:
    """Base class for Metadata descriptors (for non-array data)."""

    __slots__ = ('scale', 'is_list', 'name', 'private_name')

    def __init__(
        self,
        scale: Scale,
        is_list: bool = False,
    ):
        self.scale = scale
        self.is_list = is_list
        self.name = ""
        self.private_name = ""

    def __set_name__(self, owner, name):
        self.name = name
        self.private_name = f"_{name}"

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self  # Class-level access returns descriptor
        # Metadata returns None if not set (unlike Field which raises)
        return getattr(obj, self.private_name, None)

    def __set__(self, obj, value):
        setattr(obj, self.private_name, value)


class Metadata(_MetadataDescriptor):
    """
    Descriptor for Polymer metadata (non-array values passed through without conversion).

    Metadata describes simple values like scalars, strings, or lists that don't need
    backend conversion but still need proper slicing behavior.

    Unlike Field, accessing metadata that is None returns None (not an error).

    Args:
        scale: Scale at which the metadata is defined (Scale.CHAIN, MOLECULE).
        is_list: True for Python list fields (names, strands, descriptions).

    Example:
        >>> pdb_id = Metadata(Scale.MOLECULE)
        >>> names = Metadata(Scale.CHAIN, is_list=True)
    """

    def __set__(self, obj, value):
        # Validate size for list metadata at non-molecule scales
        if value is not None and self.scale != Scale.MOLECULE:
            hierarchy = getattr(obj, '_hierarchy', None)
            if hierarchy is not None and hasattr(value, '__len__'):
                expected = hierarchy.size(self.scale)
                actual = len(value)
                if actual != expected:
                    raise ValueError(
                        f"Size mismatch for '{self.name}': got {actual} elements, "
                        f"expected {expected} ({self.scale.name} scale)"
                    )
        setattr(obj, self.private_name, value)

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self  # Class-level access returns descriptor
        # Metadata returns None if not set (unlike Field which raises)
        return getattr(obj, self.private_name, None)

    def __repr__(self):
        return f"Metadata({self.scale.name}, is_list={self.is_list})"


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
    # Metadata Descriptors - values passed through without conversion
    # ─────────────────────────────────────────────────────────────────────────

    # Molecule-level (polymer_count is a property delegated to hierarchy)
    pdb_id = Metadata(Scale.MOLECULE)
    resolution = Metadata(Scale.MOLECULE)
    date = Metadata(Scale.MOLECULE)

    # Per-chain lists
    names = Metadata(Scale.CHAIN, is_list=True)
    strands = Metadata(Scale.CHAIN, is_list=True)
    descriptions = Metadata(Scale.CHAIN, is_list=True)

    def __init__(
        self: Polymer,
        hierarchy: _Hierarchy | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize a Polymer structure.

        This is an internal constructor. Users should use ciffy.load() or
        ciffy.from_sequence() to create Polymers.

        Args:
            hierarchy: _Hierarchy object containing scale bookkeeping.
                If None, creates an empty polymer.
            **kwargs: Field objects, Metadata values, and internal state:
                - Field objects are stored directly as attributes
                - Metadata values are assigned to their descriptors
                - connections/connection_types are stored as internal state

        Example:
            >>> empty = Polymer()
            >>> empty.size()
            0
        """
        # Extract internal state (not Field or Metadata)
        connections = kwargs.pop('connections', None)
        connection_types = kwargs.pop('connection_types', None)

        # Handle empty polymer case
        if hierarchy is None:
            # Only pdb_id allowed without hierarchy
            invalid = [k for k in kwargs if k != 'pdb_id']
            if invalid:
                raise TypeError(
                    f"Cannot pass {invalid} without hierarchy. "
                    f"Only 'pdb_id' is allowed for empty polymers."
                )
            hierarchy = _Hierarchy()

        # Must set hierarchy first (needed for validation)
        object.__setattr__(self, '_hierarchy', hierarchy)

        # Store Field objects directly as attributes (only if data is not None)
        for name, value in list(kwargs.items()):
            if isinstance(value, Field):
                if value.data is not None:
                    object.__setattr__(self, name, value)
                kwargs.pop(name)

        # Assign Metadata descriptors from remaining kwargs
        for name, desc in self._get_metadata().items():
            if name in kwargs:
                value = kwargs.pop(name)
                setattr(self, desc.private_name, value)
            else:
                setattr(self, desc.private_name, None)

        if kwargs:
            raise TypeError(
                f"__init__() got unexpected keyword arguments: {list(kwargs.keys())}"
            )

        object.__setattr__(self, '_bonds', None)
        object.__setattr__(self, '_connections', connections)
        object.__setattr__(self, '_connection_types', connection_types)

        # Validate field sizes match hierarchy
        self._validate_sizes()

    @property
    def lengths(self) -> Array:
        """Residues per chain (C,) array. Delegated to hierarchy."""
        return self._hierarchy.lengths

    @property
    def polymer_count(self) -> int:
        """Number of polymer atoms. Delegated to hierarchy."""
        return self._hierarchy.polymer_count

    # ─────────────────────────────────────────────────────────────────────────
    # Attribute Access - for Field unwrapping and validation
    # ─────────────────────────────────────────────────────────────────────────

    def __getattribute__(self, name: str):
        """Intercept attribute access to unwrap Field data.

        For Field objects, returns the data array.
        All other attributes are returned normally.
        """
        value = object.__getattribute__(self, name)
        if isinstance(value, Field):
            return value.data
        return value

    def __setattr__(self, name: str, value) -> None:
        """Intercept attribute assignment to validate and update Field data.

        For existing Field objects, validates backend/device compatibility
        and size, then updates the data. For known field names that don't
        exist yet, creates a new Field. For other attributes, uses normal
        setattr (only allowed for private attributes during init).
        """
        # Check if this is an existing Field
        try:
            existing = object.__getattribute__(self, name)
            if isinstance(existing, Field):
                # Validate if value is not None
                if value is not None:
                    hierarchy = object.__getattribute__(self, '_hierarchy')
                    check_compatible(hierarchy._ref, value, name)
                    expected = hierarchy.size(existing.scale)
                    actual = value.shape[0] if hasattr(value, 'shape') else len(value)
                    if actual != expected:
                        raise ValueError(
                            f"Shape mismatch for '{name}': got {actual} elements, "
                            f"expected {expected} ({existing.scale.name} scale)"
                        )
                existing.data = value
                return
        except AttributeError:
            pass

        # Check if this is a known field name that doesn't exist yet
        if name in _KNOWN_FIELDS and value is not None:
            hierarchy = object.__getattribute__(self, '_hierarchy')
            scale, dtype = _KNOWN_FIELDS[name]
            check_compatible(hierarchy._ref, value, name)
            expected = hierarchy.size(scale)
            actual = value.shape[0] if hasattr(value, 'shape') else len(value)
            if actual != expected:
                raise ValueError(
                    f"Shape mismatch for '{name}': got {actual} elements, "
                    f"expected {expected} ({scale.name} scale)"
                )
            new_field = Field(value, scale, dtype)
            object.__setattr__(self, name, new_field)
            return

        # Not a Field - use normal setattr
        object.__setattr__(self, name, value)

    # ─────────────────────────────────────────────────────────────────────────
    # Dynamic Fields - user-registered per-scale data
    # ─────────────────────────────────────────────────────────────────────────

    def annotate(
        self: Polymer,
        name: str,
        data: Array,
        scale: Scale = Scale.RESIDUE,
        dtype: Dtype | None = None,
    ) -> Polymer:
        """
        Register a new dynamic field on this polymer.

        Dynamic fields work exactly like built-in fields (coordinates, atoms, etc.):
        they are accessible as attributes, propagate through selections, and convert
        with backend changes.

        Args:
            name: Field name. Must not conflict with existing attributes.
            data: Array data with first dimension matching scale size.
            scale: Scale at which the field is defined (default: RESIDUE).
            dtype: Optional dtype hint for backend conversion.

        Returns:
            Self, for method chaining.

        Raises:
            ValueError: If field already exists or name starts with underscore.
            ValueError: If data size doesn't match scale.
            TypeError: If backend/device doesn't match polymer.

        Examples:
            >>> # Add per-residue reactivity data
            >>> polymer.annotate('reactivity', shape_tensor)
            >>> polymer.reactivity  # Access like built-in field

            >>> # Add per-atom embeddings
            >>> polymer.annotate('embeddings', atom_features, Scale.ATOM)
            >>> chain = polymer.chain(0)
            >>> chain.embeddings  # Sliced automatically

            >>> # Method chaining
            >>> polymer.annotate('dms', dms_data).annotate('shape', shape_data)
        """
        # Check if already exists
        try:
            object.__getattribute__(self, name)
            raise ValueError(f"Field '{name}' already exists.")
        except AttributeError:
            pass

        if name.startswith('_'):
            raise ValueError("Field names cannot start with underscore")

        # Validate backend/device compatibility
        hierarchy = object.__getattribute__(self, '_hierarchy')
        check_compatible(hierarchy._ref, data, name)

        # Validate size
        expected = hierarchy.size(scale)
        actual = data.shape[0] if hasattr(data, 'shape') else len(data)
        if actual != expected:
            raise ValueError(
                f"Shape mismatch for '{name}': got {actual} elements, "
                f"expected {expected} ({scale.name} scale)"
            )

        # Infer dtype from array if not provided
        if dtype is None:
            dtype = _infer_dtype(data)

        # Create and store Field
        field = Field(data, scale, dtype)
        object.__setattr__(self, name, field)
        return self

    # ─────────────────────────────────────────────────────────────────────────
    # Field/Metadata Helpers - for automatic slicing and backend conversion
    # ─────────────────────────────────────────────────────────────────────────

    def _get_field_data(self, name: str) -> Array | None:
        """Get field data, returning None if field doesn't exist.

        Useful for checking field availability without raising AttributeError.

        Args:
            name: Field name.

        Returns:
            Field data or None if field not set.
        """
        try:
            field = object.__getattribute__(self, name)
            if isinstance(field, Field):
                return field.data
        except AttributeError:
            pass
        return None

    def _get_fields(self) -> dict[str, Field]:
        """Return all Field objects on this instance (built-in and dynamic)."""
        return {
            name: attr
            for name, attr in object.__getattribute__(self, '__dict__').items()
            if isinstance(attr, Field)
        }

    @classmethod
    def _get_metadata(cls) -> dict[str, Metadata]:
        """Return all Metadata descriptors (class-level, values passed through)."""
        return {
            name: attr for name, attr in vars(cls).items()
            if isinstance(attr, Metadata)
        }

    def _validate_sizes(self) -> None:
        """Validate that field sizes and devices match hierarchy.

        Raises:
            ValueError: If any field size doesn't match the hierarchy.
            TypeError: If any field backend doesn't match.
            ValueError: If any field device doesn't match.
        """
        from ..backend import check_compatible

        hierarchy = object.__getattribute__(self, '_hierarchy')
        ref = hierarchy._ref

        # Validate Field objects
        for name, field in self._get_fields().items():
            if field.data is None:
                continue

            # Validate device/backend
            if ref is not None:
                check_compatible(ref, field.data, name)

            # Validate size
            expected = hierarchy.size(field.scale)
            actual = field.data.shape[0] if hasattr(field.data, 'shape') else len(field.data)
            if actual != expected:
                raise ValueError(
                    f"Size mismatch for '{name}': got {actual}, "
                    f"expected {expected} ({field.scale.name} scale)"
                )

        # Validate Metadata descriptors
        for name, desc in self._get_metadata().items():
            value = getattr(self, desc.private_name, None)
            if value is None:
                continue

            # Skip molecule-scale (no size constraint)
            if desc.scale == Scale.MOLECULE:
                continue

            expected = hierarchy.size(desc.scale)

            # Get actual size
            if desc.is_list:
                actual = len(value)
            elif hasattr(value, '__len__'):
                actual = len(value)
            else:
                continue  # Scalar metadata

            if actual != expected:
                raise ValueError(
                    f"Size mismatch for '{name}': got {actual}, "
                    f"expected {expected} ({desc.scale.name} scale)"
                )

    @staticmethod
    def _index_copy(arr: Array, selector) -> Array:
        """Index array with selector, ensuring a copy is returned.

        For boolean masks, indexing already returns a copy.
        For slices, indexing returns a view, so we explicitly copy.
        """
        result = arr[selector]
        if isinstance(selector, slice):
            # Slice returns view - copy for consistency
            return result.copy() if hasattr(result, 'copy') else result.clone()
        return result

    def _slice_all(
        self,
        atom_sel: Array | slice,
        res_sel: Array | slice,
        chain_sel: Array | slice,
        new_hierarchy: _Hierarchy | None = None,
    ) -> dict:
        """
        Slice all Field and Metadata attributes according to their scale.

        Args:
            atom_sel: Boolean mask or slice for atoms.
            res_sel: Boolean mask or slice for residues.
            chain_sel: Boolean mask or slice for chains.
            new_hierarchy: Hierarchy for the sliced polymer.

        Returns:
            Dict mapping field/metadata names to sliced values, plus '_field_meta'
            for reconstructing Field objects in _clone.
        """
        result = {}
        field_meta = {}  # Store (scale, dtype) for each field

        # Slice Field objects (only Fields with data exist on instance)
        for name, field in self._get_fields().items():
            field_meta[name] = (field.scale, field.dtype)
            if field.scale == Scale.ATOM:
                result[name] = self._index_copy(field.data, atom_sel)
            elif field.scale == Scale.RESIDUE:
                result[name] = self._index_copy(field.data, res_sel)
            elif field.scale == Scale.CHAIN:
                result[name] = self._index_copy(field.data, chain_sel)
            else:  # Scale.MOLECULE - no slicing
                result[name] = field.data

        # Slice Metadata descriptors
        for name, desc in self._get_metadata().items():
            value = getattr(self, desc.private_name, None)
            if value is None:
                result[name] = None
            elif desc.scale == Scale.CHAIN:
                if desc.is_list:
                    if isinstance(chain_sel, slice):
                        result[name] = value[chain_sel]  # list slicing copies
                    else:
                        result[name] = filter_by_mask(value, chain_sel)
                else:
                    result[name] = self._index_copy(value, chain_sel)
            else:  # Scale.MOLECULE - scalars, no slicing
                result[name] = value

        result['_field_meta'] = field_meta
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

    def _convert_backend(self, to_func, new_hierarchy: _Hierarchy | None = None) -> dict:
        """
        Convert all Field arrays to a target backend, pass through Metadata.

        Args:
            to_func: Function to convert arrays (e.g., to_numpy, to_torch).
            new_hierarchy: Hierarchy for the converted polymer.

        Returns:
            Dict mapping all field/metadata names to converted/passed values,
            plus '_field_meta' for reconstructing Field objects in _clone.
        """
        result = {}
        field_meta = {}  # Store (scale, dtype) for each field

        # Convert Field arrays (only Fields with data exist on instance)
        for name, field in self._get_fields().items():
            field_meta[name] = (field.scale, field.dtype)
            result[name] = to_func(field.data)

        # Pass through Metadata unchanged (copy lists)
        for name, meta in self._get_metadata().items():
            value = getattr(self, meta.private_name, None)
            if meta.is_list and value is not None:
                result[name] = value.copy()
            else:
                result[name] = value

        result['_field_meta'] = field_meta
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
                    if field.data is not None:
                        field_sizes.append(arr_size(field.data, 0))
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

        Collects all Field and Metadata values, applies overrides, and
        constructs a new Polymer. This is the single place that maps
        field/metadata names to constructor parameters.

        Args:
            **overrides: Field values to override. Can include any field/metadata
                name (coordinates, atoms, pdb_id, etc.), 'hierarchy' for the
                hierarchy, or '_field_meta' for field scale/dtype info.

        Returns:
            New Polymer with the specified overrides applied.

        Example:
            >>> # Create copy with new coordinates
            >>> moved = polymer._clone(coordinates=new_coords)
            >>> # Create copy with converted arrays
            >>> converted = polymer._clone(**self._convert_backend(to_numpy))
        """
        # Extract hierarchy (used for new polymer)
        hierarchy = overrides.pop('hierarchy', object.__getattribute__(self, '_hierarchy'))

        # Extract field metadata (for dynamic fields from slicing/conversion)
        field_meta = overrides.pop('_field_meta', None)

        # Create new instance bypassing __init__ for efficiency
        polymer = object.__new__(Polymer)
        object.__setattr__(polymer, '_hierarchy', hierarchy)

        # Reconstruct Fields (only if data is not None)
        current_fields = self._get_fields()
        for name, field in current_fields.items():
            # Get metadata from override or original
            if field_meta and name in field_meta:
                scale, dtype = field_meta[name]
            else:
                scale, dtype = field.scale, field.dtype

            # Get data from override or original
            if name in overrides:
                data = overrides.pop(name)
            else:
                data = field.data

            # Only set attribute if data exists
            if data is not None:
                new_field = Field(data, scale, dtype)
                object.__setattr__(polymer, name, new_field)

        # Handle overrides for known fields that don't exist on source
        # (e.g., adding coordinates to a template)
        for name in list(overrides.keys()):
            if name in _KNOWN_FIELDS:
                data = overrides.pop(name)
                if data is not None:
                    scale, dtype = _KNOWN_FIELDS[name]
                    new_field = Field(data, scale, dtype)
                    object.__setattr__(polymer, name, new_field)

        # Copy Metadata descriptors
        for name, desc in self._get_metadata().items():
            if name in overrides:
                value = overrides.pop(name)
            else:
                value = getattr(self, desc.private_name, None)
            # Copy lists to avoid mutation
            if desc.is_list and value is not None:
                value = list(value)
            setattr(polymer, desc.private_name, value)

        # Copy internal state
        object.__setattr__(polymer, '_bonds', None)  # Bonds need recomputation
        object.__setattr__(polymer, '_connections', object.__getattribute__(self, '_connections'))
        object.__setattr__(polymer, '_connection_types', object.__getattribute__(self, '_connection_types'))

        return polymer

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

    @property
    def connections(self) -> Array | None:
        """
        Non-covalent connections as atom index pairs (H-bonds, metal coordination, etc).

        Returns:
            (C, 2) int32 array where each row [i, j] represents a connection
            between atoms i and j, or None if connections were not loaded.

        Note:
            Connections must be explicitly loaded via skip parameter:
            ``ciffy.load(file, skip=[])`` or ``skip=["descriptions"]``.
            By default, connections are skipped for performance.
        """
        return self._connections

    @property
    def connection_types(self) -> Array | None:
        """
        Types of non-covalent connections.

        Returns:
            (C,) int32 array of connection type indices, or None if not loaded.
            Values: 0=UNKNOWN, 1=HYDROG (H-bond), 2=COVALE, 3=METALC, 4=DISULF.
        """
        return self._connection_types

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

    def size(self: Polymer, scale: Scale = Scale.ATOM) -> int:
        """
        Get the count at a specific scale.

        Args:
            scale: Scale level (ATOM, RESIDUE, CHAIN, MOLECULE).

        Returns:
            Number of units at the specified scale.
        """
        return self._hierarchy.size(scale)

    def __len__(self: Polymer) -> int:
        """Return the number of atoms."""
        return self.size()

    def copy(self: Polymer, **overrides) -> Polymer:
        """
        Return a copy of this Polymer, optionally with field overrides.

        When called with no arguments, returns a deep copy with all arrays
        cloned. When called with keyword arguments, those fields use the
        provided values instead of being cloned.

        Args:
            **overrides: Field values to override. Can be any field name:
                coordinates, atoms, elements, sequence, bfactors, pdb_id,
                resolution, date, names, strands, descriptions, molecule_types,
                or any dynamic fields added via annotate().

        Returns:
            New Polymer with specified fields overridden (or all cloned if none).

        Raises:
            ValueError: If override value has different device than polymer.

        Examples:
            >>> # Deep copy
            >>> copy = polymer.copy()

            >>> # Replace coordinates (e.g., after prediction)
            >>> result = template.copy(coordinates=predicted_coords)

            >>> # Replace multiple fields
            >>> modified = polymer.copy(coordinates=new_coords, bfactors=new_bfactors)
        """
        from ..backend import ops

        # Validate device compatibility for array overrides
        hierarchy = object.__getattribute__(self, '_hierarchy')
        for name, value in overrides.items():
            if value is not None and hasattr(value, 'shape'):
                check_compatible(hierarchy._ref, value, name)

        # Clone all Field data (only Fields with data exist on instance)
        cloned = {}
        for name, field in self._get_fields().items():
            if name in overrides:
                # User override takes precedence
                cloned[name] = overrides.pop(name)
            else:
                cloned[name] = ops.clone(field.data)

        # Apply remaining overrides (metadata, etc.)
        cloned.update(overrides)

        return self._clone(**cloned)

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
        new_coordinates = self.coordinates - expanded

        centered = self.copy(coordinates=new_coordinates)
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
        new_coordinates = centered.coordinates / std_expanded * size

        scaled = centered.copy(coordinates=new_coordinates)
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

    def bonded_distances(
        self: Polymer,
        atom1: int | Array,
        atom2: int | Array,
    ) -> Array:
        """
        Get distances between bonded atoms of specified types.

        Finds all covalent bonds where one atom matches `atom1` and the other
        matches `atom2`, then computes the Euclidean distance for each pair.

        Args:
            atom1: First atom type(s) as integer value(s). Can be a single
                int or an array of ints to match multiple atom types.
            atom2: Second atom type(s) as integer value(s).

        Returns:
            1D array of distances between matching bonded pairs.
            Empty array if no matching bonds found.

        Example:
            >>> from ciffy.biochemistry import Residue
            >>> # Get all phosphodiester bond lengths (O3'-P) for adenosine
            >>> polymer.bonded_distances(Residue.A.O3p, Residue.A.P)
            array([1.59, 1.61, 1.58, ...])
            >>> # Match multiple residue types using arrays
            >>> o3p_values = np.array([Residue.A.O3p, Residue.G.O3p])
            >>> p_values = np.array([Residue.A.P, Residue.G.P])
            >>> polymer.bonded_distances(o3p_values, p_values)
        """
        # Get bonds and atom types
        bonds = self.bonds  # (B, 2) array of atom indices
        atoms = self.atoms  # (N,) array of atom type values

        # Convert atom type arguments to arrays in the same backend
        def to_values(atom: int | Array) -> Array:
            if isinstance(atom, int):
                return ops.array([atom], like=atoms)
            return ops.convert_backend(atom, like=atoms)

        v1 = to_values(atom1)
        v2 = to_values(atom2)

        # Get atom types at bond endpoints
        atom_i = atoms[bonds[:, 0]]
        atom_j = atoms[bonds[:, 1]]

        # Match: (i in v1 AND j in v2) OR (i in v2 AND j in v1)
        mask1 = ops.isin(atom_i, v1) & ops.isin(atom_j, v2)
        mask2 = ops.isin(atom_i, v2) & ops.isin(atom_j, v1)
        mask = mask1 | mask2

        # Get matching bond indices
        matching_indices = ops.nonzero_1d(mask)

        if arr_size(matching_indices) == 0:
            return ops.empty(0, like=self.coordinates)

        matching_bonds = bonds[matching_indices]

        # Compute distances
        coords = self.coordinates
        p1 = coords[matching_bonds[:, 0]]
        p2 = coords[matching_bonds[:, 1]]
        diff = p1 - p2

        return ops.norm(diff, axis=1)

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
            mask = ops.zeros(max_size, like=self._hierarchy._ref, dtype='bool')
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

        mask = ops.zeros(max_size, like=self._hierarchy._ref, dtype='bool')
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

        # Step 2: Compute new hierarchy for selection
        new_per = self._hierarchy.compute_per(atom_mask, res_mask, chn_mask, scale)
        new_polymer_count = self._hierarchy.compute_polymer_count(atom_mask, res_mask, scale)
        new_hierarchy = _Hierarchy(new_per, new_polymer_count, self._hierarchy._ref)

        # Step 3: Slice all fields and annotations
        sliced = self._slice_all(atom_mask, res_mask, chn_mask, new_hierarchy)
        sliced['hierarchy'] = new_hierarchy

        return self._clone(**sliced)

    def _select_contiguous(self: Polymer, ix: int, scale: Scale) -> Polymer:
        """
        Fast path for selecting a single contiguous unit (chain or residue).

        Uses slice indexing instead of boolean masks for ~10x speedup.

        Args:
            ix: Index of the unit to select.
            scale: Scale of the selection (CHAIN or RESIDUE).

        Returns:
            New Polymer with the selected unit.
        """
        # Get slice bounds from hierarchy
        atom_slice, res_slice, chain_slice = self._hierarchy.bounds(ix, scale)

        # Get new hierarchy first (for annotations slicing)
        new_hierarchy = self._hierarchy.select_contiguous(ix, scale)

        # Slice all fields and annotations using slices (fast path)
        sliced = self._slice_all(atom_slice, res_slice, chain_slice, new_hierarchy)
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

    def chain(self: Polymer, ix: Array | int) -> Polymer:
        """
        Select chains by index.

        Args:
            ix: Chain index or indices to select.

        Returns:
            New Polymer with selected chains.

        Raises:
            IndexError: If any index is out of range.

        Example:
            >>> polymer.chain(0)           # First chain
            >>> polymer.chain([0, 2])      # First and third chains
        """
        # Fast path for single integer selection
        if isinstance(ix, int):
            return self._select_contiguous(ix, Scale.CHAIN)
        return self.select(ix, Scale.CHAIN)

    def residue(self: Polymer, ix: Array | int) -> Polymer:
        """
        Select residues by index.

        Args:
            ix: Residue index or indices to select.

        Returns:
            New Polymer with selected residues.

        Raises:
            IndexError: If any index is out of range.

        Example:
            >>> polymer.residue(0)           # First residue
            >>> polymer.residue([0, 5, 10])  # Multiple residues
        """
        # Fast path for single integer selection
        if isinstance(ix, int):
            return self._select_contiguous(ix, Scale.RESIDUE)
        return self.select(ix, Scale.RESIDUE)

    def atom(self: Polymer, ix: Array | int) -> Polymer:
        """
        Select atoms by index.

        Args:
            ix: Atom index or indices to select.

        Returns:
            New Polymer with selected atoms.

        Raises:
            IndexError: If any index is out of range.

        Example:
            >>> polymer.atom(0)              # First atom
            >>> polymer.atom([0, 1, 2])      # Multiple atoms
        """
        return self.select(ix, Scale.ATOM)

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

    def chains(self: Polymer) -> Generator[Polymer, None, None]:
        """
        Iterate over chains.

        To filter by molecule type, use `polymer.by_type(mol).chains()`.

        Yields:
            Individual chain Polymers.
        """
        from .._selection import chains
        return chains(self)

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

    def _extend_from_empty(
        self: Polymer,
        residue: Residue,
        coords: Array,
        atoms: Array | None,
        elements: Array | None,
        name: str,
    ) -> Polymer:
        """Create first residue when extending from empty polymer."""
        # Ensure numpy arrays
        coords = np.asarray(coords, dtype=np.float32)
        if atoms is not None:
            atoms = np.asarray(atoms, dtype=np.int64)
        if elements is not None:
            elements = np.asarray(elements, dtype=np.int64)

        n_atoms = coords.shape[0]

        hierarchy = _Hierarchy.from_sizes_and_lengths(
            sizes={
                Scale.RESIDUE: np.array([n_atoms], dtype=np.int64),
                Scale.CHAIN: np.array([n_atoms], dtype=np.int64),
                Scale.MOLECULE: np.array([n_atoms], dtype=np.int64),
            },
            lengths=np.array([1], dtype=np.int64),
            polymer_count=n_atoms,
            ref=coords,
        )

        polymer = Polymer(
            hierarchy,
            coordinates=Field(coords, Scale.ATOM, Dtype.FLOAT),
            atoms=Field(atoms, Scale.ATOM, Dtype.INT),
            elements=Field(elements, Scale.ATOM, Dtype.INT),
            sequence=Field(np.array([residue.value], dtype=np.int64), Scale.RESIDUE, Dtype.INT),
            molecule_types=Field(np.array([residue.molecule_type], dtype=np.int64), Scale.CHAIN, Dtype.INT),
            names=[name],
            strands=[""],
            descriptions=[""],
            pdb_id=self.pdb_id,
        )

        return polymer.torch() if self.backend == "torch" else polymer

    def extend(
        self: Polymer,
        residue: Residue,
        coords: Array,
        transform: Array | None = None,
        atoms: Array | None = None,
        elements: Array | None = None,
        name: str = "A",
    ) -> Polymer:
        """
        Append a residue to the end of a polymer (for autoregressive generation).

        Creates a new Polymer with an additional residue. If the polymer is empty,
        creates the first residue. Otherwise, positions the residue relative to
        the last residue using the provided transform.

        Note: This method requires the polymer to have coordinates. For templates
        (from from_sequence()), use copy(coordinates=...) first.

        Args:
            residue: Residue type being added (e.g., Residue.ALA, Residue.A).
            coords: (n_atoms, 3) coordinates of the residue in its local frame.
            transform: (6,) SE(3) transform [axis-angle, translation] for positioning.
                Typically predicted by a generative model.
                Required when extending a non-empty polymer, ignored for empty.
            atoms: Atom type indices. Required for non-empty polymers with atom data.
            elements: Element indices. Required for non-empty polymers with element data.
            name: Chain name (only used when extending from empty polymer).

        Returns:
            New Polymer with the residue appended.

        Raises:
            AttributeError: If polymer has no coordinates.
            ValueError: If polymer has multiple chains, has HETATM atoms,
                or required parameters are missing.

        Example:
            >>> from ciffy import Residue, Polymer
            >>> from ciffy.polymer import expand_residue
            >>>
            >>> # Start from empty polymer
            >>> poly = Polymer()
            >>> atoms, elements, coords = expand_residue(Residue.A)
            >>> poly = poly.extend(Residue.A, coords, atoms=atoms, elements=elements)
            >>>
            >>> # Extend with model-predicted coordinates and transform
            >>> atoms2, elements2, _ = expand_residue(Residue.C, start_terminal=False)
            >>> predicted_coords, predicted_transform = model.predict(...)
            >>> poly = poly.extend(Residue.C, predicted_coords, predicted_transform, atoms2, elements2)
        """
        from ..geometry import position_residue_fast
        from ..biochemistry.linking import LINKING_BY_TYPE
        from .builder import _resolve_frame_indices

        # Handle empty polymer case - create first residue
        if self.empty():
            return self._extend_from_empty(residue, coords, atoms, elements, name)

        # Validate single chain and poly-only for non-empty
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

        # Transform is required for non-empty polymers
        if transform is None:
            raise ValueError(
                "transform is required when extending a non-empty polymer."
            )

        # Validate atoms/elements are provided when required
        if self.atoms is not None and atoms is None:
            raise ValueError(
                "atoms parameter is required because this Polymer has atom data. "
                "Use expand_residue() to get atom indices."
            )
        if self.elements is not None and elements is None:
            raise ValueError(
                "elements parameter is required because this Polymer has element data. "
                "Use expand_residue() to get element indices."
            )

        # Validate backend compatibility
        check_compatible(self.coordinates, coords, "coords")
        check_compatible(self.coordinates, transform, "transform")

        # Get last residue's state
        last_res_coords, last_res_atoms_arr, _, last_res_type = self._residue_slice(-1)

        # Get linking definition
        link_def = LINKING_BY_TYPE.get(last_res_type.molecule_type)
        if link_def is None:
            raise ValueError(
                f"No linking definition for molecule type {last_res_type.molecule_type}. "
                f"Cannot extend chains of this type."
            )

        # Convert atoms to tuples for frame resolution
        last_res_atoms = tuple(int(a) for a in last_res_atoms_arr)
        new_res_atoms = tuple(int(a) for a in atoms) if atoms is not None else ()

        # Position the new residue using fast path
        prev_frame = _resolve_frame_indices(last_res_type.value, last_res_atoms)
        next_frame = _resolve_frame_indices(residue.value, new_res_atoms) if new_res_atoms else None

        if next_frame is not None:
            positioned_coords = position_residue_fast(
                last_res_coords,
                coords,
                transform,
                prev_frame.prev_cols,
                prev_frame.prev_z_toward,
                next_frame.next_cols,
                next_frame.next_z_toward,
            )
        else:
            # No atoms provided - just apply transform without frame alignment
            from ..geometry import apply_relative_transform, compute_frame_from_indices
            prev_origin, prev_R = compute_frame_from_indices(
                last_res_coords, prev_frame.prev_cols, prev_frame.prev_z_toward
            )
            target_origin, target_R = apply_relative_transform(prev_origin, prev_R, transform)
            # Simple translation to target origin
            centroid = coords.mean(axis=0)
            positioned_coords = coords + (target_origin - centroid)

        # Concatenate arrays
        new_coords = ops.cat([self.coordinates, positioned_coords], axis=0)
        n_new_atoms = coords.shape[0]

        # Update atoms if present
        if self.atoms is not None and atoms is not None:
            new_atoms_arr = ops.cat([
                self.atoms,
                ops.to_backend(np.asarray(atoms, dtype=np.int64), self.atoms)
            ], axis=0)
        else:
            new_atoms_arr = None

        # Update elements if present
        if self.elements is not None and elements is not None:
            new_elements_arr = ops.cat([
                self.elements,
                ops.to_backend(np.asarray(elements, dtype=np.int64), self.elements)
            ], axis=0)
        else:
            new_elements_arr = None

        # Update sequence
        new_sequence = ops.cat([
            self.sequence,
            ops.to_backend(np.array([residue.value], dtype=np.int64), self.sequence)
        ], axis=0)

        # Update sizes
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
        # Handle empty or minimal polymers
        names = self._names
        if names is None or len(names) == 0:
            return []

        mol_types_data = self._get_field_data('molecule_types')
        mol_types = to_numpy(mol_types_data) if mol_types_data is not None else None
        residue_counts = to_numpy(self.lengths)
        hierarchy = object.__getattribute__(self, '_hierarchy')
        atom_counts = to_numpy(hierarchy.sizes(Scale.CHAIN))
        elements_data = self._get_field_data('elements')
        elements = to_numpy(elements_data) if elements_data is not None else None

        rows = []
        atom_offset = 0

        for i, name in enumerate(names):
            mol = molecule_type(int(mol_types[i])) if mol_types is not None else Molecule.UNKNOWN
            n_residues = int(residue_counts[i])
            n_atoms = int(atom_counts[i])

            # For ions, prefix with element symbol (e.g., "MG ION")
            if mol == Molecule.ION and n_atoms > 0 and elements is not None:
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
        return format_chain_table(self.pdb_id, self.backend, rows, self.date)

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
        return get_backend(self._hierarchy._ref).value

    @property
    def device(self: Polymer) -> str | None:
        """
        Get the device of the polymer's arrays.

        Returns:
            Device string (e.g., 'cpu', 'cuda:0', 'mps:0') for PyTorch tensors,
            None for NumPy arrays.
        """
        from ..backend import get_device
        return get_device(self._hierarchy._ref)

    def numpy(self: Polymer) -> Polymer:
        """
        Convert all arrays to NumPy.

        Returns:
            New Polymer with NumPy arrays. If already NumPy, returns self.
        """
        if self.backend == "numpy":
            return self

        new_hierarchy = self._hierarchy.numpy()
        converted = self._convert_backend(to_numpy, new_hierarchy)
        converted['hierarchy'] = new_hierarchy
        return self._clone(**converted)

    def torch(self: Polymer) -> Polymer:
        """
        Convert all arrays to PyTorch tensors.

        Returns:
            New Polymer with PyTorch tensors. If already PyTorch, returns self.

        Raises:
            ImportError: If PyTorch is not installed.
        """
        from ..backend import to_torch
        if self.backend == "torch":
            return self

        new_hierarchy = self._hierarchy.torch()
        converted = self._convert_backend(to_torch, new_hierarchy)
        converted['hierarchy'] = new_hierarchy
        return self._clone(**converted)

    def to(
        self: Polymer,
        device: "str | torch.device | None" = None,
        dtype: "torch.dtype | None" = None,
    ) -> Polymer:
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
        if self.backend != "torch":
            raise ValueError("to() is only supported for torch backend. "
                           "Use polymer.torch().to(...) to convert first.")

        if device is None and dtype is None:
            return self

        # Convert Fields based on their dtype (float vs int)
        converted = {}
        for name, field in self._get_fields().items():
            if field.data is None:
                converted[name] = None
            elif field.dtype == Dtype.FLOAT:
                # Float tensors: apply device and dtype
                result = field.data
                if device is not None:
                    result = result.to(device)
                if dtype is not None:
                    result = result.to(dtype)
                converted[name] = result
            else:
                # Int tensors: apply device only
                converted[name] = field.data.to(device) if device is not None else field.data

        # Move hierarchy to device (int tensors only, no dtype change)
        hierarchy = object.__getattribute__(self, '_hierarchy')
        if device is not None:
            converted['hierarchy'] = hierarchy.to(device)

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
                if field.data is not None and is_torch(field.data):
                    field.data = field.data.detach()
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
        if not filename.lower().endswith('.cif'):
            raise ValueError(
                f"Output file must have .cif extension, got: {filename!r}"
            )
        if self.empty():
            raise ValueError("Cannot write empty polymer to CIF file")
        if self._get_field_data('coordinates') is None:
            raise ValueError(
                "Cannot write polymer without coordinates. "
                "Use copy(coordinates=...) to add coordinates to a template."
            )
        from ..io.writer import write_cif
        write_cif(self, filename)

