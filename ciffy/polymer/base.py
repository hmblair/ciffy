"""
Base class for atom containers (Polymer, HeteroAtoms).

This module provides the AtomContainer base class which encapsulates
shared functionality for Field management, backend conversion, and slicing.
"""

from __future__ import annotations
from typing import Any, TYPE_CHECKING

import numpy as np

from ..backend import Array, is_torch, size as arr_size, check_compatible, to_numpy
from ..backend import ops
from ..biochemistry import Scale

if TYPE_CHECKING:
    import torch

from .hierarchy import _Hierarchy
from ..utils import filter_by_mask


# Known field scales for creating missing fields in copy()/_clone()
# Only used when adding fields that don't yet exist on the instance
_KNOWN_FIELDS: dict[str, Scale] = {
    'coordinates': Scale.ATOM,
    'atoms': Scale.ATOM,
    'elements': Scale.ATOM,
    'bfactors': Scale.ATOM,
    'sequence': Scale.RESIDUE,
    'molecule_types': Scale.CHAIN,
}


class Field:
    """
    Container for array fields with scale metadata.

    Fields are stored as instance attributes on AtomContainer objects. Accessing a Field
    via attribute access returns its data (unwrapped by __getattribute__).
    Setting a Field validates backend/device compatibility and size.

    Attributes:
        data: The array data.
        scale: Scale at which the field is defined (ATOM, RESIDUE, CHAIN).

    Example:
        >>> field = Field(coords_array, Scale.ATOM)
        >>> field.data  # The raw array
        >>> field.scale  # Scale.ATOM
    """

    __slots__ = ('data', 'scale')

    def __init__(self, data: Array, scale: Scale):
        self.data = data
        self.scale = scale

    def __repr__(self):
        return f"Field({self.scale.name}, shape={self.data.shape})"


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
    Descriptor for non-array metadata values.

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


class AtomContainer:
    """
    Base class for atom containers (Polymer, HeteroAtoms).

    Provides shared functionality for:
    - Field management with scale metadata
    - Backend conversion (numpy/torch)
    - Device management (cpu/cuda)
    - Atom-level slicing
    - Dynamic field annotation

    Subclasses must define:
    - _allowed_scales: Set of scales this container supports
    - _init_from_kwargs(): Handle subclass-specific initialization
    """

    # Scales this container supports (subclasses override)
    _allowed_scales: set[Scale] = {Scale.ATOM, Scale.CHAIN, Scale.MOLECULE}

    # Molecule-level metadata (shared by all containers)
    pdb_id = Metadata(Scale.MOLECULE)

    def __init__(
        self,
        hierarchy: _Hierarchy | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize an AtomContainer.

        Args:
            hierarchy: _Hierarchy object containing scale bookkeeping.
                If None, creates an empty container.
            **kwargs: Field objects, Metadata values, and internal state.
        """
        # Handle empty container case
        if hierarchy is None:
            hierarchy = _Hierarchy(scales=frozenset(self._allowed_scales))

        # Must set hierarchy first (needed for validation)
        object.__setattr__(self, '_hierarchy', hierarchy)

        # Store Field objects directly as attributes (only if data is not None)
        for name, value in list(kwargs.items()):
            if isinstance(value, Field):
                if value.data is not None:
                    # Validate scale is allowed
                    if value.scale not in self._allowed_scales:
                        raise ValueError(
                            f"Field '{name}' has scale {value.scale.name} which is not "
                            f"supported by {type(self).__name__}. "
                            f"Allowed scales: {[s.name for s in self._allowed_scales]}"
                        )
                    object.__setattr__(self, name, value)
                kwargs.pop(name)

        # Assign Metadata descriptors from remaining kwargs
        for name, desc in self._get_metadata().items():
            if name in kwargs:
                value = kwargs.pop(name)
                setattr(self, desc.private_name, value)
            else:
                setattr(self, desc.private_name, None)

        # Let subclass handle remaining kwargs
        self._init_from_kwargs(kwargs)

        # Validate field sizes match hierarchy
        self._validate_sizes()

    def _init_from_kwargs(self, kwargs: dict) -> None:
        """Handle subclass-specific initialization. Override in subclasses."""
        if kwargs:
            raise TypeError(
                f"__init__() got unexpected keyword arguments: {list(kwargs.keys())}"
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Attribute Access - for Field unwrapping and validation
    # ─────────────────────────────────────────────────────────────────────────

    def __getattribute__(self, name: str):
        """Intercept attribute access to unwrap Field data."""
        value = object.__getattribute__(self, name)
        if isinstance(value, Field):
            return value.data
        return value

    def __setattr__(self, name: str, value) -> None:
        """Intercept attribute assignment to validate and update Field data."""
        # Check if this is an existing Field
        try:
            existing = object.__getattribute__(self, name)
            if isinstance(existing, Field):
                # Validate if value is not None
                if value is not None:
                    hierarchy = self._get_hierarchy()
                    check_compatible(hierarchy._ref, value, name)
                    self._validate_field_size(name, value, existing.scale)
                existing.data = value
                return
        except AttributeError:
            pass

        # Check if this is a known field name that doesn't exist yet
        if name in _KNOWN_FIELDS and value is not None:
            scale = _KNOWN_FIELDS[name]
            # Validate scale is allowed
            if scale not in self._allowed_scales:
                raise ValueError(
                    f"Field '{name}' has scale {scale.name} which is not "
                    f"supported by {type(self).__name__}"
                )
            hierarchy = self._get_hierarchy()
            check_compatible(hierarchy._ref, value, name)
            self._validate_field_size(name, value, scale)
            new_field = Field(value, scale)
            object.__setattr__(self, name, new_field)
            return

        # Not a Field - use normal setattr
        object.__setattr__(self, name, value)

    # ─────────────────────────────────────────────────────────────────────────
    # Dynamic Fields - user-registered per-scale data
    # ─────────────────────────────────────────────────────────────────────────

    def annotate(
        self,
        name: str,
        data: Array,
        scale: Scale = Scale.ATOM,
    ) -> "AtomContainer":
        """
        Register a new dynamic field on this container.

        Dynamic fields work exactly like built-in fields: they are accessible
        as attributes, propagate through selections, and convert with backend changes.

        Args:
            name: Field name. Must not conflict with existing attributes.
            data: Array data with first dimension matching scale size.
            scale: Scale at which the field is defined (default: ATOM).

        Returns:
            Self, for method chaining.

        Raises:
            ValueError: If field already exists, name starts with underscore,
                or scale is not supported by this container.
            ValueError: If data size doesn't match scale.
            TypeError: If backend/device doesn't match container.
        """
        # Validate scale is allowed
        if scale not in self._allowed_scales:
            raise ValueError(
                f"Scale {scale.name} is not supported by {type(self).__name__}. "
                f"Allowed scales: {[s.name for s in self._allowed_scales]}"
            )

        # Check if already exists
        try:
            object.__getattribute__(self, name)
            raise ValueError(f"Field '{name}' already exists.")
        except AttributeError:
            pass

        if name.startswith('_'):
            raise ValueError("Field names cannot start with underscore")

        # Validate backend/device compatibility
        hierarchy = self._get_hierarchy()
        check_compatible(hierarchy._ref, data, name)

        # Validate size
        self._validate_field_size(name, data, scale)

        # Create and store Field
        field = Field(data, scale)
        object.__setattr__(self, name, field)
        return self

    # ─────────────────────────────────────────────────────────────────────────
    # Field/Metadata Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _get_hierarchy(self) -> _Hierarchy:
        """Get the internal hierarchy object, bypassing descriptor interception."""
        return object.__getattribute__(self, '_hierarchy')

    def _get_field_data(self, name: str) -> Array | None:
        """Get field data, returning None if field doesn't exist."""
        try:
            field = object.__getattribute__(self, name)
            if isinstance(field, Field):
                return field.data
        except AttributeError:
            pass
        return None

    def _validate_field_size(self, name: str, value, scale: Scale) -> None:
        """Validate that a field value has the expected size for its scale.

        Args:
            name: Field name (for error messages).
            value: Array or sequence to validate.
            scale: Expected scale of the field.

        Raises:
            ValueError: If size doesn't match hierarchy size at scale.
        """
        expected = self._get_hierarchy().size(scale)
        actual = value.shape[0] if hasattr(value, 'shape') else len(value)
        if actual != expected:
            raise ValueError(
                f"Shape mismatch for '{name}': got {actual} elements, "
                f"expected {expected} ({scale.name} scale)"
            )

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
        result = {}
        for klass in cls.__mro__:
            for name, attr in vars(klass).items():
                if isinstance(attr, Metadata) and name not in result:
                    result[name] = attr
        return result

    def _validate_sizes(self) -> None:
        """Validate that field sizes and devices match hierarchy."""
        hierarchy = self._get_hierarchy()
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

    # ─────────────────────────────────────────────────────────────────────────
    # Size and Structure
    # ─────────────────────────────────────────────────────────────────────────

    def empty(self) -> bool:
        """Check if the container has no atoms."""
        return self._hierarchy.empty()

    def size(self, scale: Scale = Scale.ATOM) -> int:
        """
        Get the count at a specific scale.

        Args:
            scale: Scale level (ATOM, RESIDUE, CHAIN, MOLECULE).

        Returns:
            Number of units at the specified scale.

        Raises:
            ValueError: If scale is not supported by this container.
        """
        if scale not in self._allowed_scales and scale != Scale.MOLECULE:
            raise ValueError(
                f"Scale {scale.name} not supported by {type(self).__name__}"
            )
        return self._hierarchy.size(scale)

    def __len__(self) -> int:
        """Return the number of atoms."""
        return self.size()

    # ─────────────────────────────────────────────────────────────────────────
    # Backend Properties
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def backend(self) -> str:
        """Get the array backend type ('numpy' or 'torch')."""
        from ..backend import get_backend
        return get_backend(self._hierarchy._ref).value

    @property
    def device(self) -> str | None:
        """Get the device of the arrays (None for NumPy)."""
        from ..backend import get_device
        return get_device(self._hierarchy._ref)

    # ─────────────────────────────────────────────────────────────────────────
    # Slicing Helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _index_copy(arr: Array, selector) -> Array:
        """Index array with selector, ensuring a copy is returned."""
        result = arr[selector]
        if isinstance(selector, slice):
            return result.copy() if hasattr(result, 'copy') else result.clone()
        return result

    def _slice_all(
        self,
        atom_sel: Array | slice,
        res_sel: Array | slice | None,
        chain_sel: Array | slice,
    ) -> dict:
        """
        Slice all Field and Metadata attributes according to their scale.

        Args:
            atom_sel: Boolean mask or slice for atoms.
            res_sel: Boolean mask or slice for residues (None if no residues).
            chain_sel: Boolean mask or slice for chains.

        Returns:
            Dict mapping field/metadata names to sliced values.
        """
        result = {}
        field_meta = {}  # Store scale for each field

        # Slice Field objects
        for name, field in self._get_fields().items():
            field_meta[name] = field.scale
            if field.scale == Scale.ATOM:
                result[name] = self._index_copy(field.data, atom_sel)
            elif field.scale == Scale.RESIDUE:
                if res_sel is None:
                    raise ValueError(f"Cannot slice RESIDUE field '{name}' without residue data")
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

    def _convert_backend(self, to_func) -> dict:
        """
        Convert all Field arrays to a target backend, pass through Metadata.

        Args:
            to_func: Function to convert arrays (e.g., to_numpy, to_torch).

        Returns:
            Dict mapping all field/metadata names to converted/passed values.
        """
        result = {}
        field_meta = {}

        # Convert Field arrays
        for name, field in self._get_fields().items():
            field_meta[name] = field.scale
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

    def _clone(self, **overrides) -> "AtomContainer":
        """
        Create a copy of this container with optional field overrides.

        Collects all Field and Metadata values, applies overrides, and
        constructs a new instance. This unified implementation handles:
        - Field reconstruction with optional scale overrides
        - Known fields that don't exist on source (e.g., adding coordinates)
        - Metadata copying with list protection
        - Subclass-specific internal state via _copy_internal_state hook

        Args:
            **overrides: Field values to override. Can include any field/metadata
                name (coordinates, atoms, pdb_id, etc.), 'hierarchy' for the
                hierarchy, or '_field_meta' for field scale/dtype info.

        Returns:
            New container with the specified overrides applied.
        """
        # Extract hierarchy (used for new instance)
        hierarchy = overrides.pop('hierarchy', self._get_hierarchy())

        # Extract field metadata (for dynamic fields from slicing/conversion)
        field_meta = overrides.pop('_field_meta', None)

        # Create new instance bypassing __init__ for efficiency
        instance = object.__new__(type(self))
        object.__setattr__(instance, '_hierarchy', hierarchy)

        # Reconstruct Fields (only if data is not None)
        current_fields = self._get_fields()
        for name, field in current_fields.items():
            # Get scale from override metadata or original field
            if field_meta and name in field_meta:
                scale = field_meta[name]
            else:
                scale = field.scale

            # Get data from override or original
            if name in overrides:
                data = overrides.pop(name)
            else:
                data = field.data

            # Only set attribute if data exists
            if data is not None:
                new_field = Field(data, scale)
                object.__setattr__(instance, name, new_field)

        # Handle overrides for known fields that don't exist on source
        # (e.g., adding coordinates to a template)
        for name in list(overrides.keys()):
            if name in _KNOWN_FIELDS:
                data = overrides.pop(name)
                if data is not None:
                    scale = _KNOWN_FIELDS[name]
                    new_field = Field(data, scale)
                    object.__setattr__(instance, name, new_field)

        # Copy Metadata descriptors (with list protection)
        for name, desc in self._get_metadata().items():
            if name in overrides:
                value = overrides.pop(name)
            else:
                value = getattr(self, desc.private_name, None)
            # Copy lists to avoid mutation
            if desc.is_list and value is not None:
                value = list(value)
            setattr(instance, desc.private_name, value)

        # Let subclasses handle their internal state
        self._copy_internal_state(instance, overrides)

        return instance

    def _copy_internal_state(self, instance: "AtomContainer", overrides: dict) -> None:
        """
        Copy subclass-specific internal state to new instance.

        Subclasses should override this to handle their internal attributes
        (e.g., cached computed values, references to related objects).

        Args:
            instance: The new instance being created.
            overrides: Remaining overrides dict (subclass can pop values).
        """
        pass  # Base implementation has no internal state

    def copy(self, **overrides) -> "AtomContainer":
        """
        Return a copy of this container, optionally with field overrides.

        Args:
            **overrides: Field values to override.

        Returns:
            New container with specified fields overridden (or all cloned if none).
        """
        # Validate device compatibility for array overrides
        hierarchy = self._get_hierarchy()
        for name, value in overrides.items():
            if value is not None and hasattr(value, 'shape'):
                check_compatible(hierarchy._ref, value, name)

        # Clone all Field data
        cloned = {}
        for name, field in self._get_fields().items():
            if name in overrides:
                cloned[name] = overrides.pop(name)
            else:
                cloned[name] = ops.clone(field.data)

        # Apply remaining overrides (metadata, etc.)
        cloned.update(overrides)

        return self._clone(**cloned)

    # ─────────────────────────────────────────────────────────────────────────
    # Backend Conversion
    # ─────────────────────────────────────────────────────────────────────────

    def numpy(self) -> "AtomContainer":
        """
        Convert all arrays to NumPy.

        Returns:
            New container with NumPy arrays. If already NumPy, returns self.
        """
        if self.backend == "numpy":
            return self

        new_hierarchy = self._hierarchy.numpy()
        converted = self._convert_backend(to_numpy)
        converted['hierarchy'] = new_hierarchy
        return self._clone(**converted)

    def torch(self) -> "AtomContainer":
        """
        Convert all arrays to PyTorch tensors.

        Returns:
            New container with PyTorch tensors. If already PyTorch, returns self.

        Raises:
            ImportError: If PyTorch is not installed.
        """
        from ..backend import to_torch
        if self.backend == "torch":
            return self

        new_hierarchy = self._hierarchy.torch()
        converted = self._convert_backend(to_torch)
        converted['hierarchy'] = new_hierarchy
        return self._clone(**converted)

    def to(
        self,
        device: "str | torch.device | None" = None,
        dtype: "torch.dtype | None" = None,
    ) -> "AtomContainer":
        """
        Move tensors to device and/or convert dtype (torch backend only).

        Args:
            device: Target device (e.g., 'cuda', 'cpu').
            dtype: Target dtype for float tensors only.

        Returns:
            New container with tensors on the specified device/dtype.

        Raises:
            ValueError: If called on NumPy backend.
        """
        if self.backend != "torch":
            raise ValueError("to() is only supported for torch backend. "
                           "Use container.torch().to(...) to convert first.")

        if device is None and dtype is None:
            return self

        # Convert Fields based on array dtype (float vs int)
        converted = {}
        for name, field in self._get_fields().items():
            result = field.data
            if device is not None:
                result = result.to(device)
            # Only apply dtype conversion to float tensors
            if dtype is not None and result.is_floating_point():
                result = result.to(dtype)
            converted[name] = result

        # Move hierarchy to device (int tensors only, no dtype change)
        hierarchy = self._get_hierarchy()
        if device is not None:
            converted['hierarchy'] = hierarchy.to(device)

        return self._clone(**converted)

    def cpu(self) -> "AtomContainer":
        """Move tensors to CPU (torch backend only)."""
        return self.to(device="cpu")

    def cuda(self) -> "AtomContainer":
        """Move tensors to CUDA (torch backend only)."""
        return self.to(device="cuda")

    # ─────────────────────────────────────────────────────────────────────────
    # Selection Operations
    # ─────────────────────────────────────────────────────────────────────────

    def __getitem__(self, key: Array | slice) -> "AtomContainer":
        """
        Select atoms by boolean mask or slice.

        Args:
            key: Boolean mask of atoms to keep, or slice for contiguous range.

        Returns:
            New container with selected atoms.
        """
        return self.select(key, Scale.ATOM)

    def select(self, selector: Array | int | list | slice, scale: Scale) -> "AtomContainer":
        """
        Select units at the specified scale.

        Args:
            selector: Selection criteria (mask, index, indices, or slice).
            scale: Scale of selection (ATOM, RESIDUE, or CHAIN).

        Returns:
            New container with selected units.

        Raises:
            ValueError: If scale is not supported by this container.
        """
        if scale not in self._allowed_scales:
            raise ValueError(
                f"Selection at {scale.name} scale not supported by {type(self).__name__}"
            )

        # Convert indices to mask if needed
        mask = self._to_mask(selector, scale)
        return self._select(mask, scale)

    def _to_mask(self, selector: Array | int | list | slice, scale: Scale) -> Array:
        """Convert a selector to a boolean mask."""
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
            indices = selector.tolist()
        else:
            return selector

        # Validate indices and create mask
        for ix in indices:
            if ix < 0 or ix >= max_size:
                raise IndexError(
                    f"{scale.name} index {ix} out of range for container with {max_size} {scale.name.lower()}s"
                )

        mask = ops.zeros(max_size, like=self._hierarchy._ref, dtype='bool')
        for ix in indices:
            mask[ix] = True
        return mask

    def _select(self, mask: Array, scale: Scale) -> "AtomContainer":
        """
        Unified selection implementation for all scales.

        Uses the hierarchy to derive masks, compute new sizes/lengths,
        and slice fields accordingly.
        """
        # Derive masks at all scales
        remove_empty_residues = (scale == Scale.ATOM) and self._hierarchy.has_scale(Scale.RESIDUE)
        masks = self._hierarchy.derive_masks(mask, scale, remove_empty_residues)

        # Compute new hierarchy for selection
        new_per = self._hierarchy.compute_per(masks)
        new_hierarchy = _Hierarchy(new_per, self._hierarchy._ref)

        # Extract masks for _slice_all
        atom_mask = masks[Scale.ATOM]
        res_mask = masks.get(Scale.RESIDUE)
        chn_mask = masks.get(Scale.CHAIN)

        # Slice all fields and annotations
        sliced = self._slice_all(atom_mask, res_mask, chn_mask)
        sliced['hierarchy'] = new_hierarchy

        return self._clone(**sliced)
