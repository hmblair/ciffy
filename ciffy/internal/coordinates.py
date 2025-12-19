"""
Coordinate management with dual representation support.

Provides the CoordinateManager class that manages both Cartesian and internal
coordinate representations with lazy evaluation and automatic conversion.
"""

from __future__ import annotations

import numpy as np

from ..backend import Array, is_torch, to_numpy, to_torch, check_compatible, has_nan, has_inf
from ..backend.dispatch import (
    ZMatrix,
    ConnectedComponents,
    TopologyInfo,
    build_bond_graph_csr,
    cartesian_to_internal,
)
from ..backend.graph import nerf_reconstruct
from ..types import DihedralType


# ─────────────────────────────────────────────────────────────────────────────
# Backend Polymorphism Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _empty_array(dtype, backend_like: Array) -> Array:
    """Create empty 1D array matching backend of backend_like."""
    if is_torch(backend_like):
        import torch
        if dtype == np.float32:
            torch_dtype = torch.float32
        elif dtype == np.float64:
            torch_dtype = torch.float64
        else:
            torch_dtype = dtype
        return torch.tensor([], dtype=torch_dtype, device=backend_like.device)
    return np.array([], dtype=dtype)


def _concat(arrays: list, backend_like: Array) -> Array:
    """Concatenate arrays matching backend of backend_like."""
    if is_torch(backend_like):
        import torch
        return torch.cat(arrays)
    return np.concatenate(arrays)


# ─────────────────────────────────────────────────────────────────────────────
# CoordinateManager Class
# ─────────────────────────────────────────────────────────────────────────────


class CoordinateManager:
    """
    Manages dual representation of molecular coordinates with lazy evaluation.

    Stores both Cartesian (XYZ) and internal (bond lengths, angles, dihedrals)
    coordinate representations, automatically converting between them as needed.
    Uses dirty flags to track validity and avoid redundant conversions.

    Attributes:
        coordinates: (N, 3) array of Cartesian XYZ positions.
        distances: (N,) array of bond lengths in Angstroms.
        angles: (N,) array of bond angles in radians.
        dihedrals: (N,) array of dihedral angles in radians.
        zmatrix: Z-matrix structure defining coordinate references.

    Example:
        >>> # Create manager (typically done by Polymer)
        >>> manager = CoordinateManager(coordinates, polymer)
        >>>
        >>> # Access Cartesian coordinates
        >>> coords = manager.coordinates
        >>>
        >>> # Access internal coordinates (auto-computed if needed)
        >>> dihedrals = manager.dihedrals
        >>>
        >>> # Get specific named dihedrals
        >>> phi = manager.get_dihedral(DihedralType.PHI)
    """

    __slots__ = (
        # Cartesian representation
        '_coordinates',

        # Internal representation
        '_internal',
        '_zmatrix',  # ZMatrix object wrapping (M, 4) int64 array

        # Structural metadata (injected, not owned)
        '_topology',    # TopologyInfo (immutable reference)
        '_components',  # ConnectedComponents for reconstruction

        '_is_torch',  # Cached backend flag
    )

    def __init__(
        self,
        coordinates: Array,
        topology: "TopologyInfo",
    ) -> None:
        """
        Initialize coordinate manager with Cartesian coordinates.

        Args:
            coordinates: (N, 3) array of Cartesian XYZ positions.
            topology: TopologyInfo containing structural metadata.
        """
        self._topology = topology

        # Initialize Cartesian representation as valid
        self._coordinates: Array | None = coordinates
        self._is_torch = is_torch(coordinates) if coordinates is not None else False

        # Initialize internal representation as invalid (not yet computed)
        self._internal: Array | None = None
        self._zmatrix: ZMatrix | None = None

        # Connected components are built lazily in _recompute_internal
        # when the bond graph is computed for z-matrix construction
        self._components: "ConnectedComponents | None" = None

    @classmethod
    def _from_slice(
        cls,
        coordinates: Array,
        is_torch_flag: bool,
    ) -> "CoordinateManager":
        """
        Create a CoordinateManager from sliced coordinates.

        This factory method bypasses __init__ to create a manager with
        only Cartesian coordinates. Used by __getitem__ for efficient slicing.

        The caller MUST set _topology before internal coordinates can be
        accessed. This is an internal API with explicit contracts.

        Args:
            coordinates: (N, 3) sliced Cartesian coordinates.
            is_torch_flag: Whether coordinates are a PyTorch tensor.

        Returns:
            CoordinateManager with only Cartesian representation initialized.

        Contract:
            - _coordinates: Set to provided coordinates
            - _is_torch: Set to provided flag
            - _topology: Set to None (caller must set)
            - _internal: Set to None (lazy computation)
            - _zmatrix: Set to None (lazy computation)
            - _components: Set to None (lazy computation)
        """
        manager = cls.__new__(cls)
        manager._coordinates = coordinates
        manager._is_torch = is_torch_flag
        manager._topology = None  # Must be set by caller
        manager._internal = None
        manager._zmatrix = None
        manager._components = None
        return manager

    # ─────────────────────────────────────────────────────────────────────
    # Number of atoms
    # ─────────────────────────────────────────────────────────────────────

    def size(self) -> int:
        """Return the number of atoms in the CoordinateManager."""
        if self._coordinates is not None:
            return len(self._coordinates)
        if self._internal is not None:
            return len(self._internal)
        raise ValueError("Invalid CoordinateManager.")

    # ─────────────────────────────────────────────────────────────────────
    # String Representation
    # ─────────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        """Return string representation of CoordinateManager."""
        n_atoms = self.size()
        backend = "torch" if is_torch(self._get_reference_array()) else "numpy"
        status = []
        if self._coordinates is not None:
            status.append("cartesian")
        if self._internal is not None:
            status.append("internal")
        status_str = "+".join(status) if status else "empty"
        return f"CoordinateManager({n_atoms} atoms, {backend}, {status_str})"

    # ─────────────────────────────────────────────────────────────────────
    # Helper Methods
    # ─────────────────────────────────────────────────────────────────────

    def _get_reference_array(self) -> Array:
        """Get a reference array for backend detection."""
        if self._coordinates is not None:
            return self._coordinates
        elif self._internal is not None:
            return self._internal

    # ─────────────────────────────────────────────────────────────────────
    # Lazy Evaluation Properties - Cartesian
    # ─────────────────────────────────────────────────────────────────────

    @property
    def coordinates(self) -> Array:
        """
        Cartesian coordinates with lazy reconstruction.

        Returns:
            (N, 3) array of XYZ positions in Angstroms.

        Note:
            If Cartesian representation is invalid, automatically reconstructs
            from internal coordinates using the NERF algorithm.
        """
        if self._coordinates is None:
            self._recompute_cartesian()
        return self._coordinates

    @coordinates.setter
    def coordinates(self, value: Array) -> None:
        """
        Set Cartesian coordinates and invalidate internal representation.

        Args:
            value: (N, 3) array of XYZ positions.
        """
        check_compatible(self._get_reference_array(), value, "coordinates")
        self._coordinates = value
        self._internal = None

    # ─────────────────────────────────────────────────────────────────────
    # Lazy Evaluation Properties - Internal
    # ─────────────────────────────────────────────────────────────────────

    @property
    def internal(self) -> Array:
        """
        Internal coordinates with lazy computation.

        Returns:
            (N,3) array of internal coordinates

        Note:
            If internal representation is invalid, automatically computes
            from Cartesian coordinates.
        """
        if self._internal is None:
            self._recompute_internal()
        return self._internal

    @internal.setter
    def internal(self, value: Array) -> None:
        """
        Set dihedral angles and invalidate Cartesian representation.

        Args:
            value: (N,) array of dihedral angles in radians.
        """
        check_compatible(self._get_reference_array(), value, "internal")
        self._internal = value
        self._coordinates = None


    @property
    def distances(self) -> Array:
        """
        Bond lengths with lazy computation.

        Returns:
            (N,) array of bond lengths in Angstroms.

        Note:
            If internal representation is invalid, automatically computes
            from Cartesian coordinates.
        """
        return self.internal[:, 0]

    @distances.setter
    def distances(self, value: Array) -> None:
        """
        Set bond lengths and invalidate Cartesian representation.

        Args:
            value: (N,) array of bond lengths in Angstroms.
        """
        # Ensure internal coordinates are computed
        if self._internal is None:
            self._recompute_internal()

        # Check compatibility
        check_compatible(self._get_reference_array(), value, "distances")

        # Copy and detach internal array to avoid graph accumulation
        if is_torch(self._internal):
            new_internal = self._internal.detach().clone()
        else:
            new_internal = self._internal.copy()

        # Update distance column
        new_internal[:, 0] = value

        # Use internal setter to trigger Cartesian invalidation
        self.internal = new_internal

    @property
    def angles(self) -> Array:
        """
        Bond angles with lazy computation.

        Returns:
            (N,) array of bond angles in radians.

        Note:
            If internal representation is invalid, automatically computes
            from Cartesian coordinates.
        """
        return self.internal[:, 1]

    @angles.setter
    def angles(self, value: Array) -> None:
        """
        Set bond angles and invalidate Cartesian representation.

        Args:
            value: (N,) array of bond angles in radians.
        """
        # Ensure internal coordinates are computed
        if self._internal is None:
            self._recompute_internal()

        # Check compatibility
        check_compatible(self._get_reference_array(), value, "angles")

        # Copy and detach internal array to avoid graph accumulation
        if is_torch(self._internal):
            new_internal = self._internal.detach().clone()
        else:
            new_internal = self._internal.copy()

        # Update angle column
        new_internal[:, 1] = value

        # Use internal setter to trigger Cartesian invalidation
        self.internal = new_internal

    @property
    def dihedrals(self) -> Array:
        """
        Dihedral angles with lazy computation.

        Returns:
            (N,) array of dihedral angles in radians.

        Note:
            If internal representation is invalid, automatically computes
            from Cartesian coordinates.
        """
        return self.internal[:, 2]

    @dihedrals.setter
    def dihedrals(self, value: Array) -> None:
        """
        Set dihedral angles and invalidate Cartesian representation.

        Args:
            value: (N,) array of dihedral angles in radians.
        """
        # Ensure internal coordinates are computed
        if self._internal is None:
            self._recompute_internal()

        # Check compatibility
        check_compatible(self._get_reference_array(), value, "dihedrals")

        # Copy and detach internal array to avoid graph accumulation
        if is_torch(self._internal):
            new_internal = self._internal.detach().clone()
        else:
            new_internal = self._internal.copy()

        # Update dihedral column
        new_internal[:, 2] = value

        # Use internal setter to trigger Cartesian invalidation
        self.internal = new_internal

    @property
    def zmatrix(self) -> "ZMatrix":
        """
        Z-matrix structure (read-only).

        Returns:
            ZMatrix object wrapping (M, 4) int64 array.

        Note:
            If Z-matrix hasn't been built yet, triggers internal coordinate
            computation which builds the Z-matrix.
        """
        if self._zmatrix is None:
            # Trigger internal computation to build Z-matrix
            _ = self.dihedrals
        return self._zmatrix

    @property
    def zmatrix_indices(self) -> Array:
        """
        Z-matrix structure as raw array (read-only).

        Returns:
            (M, 4) int64 array [atom_idx, dist_ref, ang_ref, dih_ref]
            where -1 indicates "no reference" (for root atoms).
        """
        return self.zmatrix.indices

    # ─────────────────────────────────────────────────────────────────────
    # Recomputation Methods
    # ─────────────────────────────────────────────────────────────────────

    def _recompute_internal(self) -> None:
        """
        Recompute internal coordinates from Cartesian.

        Builds Z-matrix and connected components (if not cached) from the bond
        graph, then computes bond lengths, angles, and dihedrals from current
        Cartesian coordinates.
        """
        if self._coordinates is None:
            raise RuntimeError("Cannot compute internal coordinates: Cartesian coordinates are None")

        coords = self._coordinates
        n_atoms = coords.shape[0]

        # Build Z-matrix and connected components if not already cached
        if self._zmatrix is None:
            if self._topology is None:
                raise RuntimeError(
                    "Cannot compute internal coordinates without topology. "
                    "This CoordinateManager was created by slicing and doesn't have "
                    "access to bond information needed for Z-matrix construction."
                )

            # Build bond graph CSR once (used for both z-matrix and components)
            csr_offsets, csr_neighbors, _ = build_bond_graph_csr(self._topology)

            # Build connected components from bond graph (includes isolated atoms)
            self._components = ConnectedComponents.from_bond_graph(
                csr_offsets, csr_neighbors, coords, n_atoms
            )

            # Build z-matrix (reuse CSR to avoid redundant computation)
            self._zmatrix = ZMatrix.from_topology(
                self._topology, csr_offsets, csr_neighbors
            )
        elif self._components is not None:
            # Z-matrix exists but coordinates changed - update anchor coords
            # Uses vectorized gather for efficiency (O(1) GPU ops instead of O(n) Python loop)
            self._components.anchor_coords = self._components.get_anchor_coords(coords)

        # Use wrapper function that handles C/Python dispatch
        # Returns (M, 3) array where each row is [distance, angle, dihedral]
        self._internal = cartesian_to_internal(coords, self._zmatrix.indices)

    def _recompute_cartesian(self) -> None:
        """
        Recompute Cartesian coordinates from internal.

        Uses NERF (Natural Extension Reference Frame) algorithm to reconstruct
        3D coordinates from bond lengths, angles, and dihedrals. When anchor
        coordinates are available, atoms are placed directly in the correct
        reference frame without needing post-reconstruction Kabsch rotation.
        """
        if self._internal is None:
            raise RuntimeError("Cannot reconstruct Cartesian coordinates: internal coordinates are None")

        if self._zmatrix is None:
            raise RuntimeError("Cannot reconstruct Cartesian coordinates: Z-matrix is None")

        if self._components is None:
            raise RuntimeError("Cannot reconstruct Cartesian coordinates: connected components not computed")

        zmatrix_indices = self._zmatrix.indices

        # Detach internal coords (distances/angles columns) if they came from a previous
        # cartesian_to_internal call with requires_grad. This prevents errors when:
        # 1. User did to_internal with grad-enabled coords, called backward()
        # 2. User now does to_cartesian with grad-enabled dihedrals
        # The old graph was freed, so we must detach non-gradient columns.
        # Gradients for to_cartesian should flow through dihedrals only anyway.
        internal = self._internal
        if is_torch(internal) and internal.requires_grad:
            # Clone and selectively enable grad on dihedral column only
            internal = internal.detach().clone()
            internal[:, 2] = self._internal[:, 2]  # Preserve grad for dihedrals

        # Get anchor coordinates and component IDs for anchored NERF
        # This eliminates the need for post-reconstruction Kabsch rotation
        # Detach anchor_coords to avoid grad history from previous computations
        anchor_coords = self._components.anchor_coords
        if is_torch(anchor_coords) and anchor_coords.requires_grad:
            anchor_coords = anchor_coords.detach()
        component_ids = self._zmatrix.component_ids

        # NERF reconstruction with anchored placement
        coords = nerf_reconstruct(
            zmatrix_indices,
            internal,
            component_offsets=self._zmatrix.component_offsets,
            anchor_coords=anchor_coords,
            component_ids=component_ids,
        )

        # Clone coords for any in-place modifications below (preserves autograd graph)
        if is_torch(coords):
            coords = coords.clone()
        else:
            coords = coords.copy()

        self._coordinates = coords
        self._validate_coordinates()

    def _validate_coordinates(self) -> None:
        """
        Validate coordinates after reconstruction.

        Raises:
            ValueError: If coordinates contain NaN, Inf, or unreasonable values.
        """
        coords = self._coordinates
        if has_nan(coords):
            raise ValueError("Invalid coordinates after reconstruction (NaN detected)")
        if has_inf(coords):
            raise ValueError("Invalid coordinates after reconstruction (Inf detected)")

    # ─────────────────────────────────────────────────────────────────────
    # Named Dihedral API
    # ─────────────────────────────────────────────────────────────────────

    def get_dihedral(
        self,
        dtype: DihedralType | list[DihedralType] | tuple[DihedralType, ...],
    ) -> Array:
        """
        Get specific named dihedral angles using array masking.

        Args:
            dtype: Type(s) of dihedral to retrieve. Can be a single DihedralType
                or a list/tuple of DihedralTypes.

        Returns:
            Array of dihedral values in radians. For multiple types, values are
            concatenated in the order specified. Returns empty array if none found.

        Example:
            >>> phi = manager.get_dihedral(DihedralType.PHI)
            >>> backbone = manager.get_dihedral([DihedralType.PHI, DihedralType.PSI])
        """
        # Ensure internal coordinates are computed (use property accessor)
        dihedrals = self.dihedrals

        # Get dihedral types from ZMatrix (single source of truth)
        dihedral_types = self._zmatrix.dihedral_types if self._zmatrix else None
        if dihedral_types is None:
            return _empty_array(dihedrals.dtype, dihedrals)

        # Handle single type - DihedralType is IntEnum, use .value directly
        if isinstance(dtype, DihedralType):
            mask = dihedral_types == dtype.value
            return dihedrals[mask]

        # Handle multiple types - concatenate in order
        arrays = []
        for dt in dtype:
            mask = dihedral_types == dt.value
            values = dihedrals[mask]
            if len(values) > 0:
                arrays.append(values)

        if not arrays:
            return _empty_array(dihedrals.dtype, dihedrals)

        return _concat(arrays, dihedrals)

    def set_dihedral(
        self,
        dtype: DihedralType | list[DihedralType] | tuple[DihedralType, ...],
        values: Array,
    ) -> None:
        """
        Set specific named dihedral angles using array masking.

        Args:
            dtype: Type(s) of dihedral to set. Can be a single DihedralType
                or a list/tuple of DihedralTypes.
            values: New dihedral values in radians. For multiple types, values
                should be concatenated in the same order as dtype list.

        Raises:
            ValueError: If the specified dihedral type is not found in the structure,
                or if the number of values doesn't match the expected count.

        Example:
            >>> # Set all phi angles to -60 degrees
            >>> manager.set_dihedral(DihedralType.PHI, np.full(n_phi, -np.pi/3))
            >>> # Set multiple types at once
            >>> manager.set_dihedral([DihedralType.PHI, DihedralType.PSI], backbone_values)
        """
        # Ensure internal coordinates are computed (use property accessor)
        internal = self.internal

        # Get dihedral types from ZMatrix (single source of truth)
        dihedral_types = self._zmatrix.dihedral_types if self._zmatrix else None
        if dihedral_types is None:
            raise ValueError("No dihedral types available")

        # Copy and detach internal array to avoid graph accumulation across iterations
        # The new values will bring their own gradients; old values should be detached
        if is_torch(internal):
            new_internal = internal.detach().clone()
        else:
            new_internal = internal.copy()

        # Handle single type - DihedralType is IntEnum, use .value directly
        if isinstance(dtype, DihedralType):
            mask = dihedral_types == dtype.value

            if is_torch(mask):
                has_dihedrals = mask.any().item()
            else:
                has_dihedrals = mask.any()

            if not has_dihedrals:
                raise ValueError(
                    f"No {dtype.name} dihedrals found in structure. "
                    f"This may be because the structure doesn't contain the appropriate molecule type."
                )

            new_internal[mask, 2] = values
            self.internal = new_internal
            return

        # Handle multiple types - split values and assign each
        offset = 0
        for dt in dtype:
            mask = dihedral_types == dt.value

            if is_torch(mask):
                count = int(mask.sum().item())
            else:
                count = int(mask.sum())

            if count == 0:
                continue

            # Extract values for this type
            new_internal[mask, 2] = values[offset:offset + count]
            offset += count

        # Use setter to trigger invalidation
        self.internal = new_internal

    # ─────────────────────────────────────────────────────────────────────
    # Backend Conversion
    # ─────────────────────────────────────────────────────────────────────

    def numpy(self) -> "CoordinateManager":
        """
        Convert all arrays to NumPy backend.

        Returns:
            New CoordinateManager with NumPy arrays.
        """
        # Create new manager with converted coordinates
        # TopologyInfo is always numpy, so we can share it
        new_manager = CoordinateManager(
            to_numpy(self._coordinates) if self._coordinates is not None else None,
            self._topology,
        )

        if self._internal is not None:
            new_manager._internal = to_numpy(self._internal)

        # Convert Z-matrix
        if self._zmatrix is not None:
            new_manager._zmatrix = self._zmatrix.numpy()

        # ConnectedComponents stores numpy arrays, so just copy reference
        new_manager._components = self._components

        return new_manager

    def torch(self) -> "CoordinateManager":
        """
        Convert all arrays to PyTorch backend.

        Returns:
            New CoordinateManager with PyTorch tensors.
        """
        # Create new manager with converted coordinates
        # TopologyInfo is always numpy, so we can share it
        new_manager = CoordinateManager(
            to_torch(self._coordinates) if self._coordinates is not None else None,
            self._topology,
        )

        if self._internal is not None:
            new_manager._internal = to_torch(self._internal)

        # Convert Z-matrix
        if self._zmatrix is not None:
            new_manager._zmatrix = self._zmatrix.torch()

        # ConnectedComponents stores numpy arrays, so just copy reference
        new_manager._components = self._components

        return new_manager

    def to(self, device: str = None, dtype=None) -> "CoordinateManager":
        """
        Move tensors to specified device and/or convert dtype (PyTorch only).

        Args:
            device: Target device (e.g., "cuda", "cpu", "mps").
            dtype: Target dtype for float tensors (e.g., torch.float16).

        Returns:
            New CoordinateManager on the specified device/dtype.

        Raises:
            RuntimeError: If arrays are not PyTorch tensors.
        """
        if not is_torch(self._get_reference_array()):
            raise RuntimeError(
                "Cannot move to device: arrays are not PyTorch tensors. "
                "Use to_torch() first."
            )

        def convert(t):
            """Apply device and/or dtype conversion."""
            if t is None:
                return None
            if device is not None:
                t = t.to(device)
            if dtype is not None:
                t = t.to(dtype)
            return t

        # Create new manager with coordinates on target device/dtype
        # TopologyInfo is always numpy, so we can share it
        new_manager = CoordinateManager(
            convert(self._coordinates),
            self._topology,
        )

        # Move internal coordinates if valid
        if self._internal is not None:
            new_manager._internal = convert(self._internal)

        # Copy Z-matrix (move to device if PyTorch, keep int dtype)
        if self._zmatrix is not None:
            new_manager._zmatrix = self._zmatrix.to(device) if device is not None else self._zmatrix

        # ConnectedComponents stores numpy arrays, so just copy reference
        new_manager._components = self._components

        return new_manager

    def detach(self) -> "CoordinateManager":
        """
        Detach all tensors from their computation graphs (PyTorch only).

        This is useful after calling `backward()` on a computation that used
        this manager's coordinates or internal coordinates. After backward(),
        the cached tensors retain grad_fn pointers to freed computation graphs.
        Calling detach() clears these pointers, allowing the manager to be
        reused for new gradient computations.

        Returns:
            Self, for method chaining.

        Example:
            >>> # Compute gradients through to_internal
            >>> coords = polymer.coordinates.clone().requires_grad_(True)
            >>> polymer.coordinates = coords
            >>> loss = polymer.dihedrals.sum()
            >>> loss.backward()
            >>>
            >>> # Detach before next computation
            >>> polymer.detach()
            >>>
            >>> # Now safe to compute new gradients
            >>> dihedrals = polymer.dihedrals.detach().clone().requires_grad_(True)
            >>> polymer.dihedrals = dihedrals
            >>> new_loss = polymer.coordinates.sum()
            >>> new_loss.backward()

        Note:
            For NumPy arrays, this is a no-op since NumPy doesn't have
            computation graphs.
        """
        if self._coordinates is not None and is_torch(self._coordinates):
            if self._coordinates.requires_grad:
                self._coordinates = self._coordinates.detach()

        if self._internal is not None and is_torch(self._internal):
            if self._internal.requires_grad:
                self._internal = self._internal.detach()

        return self

    # ─────────────────────────────────────────────────────────────────────
    # Slicing
    # ─────────────────────────────────────────────────────────────────────

    def __getitem__(self, mask: Array) -> "CoordinateManager":
        """
        Slice coordinate manager by boolean atom mask.

        Ensures Cartesian coordinates are valid, slices them, and returns
        a new CoordinateManager with internal coordinates marked as invalid
        (to be lazily recomputed when accessed).

        Args:
            mask: (N,) boolean mask where True means keep the atom.

        Returns:
            New CoordinateManager for the sliced atoms.

        Note:
            Gradients flow through the Cartesian coordinate slicing.
            Internal coordinates are recomputed from the sliced Cartesian
            coordinates when accessed. The caller must set _topology on the
            returned manager before internal coordinates can be computed.
        """
        # Ensure Cartesian is valid
        if self._coordinates is None:
            self._recompute_cartesian()

        # Slice Cartesian coordinates
        sliced_coords = self._coordinates[mask]

        # Use factory method to create new manager with explicit contract
        return CoordinateManager._from_slice(sliced_coords, is_torch(sliced_coords))
