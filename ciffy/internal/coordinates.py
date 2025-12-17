"""
Coordinate management with dual representation support.

Provides the CoordinateManager class that manages both Cartesian and internal
coordinate representations with lazy evaluation and automatic conversion.
"""

from __future__ import annotations

import numpy as np

from ..backend import Array, is_torch, to_numpy, to_torch, check_compatible, has_nan, has_inf, any_abs_greater_than
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
        '_cartesian_valid',

        # Internal representation
        '_distances',
        '_angles',
        '_dihedrals',
        '_zmatrix',  # ZMatrix object wrapping (M, 4) int64 array

        # Structural metadata (injected, not owned)
        '_topology',    # TopologyInfo (immutable reference)
        '_components',  # ConnectedComponents for reconstruction
        '_components_centroids_valid',  # Whether centroids match current coordinates

        '_internal_valid',
        '_n_atoms',  # Total atom count (set from initial coordinates)
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
        self._coordinates = coordinates
        self._cartesian_valid = True
        self._n_atoms = len(coordinates) if coordinates is not None else 0
        self._is_torch = is_torch(coordinates) if coordinates is not None else False

        # Initialize internal representation as invalid (not yet computed)
        self._distances: Array | None = None
        self._angles: Array | None = None
        self._dihedrals: Array | None = None
        self._zmatrix: ZMatrix | None = None

        self._internal_valid = False

        # Connected components are built lazily in _recompute_internal
        # when the bond graph is computed for z-matrix construction
        self._components: "ConnectedComponents | None" = None
        self._components_centroids_valid = False

    # ─────────────────────────────────────────────────────────────────────
    # String Representation
    # ─────────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        """Return string representation of CoordinateManager."""
        n_atoms = len(self._coordinates) if self._coordinates is not None else 0
        backend = "torch" if is_torch(self._get_reference_array()) else "numpy"
        status = []
        if self._cartesian_valid:
            status.append("cartesian")
        if self._internal_valid:
            status.append("internal")
        status_str = "+".join(status) if status else "empty"
        return f"CoordinateManager({n_atoms} atoms, {backend}, {status_str})"

    # ─────────────────────────────────────────────────────────────────────
    # Helper Methods
    # ─────────────────────────────────────────────────────────────────────

    def _get_reference_array(self) -> Array:
        """Get a reference array for backend detection."""
        if self._cartesian_valid and self._coordinates is not None:
            return self._coordinates
        elif self._internal_valid and self._distances is not None:
            return self._distances
        else:
            # Fallback to topology atoms (always numpy, but sufficient for backend detection)
            return self._topology.atoms

    def _invalidate_cartesian(self) -> None:
        """Mark Cartesian representation as invalid."""
        self._cartesian_valid = False
        self._coordinates = None

    def _invalidate_internal(self) -> None:
        """Mark internal representation as invalid."""
        self._internal_valid = False
        self._distances = None
        self._angles = None
        self._dihedrals = None
        # Note: Keep Z-matrix and dihedral_types cached - they're structure-based

    def _invalidate_structure(self) -> None:
        """
        Invalidate structure-dependent caches.

        Call this when the polymer topology changes (bonds, residues, chains).
        This invalidates the Z-matrix and all dependent data.

        Note:
            This is more aggressive than _invalidate_internal() which preserves
            the Z-matrix. Use this only when the molecular structure itself changes.
        """
        self._zmatrix = None
        self._invalidate_internal()

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
        if not self._cartesian_valid:
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
        self._cartesian_valid = True

        # Mark component centroids as needing update (lazy - only rebuilt when needed)
        # The component structure (offsets, atoms) doesn't change, only centroids
        self._components_centroids_valid = False

        self._invalidate_internal()

    # ─────────────────────────────────────────────────────────────────────
    # Lazy Evaluation Properties - Internal
    # ─────────────────────────────────────────────────────────────────────

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
        if not self._internal_valid:
            self._recompute_internal()
        return self._distances

    @distances.setter
    def distances(self, value: Array) -> None:
        """
        Set bond lengths and invalidate Cartesian representation.

        Args:
            value: (N,) array of bond lengths in Angstroms.
        """
        check_compatible(self._get_reference_array(), value, "distances")
        self._distances = value
        self._internal_valid = True
        self._invalidate_cartesian()

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
        if not self._internal_valid:
            self._recompute_internal()
        return self._angles

    @angles.setter
    def angles(self, value: Array) -> None:
        """
        Set bond angles and invalidate Cartesian representation.

        Args:
            value: (N,) array of bond angles in radians.
        """
        check_compatible(self._get_reference_array(), value, "angles")
        self._angles = value
        self._internal_valid = True
        self._invalidate_cartesian()

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
        if not self._internal_valid:
            self._recompute_internal()
        return self._dihedrals

    @dihedrals.setter
    def dihedrals(self, value: Array) -> None:
        """
        Set dihedral angles and invalidate Cartesian representation.

        Args:
            value: (N,) array of dihedral angles in radians.
        """
        check_compatible(self._get_reference_array(), value, "dihedrals")
        self._dihedrals = value
        self._internal_valid = True
        self._invalidate_cartesian()

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
            self._components_centroids_valid = True

            # Build z-matrix (reuse CSR to avoid redundant computation)
            self._zmatrix = ZMatrix.from_topology(
                self._topology, csr_offsets, csr_neighbors
            )
        elif not self._components_centroids_valid and self._components is not None:
            # Z-matrix exists but centroids need updating (coordinates changed)
            # Update centroids for correct position restoration in to_cartesian
            self._components.update_centroids(coords)
            self._components_centroids_valid = True

        # Use wrapper function that handles C/Python dispatch
        self._distances, self._angles, self._dihedrals = cartesian_to_internal(
            coords, self._zmatrix.indices
        )

        self._internal_valid = True

    def _recompute_cartesian(self) -> None:
        """
        Recompute Cartesian coordinates from internal.

        Uses NERF (Natural Extension Reference Frame) algorithm to reconstruct
        3D coordinates from bond lengths, angles, and dihedrals. When anchor
        coordinates are available, atoms are placed directly in the correct
        reference frame without needing post-reconstruction Kabsch rotation.
        """
        if self._distances is None or self._angles is None or self._dihedrals is None:
            raise RuntimeError("Cannot reconstruct Cartesian coordinates: internal coordinates are None")

        if self._zmatrix is None:
            raise RuntimeError("Cannot reconstruct Cartesian coordinates: Z-matrix is None")

        if self._components is None:
            raise RuntimeError("Cannot reconstruct Cartesian coordinates: connected components not computed")

        zmatrix_indices = self._zmatrix.indices

        # Get atom count (stored from initial coordinates)
        n_atoms = self._n_atoms

        # Detach distances/angles if they came from a previous cartesian_to_internal
        # call with requires_grad. This prevents errors when:
        # 1. User did to_internal with grad-enabled coords, called backward()
        # 2. User now does to_cartesian with grad-enabled dihedrals
        # The old distances/angles graph was freed, so we must detach them.
        # Gradients for to_cartesian should flow through dihedrals only anyway.
        distances = self._distances
        angles = self._angles
        if is_torch(distances) and distances.requires_grad:
            distances = distances.detach()
            angles = angles.detach()

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
            distances,
            angles,
            self._dihedrals,
            n_atoms=n_atoms,
            level_offsets=self._zmatrix.level_offsets,
            anchor_coords=anchor_coords,
            component_ids=component_ids,
        )

        # Clone coords for any in-place modifications below (preserves autograd graph)
        if is_torch(coords):
            coords = coords.clone()
        else:
            coords = coords.copy()

        # Handle single-atom orphans - they need position restoration
        # (NERF places root atoms at origin for each component)
        # For single-atom components, anchor_coords[comp_idx, 0] is the atom's position
        n_components = self._components.n_components
        for comp_idx in range(n_components):
            component_size = self._components.get_component_size(comp_idx)
            if component_size == 1:
                component_atoms = self._components.get_component_atoms(comp_idx)
                atom_idx = int(component_atoms[0])
                if atom_idx < n_atoms:
                    coords[atom_idx] = anchor_coords[comp_idx, 0]

        self._coordinates = coords
        self._cartesian_valid = True
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
        if any_abs_greater_than(coords, 10000):
            raise ValueError(
                "Coordinates exceed 10000 Angstroms - likely reconstruction error. "
                "Check that internal coordinates are within reasonable bounds."
            )

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
        # Ensure internal coordinates are computed
        if not self._internal_valid:
            self._recompute_internal()

        # Get dihedral types from ZMatrix (single source of truth)
        dihedral_types = self._zmatrix.dihedral_types if self._zmatrix else None
        if dihedral_types is None:
            return _empty_array(self._dihedrals.dtype, self._dihedrals)

        # Handle single type - DihedralType is IntEnum, use .value directly
        if isinstance(dtype, DihedralType):
            mask = dihedral_types == dtype.value
            return self._dihedrals[mask]

        # Handle multiple types - concatenate in order
        arrays = []
        for dt in dtype:
            mask = dihedral_types == dt.value
            values = self._dihedrals[mask]
            if len(values) > 0:
                arrays.append(values)

        if not arrays:
            return _empty_array(self._dihedrals.dtype, self._dihedrals)

        return _concat(arrays, self._dihedrals)

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
        # Ensure internal coordinates are computed
        if not self._internal_valid:
            self._recompute_internal()

        # Get dihedral types from ZMatrix (single source of truth)
        dihedral_types = self._zmatrix.dihedral_types if self._zmatrix else None
        if dihedral_types is None:
            raise ValueError("No dihedral types available")

        # Copy and detach dihedrals array to avoid graph accumulation across iterations
        # The new values will bring their own gradients; old values should be detached
        if is_torch(self._dihedrals):
            new_dihedrals = self._dihedrals.detach().clone()
        else:
            new_dihedrals = self._dihedrals.copy()

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

            new_dihedrals[mask] = values
            self.dihedrals = new_dihedrals
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
            new_dihedrals[mask] = values[offset:offset + count]
            offset += count

        # Use setter to trigger invalidation
        self.dihedrals = new_dihedrals

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

        # Copy validity flags
        new_manager._cartesian_valid = self._cartesian_valid

        # Convert internal coordinates if valid
        if self._internal_valid:
            new_manager._distances = to_numpy(self._distances)
            new_manager._angles = to_numpy(self._angles)
            new_manager._dihedrals = to_numpy(self._dihedrals)
            new_manager._internal_valid = True

        # Convert Z-matrix
        if self._zmatrix is not None:
            new_manager._zmatrix = self._zmatrix.numpy()

        # ConnectedComponents stores numpy arrays, so just copy reference
        new_manager._components = self._components
        new_manager._components_centroids_valid = self._components_centroids_valid

        # Copy atom count
        new_manager._n_atoms = self._n_atoms

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

        # Copy validity flags
        new_manager._cartesian_valid = self._cartesian_valid

        # Convert internal coordinates if valid
        if self._internal_valid:
            new_manager._distances = to_torch(self._distances)
            new_manager._angles = to_torch(self._angles)
            new_manager._dihedrals = to_torch(self._dihedrals)
            new_manager._internal_valid = True

        # Convert Z-matrix
        if self._zmatrix is not None:
            new_manager._zmatrix = self._zmatrix.torch()

        # ConnectedComponents stores numpy arrays, so just copy reference
        new_manager._components = self._components
        new_manager._components_centroids_valid = self._components_centroids_valid

        # Copy atom count
        new_manager._n_atoms = self._n_atoms

        return new_manager

    def to(self, device: str) -> "CoordinateManager":
        """
        Move tensors to specified device (PyTorch only).

        Args:
            device: Target device (e.g., "cuda", "cpu", "mps").

        Returns:
            New CoordinateManager on the specified device.

        Raises:
            RuntimeError: If arrays are not PyTorch tensors.
        """
        if not is_torch(self._get_reference_array()):
            raise RuntimeError(
                "Cannot move to device: arrays are not PyTorch tensors. "
                "Use to_torch() first."
            )

        # Create new manager with coordinates on target device
        # TopologyInfo is always numpy, so we can share it
        new_manager = CoordinateManager(
            self._coordinates.to(device) if self._coordinates is not None else None,
            self._topology,
        )

        # Copy validity flags
        new_manager._cartesian_valid = self._cartesian_valid

        # Move internal coordinates if valid
        if self._internal_valid:
            new_manager._distances = self._distances.to(device)
            new_manager._angles = self._angles.to(device)
            new_manager._dihedrals = self._dihedrals.to(device)
            new_manager._internal_valid = True

        # Copy Z-matrix (move to device if PyTorch)
        if self._zmatrix is not None:
            new_manager._zmatrix = self._zmatrix.to(device)

        # ConnectedComponents stores numpy arrays, so just copy reference
        new_manager._components = self._components
        new_manager._components_centroids_valid = self._components_centroids_valid

        # Copy atom count
        new_manager._n_atoms = self._n_atoms

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

        if self._distances is not None and is_torch(self._distances):
            if self._distances.requires_grad:
                self._distances = self._distances.detach()

        if self._angles is not None and is_torch(self._angles):
            if self._angles.requires_grad:
                self._angles = self._angles.detach()

        if self._dihedrals is not None and is_torch(self._dihedrals):
            if self._dihedrals.requires_grad:
                self._dihedrals = self._dihedrals.detach()

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
        if not self._cartesian_valid:
            self._recompute_cartesian()

        # Slice Cartesian coordinates
        sliced_coords = self._coordinates[mask]

        # Create new manager without calling __init__
        new_manager = CoordinateManager.__new__(CoordinateManager)
        new_manager._coordinates = sliced_coords
        new_manager._cartesian_valid = True
        new_manager._n_atoms = len(sliced_coords)
        new_manager._is_torch = is_torch(sliced_coords)

        # Topology must be set by caller (sliced topology info)
        new_manager._topology = None

        # Internal representation starts invalid (lazy recomputation)
        new_manager._distances = None
        new_manager._angles = None
        new_manager._dihedrals = None
        new_manager._zmatrix = None
        new_manager._internal_valid = False

        # ConnectedComponents starts empty (rebuilt when internal coords accessed)
        new_manager._components = None
        new_manager._components_centroids_valid = False

        return new_manager
