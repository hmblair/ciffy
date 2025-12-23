"""
Coordinate management with dual representation and constraint support.

Provides the CoordinateManager class that manages both Cartesian and internal
coordinate representations with lazy evaluation, automatic conversion, and
constraint-aware minimal DOF representation.
"""

from __future__ import annotations

import numpy as np

from ..backend import Array, is_torch, to_numpy, to_torch, check_compatible, has_nan, has_inf
from ..backend.dispatch import (
    TopologyInfo,
    build_bond_graph_csr,
)
from .tree import SpanningTree


# ─────────────────────────────────────────────────────────────────────────────
# CoordinateManager Class
# ─────────────────────────────────────────────────────────────────────────────


class CoordinateManager:
    """
    Molecular coordinates with minimal DOF representation.

    Provides a clean interface for ML applications where users interact only
    with Cartesian coordinates and degrees of freedom (DOF). All internal
    machinery (internal coordinates, ring analysis, ring closure) is private.

    Constraints are derived from molecular topology:
    - All covalent bonds are fixed (bond lengths preserved)
    - All bond angles are fixed (angles preserved)
    - For rings, some dihedrals become dependent through closure constraints

    Public API:
        coordinates: (N, 3) Cartesian XYZ positions (get/set)
        dof: (K,) degrees of freedom in radians (get/set)
        n_dof: Number of degrees of freedom
        n_atoms: Number of atoms

    Example:
        >>> manager = polymer.coordinates_manager
        >>> manager.dof = model_output     # Set K DOF values
        >>> coords = manager.coordinates   # Get N×3 coordinates
    """

    __slots__ = (
        # Cartesian representation
        '_coordinates',

        # Internal representation (private)
        '_internal',
        '_tree',  # SpanningTree for coordinate conversion

        # Coordinate centering for float32 precision
        '_center_offset',
        '_fixed_coords',

        # Structural metadata
        '_topology',

        # Constraint analysis (private)
        '_independent_dof',  # IndependentDOF result
        '_csr_offsets',
        '_csr_neighbors',

        # Dirty flags
        '_coords_dirty',  # True if coordinates need recomputation from dof
        '_dof_dirty',     # True if dof needs recomputation from coordinates

        '_is_torch',
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
        self._coordinates: Array | None = coordinates
        self._is_torch = is_torch(coordinates) if coordinates is not None else False

        # Internal representation (computed lazily)
        self._internal: Array | None = None
        self._tree: SpanningTree | None = None
        self._center_offset: np.ndarray | None = None
        self._fixed_coords: np.ndarray | None = None

        # Constraint analysis (computed lazily)
        self._independent_dof = None
        self._csr_offsets = None
        self._csr_neighbors = None

        # Dirty flags - initially coordinates are valid, dof is dirty
        self._coords_dirty = False
        self._dof_dirty = True

    # ─────────────────────────────────────────────────────────────────────
    # Public API: coordinates
    # ─────────────────────────────────────────────────────────────────────

    @property
    def coordinates(self) -> Array:
        """
        (N, 3) Cartesian coordinates, lazily recomputed when dof changes.

        Returns:
            Array of XYZ positions in Angstroms.
        """
        if self._coords_dirty:
            self._recompute_cartesian_from_dof()
        return self._coordinates

    @coordinates.setter
    def coordinates(self, value: Array) -> None:
        """
        Set Cartesian coordinates, marks dof as dirty.

        Args:
            value: (N, 3) array of XYZ positions.
        """
        if self._coordinates is not None:
            check_compatible(self._coordinates, value, "coordinates")
        self._coordinates = value
        self._is_torch = is_torch(value)
        self._coords_dirty = False
        self._dof_dirty = True
        # Invalidate internal representation
        self._internal = None

    # ─────────────────────────────────────────────────────────────────────
    # Public API: dof (degrees of freedom)
    # ─────────────────────────────────────────────────────────────────────

    @property
    def dof(self) -> Array:
        """
        (K,) degrees of freedom in radians, lazily recomputed when coordinates change.

        Returns:
            Array of independent dihedral values.
        """
        if self._dof_dirty:
            self._recompute_dof_from_coordinates()
        return self._get_dof_values()

    @dof.setter
    def dof(self, values: Array) -> None:
        """
        Set DOF values, marks coordinates as dirty.

        Args:
            values: (K,) array of dihedral angles in radians.
        """
        self._set_dof_values(values)
        self._dof_dirty = False
        self._coords_dirty = True

    @property
    def n_dof(self) -> int:
        """Number of degrees of freedom."""
        self._ensure_constraint_analysis()
        return self._independent_dof.n_independent

    def _get_n_atoms(self) -> int:
        """Get number of atoms (internal use)."""
        if self._coordinates is not None:
            return len(self._coordinates)
        if self._internal is not None:
            return len(self._internal)
        raise ValueError("Invalid CoordinateManager.")

    # ─────────────────────────────────────────────────────────────────────
    # String Representation
    # ─────────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        """Return string representation."""
        n_atoms = self._get_n_atoms()
        # Avoid triggering constraint analysis just for repr
        if self._independent_dof is not None:
            n_dof = self._independent_dof.n_independent
            return f"CoordinateManager(n_atoms={n_atoms}, n_dof={n_dof})"
        return f"CoordinateManager(n_atoms={n_atoms})"

    # ─────────────────────────────────────────────────────────────────────
    # Private: Constraint Analysis
    # ─────────────────────────────────────────────────────────────────────

    def _ensure_constraint_analysis(self) -> None:
        """Ensure constraint analysis has been performed."""
        if self._independent_dof is not None:
            return

        from .ring_analysis import ConstraintSpec, RingAnalyzer

        # Ensure internal coordinates are computed (builds tree)
        self._ensure_internal()

        # Build bond graph CSR if not cached
        if self._csr_offsets is None:
            self._csr_offsets, self._csr_neighbors, _ = build_bond_graph_csr(self._topology)

        # Analyze constraints with default spec (all bonds and angles fixed)
        spec = ConstraintSpec(fixed_bonds="all", fixed_angles="all")
        zmatrix_indices = self._tree.to_zmatrix_indices()

        self._independent_dof = RingAnalyzer.analyze_constraints(
            self._csr_offsets,
            self._csr_neighbors,
            self._get_n_atoms(),
            zmatrix_indices,
            spec,
        )

    def _ensure_internal(self) -> None:
        """Ensure internal coordinates are computed."""
        if self._internal is None:
            self._recompute_internal()

    # ─────────────────────────────────────────────────────────────────────
    # Private: DOF Access Helpers
    # ─────────────────────────────────────────────────────────────────────

    def _get_atom_to_row_mapping(self) -> np.ndarray:
        """Build mapping from atom index to Z-matrix row index."""
        zmatrix_indices = self._tree.to_zmatrix_indices()
        n_atoms = self._get_n_atoms()
        atom_to_row = np.full(n_atoms, -1, dtype=np.int64)
        for row in range(len(zmatrix_indices)):
            atom = int(zmatrix_indices[row, 0])
            if atom < n_atoms:
                atom_to_row[atom] = row
        return atom_to_row

    def _get_dof_values(self) -> Array:
        """Get current values of independent DOF."""
        self._ensure_constraint_analysis()
        self._ensure_internal()

        atom_indices = self._independent_dof.independent_indices

        if len(atom_indices) == 0:
            if self._is_torch:
                import torch
                return torch.tensor([], dtype=self._internal.dtype,
                                    device=self._internal.device if hasattr(self._internal, 'device') else 'cpu')
            return np.array([], dtype=self._internal.dtype)

        # Get all dihedrals
        all_dihedrals = self._internal[:, 2]

        # Convert atom indices to Z-matrix row indices
        atom_to_row = self._get_atom_to_row_mapping()
        row_indices = atom_to_row[atom_indices]

        return all_dihedrals[row_indices]

    def _set_dof_values(self, new_values: Array) -> None:
        """Set independent DOF and solve ring closure."""
        self._ensure_constraint_analysis()
        self._ensure_internal()

        atom_indices = self._independent_dof.independent_indices

        if len(atom_indices) == 0:
            return

        # Validate shape
        expected_len = len(atom_indices)
        if len(new_values) != expected_len:
            raise ValueError(f"Expected {expected_len} values, got {len(new_values)}")

        # Convert atom indices to Z-matrix row indices
        atom_to_row = self._get_atom_to_row_mapping()
        row_indices = atom_to_row[atom_indices]

        # Clone internal coordinates
        if is_torch(self._internal):
            internal = self._internal.clone()
            internal[row_indices, 2] = new_values
        else:
            internal = self._internal.copy()
            internal[row_indices, 2] = new_values

        # Solve ring closure for dependent dihedrals
        if self._independent_dof.ring_constraints:
            from .ring_closure import solve_ring_closure

            # Need current coordinates for ring closure
            if self._coordinates is None:
                self._recompute_cartesian()

            internal = solve_ring_closure(
                internal=internal,
                zmatrix_indices=self._tree.to_zmatrix_indices(),
                coords=self._coordinates,
                ring_constraints=self._independent_dof.ring_constraints,
                tree=self._tree,
                fixed_coords=self._fixed_coords,
                offsets=self._center_offset,
                max_iterations=100,
                tolerance=0.01,
            )

        self._internal = internal

    # ─────────────────────────────────────────────────────────────────────
    # Private: Recomputation Methods
    # ─────────────────────────────────────────────────────────────────────

    def _recompute_internal(self) -> None:
        """Recompute internal coordinates from Cartesian."""
        if self._coordinates is None:
            raise RuntimeError("Cannot compute internal: coordinates are None")

        coords = self._coordinates
        n_atoms = coords.shape[0]

        # Build SpanningTree if needed
        if self._tree is None:
            if self._topology is None:
                raise RuntimeError("Cannot compute internal without topology")
            csr_offsets, csr_neighbors, _ = build_bond_graph_csr(self._topology)
            self._csr_offsets = csr_offsets
            self._csr_neighbors = csr_neighbors
            self._tree = SpanningTree.from_bond_graph(csr_offsets, csr_neighbors, n_atoms)

        # Convert to numpy for C backend
        coords_np = to_numpy(coords).astype(np.float32)

        # Convert Cartesian to internal with per-component centering
        internal, offsets = self._tree.cartesian_to_internal(coords_np, center=True)
        self._center_offset = offsets

        # Store fixed coords for NERF reconstruction
        if offsets is not None:
            self._fixed_coords = coords_np.copy()
            for comp_idx in range(self._tree.n_components):
                mask = self._tree.component_id == comp_idx
                self._fixed_coords[mask] -= offsets[comp_idx]
        else:
            self._fixed_coords = coords_np.copy()

        # Convert back to torch if needed
        if self._is_torch:
            import torch
            device = coords.device if hasattr(coords, 'device') else 'cpu'
            self._internal = torch.from_numpy(internal).to(device)
        else:
            self._internal = internal

    def _recompute_cartesian(self) -> None:
        """Recompute Cartesian coordinates from internal."""
        if self._internal is None:
            raise RuntimeError("Cannot reconstruct: internal is None")
        if self._tree is None:
            raise RuntimeError("Cannot reconstruct: tree is None")
        if self._fixed_coords is None:
            raise RuntimeError("Cannot reconstruct: fixed_coords is None")

        internal = self._internal
        was_torch = is_torch(internal)
        if was_torch:
            device = internal.device if hasattr(internal, 'device') else 'cpu'
            internal_np = to_numpy(internal).astype(np.float32)
        else:
            internal_np = internal.astype(np.float32)

        # NERF reconstruction
        coords = self._tree.internal_to_cartesian(
            internal_np, self._fixed_coords, offsets=self._center_offset
        )

        if was_torch:
            import torch
            coords = torch.from_numpy(coords).to(device)

        self._coordinates = coords
        self._validate_coordinates()

    def _recompute_dof_from_coordinates(self) -> None:
        """Recompute DOF from current coordinates."""
        # Just need to ensure internal is computed
        self._internal = None  # Force recomputation
        self._recompute_internal()
        self._dof_dirty = False

    def _recompute_cartesian_from_dof(self) -> None:
        """Recompute Cartesian from current DOF (internal already set)."""
        self._recompute_cartesian()
        self._coords_dirty = False

    def _validate_coordinates(self) -> None:
        """Validate coordinates after reconstruction."""
        coords = self._coordinates
        if has_nan(coords):
            raise ValueError("Invalid coordinates (NaN detected)")
        if has_inf(coords):
            raise ValueError("Invalid coordinates (Inf detected)")

    # ─────────────────────────────────────────────────────────────────────
    # Backend Conversion
    # ─────────────────────────────────────────────────────────────────────

    def numpy(self) -> "CoordinateManager":
        """Convert to NumPy backend."""
        new_manager = CoordinateManager(
            to_numpy(self._coordinates) if self._coordinates is not None else None,
            self._topology,
        )
        if self._internal is not None:
            new_manager._internal = to_numpy(self._internal)
        new_manager._tree = self._tree
        new_manager._center_offset = self._center_offset
        new_manager._fixed_coords = self._fixed_coords
        new_manager._independent_dof = self._independent_dof
        new_manager._csr_offsets = self._csr_offsets
        new_manager._csr_neighbors = self._csr_neighbors
        new_manager._coords_dirty = self._coords_dirty
        new_manager._dof_dirty = self._dof_dirty
        return new_manager

    def torch(self) -> "CoordinateManager":
        """Convert to PyTorch backend."""
        new_manager = CoordinateManager(
            to_torch(self._coordinates) if self._coordinates is not None else None,
            self._topology,
        )
        if self._internal is not None:
            new_manager._internal = to_torch(self._internal)
        new_manager._tree = self._tree
        new_manager._center_offset = self._center_offset
        new_manager._fixed_coords = self._fixed_coords
        new_manager._independent_dof = self._independent_dof
        new_manager._csr_offsets = self._csr_offsets
        new_manager._csr_neighbors = self._csr_neighbors
        new_manager._coords_dirty = self._coords_dirty
        new_manager._dof_dirty = self._dof_dirty
        return new_manager

    def to(self, device: str = None, dtype=None) -> "CoordinateManager":
        """Move tensors to specified device/dtype (PyTorch only)."""
        if not is_torch(self._coordinates if self._coordinates is not None else self._internal):
            raise RuntimeError("Cannot move to device: not PyTorch tensors")

        def convert(t):
            if t is None:
                return None
            if device is not None:
                t = t.to(device)
            if dtype is not None:
                t = t.to(dtype)
            return t

        new_manager = CoordinateManager(convert(self._coordinates), self._topology)
        if self._internal is not None:
            new_manager._internal = convert(self._internal)
        new_manager._tree = self._tree
        new_manager._center_offset = self._center_offset
        new_manager._fixed_coords = self._fixed_coords
        new_manager._independent_dof = self._independent_dof
        new_manager._csr_offsets = self._csr_offsets
        new_manager._csr_neighbors = self._csr_neighbors
        new_manager._coords_dirty = self._coords_dirty
        new_manager._dof_dirty = self._dof_dirty
        return new_manager

    def detach(self) -> "CoordinateManager":
        """Detach tensors from computation graphs (PyTorch only)."""
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
        """Slice by boolean atom mask."""
        if self._coordinates is None:
            self._recompute_cartesian()
        sliced_coords = self._coordinates[mask]
        return CoordinateManager._from_slice(sliced_coords, is_torch(sliced_coords))

    @classmethod
    def _from_slice(cls, coordinates: Array, is_torch_flag: bool) -> "CoordinateManager":
        """Create from sliced coordinates (internal factory)."""
        manager = cls.__new__(cls)
        manager._coordinates = coordinates
        manager._is_torch = is_torch_flag
        manager._topology = None
        manager._internal = None
        manager._tree = None
        manager._center_offset = None
        manager._fixed_coords = None
        manager._independent_dof = None
        manager._csr_offsets = None
        manager._csr_neighbors = None
        manager._coords_dirty = False
        manager._dof_dirty = True
        return manager

    # ─────────────────────────────────────────────────────────────────────
    # Internal Coordinate Access (for Polymer)
    # ─────────────────────────────────────────────────────────────────────
    @property
    def dihedrals(self) -> Array:
        """All dihedral angles (compatibility, prefer using dof)."""
        self._ensure_internal()
        return self._internal[:, 2]

    @dihedrals.setter
    def dihedrals(self, value: Array) -> None:
        """Set all dihedrals (compatibility)."""
        self._ensure_internal()
        if is_torch(self._internal):
            new_internal = self._internal.detach().clone()
        else:
            new_internal = self._internal.copy()
        new_internal[:, 2] = value
        self._internal = new_internal
        self._coords_dirty = True
        self._dof_dirty = False

    @property
    def angles(self) -> Array:
        """All bond angles (compatibility)."""
        self._ensure_internal()
        return self._internal[:, 1]

    @angles.setter
    def angles(self, value: Array) -> None:
        """Set all angles (compatibility)."""
        self._ensure_internal()
        if is_torch(self._internal):
            new_internal = self._internal.detach().clone()
        else:
            new_internal = self._internal.copy()
        new_internal[:, 1] = value
        self._internal = new_internal
        self._coords_dirty = True

    @property
    def distances(self) -> Array:
        """All bond distances (compatibility)."""
        self._ensure_internal()
        return self._internal[:, 0]

    @distances.setter
    def distances(self, value: Array) -> None:
        """Set all distances (compatibility)."""
        self._ensure_internal()
        if is_torch(self._internal):
            new_internal = self._internal.detach().clone()
        else:
            new_internal = self._internal.copy()
        new_internal[:, 0] = value
        self._internal = new_internal
        self._coords_dirty = True

    @property
    def internal(self) -> Array:
        """Full internal coordinates (compatibility)."""
        self._ensure_internal()
        return self._internal

    @internal.setter
    def internal(self, value: Array) -> None:
        """Set full internal coordinates (compatibility)."""
        self._internal = value
        self._coords_dirty = True
        self._dof_dirty = False
