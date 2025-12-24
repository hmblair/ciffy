"""
Coordinate management with dual representation and constraint support.

Provides the MolecularGeometry class that manages both Cartesian and internal
coordinate representations with lazy evaluation, automatic conversion, and
constraint-aware minimal DOF representation.

The constraint system uses a unified architecture:
1. Spanning tree handles implicit constraints (bond lengths/angles)
2. Non-tree edges (closures) create explicit constraints
3. Jacobian analysis discovers independent DOF automatically
4. Newton-Raphson solves for dependent torsions

This eliminates chemistry-specific ring classification - all constraints
are geometric, enabling arbitrary bond/angle constraints.
"""

from __future__ import annotations

import numpy as np

from ..backend import (
    Array,
    is_torch,
    to_numpy,
    to_torch,
    check_compatible,
    has_nan,
    has_inf,
    clone,
    empty,
    to_backend,
)
from ..backend.dispatch import (
    TopologyInfo,
    build_bond_graph_csr,
)
from .tree import SpanningTree, ReconstructionData
from .constraints import ConstraintSystem, solve_closure


# ─────────────────────────────────────────────────────────────────────────────
# MolecularGeometry Class
# ─────────────────────────────────────────────────────────────────────────────


class MolecularGeometry:
    """
    Molecular geometry with minimal DOF representation.

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
        bonds: (B, 2) bonded atom pairs (read-only)
        n_dof: Number of degrees of freedom

    Example:
        >>> geom = polymer.geometry
        >>> geom.dof = model_output     # Set K DOF values
        >>> coords = geom.coordinates   # Get N×3 coordinates
    """

    __slots__ = (
        # Cartesian representation
        '_coordinates',

        # Internal representation (private)
        '_internal',
        '_tree',  # SpanningTree for coordinate conversion
        '_recon_data',  # ReconstructionData bundle for NERF

        # Structural metadata
        '_topology',
        '_bonds',  # Cached (B, 2) bond array

        # Constraint analysis (private) - unified system
        '_constraint_system',  # ConstraintSystem with DOF discovery

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
        self._bonds: np.ndarray | None = None

        # Internal representation (computed lazily)
        self._internal: Array | None = None
        self._tree: SpanningTree | None = None
        self._recon_data: ReconstructionData | None = None

        # Constraint analysis (computed lazily) - unified system
        self._constraint_system: ConstraintSystem | None = None

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
        """
        Number of degrees of freedom (independent torsions).

        All DOF are generalized torsions (angles in radians). For molecules
        with rings, ring closure constraints reduce the number of independent
        DOF. Ring puckering emerges naturally from the ring geometry after
        closure - it is not a separate DOF.
        """
        self._ensure_constraint_analysis()
        return self._constraint_system.n_dof

    @property
    def bonds(self) -> np.ndarray:
        """
        (B, 2) array of bonded atom pairs.

        Each row [i, j] represents a covalent bond between atoms i and j,
        where i < j. Computed lazily from topology and cached.

        Returns:
            (B, 2) int64 array of unique bond pairs.
        """
        if self._bonds is None:
            from ..backend.graph import build_bond_graph_from_topology
            edges, _ = build_bond_graph_from_topology(self._topology)
            # Deduplicate: edges are symmetric, keep only i < j
            self._bonds = edges[edges[:, 0] < edges[:, 1]]
        return self._bonds

    def _get_n_atoms(self) -> int:
        """Get number of atoms (internal use)."""
        if self._coordinates is not None:
            return len(self._coordinates)
        if self._internal is not None:
            return len(self._internal)
        raise ValueError("Invalid MolecularGeometry.")

    # ─────────────────────────────────────────────────────────────────────
    # String Representation
    # ─────────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        """Return string representation."""
        n_atoms = self._get_n_atoms()
        # Avoid triggering constraint analysis just for repr
        if self._constraint_system is not None:
            n_dof = self._constraint_system.n_dof
            return f"MolecularGeometry(n_atoms={n_atoms}, n_dof={n_dof})"
        return f"MolecularGeometry(n_atoms={n_atoms})"

    # ─────────────────────────────────────────────────────────────────────
    # Private: Constraint Analysis
    # ─────────────────────────────────────────────────────────────────────

    def _ensure_constraint_analysis(self) -> None:
        """
        Ensure constraint analysis has been performed.

        Uses the unified ConstraintSystem which:
        1. Builds spanning tree from bond graph
        2. Identifies non-tree edges (closure bonds)
        3. Computes Jacobian to discover independent DOF
        4. No chemistry-specific ring classification needed
        """
        if self._constraint_system is not None:
            return

        # Ensure we have coordinates for constraint analysis
        if self._coordinates is None:
            raise RuntimeError("Cannot analyze constraints without coordinates")

        coords_np = to_numpy(self._coordinates).astype(np.float32)

        # Build constraint system from topology
        self._constraint_system = ConstraintSystem.from_topology(
            topology=self._topology,
            coords=coords_np,
            fix_covalent_bonds=True,
            fix_covalent_angles=True,
        )

        # Update tree and recon_data to match constraint system
        self._tree = SpanningTree(
            parent=self._constraint_system.parent,
            level=self._constraint_system.level,
            component_id=np.zeros(self._constraint_system.n_atoms, dtype=np.int32),
            n_components=1,
        )

        # Update internal coordinates if they exist
        if self._internal is not None:
            # Use base_internal from constraint system
            self._internal = to_backend(
                self._constraint_system.base_internal,
                like=self._coordinates
            )

    def _ensure_internal(self) -> None:
        """Ensure internal coordinates are computed."""
        if self._internal is None:
            self._recompute_internal()

    # ─────────────────────────────────────────────────────────────────────
    # Private: DOF Access Helpers
    # ─────────────────────────────────────────────────────────────────────

    def _get_atom_to_row_mapping(self) -> np.ndarray:
        """Build mapping from atom index to internal coordinate row index.

        With parent-based internal coordinates, atom k's data is at row k,
        so this is an identity mapping.
        """
        n_atoms = self._get_n_atoms()
        return np.arange(n_atoms, dtype=np.int64)

    def _get_dof_values(self) -> Array:
        """
        Get current values of independent DOF (all torsions).

        All DOF are generalized torsions (angles in radians). Ring puckering
        emerges from the ring geometry - it is not a separate DOF.
        """
        self._ensure_constraint_analysis()
        self._ensure_internal()

        system = self._constraint_system

        # Get all torsion atoms (level >= 3) that are not dependent
        all_torsions = np.where(system.level >= 3)[0]
        dependent_set = set(system.closures.dependent_idx.tolist())

        # Collect independent torsion values
        dof_list = []
        for atom in all_torsions:
            if atom not in dependent_set:
                dof_list.append(atom)

        if len(dof_list) == 0:
            return empty(0, like=self._internal)

        # Extract independent dihedrals
        indices = np.array(dof_list, dtype=np.int64)
        dihedral_dof = self._internal[indices, 2]

        return clone(dihedral_dof)

    def _set_dof_values(self, new_values: Array) -> None:
        """
        Set independent DOF (all torsions) and solve ring closure.

        All DOF are generalized torsions (angles in radians). After setting
        independent dihedrals, Newton-Raphson solves for dependent dihedrals
        to satisfy ring closure constraints.
        """
        self._ensure_constraint_analysis()
        self._ensure_internal()

        system = self._constraint_system
        new_values_np = to_numpy(new_values).astype(np.float32)

        # Get all torsion atoms that are independent
        all_torsions = np.where(system.level >= 3)[0]
        dependent_set = set(system.closures.dependent_idx.tolist())

        independent_atoms = [a for a in all_torsions if a not in dependent_set]
        n_dof = len(independent_atoms)

        if len(new_values_np) != n_dof:
            raise ValueError(f"Expected {n_dof} DOF values, got {len(new_values_np)}")

        if n_dof == 0:
            return

        # Get internal as numpy for modification
        internal_np = to_numpy(self._internal).astype(np.float32).copy()

        # Set independent dihedrals
        for i, atom in enumerate(independent_atoms):
            internal_np[atom, 2] = new_values_np[i]

        # Solve ring closure for dependent dihedrals using Newton-Raphson
        if system.closures.n_closures > 0:
            internal_np = solve_closure(
                internal=internal_np,
                system=system,
                max_iter=10,
                tol=1e-7,
            )

        # Convert back to original backend
        self._internal = to_backend(internal_np, like=self._coordinates)

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
            self._tree = SpanningTree.from_bond_graph(csr_offsets, csr_neighbors, n_atoms)

        # Check if we need autograd path (PyTorch tensor with requires_grad)
        use_autograd = is_torch(coords) and coords.requires_grad

        if use_autograd:
            # Use autograd path via dispatch (bridge derives zmatrix from parent)
            from ..backend.dispatch import cartesian_to_internal as dispatch_c2i

            # Compute internal coords with autograd (dispatch handles bridge)
            internal = dispatch_c2i(coords, self._tree.parent)

            # Create ReconstructionData bundle (no centering for autograd)
            coords_np = to_numpy(coords.detach()).astype(np.float32)
            self._recon_data = self._tree.get_reconstruction_data(coords_np, None)

            self._internal = internal
        else:
            # NumPy path: convert to numpy for C backend
            coords_np = to_numpy(coords).astype(np.float32)

            # Convert Cartesian to internal with per-component centering
            internal, center_offsets = self._tree.cartesian_to_internal(coords_np, center=True)

            # Compute fixed coords (centered if centering was used)
            if center_offsets is not None:
                fixed_coords = coords_np - center_offsets[self._tree.component_id]
            else:
                fixed_coords = coords_np.copy()

            # Create ReconstructionData bundle
            self._recon_data = self._tree.get_reconstruction_data(fixed_coords, center_offsets)

            # Convert back to original backend
            self._internal = to_backend(internal, like=coords)

    def _recompute_cartesian(self) -> None:
        """
        Recompute Cartesian coordinates from internal.

        Uses NERF (Natural Extension Reference Frame) to place atoms based on
        internal coordinates (distances, angles, dihedrals). Ring closure has
        already set dependent dihedrals, so ring geometry (including puckering)
        emerges naturally.
        """
        if self._internal is None:
            raise RuntimeError("Cannot reconstruct: internal is None")
        if self._tree is None:
            raise RuntimeError("Cannot reconstruct: tree is None")
        if self._recon_data is None:
            raise RuntimeError("Cannot reconstruct: recon_data is None")

        internal = self._internal
        recon = self._recon_data

        # Check if we need autograd path (PyTorch tensor with requires_grad)
        use_autograd = is_torch(internal) and internal.requires_grad

        if use_autograd:
            # Use autograd path via dispatch (bridge derives zmatrix from parent)
            from ..backend.dispatch import nerf_reconstruct as dispatch_nerf

            # NERF reconstruction with autograd via dispatch
            coords = dispatch_nerf(
                parent=recon.parent,
                levels=recon.levels,
                internal=internal,
                level_offsets=recon.level_offsets,
                level_atoms=recon.level_atoms,
                fixed_coords=recon.fixed_coords,
                anchor_coords=recon.anchor_coords,
                component_ids=recon.component_ids,
                center_offsets=recon.center_offsets,
            )

            self._coordinates = coords
        else:
            # NumPy path
            internal_np = to_numpy(internal).astype(np.float32)

            # NERF reconstruction using ReconstructionData
            coords = self._tree.internal_to_cartesian(
                internal_np, recon.fixed_coords, offsets=recon.center_offsets
            )

            # Convert back to original backend
            self._coordinates = to_backend(coords, like=internal)

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

    def numpy(self) -> "MolecularGeometry":
        """Convert to NumPy backend."""
        new_manager = MolecularGeometry(
            to_numpy(self._coordinates) if self._coordinates is not None else None,
            self._topology,
        )
        if self._internal is not None:
            new_manager._internal = to_numpy(self._internal)
        new_manager._bonds = self._bonds
        new_manager._tree = self._tree
        new_manager._recon_data = self._recon_data
        new_manager._constraint_system = self._constraint_system
        new_manager._coords_dirty = self._coords_dirty
        new_manager._dof_dirty = self._dof_dirty
        return new_manager

    def torch(self) -> "MolecularGeometry":
        """Convert to PyTorch backend."""
        new_manager = MolecularGeometry(
            to_torch(self._coordinates) if self._coordinates is not None else None,
            self._topology,
        )
        if self._internal is not None:
            new_manager._internal = to_torch(self._internal)
        new_manager._bonds = self._bonds
        new_manager._tree = self._tree
        new_manager._recon_data = self._recon_data
        new_manager._constraint_system = self._constraint_system
        new_manager._coords_dirty = self._coords_dirty
        new_manager._dof_dirty = self._dof_dirty
        return new_manager

    def to(self, device: str = None, dtype=None) -> "MolecularGeometry":
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

        new_manager = MolecularGeometry(convert(self._coordinates), self._topology)
        if self._internal is not None:
            new_manager._internal = convert(self._internal)
        new_manager._bonds = self._bonds
        new_manager._tree = self._tree
        new_manager._recon_data = self._recon_data
        new_manager._constraint_system = self._constraint_system
        new_manager._coords_dirty = self._coords_dirty
        new_manager._dof_dirty = self._dof_dirty
        return new_manager

    def detach(self) -> "MolecularGeometry":
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

    def __getitem__(self, mask: Array) -> "MolecularGeometry":
        """Slice by boolean atom mask."""
        if self._coordinates is None:
            self._recompute_cartesian()
        sliced_coords = self._coordinates[mask]
        return MolecularGeometry._from_slice(sliced_coords, is_torch(sliced_coords))

    @classmethod
    def _from_slice(cls, coordinates: Array, is_torch_flag: bool) -> "MolecularGeometry":
        """Create from sliced coordinates (internal factory)."""
        manager = cls.__new__(cls)
        manager._coordinates = coordinates
        manager._is_torch = is_torch_flag
        manager._topology = None
        manager._bonds = None
        manager._internal = None
        manager._tree = None
        manager._recon_data = None
        manager._constraint_system = None
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
        new_internal = clone(self._internal)
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
        new_internal = clone(self._internal)
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
        new_internal = clone(self._internal)
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
