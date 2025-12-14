"""
Coordinate management with dual representation support.

Provides the CoordinateManager class that manages both Cartesian and internal
coordinate representations with lazy evaluation and automatic conversion.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from ..backend import Array, is_torch, to_numpy, to_torch, check_compatible
from ..types import DihedralType

if TYPE_CHECKING:
    from ..polymer import Polymer
    from .graph import ZMatrix


# ─────────────────────────────────────────────────────────────────────────────
# Backend Polymorphism Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _copy(arr: Array) -> Array:
    """Copy array regardless of backend."""
    return arr.clone() if is_torch(arr) else arr.copy()


def _to_torch_dtype(dtype):
    """Convert numpy dtype to torch dtype."""
    import torch
    if dtype == np.int64:
        return torch.int64
    elif dtype == np.float32:
        return torch.float32
    elif dtype == np.float64:
        return torch.float64
    return dtype  # Assume already torch dtype


def _zeros(shape: tuple, dtype, backend_like: Array) -> Array:
    """Create zeros array matching backend of backend_like."""
    if is_torch(backend_like):
        import torch
        return torch.zeros(shape, dtype=_to_torch_dtype(dtype), device=backend_like.device)
    return np.zeros(shape, dtype=dtype)


def _array(data, dtype, backend_like: Array) -> Array:
    """Create array from data matching backend of backend_like."""
    if is_torch(backend_like):
        import torch
        return torch.tensor(data, dtype=_to_torch_dtype(dtype), device=backend_like.device)
    return np.array(data, dtype=dtype)


def _stack(arrays: list, backend_like: Array) -> Array:
    """Stack arrays matching backend of backend_like."""
    if is_torch(backend_like):
        import torch
        return torch.stack(arrays)
    return np.array(arrays)


def _empty_array(dtype, backend_like: Array) -> Array:
    """Create empty 1D array matching backend of backend_like."""
    if is_torch(backend_like):
        import torch
        return torch.tensor([], dtype=_to_torch_dtype(dtype), device=backend_like.device)
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

        # Connected components (CSR format)
        '_component_offsets',   # (C+1,) int64
        '_component_atoms',     # (N,) int64 (flattened atom indices)
        '_component_centroids', # (C, 3) float32/64
        '_component_reference_coords',  # List of (n_i, 3) centered coords for multi-atom chains
        '_component_contiguous',  # List of bool - whether each component's atoms are contiguous

        # Dihedral indices (CSR format)
        '_dihedral_offsets',  # (NUM_DIHEDRAL_TYPES+1,) int64
        '_dihedral_indices',  # (total_dihedrals,) int64

        '_internal_valid',
        '_n_atoms',  # Total atom count (set from initial coordinates)
        '_is_torch',  # Cached backend flag

        # Reference to parent Polymer
        '_polymer',
    )

    def __init__(
        self,
        coordinates: Array,
        polymer: "Polymer",
    ) -> None:
        """
        Initialize coordinate manager with Cartesian coordinates.

        Args:
            coordinates: (N, 3) array of Cartesian XYZ positions.
            polymer: Reference to parent Polymer for metadata access.
        """
        self._polymer = polymer

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

        # Connected components (CSR format)
        self._component_offsets: Array | None = None
        self._component_atoms: Array | None = None
        self._component_centroids: Array | None = None
        self._component_reference_coords: list | None = None  # List of centered coords per chain
        self._component_contiguous: list | None = None  # Whether each component's atoms are contiguous

        # Dihedral indices (CSR format)
        self._dihedral_offsets: Array | None = None
        self._dihedral_indices: Array | None = None

        self._internal_valid = False

        # Store connected component centroids from initial coordinates
        self._update_connected_components()

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
            # Fallback to polymer's atoms array
            return self._polymer.atoms

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
        # Note: Keep Z-matrix cached - it's structure-based, not coordinate-based
        # Only invalidate dihedral indices as they depend on Z-matrix order
        self._dihedral_offsets = None
        self._dihedral_indices = None

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
        self._dihedral_offsets = None
        self._dihedral_indices = None
        self._invalidate_internal()

    def _update_connected_components(self) -> None:
        """
        Identify connected components and store their centroids and reference coordinates.

        Connected components are groups of bonded atoms. This includes:
        - Chains (multi-atom connected components)
        - Single atoms with no bonds (orphan atoms)

        Stores in CSR format:
        - offsets: (C+1,) cumulative counts
        - atoms: (N,) flattened atom indices
        - centroids: (C, 3) centroid positions

        Also stores reference coordinates (centered) for multi-atom chains
        to enable orientation restoration after NERF reconstruction.
        """
        from ..types import Scale

        if self._coordinates is None:
            return

        coords = self._coordinates
        res_sizes = self._polymer.sizes(Scale.RESIDUE)

        components_list = []  # Temporary list of (atom_indices, centroid, centered_coords)
        atom_offset = 0
        res_offset = 0

        # Each chain is a connected component
        for chain_len in self._polymer.lengths:
            chain_len_val = int(chain_len)
            if chain_len_val == 0:
                continue

            # Get atom count for this chain
            chain_atom_count = sum(int(res_sizes[res_offset + i]) for i in range(chain_len_val))

            # Get atom indices for this chain
            atom_indices = list(range(atom_offset, atom_offset + chain_atom_count))

            # Compute centroid and centered coordinates for this chain
            chain_coords = coords[atom_offset:atom_offset + chain_atom_count]
            centroid = chain_coords.mean(axis=0)
            centered_coords = chain_coords - centroid

            components_list.append((atom_indices, _copy(centroid), _copy(centered_coords)))

            atom_offset += chain_atom_count
            res_offset += chain_len_val

        # Convert to CSR format
        int64_dtype = np.int64  # Works for both numpy and torch
        if not components_list:
            # No components - create empty arrays
            self._component_offsets = _array([0], int64_dtype, coords)
            self._component_atoms = _empty_array(int64_dtype, coords)
            self._component_centroids = _zeros((0, 3), coords.dtype, coords)
            self._component_reference_coords = []
            self._component_contiguous = []
        else:
            offsets = [0]
            all_atoms = []
            all_centroids = []
            all_reference_coords = []
            all_contiguous = []

            for atom_indices, centroid, centered_coords in components_list:
                all_atoms.extend(atom_indices)
                all_centroids.append(centroid)
                # Only store reference coords for multi-atom chains (needed for orientation)
                if len(atom_indices) > 1:
                    all_reference_coords.append(centered_coords)
                else:
                    all_reference_coords.append(None)  # Single-atom, no orientation needed
                offsets.append(len(all_atoms))
                # Chains from polymer.lengths are always contiguous
                all_contiguous.append(True)

            # Store as arrays
            self._component_offsets = _array(offsets, int64_dtype, coords)
            self._component_atoms = _array(all_atoms, int64_dtype, coords)
            self._component_centroids = _stack(all_centroids, coords)
            self._component_reference_coords = all_reference_coords
            self._component_contiguous = all_contiguous

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

        # Update connected component centroids for reconstruction
        self._update_connected_components()

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

        Builds Z-matrix (if not cached) and computes bond lengths, angles,
        and dihedrals from current Cartesian coordinates.
        """
        from .graph import ZMatrix
        from .zmatrix import _compute_distance, _compute_angle, _compute_dihedral

        if self._coordinates is None:
            raise RuntimeError("Cannot compute internal coordinates: Cartesian coordinates are None")

        coords = self._coordinates
        n_atoms = coords.shape[0]

        # Build Z-matrix if not already cached
        if self._zmatrix is None:
            if self._polymer is None:
                raise RuntimeError(
                    "Cannot compute internal coordinates without polymer reference. "
                    "This CoordinateManager was created by slicing and doesn't have "
                    "access to bond information needed for Z-matrix construction."
                )
            self._zmatrix = ZMatrix.from_polymer(self._polymer)

            # Detect orphan atoms (atoms not in Z-matrix - no bonds)
            # Add them as single-atom connected components
            zmatrix_indices = self._zmatrix.indices
            if len(zmatrix_indices) > 0:
                zmatrix_atoms = set(int(idx) for idx in zmatrix_indices[:, 0])
            else:
                zmatrix_atoms = set()

            orphan_atoms = [i for i in range(n_atoms) if i not in zmatrix_atoms]

            if orphan_atoms and self._component_offsets is not None:
                # Add each orphan atom as a single-atom connected component
                # Use vectorized operations to avoid slow per-atom tensor creation
                n_orphans = len(orphan_atoms)
                old_end = int(self._component_offsets[-1])

                # Vectorized: create all indices and centroids at once
                orphan_indices = _array(orphan_atoms, np.int64, coords)
                orphan_centroids = coords[orphan_indices]  # Single indexing op
                new_offsets = _array(list(range(old_end + 1, old_end + n_orphans + 1)), np.int64, coords)

                # Extend CSR arrays
                self._component_offsets = _concat([self._component_offsets, new_offsets], coords)
                self._component_atoms = _concat([self._component_atoms, orphan_indices], coords)
                self._component_centroids = _concat([self._component_centroids, orphan_centroids], coords)

                # Orphan atoms are single-atom components, always "contiguous"
                self._component_contiguous.extend([True] * n_orphans)
                # No reference coords needed for single-atom components
                self._component_reference_coords.extend([None] * n_orphans)

        zmatrix_indices = self._zmatrix.indices
        n_zmatrix = len(zmatrix_indices)

        # Try to use C extension
        try:
            from .._c import _cartesian_to_internal as _c_cartesian_to_internal
            use_c = True
        except ImportError:
            use_c = False

        if use_c and not (is_torch(coords) and coords.requires_grad):
            # Use C extension (but not if we need gradients)
            # Convert indices to numpy if needed
            if is_torch(zmatrix_indices):
                indices_np = zmatrix_indices.cpu().numpy()
            else:
                indices_np = np.asarray(zmatrix_indices)

            if is_torch(coords):
                import torch
                coords_f32 = coords.detach().cpu().to(torch.float32).numpy()
            else:
                coords_f32 = np.ascontiguousarray(coords, dtype=np.float32)

            # Call C extension
            distances_np, angles_np, dihedrals_np = _c_cartesian_to_internal(coords_f32, indices_np)

            if is_torch(coords):
                import torch
                self._distances = torch.from_numpy(distances_np).to(device=coords.device, dtype=coords.dtype)
                self._angles = torch.from_numpy(angles_np).to(device=coords.device, dtype=coords.dtype)
                self._dihedrals = torch.from_numpy(dihedrals_np).to(device=coords.device, dtype=coords.dtype)
            else:
                self._distances = distances_np
                self._angles = angles_np
                self._dihedrals = dihedrals_np
        else:
            # Python fallback
            if is_torch(coords):
                import torch
                self._distances = torch.zeros(n_zmatrix, dtype=coords.dtype, device=coords.device)
                self._angles = torch.zeros(n_zmatrix, dtype=coords.dtype, device=coords.device)
                self._dihedrals = torch.zeros(n_zmatrix, dtype=coords.dtype, device=coords.device)
            else:
                self._distances = np.zeros(n_zmatrix, dtype=np.float32)
                self._angles = np.zeros(n_zmatrix, dtype=np.float32)
                self._dihedrals = np.zeros(n_zmatrix, dtype=np.float32)

            # Compute internal coordinates for each atom
            for i in range(n_zmatrix):
                atom_idx = int(zmatrix_indices[i, 0])
                dist_ref = int(zmatrix_indices[i, 1])
                ang_ref = int(zmatrix_indices[i, 2])
                dih_ref = int(zmatrix_indices[i, 3])

                if dist_ref >= 0:
                    self._distances[i] = _compute_distance(
                        coords[atom_idx],
                        coords[dist_ref],
                    )

                if ang_ref >= 0:
                    self._angles[i] = _compute_angle(
                        coords[atom_idx],
                        coords[dist_ref],
                        coords[ang_ref],
                    )

                if dih_ref >= 0:
                    self._dihedrals[i] = _compute_dihedral(
                        coords[atom_idx],
                        coords[dist_ref],
                        coords[ang_ref],
                        coords[dih_ref],
                    )

        self._internal_valid = True

    def _recompute_cartesian(self) -> None:
        """
        Recompute Cartesian coordinates from internal.

        Uses NERF (Natural Extension Reference Frame) algorithm to reconstruct
        3D coordinates from bond lengths, angles, and dihedrals.
        """
        from .nerf import nerf_reconstruct

        if self._distances is None or self._angles is None or self._dihedrals is None:
            raise RuntimeError("Cannot reconstruct Cartesian coordinates: internal coordinates are None")

        if self._zmatrix is None:
            raise RuntimeError("Cannot reconstruct Cartesian coordinates: Z-matrix is None")

        if self._component_offsets is None:
            raise RuntimeError("Cannot reconstruct Cartesian coordinates: connected components not computed")

        zmatrix_indices = self._zmatrix.indices

        # Get atom count (stored from initial coordinates)
        n_atoms = self._n_atoms

        # NERF reconstruction (places each chain root at origin)
        coords = nerf_reconstruct(
            zmatrix_indices,
            self._distances,
            self._angles,
            self._dihedrals,
            n_atoms=n_atoms,
        )

        # Restore chain positions AND orientations, plus orphan atoms
        # NERF places each chain in a canonical frame - we need to rotate back to original
        from ..operations.alignment import kabsch_rotation

        n_components = len(self._component_offsets) - 1

        # Build set of atoms in Z-matrix for orphan detection
        if len(zmatrix_indices) > 0:
            zmatrix_atoms = set(zmatrix_indices[:, 0].tolist() if hasattr(zmatrix_indices, 'tolist')
                               else [int(x) for x in zmatrix_indices[:, 0]])
        else:
            zmatrix_atoms = set()

        # Process each component (chains need rotation+translation, orphans just position)
        for comp_idx in range(n_components):
            comp_start = int(self._component_offsets[comp_idx])
            comp_end = int(self._component_offsets[comp_idx + 1])
            component_size = comp_end - comp_start

            if component_size == 1:
                # Single-atom component (orphan) - just restore position
                atom_idx = int(self._component_atoms[comp_start])
                if atom_idx not in zmatrix_atoms and atom_idx < n_atoms:
                    coords[atom_idx] = self._component_centroids[comp_idx]
            else:
                # Multi-atom component (chain) - restore orientation AND position
                atom_start = int(self._component_atoms[comp_start])
                atom_end = int(self._component_atoms[comp_end - 1]) + 1

                # Skip if out of bounds
                if atom_end > n_atoms:
                    continue

                # Get reference coordinates for this chain (centered original coords)
                reference_coords = self._component_reference_coords[comp_idx]
                if reference_coords is None:
                    continue

                original_centroid = self._component_centroids[comp_idx]

                # Use pre-computed contiguity flag (avoids per-iteration check)
                if self._component_contiguous[comp_idx]:
                    # Contiguous atoms - use slice
                    component_coords = coords[atom_start:atom_end]
                    reconstructed_centroid = component_coords.mean(axis=0)
                    centered_reconstructed = component_coords - reconstructed_centroid

                    # Compute Kabsch rotation: reconstructed → original orientation
                    R = kabsch_rotation(centered_reconstructed, reference_coords)

                    # Apply rotation then translate to original centroid
                    aligned = centered_reconstructed @ R.T + original_centroid
                    coords[atom_start:atom_end] = aligned
                else:
                    # Non-contiguous atoms - use indexing
                    atom_indices = self._component_atoms[comp_start:comp_end]
                    if is_torch(coords):
                        component_coords = coords[atom_indices.long()]
                    else:
                        component_coords = coords[atom_indices]

                    reconstructed_centroid = component_coords.mean(axis=0)
                    centered_reconstructed = component_coords - reconstructed_centroid

                    # Compute Kabsch rotation: reconstructed → original orientation
                    R = kabsch_rotation(centered_reconstructed, reference_coords)

                    # Apply rotation then translate to original centroid
                    aligned = centered_reconstructed @ R.T + original_centroid
                    if is_torch(coords):
                        coords[atom_indices.long()] = aligned
                    else:
                        coords[atom_indices] = aligned

        self._coordinates = coords
        self._cartesian_valid = True
        self._validate_coordinates()

    def _validate_coordinates(self) -> None:
        """
        Validate coordinates after reconstruction.

        Raises:
            ValueError: If coordinates contain NaN, Inf, or unreasonable values.
        """
        coords = to_numpy(self._coordinates)
        if np.any(np.isnan(coords)):
            raise ValueError("Invalid coordinates after reconstruction (NaN detected)")
        if np.any(np.isinf(coords)):
            raise ValueError("Invalid coordinates after reconstruction (Inf detected)")
        if np.any(np.abs(coords) > 10000):
            raise ValueError(
                "Coordinates exceed 10000 Angstroms - likely reconstruction error. "
                "Check that internal coordinates are within reasonable bounds."
            )

    def _compute_dihedral_indices(self) -> None:
        """
        Compute indices into the dihedral array for named dihedrals in CSR format.

        Identifies which Z-matrix entries correspond to standard backbone
        dihedrals (phi, psi, omega, etc.) based on atom types and topology.

        Stores results in CSR format:
        - offsets: (NUM_DIHEDRAL_TYPES+1,) cumulative counts
        - indices: (total_dihedrals,) flattened Z-matrix indices

        Raises:
            RuntimeError: If polymer reference is not set (e.g., on sliced manager).
        """
        if self._polymer is None:
            raise RuntimeError(
                "Cannot compute named dihedral indices without polymer reference. "
                "This CoordinateManager was created by slicing and doesn't have "
                "access to residue information needed for named dihedrals."
            )

        from .dihedrals import compute_dihedral_indices

        if self._zmatrix is None:
            # Trigger internal computation to build Z-matrix
            _ = self.dihedrals

        # Get dict[int, Array] mapping dihedral type → Z-matrix indices
        result = compute_dihedral_indices(
            self._polymer,
            self._zmatrix.indices,
        )

        # Convert to CSR format
        from .dihedrals import DIHEDRAL_TYPE_TO_INDEX
        num_types = len(DIHEDRAL_TYPE_TO_INDEX)

        offsets = [0]
        all_indices = []

        for type_idx in range(num_types):
            indices = result.get(type_idx, [])
            if hasattr(indices, '__iter__') and not isinstance(indices, (str, bytes)):
                # Convert to list if it's an array
                if hasattr(indices, 'tolist'):
                    indices_list = indices.tolist()
                elif is_torch(indices):
                    indices_list = indices.cpu().tolist()
                else:
                    indices_list = list(indices)
            else:
                indices_list = []

            all_indices.extend(indices_list)
            offsets.append(len(all_indices))

        # Store as arrays
        coords = self._coordinates
        int64_dtype = np.int64
        self._dihedral_offsets = _array(offsets, int64_dtype, coords)
        self._dihedral_indices = _array(all_indices, int64_dtype, coords) if all_indices else _empty_array(int64_dtype, coords)

    # ─────────────────────────────────────────────────────────────────────
    # Named Dihedral API
    # ─────────────────────────────────────────────────────────────────────

    def get_dihedral(self, dtype: DihedralType) -> Array:
        """
        Get specific named dihedral angles using CSR lookup.

        Args:
            dtype: Type of dihedral to retrieve (e.g., DihedralType.PHI).

        Returns:
            Array of dihedral values in radians (one per applicable residue).
            Returns empty array if the specified dihedral type is not found.

        Example:
            >>> phi = manager.get_dihedral(DihedralType.PHI)
            >>> psi = manager.get_dihedral(DihedralType.PSI)
        """
        # Ensure internal coordinates are computed
        if not self._internal_valid:
            self._recompute_internal()

        # Compute dihedral indices if not already cached
        if self._dihedral_offsets is None:
            self._compute_dihedral_indices()

        # CSR lookup by DihedralType integer index
        from .dihedrals import DIHEDRAL_TYPE_TO_INDEX
        type_idx = DIHEDRAL_TYPE_TO_INDEX.get(dtype)
        if type_idx is None or type_idx >= len(self._dihedral_offsets) - 1:
            # Type index out of range
            return _empty_array(self._dihedrals.dtype, self._dihedrals)

        start = int(self._dihedral_offsets[type_idx])
        end = int(self._dihedral_offsets[type_idx + 1])

        if start == end:  # Empty range
            return _empty_array(self._dihedrals.dtype, self._dihedrals)

        indices = self._dihedral_indices[start:end]
        return self._dihedrals[indices]

    def set_dihedral(self, dtype: DihedralType, values: Array) -> None:
        """
        Set specific named dihedral angles.

        Args:
            dtype: Type of dihedral to set (e.g., DihedralType.PHI).
            values: New dihedral values in radians.

        Raises:
            ValueError: If the specified dihedral type is not found in the structure.

        Example:
            >>> # Set all phi angles to -60 degrees
            >>> manager.set_dihedral(DihedralType.PHI, np.full(n_residues, -np.pi/3))
        """
        # Ensure internal coordinates are computed
        if not self._internal_valid:
            self._recompute_internal()

        # Compute dihedral indices if not already cached
        if self._dihedral_offsets is None:
            self._compute_dihedral_indices()

        # CSR lookup by DihedralType integer index
        from .dihedrals import DIHEDRAL_TYPE_TO_INDEX
        type_idx = DIHEDRAL_TYPE_TO_INDEX.get(dtype)
        if type_idx is None or type_idx >= len(self._dihedral_offsets) - 1:
            raise ValueError(
                f"No {dtype.name} dihedrals found in structure. "
                f"This may be because the structure doesn't contain the appropriate molecule type."
            )

        start = int(self._dihedral_offsets[type_idx])
        end = int(self._dihedral_offsets[type_idx + 1])

        if start == end:  # Empty range
            raise ValueError(
                f"No {dtype.name} dihedrals found in structure. "
                f"This may be because the structure doesn't contain the appropriate molecule type."
            )

        indices = self._dihedral_indices[start:end]

        # Copy dihedrals array to avoid in-place modification issues
        new_dihedrals = _copy(self._dihedrals)

        # Update specific dihedrals
        new_dihedrals[indices] = values

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
        new_manager = CoordinateManager(
            to_numpy(self._coordinates) if self._coordinates is not None else None,
            self._polymer,
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

        # Convert connected components (CSR format)
        if self._component_offsets is not None:
            new_manager._component_offsets = to_numpy(self._component_offsets)
            new_manager._component_atoms = to_numpy(self._component_atoms)
            new_manager._component_centroids = to_numpy(self._component_centroids)
            # Convert reference coordinates list
            if self._component_reference_coords is not None:
                new_manager._component_reference_coords = [
                    to_numpy(rc) if rc is not None else None
                    for rc in self._component_reference_coords
                ]

        # Convert dihedral indices (CSR format)
        if self._dihedral_offsets is not None:
            new_manager._dihedral_offsets = to_numpy(self._dihedral_offsets)
            new_manager._dihedral_indices = to_numpy(self._dihedral_indices)

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
        new_manager = CoordinateManager(
            to_torch(self._coordinates) if self._coordinates is not None else None,
            self._polymer,
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

        # Convert connected components (CSR format)
        if self._component_offsets is not None:
            new_manager._component_offsets = to_torch(self._component_offsets)
            new_manager._component_atoms = to_torch(self._component_atoms)
            new_manager._component_centroids = to_torch(self._component_centroids)
            # Convert reference coordinates list
            if self._component_reference_coords is not None:
                new_manager._component_reference_coords = [
                    to_torch(rc) if rc is not None else None
                    for rc in self._component_reference_coords
                ]

        # Convert dihedral indices (CSR format)
        if self._dihedral_offsets is not None:
            new_manager._dihedral_offsets = to_torch(self._dihedral_offsets)
            new_manager._dihedral_indices = to_torch(self._dihedral_indices)

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
        new_manager = CoordinateManager(
            self._coordinates.to(device) if self._coordinates is not None else None,
            self._polymer,
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

        # Move connected components (CSR format)
        if self._component_offsets is not None:
            new_manager._component_offsets = self._component_offsets.to(device)
            new_manager._component_atoms = self._component_atoms.to(device)
            new_manager._component_centroids = self._component_centroids.to(device)
            # Move reference coordinates list
            if self._component_reference_coords is not None:
                new_manager._component_reference_coords = [
                    rc.to(device) if rc is not None else None
                    for rc in self._component_reference_coords
                ]

        # Move dihedral indices (CSR format)
        if self._dihedral_offsets is not None:
            new_manager._dihedral_offsets = self._dihedral_offsets.to(device)
            new_manager._dihedral_indices = self._dihedral_indices.to(device)

        # Copy atom count
        new_manager._n_atoms = self._n_atoms

        return new_manager

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
            coordinates when accessed.
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
        new_manager._polymer = None  # Must be set by caller

        # Internal representation starts invalid (lazy recomputation)
        new_manager._distances = None
        new_manager._angles = None
        new_manager._dihedrals = None
        new_manager._zmatrix = None
        new_manager._internal_valid = False

        # CSR structures start empty (rebuilt on demand)
        new_manager._component_offsets = None
        new_manager._component_atoms = None
        new_manager._component_centroids = None
        new_manager._component_reference_coords = None
        new_manager._component_contiguous = None
        new_manager._dihedral_offsets = None
        new_manager._dihedral_indices = None

        return new_manager
