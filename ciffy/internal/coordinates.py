"""
Coordinate management with dual representation support.

Provides the CoordinateManager class that manages both Cartesian and internal
coordinate representations with lazy evaluation and automatic conversion.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Dict

import numpy as np

from ..backend import Array, is_torch, to_numpy, to_torch, check_compatible
from ..types import DihedralType

if TYPE_CHECKING:
    from ..polymer import Polymer
    from .graph import ZMatrixEntry


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
        '_zmatrix',
        '_dihedral_indices',
        '_orphan_atoms',
        '_orphan_coords',
        '_internal_valid',

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

        # Initialize internal representation as invalid (not yet computed)
        self._distances: Optional[Array] = None
        self._angles: Optional[Array] = None
        self._dihedrals: Optional[Array] = None
        self._zmatrix: Optional[list["ZMatrixEntry"]] = None
        self._dihedral_indices: Optional[Dict[str, Array]] = None
        self._orphan_atoms: Optional[list[int]] = None
        self._orphan_coords: Optional[Array] = None
        self._internal_valid = False

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
        self._dihedral_indices = None

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
    def zmatrix(self) -> list["ZMatrixEntry"]:
        """
        Z-matrix structure (read-only).

        Returns:
            List of ZMatrixEntry objects defining coordinate references.

        Note:
            If Z-matrix hasn't been built yet, triggers internal coordinate
            computation which builds the Z-matrix.
        """
        if self._zmatrix is None:
            # Trigger internal computation to build Z-matrix
            _ = self.dihedrals
        return self._zmatrix

    # ─────────────────────────────────────────────────────────────────────
    # Recomputation Methods
    # ─────────────────────────────────────────────────────────────────────

    def _recompute_internal(self) -> None:
        """
        Recompute internal coordinates from Cartesian.

        Builds Z-matrix (if not cached) and computes bond lengths, angles,
        and dihedrals from current Cartesian coordinates.
        """
        from .graph import build_zmatrix, zmatrix_to_indices
        from .zmatrix import _compute_distance, _compute_angle, _compute_dihedral

        if self._coordinates is None:
            raise RuntimeError("Cannot compute internal coordinates: Cartesian coordinates are None")

        coords = self._coordinates
        n_atoms = coords.shape[0]

        # Build Z-matrix if not already cached
        if self._zmatrix is None:
            self._zmatrix = build_zmatrix(self._polymer)

            # Detect orphan atoms (not in Z-matrix - no bonds)
            zmatrix_atoms = {entry.atom_idx for entry in self._zmatrix}
            self._orphan_atoms = [i for i in range(n_atoms) if i not in zmatrix_atoms]

            # Store orphan coordinates
            if self._orphan_atoms:
                if is_torch(coords):
                    import torch
                    self._orphan_coords = torch.stack([coords[i] for i in self._orphan_atoms])
                else:
                    self._orphan_coords = np.stack([coords[i] for i in self._orphan_atoms])
            else:
                self._orphan_coords = None

        n_zmatrix = len(self._zmatrix)

        # Try to use C extension
        try:
            from .._c import _cartesian_to_internal as _c_cartesian_to_internal
            use_c = True
        except ImportError:
            use_c = False

        if use_c and not (is_torch(coords) and coords.requires_grad):
            # Use C extension (but not if we need gradients)
            indices = zmatrix_to_indices(self._zmatrix)

            if is_torch(coords):
                import torch
                coords_f32 = coords.detach().cpu().to(torch.float32).numpy()
            else:
                coords_f32 = np.ascontiguousarray(coords, dtype=np.float32)

            # Call C extension
            distances_np, angles_np, dihedrals_np = _c_cartesian_to_internal(coords_f32, indices)

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
            for i, entry in enumerate(self._zmatrix):
                atom_idx = entry.atom_idx

                if entry.distance_ref >= 0:
                    self._distances[i] = _compute_distance(
                        coords[atom_idx],
                        coords[entry.distance_ref],
                    )

                if entry.angle_ref >= 0:
                    self._angles[i] = _compute_angle(
                        coords[atom_idx],
                        coords[entry.distance_ref],
                        coords[entry.angle_ref],
                    )

                if entry.dihedral_ref >= 0:
                    self._dihedrals[i] = _compute_dihedral(
                        coords[atom_idx],
                        coords[entry.distance_ref],
                        coords[entry.angle_ref],
                        coords[entry.dihedral_ref],
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

        # Get atom count from internal coordinates (avoids circular dependency with polymer.size())
        n_atoms = len(self._zmatrix) + len(self._orphan_atoms if self._orphan_atoms else [])

        # NERF reconstruction
        coords = nerf_reconstruct(
            self._distances,
            self._angles,
            self._dihedrals,
            self._zmatrix,
            n_atoms=n_atoms,
        )

        # Restore orphan atom coordinates
        if self._orphan_atoms and self._orphan_coords is not None:
            for i, atom_idx in enumerate(self._orphan_atoms):
                coords[atom_idx] = self._orphan_coords[i]

        self._coordinates = coords
        self._cartesian_valid = True

    def _compute_dihedral_indices(self) -> None:
        """
        Compute indices into the dihedral array for named dihedrals.

        Identifies which Z-matrix entries correspond to standard backbone
        dihedrals (phi, psi, omega, etc.) based on atom types and topology.
        """
        from .dihedrals import compute_dihedral_indices

        if self._zmatrix is None:
            # Trigger internal computation to build Z-matrix
            _ = self.dihedrals

        self._dihedral_indices = compute_dihedral_indices(
            self._polymer,
            self._zmatrix,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Named Dihedral API
    # ─────────────────────────────────────────────────────────────────────

    def get_dihedral(self, dtype: DihedralType) -> Array:
        """
        Get specific named dihedral angles.

        Args:
            dtype: Type of dihedral to retrieve (e.g., DihedralType.PHI).

        Returns:
            Array of dihedral values in radians (one per applicable residue).

        Raises:
            ValueError: If the specified dihedral type is not found in the structure.

        Example:
            >>> phi = manager.get_dihedral(DihedralType.PHI)
            >>> psi = manager.get_dihedral(DihedralType.PSI)
        """
        # Ensure internal coordinates are computed
        if not self._internal_valid:
            self._recompute_internal()

        # Compute dihedral indices if not already cached
        if self._dihedral_indices is None:
            self._compute_dihedral_indices()

        # Lookup indices for this dihedral type
        indices = self._dihedral_indices.get(dtype.value)
        if indices is None or len(indices) == 0:
            raise ValueError(
                f"No {dtype.value} dihedrals found in structure. "
                f"This may be because the structure doesn't contain the appropriate molecule type."
            )

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
        if self._dihedral_indices is None:
            self._compute_dihedral_indices()

        # Lookup indices for this dihedral type
        indices = self._dihedral_indices.get(dtype.value)
        if indices is None:
            raise ValueError(
                f"No {dtype.value} dihedrals found in structure. "
                f"This may be because the structure doesn't contain the appropriate molecule type."
            )

        # Copy dihedrals array to avoid in-place modification issues
        if is_torch(self._dihedrals):
            new_dihedrals = self._dihedrals.clone()
        else:
            new_dihedrals = self._dihedrals.copy()

        # Update specific dihedrals
        new_dihedrals[indices] = values

        # Use setter to trigger invalidation
        self.dihedrals = new_dihedrals

    # ─────────────────────────────────────────────────────────────────────
    # Backend Conversion
    # ─────────────────────────────────────────────────────────────────────

    def to_numpy(self) -> "CoordinateManager":
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

        # Copy Z-matrix and dihedral indices (these are backend-independent structures)
        new_manager._zmatrix = self._zmatrix
        new_manager._orphan_atoms = self._orphan_atoms
        if self._orphan_coords is not None:
            new_manager._orphan_coords = to_numpy(self._orphan_coords)

        # Convert dihedral indices if they exist
        if self._dihedral_indices is not None:
            new_manager._dihedral_indices = {
                k: to_numpy(v) for k, v in self._dihedral_indices.items()
            }

        return new_manager

    def to_torch(self) -> "CoordinateManager":
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

        # Copy Z-matrix and orphan atoms
        new_manager._zmatrix = self._zmatrix
        new_manager._orphan_atoms = self._orphan_atoms
        if self._orphan_coords is not None:
            new_manager._orphan_coords = to_torch(self._orphan_coords)

        # Convert dihedral indices if they exist
        if self._dihedral_indices is not None:
            new_manager._dihedral_indices = {
                k: to_torch(v) for k, v in self._dihedral_indices.items()
            }

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

        # Copy Z-matrix and orphan atoms
        new_manager._zmatrix = self._zmatrix
        new_manager._orphan_atoms = self._orphan_atoms
        if self._orphan_coords is not None:
            new_manager._orphan_coords = self._orphan_coords.to(device)

        # Move dihedral indices if they exist
        if self._dihedral_indices is not None:
            new_manager._dihedral_indices = {
                k: v.to(device) for k, v in self._dihedral_indices.items()
            }

        return new_manager
