"""
Polymer class representing molecular structures.

The Polymer class provides a unified interface for working with molecular
structures loaded from CIF files. It supports RNA, DNA, proteins, and
other molecular types.
"""

from __future__ import annotations
from typing import Any, Generator, TYPE_CHECKING

import numpy as np

from ..backend import Array, is_torch, size as arr_size, check_compatible, to_numpy
from ..backend import ops
from ..biochemistry import Scale, Molecule
from ..biochemistry._generated_molecule import molecule_type

if TYPE_CHECKING:
    import torch
    from .hetero import HeteroAtoms
    from ..biochemistry.linking import FrameDefinition
from ..operations.reduction import Reduction, ReductionResult
from .hierarchy import _Hierarchy
from .base import AtomContainer, Field, Metadata, _KNOWN_FIELDS
from ..biochemistry import (
    Residue,
    Atom,
    AtomGroup,
    ELEMENT_NAMES,
)
from ..utils import filter_by_mask
from ..utils.formatting import format_chain_table


UNKNOWN = "UNKNOWN"


class Polymer(AtomContainer):
    """
    A molecular structure with coordinates, atom types, and hierarchy.

    Represents a complete molecular assembly with multiple scales of
    organization: atoms, residues, chains, and molecules. Provides
    methods for geometric operations, selection, and analysis.

    Polymer objects contain only polymer atoms. Non-polymer atoms
    (water, ions, ligands) are stored separately in HeteroAtoms
    objects, accessible via the hetero() method.

    Attributes:
        coordinates: (N, 3) tensor of atom positions.
        atoms: (N,) tensor of atom type indices.
        elements: (N,) tensor of element indices.
        sequence: (R,) tensor of residue type indices.
        names: List of chain names.
        lengths: (C,) tensor of residues per chain.
    """

    # Scales this container supports
    _allowed_scales = {Scale.ATOM, Scale.RESIDUE, Scale.CHAIN, Scale.MOLECULE}

    # ─────────────────────────────────────────────────────────────────────────
    # Metadata Descriptors - values passed through without conversion
    # ─────────────────────────────────────────────────────────────────────────

    # Molecule-level (pdb_id is inherited from AtomContainer)
    resolution = Metadata(Scale.MOLECULE)
    date = Metadata(Scale.MOLECULE)

    # Per-chain lists
    names = Metadata(Scale.CHAIN, is_list=True)
    descriptions = Metadata(Scale.CHAIN, is_list=True)

    def _init_from_kwargs(self, kwargs: dict) -> None:
        """Handle Polymer-specific initialization."""
        # Extract internal state (not Field or Metadata)
        connections = kwargs.pop('connections', None)
        connection_types = kwargs.pop('connection_types', None)
        hetero = kwargs.pop('hetero', None)

        if kwargs:
            raise TypeError(
                f"__init__() got unexpected keyword arguments: {list(kwargs.keys())}"
            )

        object.__setattr__(self, '_bonds', None)
        object.__setattr__(self, '_connections', connections)
        object.__setattr__(self, '_connection_types', connection_types)
        object.__setattr__(self, '_hetero', hetero)

    @property
    def lengths(self) -> Array:
        """Residues per chain (C,) array. Delegated to hierarchy."""
        return self._hierarchy.lengths

    def _convert_backend(self, to_func) -> dict:
        """Override to also convert HETATM data."""
        result = super()._convert_backend(to_func)

        # Convert HETATM data if present
        hetero = object.__getattribute__(self, '_hetero')
        if hetero is not None:
            from ..backend import to_numpy as _to_numpy
            if to_func is _to_numpy:
                result['hetero'] = hetero.numpy()
            else:
                result['hetero'] = hetero.torch()

        return result

    def _residue_coords(
        self: Polymer,
        idx: int,
    ) -> tuple[Array, Array, Residue]:
        """
        Extract coordinates, atoms, and residue type for a single residue.

        This is an internal helper for methods that need to work with individual
        residue data, such as align() and extend().

        Args:
            idx: Residue index. Negative indices are supported (e.g., -1 for last).

        Returns:
            Tuple of:
            - coords: (n_atoms, 3) coordinates for this residue
            - atoms: (n_atoms,) atom type indices
            - residue: Residue enum for this residue type
        """
        n_residues = self.size(Scale.RESIDUE)

        # Handle negative indices
        if idx < 0:
            idx = n_residues + idx
        if idx < 0 or idx >= n_residues:
            raise IndexError(
                f"Residue index {idx} out of range for Polymer with {n_residues} residues"
            )

        # Compute atom offset and size for this residue
        res_sizes = self.counts(Scale.RESIDUE)
        atom_offset = res_sizes[:idx].sum().item() if idx > 0 else 0
        n_atoms = res_sizes[idx].item()

        # Extract data
        coords = self.coordinates[atom_offset:atom_offset + n_atoms]
        atoms = self.atoms[atom_offset:atom_offset + n_atoms]
        residue = Residue.from_index(self.sequence[idx].item())

        return coords, atoms, residue

    def _compute_frame_matrices(
        self: Polymer,
        frame: "FrameDefinition | None" = None,
    ) -> tuple[Array, Array]:
        """
        Compute local coordinate frame origins and rotation matrices.

        Args:
            frame: FrameDefinition specifying origin, axis_ref, and plane_ref.
                Defaults to GLYCOSIDIC_FRAME.

        Returns:
            Tuple of (origins, Rs) where:
            - origins: (n_residues, 3) frame origin positions
            - Rs: (n_residues, 3, 3) rotation matrices
        """
        from ..geometry.transforms import frame_from_positions

        if frame is None:
            from ..biochemistry.linking import GLYCOSIDIC_FRAME
            frame = GLYCOSIDIC_FRAME

        frame_atoms = [frame.origin, frame.axis_ref, frame.plane_ref]
        positions = self.gather(frame_atoms)
        return frame_from_positions(positions)

    def frames(
        self: Polymer,
        frame: "FrameDefinition | None" = None,
    ) -> Array:
        """
        Compute local coordinate frames for each residue.

        Args:
            frame: FrameDefinition specifying origin, axis_ref, and plane_ref.
                Defaults to GLYCOSIDIC_FRAME (C1' origin, Z toward N9/N1).
                Common frames from ciffy.biochemistry.linking:
                - GLYCOSIDIC_FRAME: For nucleotides (C1' origin, Z toward N9/N1)
                - PROTEIN_BACKBONE_FRAME: For proteins (CA origin, Z toward N)

        Returns:
            (n_residues, 6) frames as [axis_angle (3), origin (3)] where
            axis_angle encodes the rotation (direction is axis, magnitude
            is angle in radians).

        Raises:
            ValueError: If required frame atoms are missing from any residue.

        Example:
            >>> frames = polymer.strip().frames()
            >>> # frames[i, :3] is axis-angle rotation for residue i
            >>> # frames[i, 3:] is origin position for residue i
        """
        from ..geometry.transforms import rotation_to_axis_angle

        origins, Rs = self._compute_frame_matrices(frame)
        axis_angles = rotation_to_axis_angle(Rs)
        return ops.cat([axis_angles, origins], axis=1)

    def align(
        self: Polymer,
        frame: "FrameDefinition | None" = None,
        *,
        return_origins: bool = False,
    ) -> "tuple[Polymer, Array] | tuple[Polymer, Array, Array]":
        """
        Align all residues to a specified local coordinate frame.

        This is the GLOBAL FRAME paradigm - it puts each residue in a consistent
        local frame independent of global position. Use for per-residue analysis
        and ML models that operate on individual residues.

        Args:
            frame: FrameDefinition specifying origin, axis_ref, and plane_ref
                AtomGroups. Defaults to GLYCOSIDIC_FRAME (C1' origin, Z toward
                N9/N1) which works for all nucleotides. Common frames from
                ciffy.biochemistry.linking:
                - GLYCOSIDIC_FRAME: For nucleotides (C1' origin, Z toward N9/N1)
                - PROTEIN_BACKBONE_FRAME: For proteins (CA origin, Z toward N)
            return_origins: If True, also return the frame origins needed for
                unalign(). Default False for backward compatibility.

        Returns:
            If return_origins=False (default):
                Tuple of (aligned_polymer, Rs) where:
                - aligned_polymer: New Polymer with aligned coordinates
                - Rs: (n_residues, 3, 3) rotation matrices used for alignment

            If return_origins=True:
                Tuple of (aligned_polymer, Rs, origins) where:
                - origins: (n_residues, 3) frame origin positions needed for unalign()

        Raises:
            ValueError: If required frame atoms are missing from any residue.

        Example:
            >>> # Basic usage (backward compatible)
            >>> aligned, Rs = polymer.strip().align()
            >>> # Rs[i] is the rotation matrix for residue i

            >>> # Full usage with origins for unalign()
            >>> aligned, Rs, origins = polymer.align(return_origins=True)
            >>> restored = aligned.unalign(Rs, origins)
            >>> ciffy.rmsd(restored, polymer) < 0.001  # True
        """
        origins, Rs = self._compute_frame_matrices(frame)

        # Expand origins and rotations to atom level for vectorized alignment
        membership = self.membership(Scale.RESIDUE)
        origins_expanded = origins[membership]
        Rs_expanded = Rs[membership]

        # Apply alignment: (coords - origin) @ R
        centered = self.coordinates - origins_expanded
        aligned_coords = (centered[:, None, :] @ Rs_expanded).squeeze(1)

        aligned_polymer = self.copy(coordinates=aligned_coords)

        if return_origins:
            return aligned_polymer, Rs, origins
        return aligned_polymer, Rs

    def unalign(
        self: Polymer,
        Rs: Array,
        origins: Array,
    ) -> Polymer:
        """
        Reverse the alignment operation: restore residues to their original positions.

        This is the inverse of align(). Takes a polymer with local-frame coordinates
        (each residue centered at origin) and restores the original global positions.

        This is the GLOBAL FRAME paradigm - use for restoring aligned coordinates
        back to their original spatial arrangement.

        Args:
            Rs: (n_residues, 3, 3) rotation matrices from align().
            origins: (n_residues, 3) frame origin positions from align(return_origins=True).

        Returns:
            Polymer with coordinates restored to original global frame.

        Contract:
            >>> aligned, Rs, origins = polymer.align(return_origins=True)
            >>> restored = aligned.unalign(Rs, origins)
            >>> ciffy.rmsd(restored, polymer) < 0.001  # True

        Note:
            The Rs matrices encode how each residue was rotated during alignment.
            The origins encode where each residue's frame was located.
            Both are required to fully reverse the alignment.
        """
        n_residues = self.size(Scale.RESIDUE)
        if Rs.shape[0] != n_residues:
            raise ValueError(
                f"Rs has {Rs.shape[0]} rows but polymer has {n_residues} residues"
            )
        if origins.shape[0] != n_residues:
            raise ValueError(
                f"origins has {origins.shape[0]} rows but polymer has {n_residues} residues"
            )

        # Expand to atom level
        membership = self.membership(Scale.RESIDUE)
        origins_expanded = origins[membership]
        Rs_expanded = Rs[membership]

        # Reverse alignment: coords @ R.T + origin
        # align does: (coords - origin) @ R
        # unalign does: aligned_coords @ R.T + origin
        # Rs_expanded is (N_atoms, 3, 3), transpose last two dims to get R.T
        Rs_T = ops.transpose(Rs_expanded, (0, 2, 1))
        unaligned_coords = (self.coordinates[:, None, :] @ Rs_T).squeeze(1) + origins_expanded

        return self.copy(coordinates=unaligned_coords)

    def local_transforms(
        self: Polymer,
        source_frame: "FrameDefinition",
        target_frame: "FrameDefinition",
    ) -> Array:
        """
        Compute inter-residue SE(3) transforms using specified source and target frames.

        This is the LOCAL FRAME paradigm - each transform[i] describes how to go from
        source_frame in residue i to target_frame in residue i+1. Use for chain
        reconstruction and backbone sampling.

        Args:
            source_frame: Frame definition for residue i (5'/N-terminal side).
                REQUIRED - no default to make frame choice explicit.
            target_frame: Frame definition for residue i+1 (3'/C-terminal side).
                REQUIRED - no default to make frame choice explicit.

        Returns:
            (n_residues, 6) array where:
            - transforms[i] = SE(3) from source_frame[i] TO target_frame[i+1]
            - transforms[-1] = zeros (no successor residue)
            - Each transform is [axis_angle (3), translation (3)]

        Raises:
            TypeError: If source_frame or target_frame is None (fail-fast).

        Common Frame Pairs:
            - GLYCOSIDIC→GLYCOSIDIC: For ML models (same frame, canonical orientation)
            - O3P_FRAME→P_FRAME: For physical backbone modeling (actual bond geometry)

        Example:
            >>> from ciffy.biochemistry.linking import GLYCOSIDIC_FRAME, O3P_FRAME, P_FRAME
            >>>
            >>> # ML use case: same frame for both
            >>> transforms = polymer.local_transforms(GLYCOSIDIC_FRAME, GLYCOSIDIC_FRAME)
            >>>
            >>> # Physical modeling: different frames at bond site
            >>> transforms = polymer.local_transforms(O3P_FRAME, P_FRAME)

        Invariant:
            Transforms are INVARIANT to global rotation/translation of polymer.
            The same structure in different poses produces identical transforms.

        Contract:
            >>> transforms = polymer.local_transforms(src, tgt)
            >>> aligned, _ = polymer.align(frame=tgt)
            >>> rebuilt = aligned.apply_local_transforms(transforms, src, tgt)
            >>> ciffy.rmsd(rebuilt, polymer) < 0.01  # True
        """
        from ..geometry.transforms import compute_relative_transform

        # Fail-fast: require both frames
        if source_frame is None:
            raise TypeError("source_frame is required (got None)")
        if target_frame is None:
            raise TypeError("target_frame is required (got None)")

        n_residues = self.size(Scale.RESIDUE)

        # Compute frames for source and target
        src_origins, src_Rs = self._compute_frame_matrices(source_frame)
        if source_frame is target_frame:
            # Optimization: same frame for both, reuse
            tgt_origins, tgt_Rs = src_origins, src_Rs
        else:
            tgt_origins, tgt_Rs = self._compute_frame_matrices(target_frame)

        # Compute transforms: source_frame[i] -> target_frame[i+1]
        transforms = ops.zeros_nd((n_residues, 6), like=self.coordinates)
        for i in range(n_residues - 1):
            transforms[i] = compute_relative_transform(
                src_origins[i], src_Rs[i],
                tgt_origins[i + 1], tgt_Rs[i + 1],
            )

        return transforms

    def apply_transforms(
        self: Polymer,
        transforms: Array,
    ) -> Polymer:
        """
        Apply relative per-residue transforms to build a chain.

        .. deprecated::
            Use `apply_local_transforms(transforms, source_frame, target_frame)` instead.
            This method does not specify which frames are used for the transforms,
            leading to ambiguity. The new method requires explicit frame specification.

        This is the inverse of align() - it takes a polymer with local-frame
        coordinates and chains residues together using relative transforms.

        Each transform specifies how to position residue i relative to residue i-1.
        The first residue stays at the origin; subsequent residues are positioned
        by chaining the transforms.

        Args:
            transforms: (n_residues, 6) array where each row is
                [axis-angle (3), translation (3)] specifying the relative
                SE(3) transform from the previous residue's frame.
                transforms[0] is ignored (first residue stays at origin).

        Returns:
            New Polymer with global coordinates.

        Example:
            >>> # Old API (deprecated):
            >>> aligned, _ = polymer.align()
            >>> rebuilt = aligned.apply_transforms(transforms)
            >>>
            >>> # New API (preferred):
            >>> from ciffy.biochemistry.linking import GLYCOSIDIC_FRAME
            >>> transforms = polymer.local_transforms(GLYCOSIDIC_FRAME, GLYCOSIDIC_FRAME)
            >>> aligned, _ = polymer.align(frame=GLYCOSIDIC_FRAME)
            >>> rebuilt = aligned.apply_local_transforms(
            ...     transforms, GLYCOSIDIC_FRAME, GLYCOSIDIC_FRAME
            ... )
        """
        import warnings
        warnings.warn(
            "apply_transforms() is deprecated. Use apply_local_transforms() with "
            "explicit source_frame and target_frame parameters instead. "
            "See docstring for migration example.",
            DeprecationWarning,
            stacklevel=2,
        )
        from ..geometry.transforms import apply_relative_transform

        n_residues = self.size(Scale.RESIDUE)
        if transforms.shape[0] != n_residues:
            raise ValueError(
                f"transforms has {transforms.shape[0]} rows but polymer has "
                f"{n_residues} residues"
            )

        coords = self.coordinates
        counts = self.counts(Scale.RESIDUE)

        # Build global coordinates
        global_coords = ops.clone(coords)
        current_R = ops.eye(3, like=coords)
        current_origin = ops.zeros_nd((3,), like=coords)

        # Chain residues together
        atom_offset = 0
        for i in range(n_residues):
            n_atoms_i = int(counts[i])
            local_coords = coords[atom_offset:atom_offset + n_atoms_i]

            # Transform to global frame: local @ R.T + origin
            global_coords[atom_offset:atom_offset + n_atoms_i] = (
                local_coords @ current_R.T + current_origin
            )

            # Update frame for next residue
            if i < n_residues - 1:
                current_origin, current_R = apply_relative_transform(
                    current_origin, current_R, transforms[i]
                )

            atom_offset += n_atoms_i

        return self.copy(coordinates=global_coords)

    def apply_local_transforms(
        self: Polymer,
        transforms: Array,
        source_frame: "FrameDefinition",
        target_frame: "FrameDefinition",
    ) -> Polymer:
        """
        Chain residues together using inter-residue transforms.

        This is the LOCAL FRAME paradigm - takes a polymer with LOCAL coordinates
        (each residue in its own frame, typically aligned to target_frame) and
        chains them using the transforms.

        Args:
            transforms: (n_residues, 6) array from local_transforms().
            source_frame: Frame definition for residue i (must match local_transforms).
                REQUIRED - no default to make frame choice explicit.
            target_frame: Frame definition for residue i+1 (must match local_transforms).
                REQUIRED - no default to make frame choice explicit.

        Returns:
            Polymer with GLOBAL coordinates (chain assembled in world frame).

        Raises:
            TypeError: If source_frame or target_frame is None (fail-fast).

        Algorithm:
            1. First residue placed at global origin
            2. For each subsequent residue i:
               a. Compute source_frame in residue i-1 (now in global coords)
               b. Apply transforms[i-1] to get target_frame position for residue i
               c. Compute target_frame in residue i's local coords
               d. Align local target_frame to global target_frame position

        Contract:
            >>> transforms = polymer.local_transforms(src, tgt)
            >>> aligned, _ = polymer.align(frame=tgt)
            >>> rebuilt = aligned.apply_local_transforms(transforms, src, tgt)
            >>> ciffy.rmsd(rebuilt, polymer) < 0.01  # True
        """
        from ..geometry.transforms import (
            apply_relative_transform,
            extract_frame_positions,
            frame_from_positions,
            rigid_align,
        )

        # Fail-fast: require both frames
        if source_frame is None:
            raise TypeError("source_frame is required (got None)")
        if target_frame is None:
            raise TypeError("target_frame is required (got None)")

        n_residues = self.size(Scale.RESIDUE)
        if transforms.shape[0] != n_residues:
            raise ValueError(
                f"transforms has {transforms.shape[0]} rows but polymer has "
                f"{n_residues} residues"
            )

        coords = self.coordinates
        counts = self.counts(Scale.RESIDUE)

        # Build global coordinates
        global_coords = ops.clone(coords)

        # Compute atom offsets
        atom_offsets = [0]
        for i in range(n_residues):
            atom_offsets.append(atom_offsets[-1] + int(counts[i]))

        # First residue stays at origin (its local coords are already aligned)
        # For subsequent residues, compute frame alignment
        for i in range(1, n_residues):
            # Get previous residue's atoms (now in global coords)
            prev_start, prev_end = atom_offsets[i - 1], atom_offsets[i]
            prev_global = global_coords[prev_start:prev_end]
            prev_atoms = self.atoms[prev_start:prev_end]

            # Compute source_frame in previous residue (in global coords)
            src_positions = extract_frame_positions(prev_global, prev_atoms, source_frame)
            src_origin, src_R = frame_from_positions(src_positions)

            # Apply transform to get target_frame position for current residue
            target_origin, target_R = apply_relative_transform(
                src_origin, src_R, transforms[i - 1]
            )

            # Get current residue's atoms (still in local coords)
            curr_start, curr_end = atom_offsets[i], atom_offsets[i + 1]
            curr_local = coords[curr_start:curr_end]
            curr_atoms = self.atoms[curr_start:curr_end]

            # Compute target_frame in current residue's local coords
            tgt_positions = extract_frame_positions(curr_local, curr_atoms, target_frame)
            tgt_local_origin, tgt_local_R = frame_from_positions(tgt_positions)

            # Align local target_frame to global target_frame position
            global_coords[curr_start:curr_end] = rigid_align(
                curr_local, tgt_local_origin, tgt_local_R, target_origin, target_R
            )

        return self.copy(coordinates=global_coords)

    def sort_atoms(self: Polymer) -> Polymer:
        """
        Sort atoms within each residue by atom type enum value.

        This creates a canonical atom ordering that is consistent regardless
        of the original CIF file ordering. Useful for ensuring training and
        inference use the same atom order.

        Returns:
            New Polymer with all atom-level fields reordered so atoms within
            each residue are sorted by their enum value.

        Example:
            >>> # Canonical encoding: align then sort
            >>> aligned, _ = polymer.align()
            >>> canonical = aligned.sort_atoms()
            >>> for i in range(canonical.size(Scale.RESIDUE)):
            ...     res = canonical.residue(i)
            ...     # atoms are now in sorted order
        """
        from ..backend.ops import argsort
        from ..biochemistry import Atom

        if self.size(Scale.RESIDUE) == 0:
            return self.copy()

        # Vectorized segment argsort: create combined key that keeps residues separate
        # key = residue_id * offset + atom_value, where offset > max atom value
        # argsort(key) sorts atoms within each residue in one operation
        membership = self.membership(Scale.RESIDUE)
        offset = Atom.count() + 1
        combined_key = membership * offset + self.atoms
        sort_indices = argsort(combined_key)

        # Reorder all atom-level fields
        overrides = {}
        for name, field in self._get_fields().items():
            if field.scale == Scale.ATOM:
                overrides[name] = field.data[sort_indices]

        return self.copy(**overrides)


    def _copy_internal_state(self, instance: Polymer, overrides: dict) -> None:
        """Copy Polymer-specific internal state to new instance."""
        # Handle hetero override or copy from self
        if 'hetero' in overrides:
            hetero = overrides.pop('hetero')
        else:
            hetero = object.__getattribute__(self, '_hetero')

        # Set internal state
        object.__setattr__(instance, '_bonds', None)  # Bonds need recomputation
        object.__setattr__(instance, '_connections', object.__getattribute__(self, '_connections'))
        object.__setattr__(instance, '_connection_types', object.__getattribute__(self, '_connection_types'))
        object.__setattr__(instance, '_hetero', hetero)

    # ─────────────────────────────────────────────────────────────────────────
    # Computed Properties
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def bonds(self) -> Array:
        """
        Covalent bonds as atom index pairs.

        Returns:
            (B, 2) int64 array where each row [i, j] represents a bond
            between atoms i and j (with i < j). Backend matches polymer.

        Note:
            Computed lazily and cached. Includes both intra-residue bonds
            and inter-residue linkages.
        """
        if self._bonds is None:
            from ..backend.graph import build_bond_graph
            edges, _ = build_bond_graph(self)
            # Filter to i < j to avoid duplicates
            edges = edges[edges[:, 0] < edges[:, 1]]
            # Convert to polymer's backend
            self._bonds = ops.to_backend(edges, self._hierarchy._ref)
        return self._bonds

    def adjacency(self, dtype: str = 'bool') -> Array:
        """
        Symmetric adjacency matrix of the bond graph.

        Args:
            dtype: Data type for the matrix ('bool', 'float32', 'int32').

        Returns:
            (N, N) symmetric matrix where adj[i,j] = True/1 if atoms
            i and j are bonded.

        Note:
            This is O(N²) memory. For large structures, use bonds
            property directly or CSR representation.
        """
        n_atoms = self.size()
        bonds = self.bonds

        # Create zero matrix in correct backend
        adj = ops.zeros((n_atoms, n_atoms), like=self._hierarchy._ref, dtype=dtype)

        # Set symmetric entries
        adj[bonds[:, 0], bonds[:, 1]] = 1
        adj[bonds[:, 1], bonds[:, 0]] = 1

        return adj

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

    @property
    def vdw_radii(self) -> Array:
        """
        Van der Waals radius for each atom (Angstroms).

        Returns:
            (N,) float32 array of VDW radii, one per atom.
            Uses standard radii indexed by element type.
            Unknown elements have radius 0.0.
        """
        from ..biochemistry.constants import VDW_RADII_ARRAY
        radii_arr = ops.to_backend(VDW_RADII_ARRAY, self.elements)
        return radii_arr[self.elements]

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

    # ─────────────────────────────────────────────────────────────────────────
    # Size and Structure
    # ─────────────────────────────────────────────────────────────────────────

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
            return self._hierarchy.counts(scale)
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
        to_scale: Scale,
        reduction: Reduction = Reduction.MEAN,
        from_scale: Scale = Scale.ATOM,
    ) -> ReductionResult:
        """
        Reduce features from one scale to a coarser scale.

        Aggregates features within each unit at the output scale
        using the chosen reduction operation.

        Args:
            features: Feature tensor at from_scale.
            to_scale: Target scale to reduce to.
            reduction: Reduction type (MEAN, SUM, MIN, MAX, COLLATE).
            from_scale: Scale of input features (default: ATOM).

        Returns:
            Reduced features. For MIN/MAX, returns (values, indices).

        Examples:
            >>> # Atom -> residue (default)
            >>> res_feats = polymer.reduce(coords, Scale.RESIDUE)
            >>> # Residue -> chain (with explicit from_scale)
            >>> chain_feats = polymer.reduce(res_feats, Scale.CHAIN, from_scale=Scale.RESIDUE)
            >>> # Chain -> molecule
            >>> mol_feats = polymer.reduce(chain_feats, Scale.MOLECULE, from_scale=Scale.CHAIN)

        Note:
            When reducing from ATOM to RESIDUE scale, non-polymer atoms are
            automatically excluded since they don't belong to any residue.
        """
        return self._hierarchy.reduce(features, to_scale, reduction, from_scale)

    def expand(
        self: Polymer,
        features: Array,
        from_scale: Scale,
        to_scale: Scale = Scale.ATOM,
    ) -> Array:
        """
        Expand per-scale features to a finer scale.

        Broadcasts values from a coarser scale to a finer scale by
        repeating each value for all units in the finer scale.

        Args:
            features: Per-from_scale feature tensor.
            from_scale: Source scale (coarser).
            to_scale: Destination scale (default: ATOM, finer).

        Returns:
            Expanded feature tensor.
        """
        return self._hierarchy.expand(features, from_scale, to_scale)

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
        return self._hierarchy.membership(scale)

    def gather(
        self: Polymer,
        groups: list,
    ) -> Array:
        """
        Gather coordinates for specific atoms from each residue.

        For each atom group, finds the matching atom in each residue and
        returns their coordinates. Useful for extracting frame atoms.

        Args:
            groups: List of AtomGroups (e.g., [Sugar.C1p, PurineBase.N9]).

        Returns:
            (n_residues, len(groups), 3) coordinate array.

        Raises:
            ValueError: If any residue doesn't have exactly one atom
                matching each group.

        Example:
            >>> from ciffy.biochemistry.constants import Sugar, PurineBase
            >>> positions = polymer.gather([Sugar.C1p, PurineBase.N9, PurineBase.C4])
        """
        n_groups = len(groups)
        n_residues = self.size(Scale.RESIDUE)
        membership = self.membership(Scale.RESIDUE)

        indices = ops.empty((n_residues, n_groups), like=self.atoms, dtype='int64')

        for i, group in enumerate(groups):
            values = ops.to_backend(group.index(), self.atoms)
            mask = ops.isin(self.atoms, values)
            atom_idx = ops.nonzero_1d(mask)

            if len(atom_idx) != n_residues:
                raise ValueError(
                    f"Group {i}: expected {n_residues} matches (one per residue), "
                    f"got {len(atom_idx)}. Check for missing or duplicate atoms."
                )

            residue_idx = membership[atom_idx]
            indices[residue_idx, i] = atom_idx

        return self.coordinates[indices]

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

    def pca(
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
        from_scale: Scale,
        to_scale: Scale = Scale.ATOM,
    ) -> Array:
        """
        Create a boolean mask selecting specific units.

        Internal method used by selection operations.

        Args:
            indices: Indices of units to select.
            from_scale: Scale of the indices.
            to_scale: Scale of the output mask.

        Returns:
            Boolean array at to_scale.
        """
        from .._selection import mask
        return mask(self, indices, from_scale, to_scale)

    def _select_contiguous(self: Polymer, ix: int, scale: Scale) -> Polymer:
        """
        Fast path for selecting a single contiguous unit (chain or residue).

        Uses slice indexing for Polymer arrays (~10x speedup) but delegates
        to Hierarchy.select() for correct count recalculation.

        Args:
            ix: Index of the unit to select.
            scale: Scale of the selection (CHAIN or RESIDUE).

        Returns:
            New Polymer with the selected unit.
        """
        # Get slice bounds from hierarchy
        atom_slice, res_slice, chain_slice = self._hierarchy.bounds(ix, scale)

        # Build mask for hierarchy (small arrays, correctness matters more than speed)
        n_units = self._hierarchy.size(scale)
        mask = ops.zeros(n_units, like=self._hierarchy._ref, dtype='bool')
        mask[ix] = True
        new_hierarchy = self._hierarchy.select(mask, scale)

        # Slice all fields and annotations using slices (fast path for large arrays)
        sliced = self._slice_all(atom_slice, res_slice, chain_slice)
        sliced['hierarchy'] = new_hierarchy

        return self._clone(**sliced)

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

    def atom_type(self: Polymer, atom: Array | int) -> Polymer:
        """
        Select atoms by atom type index.

        Args:
            atom: Atom type index or indices.

        Returns:
            New Polymer with matching atoms.
        """
        from .._selection import atom_type
        return atom_type(self, atom)

    def residue_type(self: Polymer, residue: Array | int) -> Polymer:
        """
        Select residues by residue type index.

        Args:
            residue: Residue type index or indices (from Residue enum).

        Returns:
            New Polymer with matching residues.

        Example:
            >>> from ciffy.biochemistry import Residue
            >>> adenosines = polymer.residue_type(Residue.A)
            >>> purines = polymer.residue_type([Residue.A, Residue.G])
        """
        from .._selection import residue_type
        return residue_type(self, residue)

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
        return self.residue_type(CANONICAL_ALL)

    def molecule_type(self: Polymer, mol: Molecule) -> Polymer:
        """
        Select chains by molecule type.

        Args:
            mol: Molecule type to select.

        Returns:
            New Polymer with chains of that type.
        """
        from .._selection import molecule_type
        return molecule_type(self, mol)

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
            ...     waters = hetero_atoms.element_type(8)  # Oxygen atoms
        """
        from .._selection import hetero
        return hetero(self)

    def chains(self: Polymer) -> Generator[Polymer, None, None]:
        """
        Iterate over chains.

        To filter by molecule type, use `polymer.molecule_type(mol).chains()`.

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

    def has_internal_gaps(self: Polymer) -> bool:
        """
        Check if polymer has unresolved residues between resolved ones.

        Internal gaps occur when there are unresolved residues flanked by
        resolved residues on both sides. After strip(), these would create
        discontinuities where adjacent residues in the array are not actually
        bonded in the original structure.

        Returns:
            True if internal gaps exist, False otherwise.

        Example:
            >>> polymer = ciffy.load("structure.cif")
            >>> if polymer.has_internal_gaps():
            ...     # Split into contiguous segments or skip
            ...     pass
            >>> else:
            ...     stripped = polymer.strip()  # Safe to use
        """
        resolved = self.resolved()
        in_gap = False
        for i in range(len(resolved)):
            if resolved[i] and in_gap:
                return True  # Found: resolved -> unresolved -> resolved
            elif not resolved[i] and not in_gap and i > 0 and resolved[i - 1]:
                in_gap = True
        return False

    # ─────────────────────────────────────────────────────────────────────────
    # Specialized Selections
    # ─────────────────────────────────────────────────────────────────────────

    def backbone(self: Polymer) -> Polymer:
        """
        Select backbone atoms.

        For RNA/DNA: selects sugar-phosphate backbone atoms (P, OP1, OP2, O5',
        C5', C4', C3', O3', C2', O2', C1', O4').
        For protein: selects N-CA-C-O backbone atoms.

        Returns:
            New Polymer containing only backbone atoms. Residues with no
            backbone atoms are removed.

        Example:
            >>> backbone_only = polymer.backbone()
            >>> print(f"Backbone: {backbone_only.size()} atoms")
        """
        from .._selection import backbone
        return backbone(self)

    def nucleobase(self: Polymer) -> Polymer:
        """
        Select nucleobase atoms from RNA/DNA residues.

        Selects the nitrogenous base atoms (purines: N1, C2, N3, C4, C5, C6,
        N7, C8, N9; pyrimidines: N1, C2, N3, C4, C5, C6, O2, O4/N4).
        Protein chains are excluded entirely.

        Returns:
            New Polymer containing only nucleobase atoms.

        Example:
            >>> bases = rna.nucleobase()
            >>> print(f"Base atoms: {bases.size()}")
        """
        from .._selection import nucleobase
        return nucleobase(self)

    def phosphate(self: Polymer) -> Polymer:
        """
        Select phosphate group atoms from RNA/DNA residues.

        Selects P, OP1, OP2, and O5' atoms that form the phosphate group.
        Protein chains are excluded entirely.

        Returns:
            New Polymer containing only phosphate atoms.

        Example:
            >>> phosphates = rna.phosphate()
        """
        from .._selection import phosphate
        return phosphate(self)

    def sidechain(self: Polymer) -> Polymer:
        """
        Select sidechain atoms from protein residues.

        Selects all atoms except the backbone (N, CA, C, O) and hydrogen.
        For glycine, returns an empty selection. RNA/DNA chains are
        excluded entirely.

        Returns:
            New Polymer containing only sidechain atoms.

        Example:
            >>> sidechains = protein.sidechain()
        """
        from .._selection import sidechain
        return sidechain(self)

    def heavy(self: Polymer) -> Polymer:
        """
        Select heavy (non-hydrogen) atoms.

        Filters out all hydrogen atoms, keeping only C, N, O, P, S, and
        other heavy elements. Useful for reducing computational cost or
        when hydrogen positions are unreliable.

        Returns:
            New Polymer containing only heavy atoms.

        Example:
            >>> heavy_atoms = polymer.heavy()
            >>> print(f"Heavy atoms: {heavy_atoms.size()}")
        """
        from .._selection import heavy
        return heavy(self)

    # ─────────────────────────────────────────────────────────────────────────
    # Chain Operations
    # ─────────────────────────────────────────────────────────────────────────

    def _extend_from_empty(
        self: Polymer,
        residue: Residue,
        coords: Array | None,
        transform: Array | None,
        name: str,
        **fields,
    ) -> Polymer:
        """Create first residue when extending from empty polymer.

        If transform is provided, applies it relative to the global identity frame
        (origin at [0,0,0] with identity rotation).
        """
        # Determine n_atoms and reference array for backend
        if coords is not None:
            n_atoms = coords.shape[0]
            ref = coords
        elif 'atoms' in fields:
            n_atoms = len(fields['atoms'])
            ref = fields['atoms']
        else:
            raise ValueError(
                "Either coordinates or atoms must be provided for _append()."
            )

        # Apply transform relative to identity frame if provided
        if coords is not None and transform is not None:
            from ..geometry.transforms import (
                apply_relative_transform,
                extract_frame_positions,
                frame_from_positions,
                rigid_align,
            )
            from ..biochemistry.linking import LINKING_BY_TYPE
            import numpy as np

            # Get linking definition for this residue type
            link_def = LINKING_BY_TYPE.get(residue.molecule_type)
            if link_def is None:
                raise ValueError(
                    f"No linking definition for molecule type {residue.molecule_type}. "
                    f"Cannot apply transform for first residue."
                )

            # Identity frame at origin
            if is_torch(coords):
                import torch
                identity_origin = torch.zeros(3, dtype=coords.dtype, device=coords.device)
                identity_R = torch.eye(3, dtype=coords.dtype, device=coords.device)
            else:
                identity_origin = np.zeros(3, dtype=np.float32)
                identity_R = np.eye(3, dtype=np.float32)

            # Apply transform to get target frame
            target_origin, target_R = apply_relative_transform(
                identity_origin, identity_R, transform
            )

            # Get the residue's linking frame from its coordinates
            atoms_arr = fields.get('atoms')
            if atoms_arr is not None:
                next_positions = extract_frame_positions(
                    coords, atoms_arr, link_def.next_frame
                )
                next_origin, next_R = frame_from_positions(next_positions)
                # Align residue so its frame matches target
                coords = rigid_align(coords, next_origin, next_R, target_origin, target_R)
            else:
                # No atoms provided - just translate to target origin
                centroid = coords.mean(axis=0)
                coords = coords + (target_origin - centroid)

        # Create hierarchy arrays in same backend as ref
        sizes = {
            Scale.RESIDUE: ops.array([n_atoms], like=ref, dtype='int64'),
            Scale.CHAIN: ops.array([n_atoms], like=ref, dtype='int64'),
            Scale.MOLECULE: ops.array([n_atoms], like=ref, dtype='int64'),
        }
        lengths = ops.array([1], like=ref, dtype='int64')

        hierarchy = _Hierarchy.from_sizes_and_lengths(
            sizes=sizes,
            lengths=lengths,
            ref=ref,
        )

        # Create sequence/molecule_type arrays in same backend
        sequence_arr = ops.array([residue.value], like=ref, dtype='int64')
        mol_type_arr = ops.array([residue.molecule_type], like=ref, dtype='int64')

        # Build fields dict for Polymer constructor
        init_fields = {
            'sequence': Field(sequence_arr, Scale.RESIDUE),
            'molecule_types': Field(mol_type_arr, Scale.CHAIN),
        }

        # Add coordinates only if provided
        if coords is not None:
            init_fields['coordinates'] = Field(coords, Scale.ATOM)

        # Add all user-provided fields (atoms, elements, etc.)
        for field_name, data in fields.items():
            if field_name in _KNOWN_FIELDS:
                scale = _KNOWN_FIELDS[field_name]
                init_fields[field_name] = Field(data, scale)

        return Polymer(
            hierarchy,
            **init_fields,
            names=[name],
            descriptions=[""],
            pdb_id=self.pdb_id,
        )

    def _append(
        self: Polymer,
        residue: Residue,
        coordinates: Array | None = None,
        transform: Array | None = None,
        name: str = "A",
        source_frame: "FrameDefinition | None" = None,
        target_frame: "FrameDefinition | None" = None,
        **fields,
    ) -> Polymer:
        """
        Low-level append: add a residue with explicit field arrays.

        Creates a new Polymer with an additional residue. If the polymer is empty,
        creates the first residue. Otherwise, positions the residue relative to
        the last residue using the provided transform.

        The caller must provide the same atom/residue-level fields that exist on
        this polymer (e.g., atoms, elements). These are concatenated automatically.

        Note:
            This is a low-level method. Prefer `append()` for the public API.

        Args:
            residue: Residue type being added (e.g., Residue.ALA, Residue.A).
            coordinates: (n_atoms, 3) coordinates of the residue.
                If transform is provided, these are local-frame coordinates that
                will be positioned relative to the previous residue.
                If transform is None, these are absolute coordinates used as-is.
                If coordinates is None, creates a template without coordinates.
            transform: (6,) SE(3) transform [axis-angle, translation] for positioning.
                If provided, positions the residue relative to the previous one.
                If None, coordinates are used as absolute positions.
            name: Chain name (only used when extending from empty polymer).
            source_frame: Frame definition for the last residue (where transform
                originates). If None, uses the default from LINKING_BY_TYPE.
            target_frame: Frame definition for the new residue (where transform
                targets). If None, uses the default from LINKING_BY_TYPE.
            **fields: Field arrays to concatenate (atoms, elements, etc.).
                Must match fields on this polymer at ATOM/RESIDUE scale.

        Returns:
            New Polymer with the residue appended.

        Raises:
            ValueError: If polymer has multiple chains, has HETATM atoms,
                or required fields are missing.
        """
        # Handle empty polymer case
        if self.empty():
            return self._extend_from_empty(residue, coordinates, transform, name, **fields)

        # Determine n_new_atoms and handle coordinates
        if coordinates is not None:
            if transform is not None:
                # Position relative to previous residue using transform
                fields['coordinates'] = self._position_new_residue(
                    coordinates, transform, fields.get('atoms'),
                    source_frame=source_frame, target_frame=target_frame,
                )
            else:
                # Use coordinates as absolute positions
                fields['coordinates'] = coordinates
            n_new_atoms = coordinates.shape[0]
        else:
            # Template mode: no coordinates, get n_atoms from atoms field
            if 'atoms' not in fields:
                raise ValueError(
                    "Either coordinates or atoms must be provided for _append()."
                )
            n_new_atoms = len(fields['atoms'])
            # Check consistency: if existing polymer has coords, new residue must too
            coords_field = self._get_fields().get('coordinates')
            if coords_field is not None and coords_field.data is not None:
                raise ValueError(
                    "Cannot add residue without coordinates to polymer with "
                    "coordinates. Pass coordinates= to _append()."
                )

        ref = self._hierarchy._ref
        fields['sequence'] = ops.array([residue.value], like=ref, dtype='int64')

        # Check if we're starting a new chain or extending the current one
        last_chain_name = self._names[-1] if self._names else "A"
        starting_new_chain = (name != last_chain_name)

        # Concatenate all fields
        new_fields = {}

        for field_name, field in self._get_fields().items():
            if field.scale in (Scale.ATOM, Scale.RESIDUE):
                if field_name in fields:
                    new_fields[field_name] = ops.cat([field.data, fields[field_name]])
                else:
                    raise ValueError(
                        f"Field '{field_name}' required but not provided. "
                        f"This polymer has {field_name} data."
                    )
            elif field.scale == Scale.CHAIN and starting_new_chain:
                # Extend chain-level fields when starting a new chain
                if field_name == 'molecule_types':
                    new_val = ops.array([residue.molecule_type], like=ref, dtype='int64')
                    new_fields[field_name] = ops.cat([field.data, new_val])

        # Build new hierarchy and metadata
        if starting_new_chain:
            new_hierarchy = self._hierarchy.extend_new_chain(n_new_atoms)
            new_names = list(self._names) + [name]
            new_descriptions = list(self._descriptions) + [""]
        else:
            new_hierarchy = self._hierarchy.extend_residue(n_new_atoms)
            # Copy lists to avoid mutation of original
            new_names = list(self._names) if self._names else []
            new_descriptions = list(self._descriptions) if self._descriptions else []

        return self._clone(
            hierarchy=new_hierarchy,
            names=new_names,
            descriptions=new_descriptions,
            **new_fields
        )

    def append(
        self: Polymer,
        atom_group: AtomGroup,
        coordinates: "Array | LocalCoordinates | None" = None,
        *,
        residue: Residue | None = None,
        name: str = "A",
        source_frame: "FrameDefinition | None" = None,
        target_frame: "FrameDefinition | None" = None,
    ) -> Polymer:
        """
        Append a residue to the polymer chain.

        Automatically derives atoms and elements from an AtomGroup. This is the
        primary method for building chains autoregressively.

        This method uses the LOCAL FRAME paradigm when positioning residues:
        - source_frame: Frame in the last residue (5'/N-terminal side)
        - target_frame: Frame in the new residue (3'/C-terminal side)

        Args:
            atom_group: AtomGroup defining the atoms (Residue.A, model.atoms, etc.)
            coordinates: One of:
                - (n_atoms, 3) array: Absolute coordinates (no transform)
                - LocalCoordinates: Local-frame coordinates with SE(3) transform
                - None: Template mode (no coordinates)
            residue: Residue type for sequence field. Required for AtomGroup
                subsets that don't have a .value attribute.
            name: Chain name (only used for first residue).
            source_frame: Frame definition for the last residue (where transform
                originates). If None, uses the default from LINKING_BY_TYPE
                (e.g., O3P_FRAME for RNA/DNA).
            target_frame: Frame definition for the new residue (where transform
                targets). If None, uses the default from LINKING_BY_TYPE
                (e.g., P_FRAME for RNA/DNA).

        Returns:
            New Polymer with the residue appended.

        Example:
            >>> from ciffy import Polymer
            >>> from ciffy.biochemistry import Residue
            >>> from ciffy.geometry import LocalCoordinates
            >>>
            >>> # Build template (no coordinates)
            >>> p = Polymer()
            >>> for res in [Residue.A, Residue.C, Residue.G, Residue.U]:
            ...     p = p.append(res)
            >>>
            >>> # Build with absolute coordinates (first residue)
            >>> p = Polymer()
            >>> p = p.append(Residue.A, coords)
            >>>
            >>> # Build with relative coordinates (subsequent residues)
            >>> p = p.append(Residue.C, LocalCoordinates(coords, transform))
            >>>
            >>> # Build with model's atom subset
            >>> p = p.append(model.atoms, LocalCoordinates(coords, transform), residue=model.residue)
            >>>
            >>> # Build with explicit frames (ML-style same-frame transforms)
            >>> from ciffy.biochemistry.linking import GLYCOSIDIC_FRAME
            >>> p = p.append(
            ...     Residue.C,
            ...     LocalCoordinates(coords, transform),
            ...     source_frame=GLYCOSIDIC_FRAME,
            ...     target_frame=GLYCOSIDIC_FRAME,
            ... )
        """
        from ..geometry import LocalCoordinates

        # Get atom indices and elements from AtomGroup
        atom_arr = atom_group.index()
        elem_arr = atom_group.elements()

        # Determine residue for sequence field
        if residue is None:
            if atom_group.value is None:
                raise ValueError(
                    "Cannot infer residue type from atom_group subset. "
                    "Pass residue= explicitly."
                )
            residue = atom_group

        # Extract coordinates and transform from LocalCoordinates if needed
        if isinstance(coordinates, LocalCoordinates):
            actual_coords = coordinates.coordinates
            actual_transform = coordinates.transform
        else:
            actual_coords = coordinates
            actual_transform = None

        return self._append(
            residue, actual_coords, actual_transform,
            atoms=atom_arr, elements=elem_arr, name=name,
            source_frame=source_frame, target_frame=target_frame,
        )

    def _position_new_residue(
        self: Polymer,
        coords: Array,
        transform: Array,
        atoms: Array | None,
        *,
        source_frame: "FrameDefinition | None" = None,
        target_frame: "FrameDefinition | None" = None,
    ) -> Array:
        """
        Position a new residue relative to the last residue using transform.

        This method uses the LOCAL FRAME paradigm:
        - source_frame: Frame in the last residue (where transform originates)
        - target_frame: Frame in the new residue (where transform targets)

        Args:
            coords: (n_atoms, 3) local coordinates of the new residue.
            transform: (6,) SE(3) transform [axis_angle, translation].
            atoms: (n_atoms,) atom type indices for the new residue.
            source_frame: Frame definition for the last residue. If None,
                uses link_def.prev_frame (e.g., O3P_FRAME for RNA).
            target_frame: Frame definition for the new residue. If None,
                uses link_def.next_frame (e.g., P_FRAME for RNA).

        Returns:
            (n_atoms, 3) global coordinates of the new residue.
        """
        from ..biochemistry.linking import LINKING_BY_TYPE
        from ..geometry.transforms import (
            apply_relative_transform,
            extract_frame_positions,
            frame_from_positions,
            rigid_align,
        )

        # Get last residue's state
        last_coords, last_atoms, last_res_type = self._residue_coords(-1)

        # Get linking definition for default frames
        link_def = LINKING_BY_TYPE.get(last_res_type.molecule_type)

        # Determine source frame (in last residue)
        if source_frame is None:
            if link_def is None:
                raise ValueError(
                    f"No linking definition for molecule type {last_res_type.molecule_type}. "
                    f"Pass source_frame= explicitly."
                )
            source_frame = link_def.prev_frame

        # Determine target frame (in new residue)
        if target_frame is None:
            if link_def is None:
                raise ValueError(
                    f"No linking definition for molecule type {last_res_type.molecule_type}. "
                    f"Pass target_frame= explicitly."
                )
            target_frame = link_def.next_frame

        # Compute source frame (e.g., O3' for RNA) from last residue
        prev_positions = extract_frame_positions(
            last_coords, last_atoms, source_frame
        )
        prev_origin, prev_R = frame_from_positions(prev_positions)

        # Apply transform to get target frame position
        target_origin, target_R = apply_relative_transform(prev_origin, prev_R, transform)

        # Position the new residue
        if atoms is not None:
            # Ensure atoms is an array
            if isinstance(atoms, list):
                atoms = np.asarray(atoms)
            # Compute target frame (e.g., P for RNA) from new residue
            next_positions = extract_frame_positions(
                coords, atoms, target_frame
            )
            next_origin, next_R = frame_from_positions(next_positions)
            # Align new residue so its target frame matches the computed position
            return rigid_align(coords, next_origin, next_R, target_origin, target_R)
        else:
            # No atoms provided - just translate to target origin
            centroid = coords.mean(axis=0)
            return coords + (target_origin - centroid)

    # ─────────────────────────────────────────────────────────────────────────
    # String Representations
    # ─────────────────────────────────────────────────────────────────────────

    def sequence_str(self: Polymer, strict: bool = False) -> str:
        """
        Get the sequence as a single-letter string.

        Args:
            strict: If True, raise ValueError on unknown residue types.
                If False (default), use 'n' for unknown types.

        Returns:
            Single-letter sequence string (e.g., "ACGU" for RNA,
            "MGKLV" for protein).

        Raises:
            ValueError: If strict=True and an unknown residue type is found.
        """
        def abbrev(x: int) -> str:
            try:
                return Residue.from_index(x).abbrev
            except (ValueError, KeyError):
                if strict:
                    raise ValueError(f"Unknown residue index: {x}")
                return 'n'
        return "".join(abbrev(ix.item()) for ix in self.sequence)

    def atom_names(self: Polymer, strict: bool = False) -> list[str]:
        """
        Get atom names as a list of strings.

        Args:
            strict: If True, raise ValueError on unknown atom types.
                If False (default), use '?' for unknown types.

        Returns:
            List of atom name strings.

        Raises:
            ValueError: If strict=True and an unknown atom type is found.
        """
        def get_name(value: int) -> str:
            try:
                return Atom.from_value(value).name
            except KeyError:
                if strict:
                    raise ValueError(f"Unknown atom value: {value}")
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
        hierarchy = self._get_hierarchy()
        atom_counts = to_numpy(hierarchy.counts(Scale.CHAIN))
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

    def _convert_internal_state_to(
        self,
        converted: dict,
        device: "str | torch.device | None",
        dtype: "torch.dtype | None",
    ) -> None:
        """Convert Polymer-specific internal state during to()."""
        hetero = object.__getattribute__(self, '_hetero')
        if hetero is not None:
            converted['hetero'] = hetero.to(device, dtype)

    def detach(self: Polymer) -> Polymer:
        """Detach all float tensors, including HeteroAtoms."""
        super().detach()
        hetero = object.__getattribute__(self, '_hetero')
        if hetero is not None:
            hetero.detach()
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

