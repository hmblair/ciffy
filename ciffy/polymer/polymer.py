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
        strands: List of strand identifiers.
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
    strands = Metadata(Scale.CHAIN, is_list=True)
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

    def _convert_backend(self, to_func, new_hierarchy: _Hierarchy | None = None) -> dict:
        """Override to also convert HETATM data."""
        result = super()._convert_backend(to_func, new_hierarchy)

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

    def align(
        self: Polymer,
        frame: "FrameDefinition | None" = None,
    ) -> tuple[Polymer, Array]:
        """
        Align all residues to a specified local coordinate frame.

        This is the preprocessing step for training generative models - it puts
        each residue in a consistent local frame independent of global position.

        Args:
            frame: FrameDefinition specifying origin, axis_ref, and plane_ref
                AtomGroups. Defaults to GLYCOSIDIC_FRAME (C1' origin, Z toward
                N9/N1) which works for all nucleotides. Common frames from
                ciffy.biochemistry.linking:
                - GLYCOSIDIC_FRAME: For nucleotides (C1' origin, Z toward N9/N1)
                - PROTEIN_BACKBONE_FRAME: For proteins (CA origin, Z toward N)

        Returns:
            Tuple of (aligned_polymer, Rs) where:
            - aligned_polymer: New Polymer with aligned coordinates
            - Rs: (n_residues, 3, 3) rotation matrices used for alignment

        Raises:
            ValueError: If required frame atoms are missing from any residue.

        Example:
            >>> aligned, Rs = polymer.strip().align()
            >>> # Rs[i] is the rotation matrix for residue i
        """
        from ..geometry.transforms import frame_from_positions

        if frame is None:
            from ..biochemistry.linking import GLYCOSIDIC_FRAME
            frame = GLYCOSIDIC_FRAME

        # Gather frame positions for all residues: (n_residues, 3, 3)
        frame_atoms = [frame.origin, frame.axis_ref, frame.plane_ref]
        positions = self.gather(frame_atoms)

        # Compute frames in batch: origins (n_residues, 3), Rs (n_residues, 3, 3)
        origins, Rs = frame_from_positions(positions)

        # Expand origins and rotations to atom level for vectorized alignment
        # membership[i] = residue index for atom i
        membership = self.membership(Scale.RESIDUE)
        origins_expanded = origins[membership]  # (n_atoms, 3)
        Rs_expanded = Rs[membership]  # (n_atoms, 3, 3)

        # Apply alignment: (coords - origin) @ R
        # For each atom: (1, 3) @ (3, 3) -> (1, 3), squeeze to (3,)
        centered = self.coordinates - origins_expanded
        # Batched matrix multiply: einsum or manual
        # centered[:, None, :] @ Rs_expanded -> (n_atoms, 1, 3) -> squeeze
        aligned_coords = (centered[:, None, :] @ Rs_expanded).squeeze(1)

        return self.copy(coordinates=aligned_coords), Rs

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

    def _validate_consistency(self, sizes: dict[Scale, Array]) -> None:
        """
        Validate that field sizes are consistent at each scale and across scales.

        Checks:
        1. All Fields at each scale (ATOM, RESIDUE, CHAIN) have the same size.
        2. Atom counts are consistent across hierarchy (residue = chain = molecule).

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

        if not all_equal(res_count, chn_count, mol_count):
            id_str = f" for PDB {self.pdb_id}" if self.pdb_id else ""
            raise ValueError(
                f"Atom counts do not match: residues ({res_count}), "
                f"chains ({chn_count}), molecule ({mol_count}){id_str}."
            )

    def annotate(
        self,
        name: str,
        data: Array,
        scale: Scale = Scale.RESIDUE,
    ) -> "Polymer":
        """
        Register a new dynamic field on this polymer.

        Dynamic fields work exactly like built-in fields: they are accessible
        as attributes, propagate through selections, and convert with backend changes.

        Args:
            name: Field name. Must not conflict with existing attributes.
            data: Array data with first dimension matching scale size.
            scale: Scale at which the field is defined (default: RESIDUE).

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
        return super().annotate(name, data, scale)

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

        # Extract Polymer-specific internal state
        if 'hetero' in overrides:
            hetero = overrides.pop('hetero')
        else:
            hetero = object.__getattribute__(self, '_hetero')

        # Create new instance bypassing __init__ for efficiency
        polymer = object.__new__(Polymer)
        object.__setattr__(polymer, '_hierarchy', hierarchy)

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
                object.__setattr__(polymer, name, new_field)

        # Handle overrides for known fields that don't exist on source
        # (e.g., adding coordinates to a template)
        for name in list(overrides.keys()):
            if name in _KNOWN_FIELDS:
                data = overrides.pop(name)
                if data is not None:
                    scale = _KNOWN_FIELDS[name]
                    new_field = Field(data, scale)
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

        # Copy Polymer-specific internal state
        object.__setattr__(polymer, '_bonds', None)  # Bonds need recomputation
        object.__setattr__(polymer, '_connections', object.__getattribute__(self, '_connections'))
        object.__setattr__(polymer, '_connection_types', object.__getattribute__(self, '_connection_types'))
        object.__setattr__(polymer, '_hetero', hetero)

        return polymer

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
        atom_mask, res_mask, chn_mask = self._hierarchy.derive_masks(mask, scale, remove_empty)

        # Step 2: Compute new hierarchy for selection
        new_per = self._hierarchy.compute_per(atom_mask, res_mask, chn_mask, scale)
        new_hierarchy = _Hierarchy(new_per, self._hierarchy._ref,
                                   with_residues=self._hierarchy.has_residues)

        # Step 3: Slice all fields and annotations
        sliced = self._slice_all(atom_mask, res_mask, chn_mask, new_hierarchy)
        sliced['hierarchy'] = new_hierarchy

        return self._clone(**sliced)

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

    def heavy(self: Polymer) -> Polymer:
        """Select heavy (non-hydrogen) atoms."""
        from .._selection import heavy
        return heavy(self)

    # ─────────────────────────────────────────────────────────────────────────
    # Chain Operations
    # ─────────────────────────────────────────────────────────────────────────

    def _extend_from_empty(
        self: Polymer,
        residue: Residue,
        coords: Array | None,
        name: str,
        **fields,
    ) -> Polymer:
        """Create first residue when extending from empty polymer."""
        # Determine n_atoms and reference array for backend
        if coords is not None:
            n_atoms = coords.shape[0]
            ref = coords
        elif 'atoms' in fields:
            n_atoms = len(fields['atoms'])
            ref = fields['atoms']
        else:
            raise ValueError(
                "Either coordinates or atoms must be provided for extend()."
            )

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
            strands=[""],
            descriptions=[""],
            pdb_id=self.pdb_id,
        )

    def extend(
        self: Polymer,
        residue: Residue,
        coordinates: Array | None = None,
        transform: Array | None = None,
        name: str = "A",
        **fields,
    ) -> Polymer:
        """
        Append a residue to the end of a polymer (for autoregressive generation).

        Creates a new Polymer with an additional residue. If the polymer is empty,
        creates the first residue. Otherwise, positions the residue relative to
        the last residue using the provided transform.

        The caller must provide the same atom/residue-level fields that exist on
        this polymer (e.g., atoms, elements). These are concatenated automatically.

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
            **fields: Field arrays to concatenate (atoms, elements, etc.).
                Must match fields on this polymer at ATOM/RESIDUE scale.

        Returns:
            New Polymer with the residue appended.

        Raises:
            ValueError: If polymer has multiple chains, has HETATM atoms,
                or required fields are missing.

        Example:
            >>> from ciffy import Residue, Polymer
            >>>
            >>> # Start from empty polymer (first residue gets 5' terminal atoms)
            >>> poly = Polymer()
            >>> atom_group = Residue.A.terminal(start=True, end=False)
            >>> atoms, elements, coords = atom_group.index(), atom_group.elements(), atom_group.ideal
            >>> poly = poly.extend(Residue.A, coords, atoms=atoms, elements=elements)
            >>>
            >>> # Extend with relative transform (positions relative to previous residue)
            >>> atom_group = Residue.C.terminal(start=False, end=False)
            >>> atoms, elements = atom_group.index(), atom_group.elements()
            >>> local_coords, transform = model.predict_relative(...)
            >>> poly = poly.extend(Residue.C, local_coords, transform, atoms=atoms, elements=elements)
            >>>
            >>> # Extend with absolute coordinates (no transform needed)
            >>> abs_coords = model.predict_absolute(...)
            >>> poly = poly.extend(Residue.G, abs_coords, atoms=atoms, elements=elements)
        """
        # Convert list inputs to arrays
        if 'atoms' in fields and isinstance(fields['atoms'], list):
            fields['atoms'] = np.asarray(fields['atoms'])
        if 'elements' in fields and isinstance(fields['elements'], list):
            fields['elements'] = np.asarray(fields['elements'])

        # Handle empty polymer case
        if self.empty():
            return self._extend_from_empty(residue, coordinates, name, **fields)

        # Determine n_new_atoms and handle coordinates
        if coordinates is not None:
            if transform is not None:
                # Position relative to previous residue using transform
                fields['coordinates'] = self._position_new_residue(
                    coordinates, transform, fields.get('atoms')
                )
            else:
                # Use coordinates as absolute positions
                fields['coordinates'] = coordinates
            n_new_atoms = coordinates.shape[0]
        else:
            # Template mode: no coordinates, get n_atoms from atoms field
            if 'atoms' not in fields:
                raise ValueError(
                    "Either coordinates or atoms must be provided for extend()."
                )
            n_new_atoms = len(fields['atoms'])
            # Check consistency: if existing polymer has coords, new residue must too
            existing_coords = getattr(self, '_coordinates', None)
            if existing_coords is not None:
                raise ValueError(
                    "Cannot add residue without coordinates to polymer with "
                    "coordinates. Pass coordinates= to extend()."
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
            new_strands = list(self._strands) + [name]
            new_descriptions = list(self._descriptions) + [""]
        else:
            new_hierarchy = self._hierarchy.extend_residue(n_new_atoms)
            new_names = self._names
            new_strands = self._strands
            new_descriptions = self._descriptions

        return self._clone(
            hierarchy=new_hierarchy,
            names=new_names,
            strands=new_strands,
            descriptions=new_descriptions,
            **new_fields
        )

    def extend_new(
        self: Polymer,
        atom_group: AtomGroup,
        coordinates: Array | None = None,
        transform: Array | None = None,
        *,
        residue: Residue | None = None,
        name: str = "A",
    ) -> Polymer:
        """
        Extend polymer with a residue, auto-generating atoms and elements.

        A convenience wrapper around extend() that automatically derives atoms
        and elements from an AtomGroup. Use this when building chains from
        residue models that store their own AtomGroup.

        Args:
            atom_group: AtomGroup defining the atoms (Residue.A, model.atoms, etc.)
            coordinates: Optional (n_atoms, 3) coordinates. None for template mode.
            transform: Optional (6,) SE(3) transform for positioning.
            residue: Residue type for sequence field. Required for AtomGroup subsets
                that don't have a .value attribute.
            name: Chain name (only used for first residue).

        Returns:
            New Polymer with the residue appended.

        Example:
            >>> # Build with full residue
            >>> p = Polymer()
            >>> p = p.extend_new(Residue.A, coords)
            >>> p = p.extend_new(Residue.C, coords2, transform)
            >>>
            >>> # Build template (no coordinates)
            >>> p = Polymer()
            >>> for res in [Residue.A, Residue.C, Residue.G, Residue.U]:
            ...     p = p.extend_new(res)
            >>>
            >>> # Build with model's atom subset
            >>> p = p.extend_new(model.atoms, coords, transform, residue=model.residue)
        """
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

        return self.extend(
            residue, coordinates, transform,
            atoms=atom_arr, elements=elem_arr, name=name
        )

    def _position_new_residue(
        self: Polymer,
        coords: Array,
        transform: Array,
        atoms: Array | None,
    ) -> Array:
        """Position a new residue relative to the last residue using transform."""
        from ..biochemistry.linking import LINKING_BY_TYPE
        from ..geometry.transforms import (
            apply_relative_transform,
            extract_frame_positions,
            frame_from_positions,
            rigid_align,
        )

        # Get last residue's state
        last_coords, last_atoms, last_res_type = self._residue_coords(-1)

        # Get linking definition
        link_def = LINKING_BY_TYPE.get(last_res_type.molecule_type)
        if link_def is None:
            raise ValueError(
                f"No linking definition for molecule type {last_res_type.molecule_type}. "
                f"Cannot extend chains of this type."
            )

        # Compute prev frame (e.g., O3' for RNA) from last residue
        prev_positions = extract_frame_positions(
            last_coords, last_atoms, link_def.prev_frame
        )
        prev_origin, prev_R = frame_from_positions(prev_positions)

        # Apply transform to get target next frame position
        target_origin, target_R = apply_relative_transform(prev_origin, prev_R, transform)

        # Position the new residue
        if atoms is not None:
            # Ensure atoms is an array
            if isinstance(atoms, list):
                atoms = np.asarray(atoms)
            # Compute next frame (e.g., P for RNA) from new residue
            next_positions = extract_frame_positions(
                coords, atoms, link_def.next_frame
            )
            next_origin, next_R = frame_from_positions(next_positions)
            # Align new residue so its next frame matches target
            return rigid_align(coords, next_origin, next_R, target_origin, target_R)
        else:
            # No atoms provided - just translate to target origin
            centroid = coords.mean(axis=0)
            return coords + (target_origin - centroid)

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
        hierarchy = object.__getattribute__(self, '_hierarchy')
        if device is not None:
            converted['hierarchy'] = hierarchy.to(device)

        # Move HETATM data to device if present
        hetero = object.__getattribute__(self, '_hetero')
        if hetero is not None:
            converted['hetero'] = hetero.to(device, dtype)

        return self._clone(**converted)

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
            if is_torch(field.data) and field.data.requires_grad:
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

