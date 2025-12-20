"""
Polymer class representing molecular structures.

The Polymer class provides a unified interface for working with molecular
structures loaded from CIF files. It supports RNA, DNA, proteins, and
other molecular types.
"""

from __future__ import annotations
from typing import Generator, TYPE_CHECKING
from copy import copy

import numpy as np

from .backend import Array, is_torch, get_backend, size as arr_size, check_compatible, to_numpy
from .backend import ops
from .types import Scale, Molecule
from .biochemistry._generated_molecule import molecule_type

if TYPE_CHECKING:
    import torch
from .operations.reduction import Reduction, REDUCTIONS, ReductionResult, create_reduction_index
from .biochemistry import (
    Residue,
    ATOM_NAMES,
    ELEMENT_NAMES,
    Backbone,
    Nucleobase,
    Phosphate,
    Sidechain,
)
from .utils import all_equal, filter_by_mask
from .utils.formatting import format_chain_table


UNKNOWN = "UNKNOWN"


def _classify_chain_type(min_idx: int, max_idx: int,
                         large_sentinel: int, small_sentinel: int) -> int:
    """
    Classify a chain's molecule type from its min/max residue indices.

    Args:
        min_idx: Minimum residue index in the chain.
        max_idx: Maximum residue index in the chain.
        large_sentinel: Sentinel value indicating all residues were unknown (for min).
        small_sentinel: Sentinel value indicating all residues were unknown (for max).

    Returns:
        Molecule enum value as int.
    """
    # Handle case where all residues were unknown
    if min_idx == large_sentinel or max_idx == small_sentinel:
        return Molecule.UNKNOWN.value

    try:
        min_type = Residue(min_idx).molecule_type
    except ValueError:
        min_type = Molecule.UNKNOWN
    try:
        max_type = Residue(max_idx).molecule_type
    except ValueError:
        max_type = Molecule.UNKNOWN

    # If min and max agree, use that type; otherwise mark as OTHER (mixed)
    if min_type == max_type:
        return min_type.value
    return Molecule.OTHER.value


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
        nonpoly: Count of non-polymer atoms (last nonpoly atoms).
    """

    def __init__(
        self: Polymer,
        coordinates: Array,
        atoms: Array,
        elements: Array,
        sequence: Array,
        sizes: dict[Scale, Array],
        id: str,
        names: list[str],
        strands: list[str],
        lengths: Array,
        polymer_count: int | None = None,
        molecule_types: Array | None = None,
        descriptions: list[str] | None = None,
    ) -> None:
        """
        Initialize a Polymer structure.

        Args:
            coordinates: (N, 3) tensor of atom positions.
            atoms: (N,) tensor of atom type indices.
            elements: (N,) tensor of element indices.
            sequence: (R,) tensor of residue type indices.
            sizes: Dict mapping Scale to atom counts per unit.
            id: PDB identifier.
            names: List of chain names.
            strands: List of strand identifiers.
            lengths: (C,) tensor of residues per chain.
            polymer_count: Number of polymer atoms. If None, all atoms
                are assumed to be polymer atoms.
            molecule_types: (C,) array of molecule types per chain from CIF.
                If None, molecule types will be inferred from residue indices.
            descriptions: List of entity descriptions per chain, or None.

        Raises:
            ValueError: If tensor sizes are inconsistent.
        """
        self.pdb_id = id or UNKNOWN
        self.names = names
        self.strands = strands

        # Store polymer/nonpoly counts
        # If polymer_count is None, assume all atoms are polymer (backward compat)
        total_atoms = arr_size(coordinates, 0)
        if polymer_count is not None:
            self.polymer_count = polymer_count
            self.nonpoly = total_atoms - polymer_count
        else:
            self.polymer_count = total_atoms
            self.nonpoly = 0

        if not all_equal(
            arr_size(coordinates, 0),
            arr_size(atoms, 0),
            arr_size(elements, 0),
        ):
            raise ValueError(
                f"Coordinate, atom, and element tensors must have equal size "
                f"for PDB {self.pdb_id}."
            )

        res_count = sizes[Scale.RESIDUE].sum().item()
        chn_count = sizes[Scale.CHAIN].sum().item()
        mol_count = sizes[Scale.MOLECULE].sum().item()

        if not all_equal(res_count + self.nonpoly, chn_count, mol_count):
            raise ValueError(
                f"Atom counts do not match: residues ({res_count} + {self.nonpoly}), "
                f"chains ({chn_count}), molecule ({mol_count}) for PDB {self.pdb_id}."
            )

        # Store atomic properties
        self._atoms = atoms
        self._elements = elements
        self._sequence = sequence
        self._sizes = sizes
        self._lengths = lengths
        self._molecule_types = molecule_types
        self.descriptions = descriptions

        # Create topology info for coordinate manager
        from .backend.dispatch import TopologyInfo
        self._topology = TopologyInfo.from_polymer(self)

        # Initialize coordinate manager with Cartesian coordinates
        from .internal.coordinates import CoordinateManager
        self._coord_manager = CoordinateManager(coordinates, self._topology)

    # ─────────────────────────────────────────────────────────────────────────
    # Factory Methods
    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def create_empty(cls, id: str = "empty", backend: str = "numpy") -> "Polymer":
        """
        Create an empty Polymer with 0 atoms and 0 chains.

        Useful as a base case for operations that may produce empty results,
        or for testing edge cases.

        Args:
            id: PDB identifier for the empty polymer.
            backend: Array backend, either "numpy" or "torch".

        Returns:
            An empty Polymer with no atoms, residues, or chains.

        Example:
            >>> empty = Polymer.create_empty()
            >>> empty.size()
            0
            >>> empty.size(Scale.CHAIN)
            0
        """
        polymer = cls(
            coordinates=np.zeros((0, 3), dtype=np.float32),
            atoms=np.array([], dtype=np.int64),
            elements=np.array([], dtype=np.int64),
            sequence=np.array([], dtype=np.int64),
            sizes={
                Scale.RESIDUE: np.array([], dtype=np.int64),
                Scale.CHAIN: np.array([], dtype=np.int64),
                Scale.MOLECULE: np.array([0], dtype=np.int64),
            },
            id=id,
            names=[],
            strands=[],
            lengths=np.array([], dtype=np.int64),
            polymer_count=0,
        )
        return polymer.torch() if backend == "torch" else polymer

    # ─────────────────────────────────────────────────────────────────────────
    # Array Properties (with backend/device validation)
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def coordinates(self) -> Array:
        """
        (N, 3) tensor of atom positions.

        Automatically reconstructed from internal coordinates if needed.
        """
        return self._coord_manager.coordinates

    @coordinates.setter
    def coordinates(self, value: Array) -> None:
        """
        Set coordinates with backend/device validation.

        Invalidates internal coordinate representation.
        """
        # Validate backend compatibility
        if hasattr(self, '_coord_manager') and self._coord_manager._coordinates is not None:
            check_compatible(self._coord_manager._coordinates, value, "coordinates")
        self._coord_manager.coordinates = value

    @property
    def atoms(self) -> Array:
        """(N,) tensor of atom type indices."""
        return self._atoms

    @atoms.setter
    def atoms(self, value: Array) -> None:
        """Set atoms with backend/device validation."""
        if hasattr(self, '_coord_manager') and self._coord_manager._coordinates is not None:
            check_compatible(self._coord_manager._coordinates, value, "atoms")
        self._atoms = value

    @property
    def elements(self) -> Array:
        """(N,) tensor of element indices."""
        return self._elements

    @elements.setter
    def elements(self, value: Array) -> None:
        """Set elements with backend/device validation."""
        if hasattr(self, '_coord_manager') and self._coord_manager._coordinates is not None:
            check_compatible(self._coord_manager._coordinates, value, "elements")
        self._elements = value

    @property
    def sequence(self) -> Array:
        """(R,) tensor of residue type indices."""
        return self._sequence

    @sequence.setter
    def sequence(self, value: Array) -> None:
        """Set sequence with backend/device validation."""
        if hasattr(self, '_coord_manager') and self._coord_manager._coordinates is not None:
            check_compatible(self._coord_manager._coordinates, value, "sequence")
        self._sequence = value

    @property
    def lengths(self) -> Array:
        """(C,) tensor of residues per chain."""
        return self._lengths

    @lengths.setter
    def lengths(self, value: Array) -> None:
        """Set lengths with backend/device validation."""
        if hasattr(self, '_coord_manager') and self._coord_manager._coordinates is not None:
            check_compatible(self._coord_manager._coordinates, value, "lengths")
        self._lengths = value

    # ─────────────────────────────────────────────────────────────────────────
    # Internal Coordinate Properties
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def distances(self) -> Array:
        """
        Bond lengths in internal coordinate representation.

        Returns:
            (N,) array of bond lengths in Angstroms.

        Note:
            Automatically computed from Cartesian coordinates if needed.
        """
        return self._coord_manager.distances

    @distances.setter
    def distances(self, value: Array) -> None:
        """
        Set bond lengths.

        Invalidates Cartesian coordinate representation.
        """
        self._coord_manager.distances = value

    @property
    def angles(self) -> Array:
        """
        Bond angles in internal coordinate representation.

        Returns:
            (N,) array of bond angles in radians.

        Note:
            Automatically computed from Cartesian coordinates if needed.
        """
        return self._coord_manager.angles

    @angles.setter
    def angles(self, value: Array) -> None:
        """
        Set bond angles.

        Invalidates Cartesian coordinate representation.
        """
        self._coord_manager.angles = value

    @property
    def dihedrals(self) -> Array:
        """
        Dihedral angles in internal coordinate representation.

        Returns:
            (N,) array of dihedral angles in radians.

        Note:
            Automatically computed from Cartesian coordinates if needed.
        """
        return self._coord_manager.dihedrals

    @dihedrals.setter
    def dihedrals(self, value: Array) -> None:
        """
        Set dihedral angles.

        Invalidates Cartesian coordinate representation.
        """
        self._coord_manager.dihedrals = value

    def dihedral(
        self,
        dtype: "DihedralType | list[DihedralType] | tuple[DihedralType, ...]",
    ) -> Array:
        """
        Get specific named dihedral angles.

        Returns the dihedral values for atoms that "own" the specified dihedral
        type(s) in the Z-matrix representation. Uses the same mechanism as
        set_dihedral() for symmetric get/set behavior.

        Args:
            dtype: Type(s) of dihedral to retrieve. Can be a single DihedralType
                or a list/tuple of DihedralTypes. For multiple types, values are
                concatenated in the order specified.

        Returns:
            Array of dihedral values in radians. Length depends on number of
            atoms that own the specified dihedral type(s).

        Example:
            >>> from ciffy import DihedralType
            >>> phi = polymer.dihedral(DihedralType.PHI)
            >>> # Get multiple types at once
            >>> backbone = polymer.dihedral([DihedralType.PHI, DihedralType.PSI])
        """
        return self._coord_manager.get_dihedral(dtype)

    def set_dihedral(
        self,
        dtype: "DihedralType | list[DihedralType] | tuple[DihedralType, ...]",
        values: Array,
    ) -> None:
        """
        Set specific named dihedral angles.

        Args:
            dtype: Type(s) of dihedral to set. Can be a single DihedralType
                or a list/tuple of DihedralTypes.
            values: New dihedral values in radians. For multiple types, values
                should be concatenated in the same order as the dtype list.

        Raises:
            ValueError: If the specified dihedral type is not found.

        Example:
            >>> from ciffy import DihedralType
            >>> import numpy as np
            >>> # Set all phi angles to -60 degrees
            >>> polymer.set_dihedral(DihedralType.PHI, np.full(n_phi, -np.pi/3))
            >>> # Set multiple types at once
            >>> polymer.set_dihedral([DihedralType.PHI, DihedralType.PSI], backbone_values)
        """
        self._coord_manager.set_dihedral(dtype, values)

    # ─────────────────────────────────────────────────────────────────────────
    # Identification
    # ─────────────────────────────────────────────────────────────────────────

    def chain_id(self: Polymer, ix: int) -> str:
        """
        Get a unique identifier for a specific chain.

        Args:
            ix: Chain index.

        Returns:
            String combining PDB ID and chain name (e.g., "1ABC_A").
        """
        return f"{self.pdb_id}_{self.names[ix]}"

    def strand_id(self: Polymer, ix: int) -> str:
        """
        Get the strand identifier for a specific chain.

        Args:
            ix: Chain index.

        Returns:
            String combining PDB ID and strand name.
        """
        return f"{self.pdb_id}_{self.strands[ix]}"

    # ─────────────────────────────────────────────────────────────────────────
    # Size and Structure
    # ─────────────────────────────────────────────────────────────────────────

    def empty(self: Polymer) -> bool:
        """Check if the polymer has no atoms."""
        return arr_size(self.coordinates, 0) == 0

    def size(self: Polymer, scale: Scale | None = None) -> int:
        """
        Get the count at a specific scale.

        Args:
            scale: Scale level (ATOM, RESIDUE, CHAIN, MOLECULE).
                   If None, returns atom count.

        Returns:
            Number of units at the specified scale.
        """
        if scale is None:
            return arr_size(self.coordinates, 0)
        return arr_size(self._sizes[scale], 0)

    def sizes(self: Polymer, scale: Scale) -> Array:
        """
        Get the sizes tensor for a scale.

        Args:
            scale: Scale level.

        Returns:
            Tensor of atom counts per unit at this scale.
        """
        return self._sizes[scale]

    def per(self: Polymer, inner: Scale, outer: Scale) -> Array:
        """
        Get the count of inner units per outer unit.

        Args:
            inner: Inner scale (e.g., RESIDUE).
            outer: Outer scale (e.g., CHAIN).

        Returns:
            Array with count of inner units per outer unit.

        Example:
            >>> polymer.per(Scale.RESIDUE, Scale.CHAIN)
            array([150, 200, 175])  # residues per chain
        """
        if inner == outer:
            return ops.ones(self.size(inner), like=self.coordinates)

        # Atoms per {residue, chain, molecule} are stored in _sizes
        if inner == Scale.ATOM:
            return self._sizes[outer]

        # Residues per chain are stored in lengths
        if inner == Scale.RESIDUE and outer == Scale.CHAIN:
            return self.lengths

        # Single-value cases: total count as 1-element array
        if outer == Scale.MOLECULE:
            return ops.array([self.size(inner)], like=self.coordinates)

        raise ValueError(f"Cannot compute {inner.name} per {outer.name}")

    @property
    def molecule_type(self: Polymer) -> Array:
        """
        Get the molecule type of each chain.

        If molecule types were parsed from the CIF file (_entity_poly.type),
        returns those directly. Otherwise, infers types from residue indices:
        - RNA: indices 0-3 (A, C, G, U)
        - DNA: index 4 (T/DT)
        - Protein: indices 5-24 (amino acids)
        - Water: index 25 (HOH)
        - Ion: indices 26-27 (MG, CS)
        - Other: modified nucleotides (28+)

        Returns:
            Array of Molecule enum values, one per chain.
        """
        # Use stored molecule types if available (from CIF parsing)
        if self._molecule_types is not None:
            return self._molecule_types

        # Fallback: infer from residue indices
        return self._infer_molecule_type()

    def _infer_molecule_type(self: Polymer) -> Array:
        """
        Infer molecule type from residue indices (fallback when CIF doesn't have _entity_poly).

        Uses both MIN and MAX residue index per chain to robustly detect type.
        Unknown residues (index -1) are ignored when determining molecule type.
        If min and max map to different molecule types, the chain is classified
        as OTHER (mixed/heterogeneous composition).

        Returns:
            Array of Molecule enum values, one per chain.
        """
        n_chains = self.size(Scale.CHAIN)

        # Sentinel values for masking unknown residues (-1) during min/max reduction.
        # Values chosen to be outside valid Residue enum range (~0-500), ensuring
        # unknowns are never selected as min/max when valid residues exist.
        LARGE_SENTINEL = 9999
        SMALL_SENTINEL = -9999

        # Create masked copies for min/max reduction
        unknown_mask = self.sequence == -1
        seq_for_min = ops.to_backend(
            np.where(to_numpy(unknown_mask), LARGE_SENTINEL, to_numpy(self.sequence)),
            self.sequence
        )
        seq_for_max = ops.to_backend(
            np.where(to_numpy(unknown_mask), SMALL_SENTINEL, to_numpy(self.sequence)),
            self.sequence
        )

        # Get min and max residue index per chain (ignoring unknowns)
        min_res, _ = self.rreduce(seq_for_min, Scale.CHAIN, Reduction.MIN)
        max_res, _ = self.rreduce(seq_for_max, Scale.CHAIN, Reduction.MAX)

        # Convert to numpy for classification (simpler than per-element backend checks)
        min_np = to_numpy(min_res)
        max_np = to_numpy(max_res)

        # Classify each chain
        result = np.empty(n_chains, dtype=np.int64)
        for i in range(n_chains):
            result[i] = _classify_chain_type(int(min_np[i]), int(max_np[i]),
                                              LARGE_SENTINEL, SMALL_SENTINEL)

        return ops.to_backend(result, self.coordinates)

    def istype(self: Polymer, mol: Molecule) -> bool:
        """
        Check if this is a single chain of the specified type.

        Args:
            mol: Molecule type to check.

        Returns:
            True if single chain matches type, False otherwise.
        """
        types = self.molecule_type
        if arr_size(types, 0) != 1:
            return False
        return types[0].item() == mol.value

    # ─────────────────────────────────────────────────────────────────────────
    # Reduction Operations
    # ─────────────────────────────────────────────────────────────────────────

    def reduce(
        self: Polymer,
        features: Array,
        scale: Scale,
        rtype: Reduction = Reduction.MEAN,
    ) -> ReductionResult:
        """
        Reduce per-atom features to per-scale values.

        Aggregates atom-level features within each unit at the specified
        scale using the chosen reduction operation.

        Args:
            features: Per-atom feature tensor.
            scale: Scale at which to aggregate.
            rtype: Reduction type (MEAN, SUM, MIN, MAX, COLLATE).

        Returns:
            Reduced features. For MIN/MAX, returns (values, indices).

        Note:
            When reducing to RESIDUE scale, non-polymer atoms are excluded
            since they don't belong to any residue.
        """
        # Non-polymer atoms don't belong to residues, so slice them out
        # when reducing to RESIDUE scale. With reordered atoms, polymer
        # atoms are always first [0, polymer_count), so we can use simple slicing.
        if scale == Scale.RESIDUE and self.nonpoly > 0:
            features = features[:self.polymer_count]

        count = self.size(scale)
        sizes = self._sizes[scale]
        # Pass device to ensure index is on same device as features
        device = getattr(features, 'device', None)
        ix = create_reduction_index(count, sizes, device=device)

        return REDUCTIONS[rtype](features, ix, dim=0, dim_size=count)

    def rreduce(
        self: Polymer,
        features: Array,
        scale: Scale,
        rtype: Reduction = Reduction.MEAN,
    ) -> ReductionResult:
        """
        Reduce per-residue features to per-scale values.

        Like reduce(), but for features with one value per residue
        instead of per atom.

        Args:
            features: Per-residue feature tensor.
            scale: Scale at which to aggregate.
            rtype: Reduction type.

        Returns:
            Reduced features.
        """
        count = self.size(scale)
        # Pass device to ensure index is on same device as features
        device = getattr(features, 'device', None)
        ix = create_reduction_index(count, self.lengths, device=device)

        return REDUCTIONS[rtype](features, ix, dim=0, dim_size=count)

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
        # Device mismatch is handled by ops.repeat_interleave
        if dest == Scale.ATOM:
            return ops.repeat_interleave(features, self._sizes[source])
        if dest == Scale.RESIDUE:
            return ops.repeat_interleave(features, self.lengths)
        raise ValueError(f"Cannot expand to {dest.name}")

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
        return self.reduce(ops.to_int64(mask), scale, Reduction.SUM)

    def index(self: Polymer, scale: Scale) -> Array:
        """
        Get the index of each atom within units at the specified scale.

        Creates an integer array where each atom is labeled with its
        containing unit's index at the given scale. Useful for positional
        encodings, attention masking, and grouping operations.

        Args:
            scale: Scale at which to compute indices.
                - RESIDUE: atom -> residue index (0 to num_residues-1)
                - CHAIN: atom -> chain index (0 to num_chains-1)
                - MOLECULE: all atoms get index 0

        Returns:
            Integer array of shape (num_atoms,) with indices.

        Examples:
            >>> polymer = ciffy.load("structure.cif")
            >>> res_idx = polymer.index(Scale.RESIDUE)  # atom -> residue
            >>> chain_idx = polymer.index(Scale.CHAIN)  # atom -> chain

            # Use for attention masking (same-residue attention)
            >>> mask = res_idx[:, None] == res_idx[None, :]
        """
        n = self.size(scale)
        idx = ops.arange(n, like=self.coordinates)
        return self.expand(idx, scale, Scale.ATOM)

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
        coordinates = self.coordinates - expanded

        centered = copy(self)
        # Create a new coordinate manager for the copy (reuse topology from copy)
        from .internal.coordinates import CoordinateManager
        centered._coord_manager = CoordinateManager(coordinates, centered._topology)

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
        coordinates = centered.coordinates / std_expanded * size

        scaled = copy(centered)
        scaled.coordinates = coordinates

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

    def mask(
        self: Polymer,
        indices: Array | int,
        source: Scale,
        dest: Scale = Scale.ATOM,
    ) -> Array:
        """
        Create a boolean mask selecting specific units.

        Args:
            indices: Indices of units to select.
            source: Scale of the indices.
            dest: Scale of the output mask.

        Returns:
            Boolean array at dest scale.
        """
        counts = self.size(source)
        objects = ops.zeros(counts, like=self.coordinates, dtype='bool')
        objects[indices] = True
        return self.expand(objects, source, dest)

    def __getitem__(self: Polymer, key: Array | slice) -> Polymer:
        """
        Select atoms by boolean mask or slice.

        Args:
            key: Boolean mask of atoms to keep, or slice for contiguous range.

        Returns:
            New Polymer with selected atoms.
        """
        # Handle slice by converting to boolean mask
        if isinstance(key, slice):
            mask = ops.zeros(self.size(), like=self.coordinates, dtype='bool')
            mask[key] = True
            return self[mask]

        mask = key

        # Slice coordinate manager (ensures Cartesian valid, marks internal dirty)
        sliced_manager = self._coord_manager[mask]
        coordinates = sliced_manager._coordinates

        atoms = self.atoms[mask]
        elements = self.elements[mask]

        chn_sizes = self.count(mask, Scale.CHAIN)
        res_sizes = self.count(mask, Scale.RESIDUE)
        mol_sizes = self.count(mask, Scale.MOLECULE)

        # Determine which residues have atoms
        chn_mask = chn_sizes > 0
        residues = ops.repeat_interleave(chn_mask, self.lengths)

        lengths = self.lengths[chn_mask]

        sizes = {
            Scale.RESIDUE: res_sizes[residues],
            Scale.CHAIN: chn_sizes[chn_mask],
            Scale.MOLECULE: mol_sizes,
        }

        sequence = self.sequence[residues]
        names = filter_by_mask(self.names, chn_mask)
        strands = filter_by_mask(self.strands, chn_mask)

        # Calculate new polymer_count: count how many of the first
        # polymer_count atoms survive the mask (direct slice avoids O(N) allocation)
        new_polymer_count = mask[:self.polymer_count].sum().item()

        result = Polymer(
            coordinates, atoms, elements, sequence, sizes,
            self.pdb_id, names, strands, lengths, new_polymer_count,
        )

        # Replace default coord manager with sliced one and set topology
        result._coord_manager = sliced_manager
        sliced_manager._topology = result._topology

        return result

    def by_index(self: Polymer, ix: Array | int) -> Polymer:
        """
        Select chains by index.

        Args:
            ix: Chain index or indices to select.

        Returns:
            New Polymer with selected chains.

        Raises:
            IndexError: If any index is out of range.
        """
        if isinstance(ix, int):
            ix = ops.array([ix], like=self.coordinates)

        # Validate indices
        max_chain = self.size(Scale.CHAIN)
        ix_list = ix.tolist() if hasattr(ix, 'tolist') else list(ix)
        for j in ix_list:
            if j < 0 or j >= max_chain:
                raise IndexError(
                    f"Chain index {j} out of range for Polymer with {max_chain} chains"
                )

        atm_ix = self.mask(ix, Scale.CHAIN, Scale.ATOM)
        res_ix = self.mask(ix, Scale.CHAIN, Scale.RESIDUE)

        coordinates = self.coordinates[atm_ix]
        atoms = self.atoms[atm_ix]
        elements = self.elements[atm_ix]
        lengths = self.lengths[ix]

        sizes = {
            Scale.RESIDUE: self._sizes[Scale.RESIDUE][res_ix],
            Scale.CHAIN: self._sizes[Scale.CHAIN][ix],
            Scale.MOLECULE: ops.array([len(coordinates)], like=self.coordinates),
        }

        sequence = self.sequence[res_ix]
        names = [self.names[j] for j in ix]
        strands = [self.strands[j] for j in ix]

        # Calculate new polymer_count from residue sizes
        # (residue atoms are always polymer atoms)
        new_polymer_count = sizes[Scale.RESIDUE].sum().item()

        # Preserve molecule types if available
        mol_types = self._molecule_types[ix] if self._molecule_types is not None else None

        return Polymer(
            coordinates, atoms, elements, sequence, sizes,
            self.pdb_id, names, strands, lengths, new_polymer_count,
            mol_types,
        )

    def by_atom(self: Polymer, name: Array | int) -> Polymer:
        """
        Select atoms by atom type index.

        Args:
            name: Atom type index or indices.

        Returns:
            New Polymer with matching atoms.
        """
        name = ops.convert_backend(name, self.atoms)
        mask = (self.atoms[:, None] == name).any(1)
        return self[mask]

    def by_residue(self: Polymer, res: Array | int) -> Polymer:
        """
        Select residues by residue type index.

        Args:
            res: Residue type index or indices (from Residue enum).

        Returns:
            New Polymer with matching residues.

        Example:
            >>> from ciffy.biochemistry import Residue
            >>> adenosines = polymer.by_residue(Residue.ADE)
            >>> purines = polymer.by_residue([Residue.ADE, Residue.GUA])
        """
        res = ops.convert_backend(res, self.sequence)
        res_mask = (self.sequence[:, None] == res).any(1)
        atom_mask = self.expand(res_mask, Scale.RESIDUE, Scale.ATOM)
        return self[atom_mask]

    def by_residue_index(self: Polymer, ix: Array | int) -> Polymer:
        """
        Select residues by positional index.

        Unlike by_residue() which selects by residue TYPE (e.g., all adenines),
        this method selects by positional INDEX (e.g., residue 0, 1, 2...).

        Args:
            ix: Residue index or indices (0-indexed position in polymer).

        Returns:
            New Polymer with selected residues.

        Raises:
            IndexError: If any index is out of range.

        Example:
            >>> # Select first residue
            >>> first = polymer.by_residue_index(0)
            >>> # Select residues 0, 2, 4
            >>> subset = polymer.by_residue_index([0, 2, 4])
            >>> # Combine with by_atom to get specific atoms
            >>> from ciffy.biochemistry import Sugar
            >>> first_c5 = polymer.by_residue_index(0).by_atom(Sugar.C5p.index())
        """
        if isinstance(ix, int):
            ix = ops.array([ix], like=self.coordinates)

        # Validate indices
        max_res = self.size(Scale.RESIDUE)
        ix_list = ix.tolist() if hasattr(ix, 'tolist') else list(ix)
        for j in ix_list:
            if j < 0 or j >= max_res:
                raise IndexError(
                    f"Residue index {j} out of range for Polymer with {max_res} residues"
                )

        atom_mask = self.mask(ix, Scale.RESIDUE, Scale.ATOM)
        return self[atom_mask]

    def by_type(self: Polymer, mol: Molecule) -> Polymer:
        """
        Select chains by molecule type.

        Args:
            mol: Molecule type to select.

        Returns:
            New Polymer with chains of that type.
        """
        ix = ops.nonzero_1d(self.molecule_type == mol.value)
        return self.by_index(ix)

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
        if self.nonpoly == 0:
            return self

        # Slice to polymer atoms only
        coordinates = self.coordinates[:self.polymer_count]
        atoms = self.atoms[:self.polymer_count]
        elements = self.elements[:self.polymer_count]

        # Keep only chains that have residues (polymer chains)
        chain_mask = self.lengths > 0
        lengths = self.lengths[chain_mask]
        names = filter_by_mask(self.names, chain_mask)
        strands = filter_by_mask(self.strands, chain_mask)

        # Calculate chain sizes from residue sizes (atoms per chain = sum of
        # atoms per residue for that chain)
        chn_sizes = self.rreduce(self._sizes[Scale.RESIDUE], Scale.CHAIN, Reduction.SUM)
        chn_sizes = chn_sizes[chain_mask]

        sizes = {
            Scale.RESIDUE: self._sizes[Scale.RESIDUE],  # Unchanged
            Scale.CHAIN: chn_sizes,
            Scale.MOLECULE: ops.array([self.polymer_count], like=self.coordinates),
        }

        # Filter molecule types if available
        mol_types = self._molecule_types[chain_mask] if self._molecule_types is not None else None

        return Polymer(
            coordinates, atoms, elements, self.sequence, sizes,
            self.pdb_id, names, strands, lengths, self.polymer_count,
            mol_types,
        )

    def hetero(self: Polymer) -> Polymer:
        """
        Return non-polymer atoms only (HETATM: water, ions, ligands).

        Warning:
            The returned Polymer has no valid residue information.
            Residue-scale operations like reduce(scale=Scale.RESIDUE)
            will return empty results.

        Returns:
            New Polymer with only HETATM atoms. If there are no HETATM atoms,
            returns a Polymer with 0 atoms.

        Example:
            >>> p = load("file.cif")
            >>> ligands = p.hetero()  # Get waters/ions/ligands
            >>> if not ligands.empty():
            ...     ligands.center(Scale.ATOM)  # Works on atom scale
        """
        return self[self.polymer_count:]

    def chains(
        self: Polymer,
        mol: Molecule | None = None,
    ) -> Generator[Polymer, None, None]:
        """
        Iterate over chains, optionally filtered by type.

        Args:
            mol: Optional molecule type filter.

        Yields:
            Individual chain Polymers.
        """
        for ix in range(self.size(Scale.CHAIN)):
            chain = self.by_index(ix)
            if mol is None or chain.istype(mol):
                yield chain

    def resolved(self: Polymer, scale: Scale = Scale.RESIDUE) -> Array:
        """
        Get mask of resolved (non-empty) units.

        Args:
            scale: Scale to check.

        Returns:
            Boolean tensor where True indicates resolved units.
        """
        return self._sizes[scale] != 0

    def strip(self: Polymer, scale: Scale = Scale.RESIDUE) -> Polymer:
        """
        Remove unresolved units at a scale.

        Args:
            scale: Scale at which to strip.

        Returns:
            New Polymer without empty units.
        """
        poly = copy(self)

        resolved = self._sizes[scale] > 0
        poly._sizes = copy(self._sizes)
        poly._sizes[scale] = poly._sizes[scale][resolved]

        poly.lengths = self.rreduce(ops.to_int64(resolved), Scale.CHAIN, Reduction.SUM)
        poly.sequence = self.sequence[resolved]

        return poly

    # ─────────────────────────────────────────────────────────────────────────
    # Specialized Selections
    # ─────────────────────────────────────────────────────────────────────────

    def backbone(self: Polymer) -> Polymer:
        """Select backbone atoms (sugar-phosphate for RNA/DNA, N-CA-C-O for protein)."""
        return self.by_atom(Backbone.index())

    def nucleobase(self: Polymer) -> Polymer:
        """Select RNA nucleobase atoms."""
        return self.by_atom(Nucleobase.index())

    def phosphate(self: Polymer) -> Polymer:
        """Select RNA/DNA phosphate atoms."""
        return self.by_atom(Phosphate.index())

    def sidechain(self: Polymer) -> Polymer:
        """Select protein sidechain atoms."""
        return self.by_atom(Sidechain.index())

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
                return Residue(x).abbrev
            except ValueError:
                return 'n'
        return "".join(abbrev(ix.item()) for ix in self.sequence)

    def atom_names(self: Polymer) -> list[str]:
        """
        Get atom names as a list of strings.

        Returns:
            List of atom name strings.
        """
        return [ATOM_NAMES.get(ix.item(), '?') for ix in self.atoms]

    def chain_info(self: Polymer) -> list[dict]:
        """
        Get information about each chain.

        Returns:
            List of dicts with keys: 'chain', 'type', 'res', 'atoms'.
        """
        types_np = to_numpy(self.molecule_type)
        lengths_np = to_numpy(self.lengths)
        atoms_np = to_numpy(self._sizes[Scale.CHAIN])
        elements_np = to_numpy(self.elements)

        rows = []
        atom_offset = 0
        for ix in range(self.size(Scale.CHAIN)):
            mol = molecule_type(int(types_np[ix]))
            res = int(lengths_np[ix])
            atoms = int(atoms_np[ix])

            # For ION chains, show element name (e.g., "MG ION")
            type_str = mol.name
            if mol == Molecule.ION and atoms > 0:
                elem_idx = int(elements_np[atom_offset])
                elem_name = ELEMENT_NAMES.get(elem_idx, "")
                if elem_name:
                    type_str = f"{elem_name} {mol.name}"

            rows.append({
                'chain': self.names[ix],
                'type': type_str,
                'res': res,
                'atoms': atoms,
            })
            atom_offset += atoms

        return rows

    def __repr__(self: Polymer) -> str:
        """String representation with structure summary."""
        rows = self.chain_info()
        return format_chain_table(self.pdb_id, self.backend, rows)

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
        from .backend import get_backend
        return get_backend(self.coordinates).value

    @property
    def device(self: Polymer) -> str | None:
        """
        Get the device of the polymer's arrays.

        Returns:
            Device string (e.g., 'cpu', 'cuda:0', 'mps:0') for PyTorch tensors,
            None for NumPy arrays.
        """
        from .backend import get_device
        return get_device(self.coordinates)

    def numpy(self: Polymer) -> Polymer:
        """
        Convert all arrays to NumPy.

        Returns:
            New Polymer with NumPy arrays. If already NumPy, returns self.
        """
        from .backend import is_numpy
        if is_numpy(self.coordinates):
            return self

        # Create new polymer with converted arrays
        result = Polymer(
            coordinates=to_numpy(self.coordinates),
            atoms=to_numpy(self.atoms),
            elements=to_numpy(self.elements),
            sequence=to_numpy(self.sequence),
            sizes={k: to_numpy(v) for k, v in self._sizes.items()},
            id=self.pdb_id,
            names=self.names.copy(),
            strands=self.strands.copy(),
            lengths=to_numpy(self.lengths),
            polymer_count=self.polymer_count,
            molecule_types=to_numpy(self._molecule_types) if self._molecule_types is not None else None,
        )

        # Replace coordinate manager with converted one
        result._coord_manager = self._coord_manager.numpy()
        result._coord_manager._topology = result._topology

        return result

    def torch(self: Polymer) -> Polymer:
        """
        Convert all arrays to PyTorch tensors.

        Returns:
            New Polymer with PyTorch tensors. If already PyTorch, returns self.

        Raises:
            ImportError: If PyTorch is not installed.
        """
        from .backend import to_torch, is_torch
        if is_torch(self.coordinates):
            return self

        # Create new polymer with converted arrays
        result = Polymer(
            coordinates=to_torch(self.coordinates).float(),
            atoms=to_torch(self.atoms).long(),
            elements=to_torch(self.elements).long(),
            sequence=to_torch(self.sequence).long(),
            sizes={k: to_torch(v).long() for k, v in self._sizes.items()},
            id=self.pdb_id,
            names=self.names.copy(),
            strands=self.strands.copy(),
            lengths=to_torch(self.lengths).long(),
            polymer_count=self.polymer_count,
            molecule_types=to_torch(self._molecule_types).long() if self._molecule_types is not None else None,
        )

        # Replace coordinate manager with converted one
        result._coord_manager = self._coord_manager.torch()
        result._coord_manager._topology = result._topology

        return result

    def to(self: Polymer, device=None, dtype=None) -> Polymer:
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
        from .backend import is_torch
        if not is_torch(self.coordinates):
            raise ValueError("to() is only supported for torch backend. "
                           "Use polymer.torch().to(...) to convert first.")

        if device is None and dtype is None:
            return self

        # For coordinates (float), apply both device and dtype
        coords = self.coordinates
        if device is not None:
            coords = coords.to(device)
        if dtype is not None:
            coords = coords.to(dtype)

        # For integer tensors, only apply device (keep as long)
        def move_int(t):
            return t.to(device) if device is not None else t

        # Create new polymer with moved arrays
        result = Polymer(
            coordinates=coords,
            atoms=move_int(self.atoms),
            elements=move_int(self.elements),
            sequence=move_int(self.sequence),
            sizes={k: move_int(v) for k, v in self._sizes.items()},
            id=self.pdb_id,
            names=self.names.copy(),
            strands=self.strands.copy(),
            lengths=move_int(self.lengths),
            polymer_count=self.polymer_count,
        )

        result._coord_manager = self._coord_manager.to(device, dtype)

        return result

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
        Detach all tensors from their computation graphs (torch backend only).

        This is useful after calling `backward()` on a computation that used
        this polymer's coordinates or internal coordinates. After backward(),
        the cached tensors retain grad_fn pointers to freed computation graphs.
        Calling detach() clears these pointers, allowing the polymer to be
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
            >>> # Now safe to compute new gradients through to_cartesian
            >>> dihedrals = polymer.dihedrals.detach().clone().requires_grad_(True)
            >>> polymer.dihedrals = dihedrals
            >>> new_loss = polymer.coordinates.sum()
            >>> new_loss.backward()

        Note:
            For NumPy arrays, this is a no-op since NumPy doesn't have
            computation graphs.
        """
        self._coord_manager.detach()
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
        if self.empty():
            raise ValueError("Cannot write empty polymer to CIF file")
        if not filename.lower().endswith('.cif'):
            raise ValueError(
                f"Output file must have .cif extension, got: {filename!r}"
            )
        from .io.writer import write_cif
        write_cif(self, filename)

    # ─────────────────────────────────────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────────────────────────────────────

    def with_coordinates(self: Polymer, coordinates: Array) -> Polymer:
        """
        Create a copy with new coordinates.

        Args:
            coordinates: New coordinate tensor. Must match the polymer's
                backend and device.

        Returns:
            New Polymer with updated coordinates.

        Raises:
            TypeError: If backend doesn't match.
            ValueError: If device doesn't match (for PyTorch tensors).
        """
        # Validate backend and device compatibility
        check_compatible(self.coordinates, coordinates, "coordinates")

        result = copy(self)
        # Create a new coordinate manager for the copy (to avoid sharing state, reuse topology)
        from .internal.coordinates import CoordinateManager
        result._coord_manager = CoordinateManager(coordinates, result._topology)
        return result
