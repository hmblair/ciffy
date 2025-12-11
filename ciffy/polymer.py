"""
Polymer class representing molecular structures.

The Polymer class provides a unified interface for working with molecular
structures loaded from CIF files. It supports RNA, DNA, proteins, and
other molecular types.
"""

from __future__ import annotations
from typing import Generator, Union, TYPE_CHECKING
from copy import copy

import numpy as np

from .backend import Array, is_torch, get_backend, size as arr_size
from .backend import ops as backend
from .types import Scale, Molecule

if TYPE_CHECKING:
    import torch
from .types.molecule import molecule_type
from .operations.reduction import Reduction, REDUCTIONS, ReductionResult, create_reduction_index
from .biochemistry import (
    Residue,
    RES_ABBREV,
    RESIDUE_MOLECULE_TYPE,
    ATOM_NAMES,
    Element,
    FRAMES,
    Backbone,
)
from .utils import all_equal, filter_by_mask


UNKNOWN = "UNKNOWN"


def _ones_like_backend(template: Array, size: int) -> Array:
    """Create a ones array matching the backend of template."""
    if is_torch(template):
        import torch
        return torch.ones(size, dtype=torch.long)
    return np.ones(size, dtype=np.int64)


def _zeros_like_backend(template: Array, size: int) -> Array:
    """Create a zeros array matching the backend of template."""
    if is_torch(template):
        import torch
        return torch.zeros(size, dtype=torch.long)
    return np.zeros(size, dtype=np.int64)


def _array_like_backend(template: Array, data: list) -> Array:
    """Create an array from data matching the backend of template."""
    if is_torch(template):
        import torch
        return torch.tensor(data, dtype=torch.long)
    return np.array(data, dtype=np.int64)


def _bool_zeros_like_backend(template: Array, size: int) -> Array:
    """Create a boolean zeros array matching the backend of template."""
    if is_torch(template):
        import torch
        return torch.zeros(size, dtype=torch.bool)
    return np.zeros(size, dtype=bool)


def _as_backend(template: Array, arr: Array) -> Array:
    """Convert arr to match the backend of template."""
    if is_torch(template):
        if not is_torch(arr):
            import torch
            return torch.from_numpy(np.asarray(arr))
        return arr
    else:
        if is_torch(arr):
            return arr.numpy()
        return np.asarray(arr)


def _cdist(x1: Array, x2: Array) -> Array:
    """Compute pairwise distances, backend-agnostic."""
    return backend.cdist(x1, x2)


def _eigh(arr: Array) -> tuple[Array, Array]:
    """
    Compute eigendecomposition, backend-agnostic.

    Returns:
        Tuple of (eigenvalues, eigenvectors).
    """
    if is_torch(arr):
        import torch
        return torch.linalg.eigh(arr)
    return np.linalg.eigh(arr)


def _det(arr: Array) -> Array:
    """Compute determinant, backend-agnostic."""
    if is_torch(arr):
        import torch
        return torch.linalg.det(arr)
    return np.linalg.det(arr)


def _nonzero_1d(arr: Array) -> Array:
    """Get indices of non-zero elements in a 1D array, backend-agnostic."""
    if is_torch(arr):
        return arr.nonzero().squeeze(-1)
    # NumPy nonzero returns a tuple of arrays, one per dimension
    return arr.nonzero()[0]


def _to_int64(arr: Array) -> Array:
    """Convert array to int64 dtype, backend-agnostic."""
    if is_torch(arr):
        return arr.long()
    return arr.astype(np.int64)


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

        Raises:
            ValueError: If tensor sizes are inconsistent.
        """
        self._id = id or UNKNOWN
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
                f"for PDB {self.id()}."
            )

        res_count = sizes[Scale.RESIDUE].sum().item()
        chn_count = sizes[Scale.CHAIN].sum().item()
        mol_count = sizes[Scale.MOLECULE].sum().item()

        if not all_equal(res_count + self.nonpoly, chn_count, mol_count):
            raise ValueError(
                f"Atom counts do not match: residues ({res_count} + {self.nonpoly}), "
                f"chains ({chn_count}), molecule ({mol_count}) for PDB {self.id()}."
            )

        self.coordinates = coordinates
        self.atoms = atoms
        self.elements = elements
        self.sequence = sequence
        self._sizes = sizes
        self.lengths = lengths

    # ─────────────────────────────────────────────────────────────────────────
    # Identification
    # ─────────────────────────────────────────────────────────────────────────

    def id(self: Polymer, ix: int | None = None) -> str:
        """
        Get the PDB ID, optionally with chain suffix.

        Args:
            ix: Optional chain index for chain-specific ID.

        Returns:
            PDB ID string, with chain name suffix if ix is provided.
        """
        if ix is None:
            return self._id
        return f"{self._id}_{self.names[ix]}"

    def strand(self: Polymer, ix: int) -> str:
        """
        Get the strand ID for a specific chain.

        Args:
            ix: Chain index.

        Returns:
            Strand identifier string.
        """
        return f"{self._id}_{self.strands[ix]}"

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
            return _ones_like_backend(self.coordinates, self.size(inner))

        # Atoms per {residue, chain, molecule} are stored in _sizes
        if inner == Scale.ATOM:
            return self._sizes[outer]

        # Residues per chain are stored in lengths
        if inner == Scale.RESIDUE and outer == Scale.CHAIN:
            return self.lengths

        # Single-value cases: total count as 1-element array
        if outer == Scale.MOLECULE:
            return _array_like_backend(self.coordinates, [self.size(inner)])

        raise ValueError(f"Cannot compute {inner.name} per {outer.name}")

    @property
    def molecule_type(self: Polymer) -> Array:
        """
        Get the molecule type of each chain.

        Determines molecule type by examining residue indices in each chain:
        - RNA: indices 0-3 (A, C, G, U)
        - DNA: index 4 (T/DT)
        - Protein: indices 5-24 (amino acids)
        - Water: index 25 (HOH)
        - Ion: indices 26-27 (MG, CS)
        - Other: modified nucleotides (28+)

        Uses both MIN and MAX residue index per chain to robustly detect type.
        If min and max map to different molecule types, the chain is classified
        as OTHER (mixed/heterogeneous composition).

        Returns:
            Array of Molecule enum values, one per chain.
        """
        n_chains = self.size(Scale.CHAIN)
        types = _zeros_like_backend(self.coordinates, n_chains)

        # Get min and max residue index per chain
        min_res, _ = self.rreduce(self.sequence, Scale.CHAIN, Reduction.MIN)
        max_res, _ = self.rreduce(self.sequence, Scale.CHAIN, Reduction.MAX)

        # Classify based on residue index ranges
        for i in range(n_chains):
            min_idx = int(min_res[i].item() if hasattr(min_res[i], 'item') else min_res[i])
            max_idx = int(max_res[i].item() if hasattr(max_res[i], 'item') else max_res[i])

            min_type = RESIDUE_MOLECULE_TYPE.get(min_idx, Molecule.UNKNOWN)
            max_type = RESIDUE_MOLECULE_TYPE.get(max_idx, Molecule.UNKNOWN)

            # If min and max agree, use that type; otherwise mark as OTHER (mixed)
            if min_type == max_type:
                types[i] = min_type.value
            else:
                types[i] = Molecule.OTHER.value

        return types

    def type(self: Polymer) -> Array:
        """
        Get the molecule type of each chain.

        Deprecated: Use molecule_type property instead.

        Returns:
            Tensor of Molecule enum values.
        """
        import warnings
        warnings.warn(
            "Polymer.type() is deprecated and will be removed in v0.8.0. "
            "Use the molecule_type property instead: polymer.molecule_type",
            DeprecationWarning,
            stacklevel=2
        )
        return self.molecule_type

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
        ix = create_reduction_index(count, sizes)

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
        ix = create_reduction_index(count, self.lengths)

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
        if dest == Scale.ATOM:
            return backend.repeat_interleave(features, self._sizes[source])
        if dest == Scale.RESIDUE:
            return backend.repeat_interleave(features, self.lengths)
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
        return self.reduce(_to_int64(mask), scale, Reduction.SUM)

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
        centered.coordinates = coordinates

        return centered, means

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
        if scale is not None:
            coords = self.reduce(self.coordinates, scale)
        else:
            coords = self.coordinates

        return _cdist(coords, coords)

    def pd(self: Polymer, scale: Scale | None = None) -> Array:
        """
        Compute pairwise distances.

        Deprecated: Use pairwise_distances() instead.
        """
        import warnings
        warnings.warn(
            "Polymer.pd() is deprecated. Use pairwise_distances() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.pairwise_distances(scale)

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
        return _eigh(cov)

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
        signs = aligned.moment(3, scale).sign()
        signs[:, 0] = signs[:, 1] * signs[:, 2] * _det(Q)
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
        objects = _bool_zeros_like_backend(self.coordinates, counts)
        objects[indices] = True
        return self.expand(objects, source, dest)

    def __getitem__(self: Polymer, mask: Array) -> Polymer:
        """
        Select atoms by boolean mask.

        Args:
            mask: Boolean mask of atoms to keep.

        Returns:
            New Polymer with selected atoms.
        """
        # NOTE: Potential optimization target - mask generation and slicing overhead
        coordinates = self.coordinates[mask]
        atoms = self.atoms[mask]
        elements = self.elements[mask]

        chn_sizes = self.count(mask, Scale.CHAIN)
        res_sizes = self.count(mask, Scale.RESIDUE)
        mol_sizes = self.count(mask, Scale.MOLECULE)

        # Determine which residues have atoms
        chn_mask = chn_sizes > 0
        residues = backend.repeat_interleave(chn_mask, self.lengths)

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
        # polymer_count atoms survive the mask
        polymer_mask = _bool_zeros_like_backend(self.coordinates, self.size())
        polymer_mask[:self.polymer_count] = True
        new_polymer_count = (mask & polymer_mask).sum().item()

        return Polymer(
            coordinates, atoms, elements, sequence, sizes,
            self._id, names, strands, lengths, new_polymer_count,
        )

    def select(self: Polymer, ix: Array | int) -> Polymer:
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
            ix = _array_like_backend(self.coordinates, [ix])

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
            Scale.MOLECULE: _array_like_backend(self.coordinates, [len(coordinates)]),
        }

        sequence = self.sequence[res_ix]
        names = [self.names[j] for j in ix]
        strands = [self.strands[j] for j in ix]

        # Calculate new polymer_count from residue sizes
        # (residue atoms are always polymer atoms)
        new_polymer_count = sizes[Scale.RESIDUE].sum().item()

        return Polymer(
            coordinates, atoms, elements, sequence, sizes,
            self._id, names, strands, lengths, new_polymer_count,
        )

    def get_by_name(self: Polymer, name: Array | int) -> Polymer:
        """
        Select atoms by atom type name.

        Args:
            name: Atom type index or indices.

        Returns:
            New Polymer with matching atoms.
        """
        name = _as_backend(self.atoms, name)
        mask = (self.atoms[:, None] == name).any(1)
        return self[mask]

    def subset(self: Polymer, mol: Molecule) -> Polymer:
        """
        Select chains by molecule type.

        Args:
            mol: Molecule type to select.

        Returns:
            New Polymer with chains of that type.
        """
        ix = _nonzero_1d(self.molecule_type == mol.value)
        return self.select(ix)

    def polymer_only(self: Polymer) -> Polymer:
        """
        Return a new Polymer with non-polymer atoms removed.

        Deprecated: Use poly() instead, which is simpler and doesn't filter
        unknown atom types (useful for modified residues).

        Returns:
            New Polymer containing only recognized polymer atoms.
        """
        import warnings
        warnings.warn(
            "Polymer.polymer_only() is deprecated. Use poly() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        # With reordered atoms, polymer atoms are [0, polymer_count)
        # Also filter out unknown atom types (-1)
        if self.nonpoly == 0:
            # No non-polymer atoms, but may still have unknown types
            if (self.atoms < 0).any():
                mask = self.atoms >= 0
                return self[mask]
            return self

        # Create mask for polymer atoms with known types
        mask = _bool_zeros_like_backend(self.coordinates, self.size())
        mask[:self.polymer_count] = True
        mask = mask & (self.atoms >= 0)
        return self[mask]

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

        # Create mask for polymer atoms (first polymer_count atoms)
        mask = _bool_zeros_like_backend(self.coordinates, self.size())
        mask[:self.polymer_count] = True
        return self[mask]

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
        # Create mask for non-polymer atoms (last nonpoly atoms)
        mask = _bool_zeros_like_backend(self.coordinates, self.size())
        mask[self.polymer_count:] = True
        return self[mask]

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
            chain = self.select(ix)
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

        poly.lengths = self.rreduce(_to_int64(resolved), Scale.CHAIN, Reduction.SUM)
        poly.sequence = self.sequence[resolved]

        return poly

    # ─────────────────────────────────────────────────────────────────────────
    # Specialized Selections
    # ─────────────────────────────────────────────────────────────────────────

    def frame(self: Polymer) -> Polymer:
        """
        Select frame atoms for structural alignment.

        Deprecated: Use get_by_name(FRAMES) instead.
        """
        import warnings
        warnings.warn(
            "Polymer.frame() is deprecated. Use get_by_name(FRAMES) instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.get_by_name(FRAMES)

    def backbone(self: Polymer) -> Polymer:
        """Select backbone atoms."""
        return self.get_by_name(Backbone.index())

    # ─────────────────────────────────────────────────────────────────────────
    # String Representations
    # ─────────────────────────────────────────────────────────────────────────

    def str(self: Polymer) -> str:
        """
        Get the sequence as a string.

        Returns:
            Single-letter sequence string.
        """
        def abbrev(x):
            return RES_ABBREV.get(Residue.revdict().get(x, 'N'), 'n')
        return "".join(abbrev(ix.item()) for ix in self.sequence)

    def atom_names(self: Polymer) -> list[str]:
        """
        Get atom names as a list of strings.

        Returns:
            List of atom name strings.
        """
        return [ATOM_NAMES.get(ix.item(), '?') for ix in self.atoms]

    def __repr__(self: Polymer) -> str:
        """String representation with structure summary."""
        out = f"PDB {self.id()} with {self.size()} atoms ({self.backend}).\n"
        out += "-" * 39 + "\n"

        header_pad = len(str(self.size(Scale.CHAIN)))
        out += " " * header_pad + "  Type     # Res  # Atom\n"

        types = self.molecule_type

        for ix in range(self.size(Scale.CHAIN)):
            mol = molecule_type(types[ix].item())
            chain = self.names[ix]
            mol_name = mol.name
            residues = str(self.lengths[ix].item())
            atoms = str(self._sizes[Scale.CHAIN][ix].item())

            out += f"{chain:2s}  {mol_name:9s}{residues:7s}{atoms}\n"

        return out

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

    def numpy(self: Polymer) -> Polymer:
        """
        Convert all arrays to NumPy.

        Returns:
            New Polymer with NumPy arrays. If already NumPy, returns self.
        """
        from .backend import to_numpy, is_numpy
        if is_numpy(self.coordinates):
            return self
        return Polymer(
            coordinates=to_numpy(self.coordinates),
            atoms=to_numpy(self.atoms),
            elements=to_numpy(self.elements),
            sequence=to_numpy(self.sequence),
            sizes={k: to_numpy(v) for k, v in self._sizes.items()},
            id=self._id,
            names=self.names.copy(),
            strands=self.strands.copy(),
            lengths=to_numpy(self.lengths),
            polymer_count=self.polymer_count,
        )

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
        return Polymer(
            coordinates=to_torch(self.coordinates).float(),
            atoms=to_torch(self.atoms).long(),
            elements=to_torch(self.elements).long(),
            sequence=to_torch(self.sequence).long(),
            sizes={k: to_torch(v).long() for k, v in self._sizes.items()},
            id=self._id,
            names=self.names.copy(),
            strands=self.strands.copy(),
            lengths=to_torch(self.lengths).long(),
            polymer_count=self.polymer_count,
        )

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
            ValueError: If filename does not end with .cif extension.

        Example:
            >>> polymer = ciffy.load("structure.cif", backend="numpy")
            >>> polymer.write("output.cif")
        """
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
            coordinates: New coordinate tensor.

        Returns:
            New Polymer with updated coordinates.
        """
        #TODO: rename to something more informative
        result = copy(self)
        result.coordinates = coordinates
        return result
