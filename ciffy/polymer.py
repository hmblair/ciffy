"""
Polymer class representing molecular structures.

The Polymer class provides a unified interface for working with molecular
structures loaded from CIF files. It supports RNA, DNA, proteins, and
other molecular types.
"""

from __future__ import annotations
from typing import Generator
from copy import copy
import torch

from .types import Scale, Molecule
from .types.molecule import molecule_type
from .operations.reduction import Reduction, REDUCTIONS, ReductionResult, create_reduction_index
from .biochemistry import (
    Residue,
    RES_ABBREV,
    RibonucleicAcid,
    Adenosine,
    Guanosine,
    Cytosine,
    Uridine,
    Element,
    FRAMES,
    Backbone,
)
from .utils import all_equal, filter_by_mask


UNKNOWN = "UNKNOWN"


class Polymer:
    """
    A molecular structure with coordinates, atom types, and hierarchy.

    Represents a complete molecular assembly with multiple scales of
    organization: atoms, residues, chains, and molecules. Provides
    methods for geometric operations, selection, and analysis.

    Attributes:
        coordinates: (N, 3) tensor of atom positions.
        atoms: (N,) tensor of atom type indices.
        elements: (N,) tensor of element indices.
        sequence: (R,) tensor of residue type indices.
        is_nonpoly: (N,) boolean tensor marking non-polymer atoms.
        names: List of chain names.
        strands: List of strand identifiers.
        lengths: (C,) tensor of residues per chain.
        nonpoly: Count of non-polymer atoms.
    """

    def __init__(
        self: Polymer,
        coordinates: torch.Tensor,
        atoms: torch.Tensor,
        elements: torch.Tensor,
        sequence: torch.Tensor,
        sizes: dict[Scale, torch.Tensor],
        id: str,
        names: list[str],
        strands: list[str],
        lengths: torch.Tensor,
        nonpoly: int = 0,
        is_nonpoly: torch.Tensor | None = None,
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
            nonpoly: Count of non-polymer atoms.
            is_nonpoly: Boolean mask marking non-polymer atoms.

        Raises:
            ValueError: If tensor sizes are inconsistent.
        """
        self._id = id or UNKNOWN
        self.names = names
        self.strands = strands
        self.nonpoly = nonpoly

        if not all_equal(
            coordinates.size(0),
            atoms.size(0),
            elements.size(0),
        ):
            raise ValueError(
                f"Coordinate, atom, and element tensors must have equal size "
                f"for PDB {self.id()}."
            )

        res_count = sizes[Scale.RESIDUE].sum().item()
        chn_count = sizes[Scale.CHAIN].sum().item()
        mol_count = sizes[Scale.MOLECULE].sum().item()

        if not all_equal(res_count + nonpoly, chn_count, mol_count):
            raise ValueError(
                f"Atom counts do not match: residues ({res_count} + {nonpoly}), "
                f"chains ({chn_count}), molecule ({mol_count}) for PDB {self.id()}."
            )

        self.coordinates = coordinates
        self.atoms = atoms
        self.elements = elements
        self.sequence = sequence
        self._sizes = sizes
        self.lengths = lengths

        # Initialize is_nonpoly mask (default to all False if not provided)
        if is_nonpoly is not None:
            self.is_nonpoly = is_nonpoly
        else:
            self.is_nonpoly = torch.zeros(coordinates.size(0), dtype=torch.bool)

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
        return self.coordinates.size(0) == 0

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
            return self.coordinates.size(0)
        return self._sizes[scale].size(0)

    def sizes(self: Polymer, scale: Scale) -> torch.Tensor:
        """
        Get the sizes tensor for a scale.

        Args:
            scale: Scale level.

        Returns:
            Tensor of atom counts per unit at this scale.
        """
        return self._sizes[scale]

    def per(self: Polymer, inner: Scale, outer: Scale) -> torch.Tensor:
        """
        Get the count of inner units per outer unit.

        Args:
            inner: Inner scale (e.g., RESIDUE).
            outer: Outer scale (e.g., CHAIN).

        Returns:
            Tensor with count of inner units per outer unit.

        Example:
            >>> polymer.per(Scale.RESIDUE, Scale.CHAIN)
            tensor([150, 200, 175])  # residues per chain
        """
        if inner == outer:
            return torch.ones(self.size(inner), dtype=torch.long)

        if inner == Scale.ATOM:
            if outer == Scale.RESIDUE:
                return self._sizes[Scale.RESIDUE]
            if outer == Scale.CHAIN:
                return self._sizes[Scale.CHAIN]
            if outer == Scale.MOLECULE:
                return self._sizes[Scale.MOLECULE]

        if inner == Scale.RESIDUE:
            if outer == Scale.CHAIN:
                return self.lengths
            if outer == Scale.MOLECULE:
                return torch.tensor([self.size(Scale.RESIDUE)])

        if inner == Scale.CHAIN:
            if outer == Scale.MOLECULE:
                return torch.tensor([self.size(Scale.CHAIN)])

        raise ValueError(f"Cannot compute {inner.name} per {outer.name}")

    @property
    def molecule_type(self: Polymer) -> torch.Tensor:
        """
        Get the molecule type of each chain.

        Returns:
            Tensor of Molecule enum values, one per chain.
        """
        types = torch.zeros(self.size(Scale.CHAIN), dtype=torch.long)
        atoms, _ = self.rreduce(self.sequence, Scale.CHAIN, Reduction.MAX)
        types[atoms < 5] = Molecule.RNA.value
        return types

    def type(self: Polymer) -> torch.Tensor:
        """
        Get the molecule type of each chain.

        Deprecated: Use molecule_type property instead.

        Returns:
            Tensor of Molecule enum values.
        """
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
        if types.size(0) != 1:
            return False
        return types[0].item() == mol.value

    # ─────────────────────────────────────────────────────────────────────────
    # Reduction Operations
    # ─────────────────────────────────────────────────────────────────────────

    def reduce(
        self: Polymer,
        features: torch.Tensor,
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
        """
        count = self.size(scale)
        sizes = self._sizes[scale]
        ix = create_reduction_index(count, sizes)

        return REDUCTIONS[rtype](features, ix, dim=0, dim_size=count)

    def rreduce(
        self: Polymer,
        features: torch.Tensor,
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
        features: torch.Tensor,
        source: Scale,
        dest: Scale = Scale.ATOM,
    ) -> torch.Tensor:
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
            return features.repeat_interleave(self._sizes[source], dim=0)
        if dest == Scale.RESIDUE:
            return features.repeat_interleave(self.lengths, dim=0)
        raise ValueError(f"Cannot expand to {dest.name}")

    def count(
        self: Polymer,
        mask: torch.Tensor,
        scale: Scale,
    ) -> torch.Tensor:
        """
        Count True values in mask per scale unit.

        Args:
            mask: Boolean mask tensor.
            scale: Scale at which to count.

        Returns:
            Count tensor with one value per scale unit.
        """
        return self.reduce(mask.long(), scale, Reduction.SUM)

    # ─────────────────────────────────────────────────────────────────────────
    # Geometry Operations
    # ─────────────────────────────────────────────────────────────────────────

    def center(
        self: Polymer,
        scale: Scale = Scale.MOLECULE,
    ) -> tuple[Polymer, torch.Tensor]:
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

    def pd(self: Polymer, scale: Scale | None = None) -> torch.Tensor:
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

        return torch.cdist(coords, coords)

    def _pc(
        self: Polymer,
        scale: Scale,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        return torch.linalg.eigh(cov)

    def align(
        self: Polymer,
        scale: Scale,
    ) -> tuple[Polymer, torch.Tensor]:
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
        signs[:, 0] = signs[:, 1] * signs[:, 2] * torch.linalg.det(Q)
        signs_exp = aligned.expand(signs, scale)

        aligned.coordinates = aligned.coordinates * signs_exp
        Q = Q * signs[..., None]

        return aligned, Q

    def moment(
        self: Polymer,
        n: int,
        scale: Scale,
    ) -> torch.Tensor:
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
        indices: torch.Tensor | int,
        source: Scale,
        dest: Scale = Scale.ATOM,
    ) -> torch.Tensor:
        """
        Create a boolean mask selecting specific units.

        Args:
            indices: Indices of units to select.
            source: Scale of the indices.
            dest: Scale of the output mask.

        Returns:
            Boolean tensor at dest scale.
        """
        counts = self.size(source)
        objects = torch.zeros(counts, dtype=torch.bool)
        objects[indices] = True
        return self.expand(objects, source, dest)

    def __getitem__(self: Polymer, mask: torch.Tensor) -> Polymer:
        """
        Select atoms by boolean mask.

        Args:
            mask: Boolean mask of atoms to keep.

        Returns:
            New Polymer with selected atoms.
        """
        coordinates = self.coordinates[mask]
        atoms = self.atoms[mask]
        elements = self.elements[mask]

        chn_sizes = self.count(mask, Scale.CHAIN)
        res_sizes = self.count(mask, Scale.RESIDUE)
        mol_sizes = self.count(mask, Scale.MOLECULE)

        # Determine which residues have atoms
        chn_mask = chn_sizes > 0
        residues = chn_mask.repeat_interleave(self.lengths, dim=0)

        lengths = self.lengths[chn_mask]

        sizes = {
            Scale.RESIDUE: res_sizes[residues],
            Scale.CHAIN: chn_sizes[chn_mask],
            Scale.MOLECULE: mol_sizes,
        }

        sequence = self.sequence[residues]
        names = filter_by_mask(self.names, chn_mask)
        strands = filter_by_mask(self.strands, chn_mask)

        # Calculate nonpoly atoms (atoms not belonging to residues)
        res_atoms = sizes[Scale.RESIDUE].sum().item()
        chn_atoms = sizes[Scale.CHAIN].sum().item()
        nonpoly = chn_atoms - res_atoms

        # Filter is_nonpoly mask
        is_nonpoly = self.is_nonpoly[mask]

        return Polymer(
            coordinates, atoms, elements, sequence, sizes,
            self._id, names, strands, lengths, nonpoly, is_nonpoly,
        )

    def select(self: Polymer, ix: torch.Tensor | int) -> Polymer:
        """
        Select chains by index.

        Args:
            ix: Chain index or indices to select.

        Returns:
            New Polymer with selected chains.
        """
        if isinstance(ix, int):
            ix = torch.tensor([ix])

        atm_ix = self.mask(ix, Scale.CHAIN, Scale.ATOM)
        res_ix = self.mask(ix, Scale.CHAIN, Scale.RESIDUE)

        coordinates = self.coordinates[atm_ix]
        atoms = self.atoms[atm_ix]
        elements = self.elements[atm_ix]
        lengths = self.lengths[ix]

        sizes = {
            Scale.RESIDUE: self._sizes[Scale.RESIDUE][res_ix],
            Scale.CHAIN: self._sizes[Scale.CHAIN][ix],
            Scale.MOLECULE: torch.tensor([len(coordinates)]),
        }

        sequence = self.sequence[res_ix]
        names = [self.names[j] for j in ix]
        strands = [self.strands[j] for j in ix]

        # Calculate nonpoly atoms (atoms not belonging to residues)
        res_atoms = sizes[Scale.RESIDUE].sum().item()
        chn_atoms = sizes[Scale.CHAIN].sum().item()
        nonpoly = chn_atoms - res_atoms

        # Filter is_nonpoly mask
        is_nonpoly = self.is_nonpoly[atm_ix]

        return Polymer(
            coordinates, atoms, elements, sequence, sizes,
            self._id, names, strands, lengths, nonpoly, is_nonpoly,
        )

    def get_by_name(self: Polymer, name: torch.Tensor | int) -> Polymer:
        """
        Select atoms by atom type name.

        Args:
            name: Atom type index or indices.

        Returns:
            New Polymer with matching atoms.
        """
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
        ix = (self.molecule_type == mol.value).nonzero().squeeze(-1)
        return self.select(ix)

    def polymer_only(self: Polymer) -> Polymer:
        """
        Return a new Polymer with non-polymer atoms removed.

        Non-polymer atoms include water, ions, ligands, and any atoms
        with unknown types (e.g., modified residues not in standard tables).

        Returns:
            New Polymer containing only recognized polymer atoms.
        """
        # Filter out both non-polymer atoms and atoms with unknown types (-1)
        mask = ~self.is_nonpoly & (self.atoms >= 0)
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

    def resolved(self: Polymer, scale: Scale = Scale.RESIDUE) -> torch.Tensor:
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

        poly.lengths = self.rreduce(resolved.long(), Scale.CHAIN, Reduction.SUM)
        poly.sequence = self.sequence[resolved]

        return poly

    # ─────────────────────────────────────────────────────────────────────────
    # Specialized Selections
    # ─────────────────────────────────────────────────────────────────────────

    def frame(self: Polymer) -> Polymer:
        """Select frame atoms for structural alignment."""
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
        revdict = (
            Adenosine.revdict() |
            Guanosine.revdict() |
            Cytosine.revdict() |
            Uridine.revdict()
        )
        return [revdict.get(ix.item(), '?') for ix in self.atoms]

    def __repr__(self: Polymer) -> str:
        """String representation with structure summary."""
        out = f"PDB {self.id()} with {self.size()} atoms.\n"
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
    # I/O
    # ─────────────────────────────────────────────────────────────────────────

    def write(self: Polymer, filename: str) -> None:
        """
        Write structure to a PDB file.

        Args:
            filename: Output file path.
        """
        from .io.writer import write_pdb
        write_pdb(self, filename)

    # ─────────────────────────────────────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────────────────────────────────────

    def with_coordinates(self: Polymer, coordinates: torch.Tensor) -> Polymer:
        """
        Create a copy with new coordinates.

        Args:
            coordinates: New coordinate tensor.

        Returns:
            New Polymer with updated coordinates.
        """
        result = copy(self)
        result.coordinates = coordinates
        return result
