"""
Topology information for coordinate operations.

This module provides immutable data structures that capture the structural
topology of molecular systems, enabling coordinate operations without
requiring circular references to Polymer objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..backend import Array, is_torch, to_numpy

if TYPE_CHECKING:
    from ..polymer import Polymer


@dataclass(frozen=True)
class TopologyInfo:
    """
    Immutable topology information for coordinate operations.

    Captures all structural information needed for Z-matrix building and
    coordinate reconstruction without requiring a Polymer reference.

    Attributes:
        atoms: (N,) int32 array of atom type indices.
        sequence: (R,) int32 array of residue type indices.
        residue_sizes: (R,) int32 array of atom counts per residue.
        chain_lengths: (C,) int32 array of residue counts per chain.
        chain_atom_offsets: (C+1,) int64 array of cumulative atom counts per chain.
        chain_residue_offsets: (C+1,) int64 array of cumulative residue counts per chain.
        n_atoms: Total number of atoms.
        n_residues: Total number of residues.
        n_chains: Total number of chains.

    Example:
        >>> topology = TopologyInfo.from_polymer(polymer)
        >>> coord_manager = CoordinateManager(coordinates, topology)
    """

    atoms: np.ndarray
    sequence: np.ndarray
    residue_sizes: np.ndarray
    chain_lengths: np.ndarray
    chain_atom_offsets: np.ndarray
    chain_residue_offsets: np.ndarray
    n_atoms: int
    n_residues: int
    n_chains: int

    @classmethod
    def from_polymer(cls, polymer: "Polymer") -> "TopologyInfo":
        """
        Create TopologyInfo from a Polymer instance.

        Args:
            polymer: Polymer structure to extract topology from.

        Returns:
            TopologyInfo with all structural information.
        """
        from ..types import Scale

        # Convert to numpy for storage (topology is always CPU)
        atoms = to_numpy(polymer.atoms).astype(np.int32)
        sequence = to_numpy(polymer.sequence).astype(np.int32)
        residue_sizes = to_numpy(polymer.sizes(Scale.RESIDUE)).astype(np.int32)
        chain_lengths = to_numpy(polymer.lengths).astype(np.int32)

        n_atoms = len(atoms)
        n_residues = len(sequence)
        n_chains = len(chain_lengths)

        # Compute cumulative offsets
        chain_residue_offsets = np.zeros(n_chains + 1, dtype=np.int64)
        chain_residue_offsets[1:] = np.cumsum(chain_lengths)

        chain_atom_offsets = np.zeros(n_chains + 1, dtype=np.int64)
        res_offset = 0
        for chain_idx in range(n_chains):
            chain_len = int(chain_lengths[chain_idx])
            chain_atom_count = int(residue_sizes[res_offset:res_offset + chain_len].sum())
            chain_atom_offsets[chain_idx + 1] = chain_atom_offsets[chain_idx] + chain_atom_count
            res_offset += chain_len

        return cls(
            atoms=atoms,
            sequence=sequence,
            residue_sizes=residue_sizes,
            chain_lengths=chain_lengths,
            chain_atom_offsets=chain_atom_offsets,
            chain_residue_offsets=chain_residue_offsets,
            n_atoms=n_atoms,
            n_residues=n_residues,
            n_chains=n_chains,
        )

    def get_chain_atom_range(self, chain_idx: int) -> tuple[int, int]:
        """
        Get atom index range for a chain.

        Args:
            chain_idx: Chain index.

        Returns:
            Tuple of (start, end) atom indices.
        """
        return int(self.chain_atom_offsets[chain_idx]), int(self.chain_atom_offsets[chain_idx + 1])

    def get_chain_residue_range(self, chain_idx: int) -> tuple[int, int]:
        """
        Get residue index range for a chain.

        Args:
            chain_idx: Chain index.

        Returns:
            Tuple of (start, end) residue indices.
        """
        return int(self.chain_residue_offsets[chain_idx]), int(self.chain_residue_offsets[chain_idx + 1])

    def get_residue_atom_range(self, residue_idx: int) -> tuple[int, int]:
        """
        Get atom index range for a residue.

        Args:
            residue_idx: Residue index.

        Returns:
            Tuple of (start, end) atom indices.
        """
        # Compute residue atom offsets
        residue_atom_offsets = np.zeros(self.n_residues + 1, dtype=np.int64)
        residue_atom_offsets[1:] = np.cumsum(self.residue_sizes)
        return int(residue_atom_offsets[residue_idx]), int(residue_atom_offsets[residue_idx + 1])

    def slice_atoms(self, mask: np.ndarray, new_residue_sizes: np.ndarray, new_chain_lengths: np.ndarray) -> "TopologyInfo":
        """
        Create sliced TopologyInfo for a subset of atoms.

        Args:
            mask: (N,) boolean mask of atoms to keep.
            new_residue_sizes: (R',) residue sizes after slicing.
            new_chain_lengths: (C',) chain lengths after slicing.

        Returns:
            New TopologyInfo for the sliced structure.
        """
        # Slice atoms
        mask_np = to_numpy(mask)
        new_atoms = self.atoms[mask_np].astype(np.int32)

        # Build new sequence from residue sizes
        # We need to figure out which residues survived
        residue_atom_offsets = np.zeros(self.n_residues + 1, dtype=np.int64)
        residue_atom_offsets[1:] = np.cumsum(self.residue_sizes)

        # Find which residues have atoms remaining
        new_sequence_list = []
        for res_idx in range(self.n_residues):
            start = int(residue_atom_offsets[res_idx])
            end = int(residue_atom_offsets[res_idx + 1])
            if mask_np[start:end].any():
                new_sequence_list.append(self.sequence[res_idx])

        new_sequence = np.array(new_sequence_list, dtype=np.int32) if new_sequence_list else np.array([], dtype=np.int32)
        new_residue_sizes = to_numpy(new_residue_sizes).astype(np.int32)
        new_chain_lengths = to_numpy(new_chain_lengths).astype(np.int32)

        n_atoms = len(new_atoms)
        n_residues = len(new_sequence)
        n_chains = len(new_chain_lengths)

        # Compute new offsets
        chain_residue_offsets = np.zeros(n_chains + 1, dtype=np.int64)
        chain_residue_offsets[1:] = np.cumsum(new_chain_lengths)

        chain_atom_offsets = np.zeros(n_chains + 1, dtype=np.int64)
        res_offset = 0
        for chain_idx in range(n_chains):
            chain_len = int(new_chain_lengths[chain_idx])
            if chain_len > 0:
                chain_atom_count = int(new_residue_sizes[res_offset:res_offset + chain_len].sum())
            else:
                chain_atom_count = 0
            chain_atom_offsets[chain_idx + 1] = chain_atom_offsets[chain_idx] + chain_atom_count
            res_offset += chain_len

        return TopologyInfo(
            atoms=new_atoms,
            sequence=new_sequence,
            residue_sizes=new_residue_sizes,
            chain_lengths=new_chain_lengths,
            chain_atom_offsets=chain_atom_offsets,
            chain_residue_offsets=chain_residue_offsets,
            n_atoms=n_atoms,
            n_residues=n_residues,
            n_chains=n_chains,
        )


@dataclass
class ConnectedComponents:
    """
    Connected component storage in CSR format.

    Stores which atoms belong to which connected components (chains),
    along with reference data for position/orientation restoration
    during NERF reconstruction.

    Attributes:
        offsets: (C+1,) int64 CSR offsets array.
        atoms: (N,) int64 atom indices per component (flattened).
        centroids: (C, 3) float array of component centroids.
        reference_coords: List of (n_i, 3) centered coordinates for alignment.
            None for single-atom components.
        contiguous: List of bool indicating if component atoms are contiguous.

    Example:
        >>> components = ConnectedComponents.from_coordinates_and_topology(coords, topology)
        >>> # Access component 0
        >>> start, end = components.offsets[0], components.offsets[1]
        >>> atom_indices = components.atoms[start:end]
    """

    offsets: np.ndarray
    atoms: np.ndarray
    centroids: np.ndarray
    reference_coords: list[np.ndarray | None]
    contiguous: list[bool]

    @classmethod
    def from_coordinates_and_topology(
        cls,
        coordinates: Array,
        topology: TopologyInfo,
    ) -> "ConnectedComponents":
        """
        Build connected components from coordinates and topology.

        Each chain is treated as a connected component. Component centroids
        and reference coordinates are stored for orientation restoration
        during NERF reconstruction.

        Args:
            coordinates: (N, 3) array of Cartesian coordinates.
            topology: TopologyInfo for the structure.

        Returns:
            ConnectedComponents with CSR storage and reference data.
        """
        coords_np = to_numpy(coordinates)
        n_chains = topology.n_chains

        if n_chains == 0:
            return cls(
                offsets=np.array([0], dtype=np.int64),
                atoms=np.array([], dtype=np.int64),
                centroids=np.zeros((0, 3), dtype=coords_np.dtype),
                reference_coords=[],
                contiguous=[],
            )

        offsets_list = [0]
        atoms_list = []
        centroids_list = []
        reference_coords_list = []
        contiguous_list = []

        for chain_idx in range(n_chains):
            atom_start, atom_end = topology.get_chain_atom_range(chain_idx)
            chain_atom_count = atom_end - atom_start

            if chain_atom_count == 0:
                continue

            # Get atom indices for this chain
            atom_indices = list(range(atom_start, atom_end))

            # Get coordinates and compute centroid
            chain_coords = coords_np[atom_start:atom_end]
            centroid = chain_coords.mean(axis=0)
            centered_coords = chain_coords - centroid

            atoms_list.extend(atom_indices)
            centroids_list.append(centroid.copy())
            offsets_list.append(len(atoms_list))
            contiguous_list.append(True)  # Chains are always contiguous

            # Store reference coords for multi-atom chains (for orientation)
            if chain_atom_count > 1:
                reference_coords_list.append(centered_coords.copy())
            else:
                reference_coords_list.append(None)

        return cls(
            offsets=np.array(offsets_list, dtype=np.int64),
            atoms=np.array(atoms_list, dtype=np.int64),
            centroids=np.array(centroids_list, dtype=coords_np.dtype) if centroids_list else np.zeros((0, 3), dtype=coords_np.dtype),
            reference_coords=reference_coords_list,
            contiguous=contiguous_list,
        )

    def add_orphan_atoms(
        self,
        orphan_indices: np.ndarray,
        coordinates: Array,
    ) -> "ConnectedComponents":
        """
        Add orphan atoms (atoms without bonds) as single-atom components.

        Args:
            orphan_indices: (K,) int64 array of orphan atom indices.
            coordinates: (N, 3) array of coordinates for centroid lookup.

        Returns:
            New ConnectedComponents with orphan atoms added.
        """
        if len(orphan_indices) == 0:
            return self

        coords_np = to_numpy(coordinates)
        orphan_indices_np = to_numpy(orphan_indices).astype(np.int64)
        n_orphans = len(orphan_indices_np)

        # Get orphan centroids (just their positions)
        orphan_centroids = coords_np[orphan_indices_np]

        # Extend offsets
        old_end = int(self.offsets[-1])
        new_offsets = np.concatenate([
            self.offsets,
            np.arange(old_end + 1, old_end + n_orphans + 1, dtype=np.int64)
        ])

        # Extend atoms
        new_atoms = np.concatenate([self.atoms, orphan_indices_np])

        # Extend centroids
        new_centroids = np.concatenate([self.centroids, orphan_centroids])

        # Extend reference_coords (None for single atoms)
        new_reference_coords = self.reference_coords + [None] * n_orphans

        # Extend contiguous (True for single atoms)
        new_contiguous = self.contiguous + [True] * n_orphans

        return ConnectedComponents(
            offsets=new_offsets,
            atoms=new_atoms,
            centroids=new_centroids,
            reference_coords=new_reference_coords,
            contiguous=new_contiguous,
        )

    @property
    def n_components(self) -> int:
        """Number of connected components."""
        return len(self.offsets) - 1

    def get_component_atoms(self, comp_idx: int) -> np.ndarray:
        """
        Get atom indices for a component.

        Args:
            comp_idx: Component index.

        Returns:
            Array of atom indices in this component.
        """
        start = int(self.offsets[comp_idx])
        end = int(self.offsets[comp_idx + 1])
        return self.atoms[start:end]

    def get_component_size(self, comp_idx: int) -> int:
        """
        Get number of atoms in a component.

        Args:
            comp_idx: Component index.

        Returns:
            Number of atoms.
        """
        return int(self.offsets[comp_idx + 1] - self.offsets[comp_idx])
