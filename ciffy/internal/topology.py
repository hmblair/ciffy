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
        >>> components = ConnectedComponents.from_bond_graph(csr_offsets, csr_neighbors, coords, n_atoms)
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
    def from_bond_graph(
        cls,
        csr_offsets: np.ndarray,
        csr_neighbors: np.ndarray,
        coordinates: Array,
        n_atoms: int,
    ) -> "ConnectedComponents":
        """
        Build connected components from bond graph in CSR format.

        Finds all connected components including isolated atoms (no bonds).
        Component centroids and reference coordinates are stored for
        position/orientation restoration during NERF reconstruction.

        Args:
            csr_offsets: (N+1,) CSR offsets array for bond graph.
            csr_neighbors: (E,) CSR neighbor indices.
            coordinates: (N, 3) array of Cartesian coordinates.
            n_atoms: Total number of atoms.

        Returns:
            ConnectedComponents with all components (bonded and isolated).
        """
        from .graph import find_connected_components

        coords_np = to_numpy(coordinates)

        if n_atoms == 0:
            return cls(
                offsets=np.array([0], dtype=np.int64),
                atoms=np.array([], dtype=np.int64),
                centroids=np.zeros((0, 3), dtype=coords_np.dtype),
                reference_coords=[],
                contiguous=[],
            )

        # Find all connected components (includes isolated atoms as single-atom components)
        comp_atoms, comp_offsets, n_components = find_connected_components(
            csr_offsets, csr_neighbors, n_atoms
        )

        if n_components == 0:
            return cls(
                offsets=np.array([0], dtype=np.int64),
                atoms=np.array([], dtype=np.int64),
                centroids=np.zeros((0, 3), dtype=coords_np.dtype),
                reference_coords=[],
                contiguous=[],
            )

        # Process each component to compute centroids and reference coords
        centroids_list = []
        reference_coords_list = []
        contiguous_list = []

        for i in range(n_components):
            start = comp_offsets[i]
            end = comp_offsets[i + 1]
            component_atoms = comp_atoms[start:end]
            component_coords = coords_np[component_atoms]
            centroid = component_coords.mean(axis=0)

            # Check if atoms are contiguous in memory
            is_contiguous = (
                len(component_atoms) > 0 and
                (len(component_atoms) == 1 or np.all(np.diff(component_atoms) == 1))
            )

            centroids_list.append(centroid.copy())
            contiguous_list.append(is_contiguous)

            # Store reference coords for multi-atom components
            if len(component_atoms) > 1:
                centered_coords = component_coords - centroid
                reference_coords_list.append(centered_coords.copy())
            else:
                reference_coords_list.append(None)

        return cls(
            offsets=comp_offsets,
            atoms=comp_atoms,
            centroids=np.array(centroids_list, dtype=coords_np.dtype),
            reference_coords=reference_coords_list,
            contiguous=contiguous_list,
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

    def update_centroids(self, coordinates: "Array") -> None:
        """
        Update centroids and reference coordinates from new Cartesian coordinates.

        This is called when coordinates change but component structure stays the same.
        Only updates the coordinate-dependent data (centroids, reference_coords),
        not the topology-dependent data (offsets, atoms, contiguous).

        Args:
            coordinates: (N, 3) array of new Cartesian coordinates.
        """
        coords_np = to_numpy(coordinates)
        n_components = self.n_components

        for i in range(n_components):
            component_atoms = self.get_component_atoms(i)
            component_coords = coords_np[component_atoms]
            centroid = component_coords.mean(axis=0)
            self.centroids[i] = centroid

            # Update reference coords for multi-atom components
            if len(component_atoms) > 1:
                centered_coords = component_coords - centroid
                self.reference_coords[i] = centered_coords.copy()
