"""
PyTorch Dataset for loading CIF files.

Provides PolymerDataset for loading and iterating over CIF structures
with filtering and optional caching support.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..polymer import Polymer

try:
    import torch
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Dataset = object  # Placeholder for type hints

from ..types import Scale, Molecule

logger = logging.getLogger(__name__)


def _process_file(args: tuple) -> list[tuple[str, int | None]]:
    """
    Process a single CIF file for indexing (runs in worker process).

    Returns list of (filepath_str, chain_idx or None) tuples that pass filters.
    """
    from ciffy import load_metadata

    (filepath, scale_value, min_atoms, max_atoms, type_filter_values, exclude_ids) = args

    results = []
    try:
        meta = load_metadata(filepath)
    except Exception as e:
        logger.warning(f"Failed to load metadata from {filepath}: {e}")
        return results

    # Check exclude_ids
    pdb_id = meta.get("id", "").upper()
    if exclude_ids and pdb_id in exclude_ids:
        return results

    atoms_per_chain = meta["atoms_per_chain"]
    mol_types = meta["molecule_types"]
    total_atoms = meta["atoms"]

    # Convert type_filter_values back to set if provided
    type_filter = set(type_filter_values) if type_filter_values else None

    if scale_value == Scale.MOLECULE.value:
        # For molecule scale, check if structure has any matching chains
        if type_filter is not None:
            has_matching = any(int(t) in type_filter for t in mol_types)
            if not has_matching:
                return results

        # Check atom count bounds
        if min_atoms is not None and total_atoms < min_atoms:
            return results
        if max_atoms is not None and total_atoms > max_atoms:
            return results

        results.append((filepath, None))

    else:  # Scale.CHAIN
        for chain_idx in range(meta["chains"]):
            chain_mol_type = int(mol_types[chain_idx])

            # Skip chains not matching molecule_types filter
            if type_filter is not None and chain_mol_type not in type_filter:
                continue

            # Check atom count bounds
            chain_atoms = int(atoms_per_chain[chain_idx])
            if min_atoms is not None and chain_atoms < min_atoms:
                continue
            if max_atoms is not None and chain_atoms > max_atoms:
                continue

            results.append((filepath, chain_idx))

    return results


class PolymerDataset(Dataset):
    """
    PyTorch Dataset for loading CIF files from a directory.

    Supports iteration at molecule or chain scale, with optional
    filtering by atom count, molecule type, and PDB ID exclusion.

    Example:
        >>> from ciffy.nn import PolymerDataset
        >>> from ciffy import Scale, Molecule
        >>> # Basic usage
        >>> dataset = PolymerDataset("./structures/", scale=Scale.CHAIN, max_atoms=5000)
        >>> print(f"Found {len(dataset)} chains")
        >>> chain = dataset[0]  # Load first chain
        >>>
        >>> # Only RNA chains with at least 10 atoms
        >>> dataset = PolymerDataset("./structures/", molecule_types=Molecule.RNA, min_atoms=10)
        >>>
        >>> # Exclude specific PDB IDs (e.g., test set)
        >>> dataset = PolymerDataset("./structures/", exclude_ids=["1ABC", "2XYZ"])
        >>>
        >>> # Parallel scanning for large directories
        >>> dataset = PolymerDataset("./pdb/", num_workers=8)
    """

    def __init__(
        self,
        directory: str | Path,
        scale: Scale = Scale.MOLECULE,
        min_atoms: int | None = None,
        max_atoms: int | None = None,
        backend: str = "torch",
        molecule_types: Molecule | tuple[Molecule, ...] | None = None,
        exclude_ids: list[str] | set[str] | None = None,
        num_workers: int = 0,
        limit: int | None = None,
    ):
        """
        Initialize dataset by scanning directory for CIF files.

        Args:
            directory: Path to directory containing .cif files.
            scale: Iteration scale (MOLECULE or CHAIN only).
                - MOLECULE: iterate over full structures
                - CHAIN: iterate over individual chains
            min_atoms: Minimum atoms per item. Items with fewer atoms
                are filtered out. None = no minimum.
            max_atoms: Maximum atoms per item. Items exceeding this
                are filtered out. None = no limit.
            backend: Backend for loaded polymers ("torch" or "numpy").
            molecule_types: Filter to specific molecule type(s). Can be
                a single Molecule or tuple of Molecules. Chains not matching
                any specified type are excluded. None = no filtering.
                Common types: Molecule.PROTEIN, Molecule.RNA, Molecule.DNA
            exclude_ids: PDB IDs to exclude (case-insensitive). Useful for
                held-out test sets or known problematic structures.
            num_workers: Number of worker processes for parallel file scanning.
                0 = single-threaded (default). Higher values speed up scanning
                of large directories.
            limit: Maximum number of samples to include. Useful for overfitting
                tests or quick iteration. None = no limit (use all samples).

        Raises:
            ImportError: If PyTorch is not installed.
            ValueError: If scale is not MOLECULE or CHAIN.
            FileNotFoundError: If directory does not exist.
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for PolymerDataset. "
                "Install with: pip install torch"
            )

        if scale not in (Scale.MOLECULE, Scale.CHAIN):
            raise ValueError(
                f"scale must be MOLECULE or CHAIN, got {scale.name}"
            )

        directory = Path(directory)
        if not directory.is_dir():
            raise FileNotFoundError(f"Directory not found: {directory}")

        self.scale = scale
        self.min_atoms = min_atoms
        self.max_atoms = max_atoms
        self.backend = backend
        self.num_workers = num_workers
        self.limit = limit

        # Normalize molecule_types to tuple or None
        if molecule_types is None:
            self.molecule_types = None
        elif isinstance(molecule_types, Molecule):
            self.molecule_types = (molecule_types,)
        else:
            self.molecule_types = tuple(molecule_types)

        # Normalize exclude_ids to uppercase set
        if exclude_ids is None:
            self.exclude_ids = None
        else:
            self.exclude_ids = {pid.upper() for pid in exclude_ids}

        # Build index: list of (file_path, chain_idx or None)
        self._index: list[tuple[Path, int | None]] = []
        self._build_index(directory)

        # Apply limit after building index
        if self.limit is not None and len(self._index) > self.limit:
            self._index = self._index[:self.limit]

    def _build_index(self, directory: Path) -> None:
        """Scan directory and build index of valid items."""
        cif_files = sorted(directory.glob("*.cif"))

        if not cif_files:
            return

        # Pre-compute type filter values for serialization
        type_filter_values = None
        if self.molecule_types is not None:
            type_filter_values = tuple(m.value for m in self.molecule_types)

        # Convert exclude_ids to frozenset for serialization
        exclude_ids = frozenset(self.exclude_ids) if self.exclude_ids else None

        if self.num_workers > 0:
            self._build_index_parallel(cif_files, type_filter_values, exclude_ids)
        else:
            self._build_index_sequential(cif_files, type_filter_values, exclude_ids)

    def _build_index_sequential(self, cif_files: list[Path],
                                 type_filter_values: tuple | None,
                                 exclude_ids: frozenset | None) -> None:
        """Build index using single-threaded scanning."""
        from .. import load_metadata

        type_filter = set(type_filter_values) if type_filter_values else None

        for path in cif_files:
            try:
                meta = load_metadata(str(path))
            except Exception as e:
                logger.warning(f"Failed to load metadata from {path}: {e}")
                continue

            # Check exclude_ids
            pdb_id = meta.get("id", "").upper()
            if exclude_ids and pdb_id in exclude_ids:
                continue

            atoms_per_chain = meta["atoms_per_chain"]
            mol_types = meta["molecule_types"]

            if self.scale == Scale.MOLECULE:
                # For molecule scale, check if structure has any matching chains
                if type_filter is not None:
                    has_matching = any(int(t) in type_filter for t in mol_types)
                    if not has_matching:
                        continue

                # Check atom count bounds
                total_atoms = meta["atoms"]
                if self.min_atoms is not None and total_atoms < self.min_atoms:
                    continue
                if self.max_atoms is not None and total_atoms > self.max_atoms:
                    continue

                self._index.append((path, None))

            else:  # Scale.CHAIN
                for chain_idx in range(meta["chains"]):
                    chain_mol_type = int(mol_types[chain_idx])

                    # Skip chains not matching molecule_types filter
                    if type_filter is not None and chain_mol_type not in type_filter:
                        continue

                    # Check atom count bounds
                    chain_atoms = int(atoms_per_chain[chain_idx])
                    if self.min_atoms is not None and chain_atoms < self.min_atoms:
                        continue
                    if self.max_atoms is not None and chain_atoms > self.max_atoms:
                        continue

                    self._index.append((path, chain_idx))

    def _build_index_parallel(self, cif_files: list[Path],
                               type_filter_values: tuple | None,
                               exclude_ids: frozenset | None) -> None:
        """Build index using parallel worker processes."""
        from concurrent.futures import ProcessPoolExecutor, as_completed

        # Prepare arguments for each file
        args_list = [
            (str(path), self.scale.value, self.min_atoms, self.max_atoms,
             type_filter_values, exclude_ids)
            for path in cif_files
        ]

        # Process files in parallel
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(_process_file, args) for args in args_list]

            for future in as_completed(futures):
                try:
                    results = future.result()
                    for filepath_str, chain_idx in results:
                        self._index.append((Path(filepath_str), chain_idx))
                except Exception as e:
                    logger.warning(f"Worker process failed: {e}")
                    continue

        # Sort index by filepath for deterministic ordering
        self._index.sort(key=lambda x: (x[0], x[1] if x[1] is not None else -1))

    def __len__(self) -> int:
        """Return number of valid items (structures or chains)."""
        return len(self._index)

    def __getitem__(self, idx: int) -> Optional["Polymer"]:
        """
        Load and return polymer/chain at index.

        Args:
            idx: Index into dataset.

        Returns:
            Polymer object (full structure or single chain),
            with any configured filtering applied.
            Returns None if loading fails, allowing DataLoader to
            skip the sample via a custom collate_fn.

        Note:
            At CHAIN scale, molecule_types filtering is done during
            index building, so no filtering is needed here.
            At MOLECULE scale, chain filtering is applied after loading
            to remove non-matching chains from mixed structures.

            Returns None instead of raising on errors to support
            DataLoader with num_workers > 0 without crashing workers.
        """
        from .. import load

        try:
            path, chain_idx = self._index[idx]
            polymer = load(str(path), backend=self.backend)

            if chain_idx is not None:
                # Chain scale: filtering already done during indexing
                polymer = polymer.by_index(chain_idx)
            elif self.molecule_types is not None:
                # Molecule scale: filter out non-matching chains
                polymer = self._filter_by_molecule_type(polymer)

            return polymer

        except Exception as e:
            logger.debug(f"Failed to load item {idx}: {e}")
            return None

    def _filter_by_molecule_type(self, polymer: Polymer) -> Polymer:
        """Filter polymer to only include chains of specified molecule types."""
        from ..backend import ops

        # Build mask for matching types (backend-agnostic)
        type_values = [m.value for m in self.molecule_types]
        mask = ops.isin(polymer.molecule_type, type_values)

        # Get matching chain indices (backend-agnostic)
        matching_indices = ops.nonzero_1d(mask)

        if len(matching_indices) == 0:
            # Return empty polymer (first 0 atoms)
            return polymer[:0]

        return polymer.by_index(matching_indices)
