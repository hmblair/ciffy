"""
PyTorch Dataset for loading CIF files.

Provides PolymerDataset for loading and iterating over CIF structures
with filtering and optional caching support.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence, Union

if TYPE_CHECKING:
    from ciffy.polymer import Polymer

import torch
from torch.utils.data import Dataset

from ciffy.biochemistry import Scale, Molecule

logger = logging.getLogger(__name__)


class PolymerCache:
    """
    Two-level cache for polymers with automatic file eviction.

    Caches at two levels:
    - File level: full polymers keyed by path (avoids redundant disk I/O)
    - Item level: extracted chains keyed by index (returns same object on repeat access)

    When all chains from a file have been loaded into the item cache,
    the file is automatically evicted to free memory.
    """

    def __init__(self):
        self._files: dict[Path, "Polymer"] = {}
        self._items: dict[int, "Polymer"] = {}
        self._total: dict[Path, int] = {}
        self._loaded: dict[Path, int] = {}

    def register(self, path: Path) -> None:
        """Register a chain from a file. Call once per chain during indexing."""
        self._total[path] = self._total.get(path, 0) + 1

    def get(
        self,
        idx: int,
        path: Path,
        loader: callable,
        extractor: callable,
    ) -> "Polymer":
        """
        Get item from cache, loading and extracting if needed.

        Args:
            idx: Item index (for item cache key)
            path: File path (for file cache key)
            loader: Callable that loads the full polymer from disk
            extractor: Callable that extracts the item from the full polymer

        Returns:
            The extracted polymer (chain or filtered molecule)
        """
        # Item cache hit - return immediately
        if idx in self._items:
            return self._items[idx]

        # Load file if not cached
        if path not in self._files:
            self._files[path] = loader()

        # Extract item and cache it
        result = extractor(self._files[path])
        self._items[idx] = result

        # Track loaded chains, evict file when complete
        self._loaded[path] = self._loaded.get(path, 0) + 1
        if self._loaded[path] >= self._total.get(path, 0):
            self._files.pop(path, None)

        return result

    def __contains__(self, idx: int) -> bool:
        """Check if item is cached."""
        return idx in self._items


class PolymerDataset(Dataset):
    """
    PyTorch Dataset for loading CIF files.

    Accepts either a directory path or a sequence of file paths. Supports
    iteration at molecule or chain scale, with optional filtering by atom
    count, molecule type, and PDB ID exclusion.

    Example:
        >>> from ciffy.nn import PolymerDataset
        >>> from ciffy import Scale, Molecule
        >>>
        >>> # From directory
        >>> dataset = PolymerDataset("./structures/", scale=Scale.CHAIN, max_atoms=5000)
        >>> print(f"Found {len(dataset)} chains")
        >>> chain = dataset[0]  # Load first chain
        >>>
        >>> # From list of paths (e.g., from split_items)
        >>> from ciffy.utils.split import split_items
        >>> split = split_items(paths, train=0.8, val=0.1, test=0.1)
        >>> train_dataset = PolymerDataset(split.train, scale=Scale.CHAIN)
        >>>
        >>> # Only RNA chains with at least 10 atoms
        >>> dataset = PolymerDataset("./structures/", molecule_types=Molecule.RNA, min_atoms=10)
        >>>
        >>> # Exclude specific PDB IDs (e.g., test set)
        >>> dataset = PolymerDataset("./structures/", exclude_ids=["1ABC", "2XYZ"])
    """

    def __init__(
        self,
        paths: Union[str, Path, Sequence[Union[str, Path]]],
        scale: Scale = Scale.MOLECULE,
        min_atoms: int | None = None,
        max_atoms: int | None = None,
        min_residues: int | None = None,
        max_residues: int | None = None,
        min_chains: int | None = None,
        max_chains: int | None = None,
        backend: str = "torch",
        molecule_types: Molecule | tuple[Molecule, ...] | None = None,
        exclude_ids: list[str] | set[str] | None = None,
        num_workers: int = 0,
        limit: int | None = None,
        cache: bool = True,
    ):
        """
        Initialize dataset from a directory or list of CIF files.

        Args:
            paths: Either a directory path (will glob for *.cif files), or a
                sequence of .cif file paths (e.g., from DataSplit.train).
            scale: Iteration scale (MOLECULE or CHAIN only).
                - MOLECULE: iterate over full structures
                - CHAIN: iterate over individual chains
            min_atoms: Minimum atoms per item. Items with fewer atoms
                are filtered out during indexing. None = no minimum.
            max_atoms: Maximum atoms per item. Items exceeding this
                are filtered out during indexing. None = no limit.
            min_residues: Minimum residues per item. Items with fewer residues
                are filtered out during indexing. None = no minimum.
            max_residues: Maximum residues per item. Items exceeding this
                are filtered out during indexing. None = no limit.
            min_chains: Minimum chains per structure. Structures with fewer
                chains are filtered out. None = no minimum.
            max_chains: Maximum chains per structure. Structures exceeding
                this are filtered out. None = no limit. Use max_chains=1 to
                only include single-chain structures (no complexes).
            backend: Backend for loaded polymers ("torch" or "numpy").
            molecule_types: Filter to specific molecule type(s). Can be
                a single Molecule or tuple of Molecules. Chains not matching
                any specified type are excluded. None = no filtering.
                Common types: Molecule.PROTEIN, Molecule.RNA, Molecule.DNA
            exclude_ids: PDB IDs to exclude (case-insensitive). Useful for
                held-out test sets or known problematic structures.
            num_workers: Deprecated, does nothing. Kept for backward compatibility.
            limit: Maximum number of samples to include. Useful for overfitting
                tests or quick iteration. None = no limit (use all samples).
            cache: If True, cache loaded structures in memory. Uses two-level
                caching (file and item) with automatic file eviction once all
                chains from a file are loaded. Particularly effective for
                chain-scale datasets where multiple chains come from the same
                structure. Default True.

        Raises:
            ValueError: If scale is not MOLECULE or CHAIN.
            FileNotFoundError: If directory does not exist or paths are invalid.
        """
        if scale not in (Scale.MOLECULE, Scale.CHAIN):
            raise ValueError(
                f"scale must be MOLECULE or CHAIN, got {scale.name}"
            )

        if num_workers > 0:
            warnings.warn(
                "num_workers is deprecated and has no effect. "
                "Single-threaded indexing is faster for typical datasets.",
                DeprecationWarning,
                stacklevel=2,
            )

        # Determine if paths is a directory or a sequence of files
        if isinstance(paths, (str, Path)):
            # Single path: must be a directory
            directory = Path(paths)
            if not directory.is_dir():
                raise FileNotFoundError(f"Directory not found: {directory}")
            # Glob both .cif and .cif.gz files
            cif_files = sorted(
                list(directory.glob("*.cif")) + list(directory.glob("*.cif.gz"))
            )
        else:
            # Sequence of file paths
            cif_files = sorted(Path(p) for p in paths)

        self.scale = scale
        self.min_atoms = min_atoms
        self.max_atoms = max_atoms
        self.min_residues = min_residues
        self.max_residues = max_residues
        self.backend = backend
        self.limit = limit
        self._cache: PolymerCache | None = PolymerCache() if cache else None

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

        self.min_chains = min_chains
        self.max_chains = max_chains

        # Build index: list of (file_path, chain_idx or None)
        self._index: list[tuple[Path, int | None]] = []
        self._build_index(cif_files)

    def _build_index(self, cif_files: list[Path]) -> None:
        """Build index by scanning CIF files and applying filters.

        Uses parallel metadata loading in batches of 128 for faster indexing
        (~4x speedup) while avoiding loading all metadata at once.
        """
        from ciffy import load_metadata

        type_filter = None
        if self.molecule_types is not None:
            type_filter = {m.value for m in self.molecule_types}

        batch_size = 128
        for i in range(0, len(cif_files), batch_size):
            if self.limit is not None and len(self._index) >= self.limit:
                break

            batch = cif_files[i : i + batch_size]
            metas = load_metadata(batch)

            for path, meta in zip(batch, metas):
                if self.limit is not None and len(self._index) >= self.limit:
                    break

                if meta is None:
                    logger.debug(f"Failed to load metadata from {path}")
                    continue

                self._process_metadata_for_index(path, meta, type_filter)

    def _process_metadata_for_index(
        self, path: Path, meta: dict, type_filter: set | None
    ) -> None:
        """Process a single metadata dict and add to index if it passes filters."""
        # Check exclude_ids
        pdb_id = meta.get("id", "").upper()
        if self.exclude_ids and pdb_id in self.exclude_ids:
            return

        if self.scale == Scale.MOLECULE:
            if self._file_passes_filters(meta, type_filter):
                self._index.append((path, None))
                if self._cache is not None:
                    self._cache.register(path)
        else:  # Scale.CHAIN
            for chain_idx in range(meta["chains"]):
                if self._chain_passes_filters(meta, chain_idx, type_filter):
                    self._index.append((path, chain_idx))
                    if self._cache is not None:
                        self._cache.register(path)
                    # Early termination for chain scale too
                    if self.limit is not None and len(self._index) >= self.limit:
                        break

    def _file_passes_filters(self, meta: dict, type_filter: set | None) -> bool:
        """Check if a file passes molecule-level filters."""
        num_chains = meta["chains"]
        if self.min_chains is not None and num_chains < self.min_chains:
            return False
        if self.max_chains is not None and num_chains > self.max_chains:
            return False

        if type_filter is not None:
            has_matching = any(int(t) in type_filter for t in meta["molecule_types"])
            if not has_matching:
                return False

        total_atoms = meta["atoms"]
        if self.min_atoms is not None and total_atoms < self.min_atoms:
            return False
        if self.max_atoms is not None and total_atoms > self.max_atoms:
            return False

        total_residues = meta["residues"]
        if self.min_residues is not None and total_residues < self.min_residues:
            return False
        if self.max_residues is not None and total_residues > self.max_residues:
            return False

        return True

    def _chain_passes_filters(self, meta: dict, chain_idx: int, type_filter: set | None) -> bool:
        """Check if a chain passes chain-level filters."""
        num_chains = meta["chains"]
        if self.min_chains is not None and num_chains < self.min_chains:
            return False
        if self.max_chains is not None and num_chains > self.max_chains:
            return False

        if type_filter is not None:
            chain_mol_type = int(meta["molecule_types"][chain_idx])
            if chain_mol_type not in type_filter:
                return False

        chain_atoms = int(meta["atoms_per_chain"][chain_idx])
        if self.min_atoms is not None and chain_atoms < self.min_atoms:
            return False
        if self.max_atoms is not None and chain_atoms > self.max_atoms:
            return False

        chain_residues = int(meta["residues_per_chain"][chain_idx])
        if self.min_residues is not None and chain_residues < self.min_residues:
            return False
        if self.max_residues is not None and chain_residues > self.max_residues:
            return False

        return True

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

            When cache=True, caching is done at two levels:
            - File level: avoids redundant disk I/O for different chains from same file
            - Item level: returns same object on repeated access to same index
            Files are automatically evicted from cache once all their chains are loaded.
        """
        import ciffy

        try:
            path, chain_idx = self._index[idx]

            def loader():
                return ciffy.load(str(path), backend=self.backend)

            def extractor(polymer):
                if chain_idx is not None:
                    # Chain scale: filtering already done during indexing
                    return polymer.chain(chain_idx)
                elif self.molecule_types is not None:
                    # Molecule scale: filter out non-matching chains
                    return self._filter_by_molecule_type(polymer)
                return polymer

            if self._cache is not None:
                return self._cache.get(idx, path, loader, extractor)

            # Non-cached path
            polymer = loader()
            return extractor(polymer)

        except Exception as e:
            logger.debug(f"Failed to load item {idx}: {e}")
            return None

    def _filter_by_molecule_type(self, polymer: Polymer) -> Polymer:
        """Filter polymer to only include chains of specified molecule types."""
        from ciffy.backend import ops

        # Build mask for matching types (backend-agnostic)
        if polymer.molecule_types is None:
            raise ValueError("Cannot filter by type: molecule_types not available on this polymer")
        type_values = [m.value for m in self.molecule_types]
        mask = ops.isin(polymer.molecule_types, type_values)

        # Get matching chain indices (backend-agnostic)
        matching_indices = ops.nonzero_1d(mask)

        if len(matching_indices) == 0:
            # Return empty polymer (first 0 atoms)
            return polymer[:0]

        return polymer.chain(matching_indices)

    def load(self) -> "PolymerDataset":
        """
        Preload all structures into cache using parallel I/O.

        Loads all unique files in parallel (~5x faster than sequential access),
        then extracts and caches individual items. Subsequent __getitem__ calls
        return cached items instantly.

        Returns:
            self, for method chaining (e.g., `dataset = PolymerDataset(...).load()`)

        Raises:
            ValueError: If cache is disabled (cache=False in constructor).

        Example:
            >>> dataset = PolymerDataset("./structures/", scale=Scale.CHAIN).load()
            >>> for i in range(len(dataset)):  # Fast iteration, all cached
            ...     polymer = dataset[i]
        """
        if self._cache is None:
            raise ValueError("Cannot preload: cache is disabled")

        from ciffy import load as ciffy_load

        # Get unique file paths preserving order
        unique_paths = list(dict.fromkeys(path for path, _ in self._index))

        # Parallel load all files
        polymers = ciffy_load(unique_paths, backend=self.backend)

        # Build path -> polymer mapping
        path_to_polymer = {}
        for path, polymer in zip(unique_paths, polymers):
            if polymer is not None:
                path_to_polymer[path] = polymer

        # Extract and cache each item
        for idx, (path, chain_idx) in enumerate(self._index):
            if idx in self._cache:
                continue

            polymer = path_to_polymer.get(path)
            if polymer is None:
                continue

            if chain_idx is not None:
                item = polymer.chain(chain_idx)
            elif self.molecule_types is not None:
                item = self._filter_by_molecule_type(polymer)
            else:
                item = polymer

            self._cache._items[idx] = item

        return self

    @classmethod
    def _from_index(
        cls,
        index: list[tuple[Path, int | None]],
        *,
        scale: Scale,
        min_atoms: int | None,
        max_atoms: int | None,
        min_residues: int | None,
        max_residues: int | None,
        min_chains: int | None,
        max_chains: int | None,
        backend: str,
        molecule_types: tuple[Molecule, ...] | None,
        exclude_ids: set[str] | None,
        limit: int | None,
        cache: bool,
    ) -> "PolymerDataset":
        """
        Create dataset from pre-built index (internal use).

        This bypasses the normal file scanning and filtering, using a
        pre-filtered index directly. Used by split() to create child datasets.
        """
        # Create instance without calling __init__
        instance = object.__new__(cls)
        instance.scale = scale
        instance.min_atoms = min_atoms
        instance.max_atoms = max_atoms
        instance.min_residues = min_residues
        instance.max_residues = max_residues
        instance.min_chains = min_chains
        instance.max_chains = max_chains
        instance.backend = backend
        instance.molecule_types = molecule_types
        instance.exclude_ids = exclude_ids
        instance.limit = limit
        instance._index = index
        instance._cache = PolymerCache() if cache else None

        # Register paths in cache
        if instance._cache is not None:
            for path, _ in index:
                instance._cache.register(path)

        return instance

    def split(
        self,
        train: float = 0.8,
        val: float = 0.1,
        test: float = 0.0,
        seed: int | None = 42,
        by_sequence: bool = False,
        threshold: float = 0.5,
        coverage: float = 0.8,
        threads: int = 4,
    ) -> tuple["PolymerDataset", ...]:
        """
        Split dataset into train/val/test subsets.

        Splitting is done at the file level to preserve model boundaries.
        Files are assigned to splits (not individual chains), ensuring all
        chains from a file stay in the same split.

        Args:
            train: Fraction for training set (default: 0.8).
            val: Fraction for validation set (default: 0.1).
            test: Fraction for test set (default: 0.0).
            seed: Random seed for reproducibility (default: 42).
            by_sequence: If True, cluster by sequence identity before splitting
                to prevent homologous sequences in different splits. Requires
                MMseqs2 (default: False).
            threshold: Sequence identity threshold for clustering when
                by_sequence=True (default: 0.5).
            coverage: Minimum alignment coverage for clustering (default: 0.8).
            threads: Number of threads for MMseqs2 (default: 4).

        Returns:
            Tuple of 1-3 PolymerDataset instances, one for each nonzero split.
            Order is always (train, val, test), but zero-fraction splits are
            omitted from the tuple.

        Raises:
            ValueError: If split ratios don't sum to ~1.0 or are negative.
            RuntimeError: If by_sequence=True and MMseqs2 is not installed.

        Example:
            >>> dataset = PolymerDataset("./structures/", scale=Scale.CHAIN)
            >>> train, val, test = dataset.split(0.8, 0.1, 0.1)
            >>> train, test = dataset.split(0.9, 0.0, 0.1)  # 2-way split
        """
        from ._split import _split_items, _split_by_sequence_identity

        if any(r < 0 for r in [train, val, test]):
            raise ValueError("Split ratios must be non-negative")

        total = train + val + test
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")

        # Extract unique paths from index (preserves model boundaries)
        unique_paths = list(dict.fromkeys(path for path, _ in self._index))

        if len(unique_paths) == 0:
            # Empty dataset - return empty splits for nonzero fractions
            results = []
            for frac in [train, val, test]:
                if frac > 0:
                    results.append(self._create_child_dataset([]))
            return tuple(results)

        # Split paths
        if by_sequence:
            train_paths, val_paths, test_paths = _split_by_sequence_identity(
                unique_paths,
                threshold=threshold,
                train=train,
                val=val,
                test=test,
                seed=seed,
                coverage=coverage,
                threads=threads,
            )
        else:
            train_paths, val_paths, test_paths = _split_items(
                unique_paths,
                train=train,
                val=val,
                test=test,
                seed=seed,
            )

        # Create child datasets for nonzero fractions
        results = []
        for frac, paths in [(train, train_paths), (val, val_paths), (test, test_paths)]:
            if frac > 0:
                # Filter index to only include entries from these paths
                path_set = set(paths)
                child_index = [(p, idx) for p, idx in self._index if p in path_set]
                results.append(self._create_child_dataset(child_index))

        return tuple(results)

    def _create_child_dataset(
        self, index: list[tuple[Path, int | None]]
    ) -> "PolymerDataset":
        """Create a child dataset with the same configuration but different index."""
        return PolymerDataset._from_index(
            index,
            scale=self.scale,
            min_atoms=self.min_atoms,
            max_atoms=self.max_atoms,
            min_residues=self.min_residues,
            max_residues=self.max_residues,
            min_chains=self.min_chains,
            max_chains=self.max_chains,
            backend=self.backend,
            molecule_types=self.molecule_types,
            exclude_ids=self.exclude_ids,
            limit=self.limit,
            cache=self._cache is not None,
        )
