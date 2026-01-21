"""
Reusable train/validation/test splitting for molecular structure data.

Provides utilities for splitting datasets to avoid data leakage:

- `split_items`: Simple random split of any sequence (fast, no dependencies)
- `split_by_sequence_identity`: Cluster by sequence identity, then split clusters
  (prevents homologous sequences in different splits, requires MMseqs2)

Example - Simple split:
    >>> from ciffy.nn.training import split_items
    >>> split = split_items(cif_files, train=0.8, val=0.1, test=0.1)

Example - Sequence-identity split (recommended for ML):
    >>> from ciffy.nn.training import split_by_sequence_identity
    >>> split = split_by_sequence_identity(cif_files, threshold=0.5)
    >>> # Homologous structures guaranteed to be in same split
"""

from __future__ import annotations

import random
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, Sequence, TypeVar

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class DataSplit(Generic[T]):
    """
    Container for train/validation/test splits.

    Attributes:
        train: Training set items.
        val: Validation set items.
        test: Test set items.
        seed: Random seed used for splitting.
    """

    train: list[T]
    val: list[T]
    test: list[T]
    seed: int | None = None

    def __len__(self) -> int:
        """Total number of items across all splits."""
        return len(self.train) + len(self.val) + len(self.test)


def _validate_no_overlap(split: DataSplit) -> None:
    """Check if any items appear in multiple splits."""
    try:
        train_set = set(split.train)
        val_set = set(split.val)
        test_set = set(split.test)
        if train_set & val_set or train_set & test_set or val_set & test_set:
            raise ValueError("Splits have overlapping items")
    except TypeError:
        # Items not hashable, skip validation
        pass


def split_items(
    items: Sequence[T],
    train: float = 0.8,
    val: float = 0.1,
    test: float = 0.1,
    seed: int | None = 42,
) -> DataSplit[T]:
    """
    Split a sequence of items into train/val/test sets.

    Items are shuffled and divided according to the specified ratios.
    The split is deterministic when a seed is provided.

    Args:
        items: Sequence of items to split (paths, IDs, indices, etc.).
        train: Fraction for training set (default: 0.8).
        val: Fraction for validation set (default: 0.1).
        test: Fraction for test set (default: 0.1).
        seed: Random seed for reproducibility (default: 42).

    Returns:
        DataSplit with train, val, test lists.

    Raises:
        ValueError: If ratios don't sum to ~1.0 or are negative.

    Example:
        >>> paths = [Path(f"data/{i}.cif") for i in range(100)]
        >>> split = split_items(paths, train=0.8, val=0.1, test=0.1)
        >>> len(split.train), len(split.val), len(split.test)
        (80, 10, 10)
    """
    if any(r < 0 for r in [train, val, test]):
        raise ValueError("Split ratios must be non-negative")

    total = train + val + test
    if not (0.99 <= total <= 1.01):
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")

    items_list = list(items)
    n = len(items_list)

    if n == 0:
        return DataSplit(train=[], val=[], test=[], seed=seed)

    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(items_list)
    else:
        random.shuffle(items_list)

    n_train = int(n * train)
    n_val = int(n * val)

    train_items = items_list[:n_train]
    val_items = items_list[n_train : n_train + n_val]
    test_items = items_list[n_train + n_val :]

    result = DataSplit(train=train_items, val=val_items, test=test_items, seed=seed)
    _validate_no_overlap(result)
    return result


def split_train_test(
    items: Sequence[T],
    train: float = 0.8,
    seed: int | None = 42,
) -> DataSplit[T]:
    """
    Create a simple train/test split (no validation set).

    Args:
        items: Sequence of items to split.
        train: Fraction for training set (default: 0.8).
        seed: Random seed for reproducibility.

    Returns:
        DataSplit with train and test lists (val is empty).

    Example:
        >>> split = split_train_test(paths, train=0.9)
        >>> len(split.val)
        0
    """
    return split_items(items, train=train, val=0.0, test=1.0 - train, seed=seed)


def split_by_clusters(
    items: Sequence[T],
    labels: Sequence[int],
    train: float = 0.8,
    val: float = 0.1,
    test: float = 0.1,
    seed: int | None = 42,
) -> DataSplit[T]:
    """
    Split items where items in the same cluster stay together.

    Clusters are shuffled and assigned to splits. All items within
    a cluster go to the same split, preventing data leakage between
    similar items.

    Args:
        items: Sequence of items to split.
        labels: Cluster label for each item (same length as items).
        train: Fraction for training set (default: 0.8).
        val: Fraction for validation set (default: 0.1).
        test: Fraction for test set (default: 0.1).
        seed: Random seed for reproducibility (default: 42).

    Returns:
        DataSplit with train, val, test lists.

    Example:
        >>> from ciffy.operations import cluster
        >>> result = cluster(paths, threshold=0.5)
        >>> split = split_by_clusters(
        ...     result.paths, result.labels,
        ...     train=0.8, val=0.1, test=0.1
        ... )
    """
    if len(items) != len(labels):
        raise ValueError(
            f"items and labels must have same length: {len(items)} vs {len(labels)}"
        )

    if any(r < 0 for r in [train, val, test]):
        raise ValueError("Split ratios must be non-negative")

    total_ratio = train + val + test
    if not (0.99 <= total_ratio <= 1.01):
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")

    items_list = list(items)
    labels_list = list(labels)

    if len(items_list) == 0:
        return DataSplit(train=[], val=[], test=[], seed=seed)

    # Group items by cluster
    cluster_to_items: dict[int, list[T]] = {}
    for item, label in zip(items_list, labels_list):
        if label not in cluster_to_items:
            cluster_to_items[label] = []
        cluster_to_items[label].append(item)

    # Shuffle cluster IDs
    cluster_ids = list(cluster_to_items.keys())
    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(cluster_ids)
    else:
        random.shuffle(cluster_ids)

    # Split clusters according to ratios
    n_clusters = len(cluster_ids)
    n_train = int(n_clusters * train)
    n_val = int(n_clusters * val)

    min_clusters_needed = (
        (1 if train > 0 else 0) + (1 if val > 0 else 0) + (1 if test > 0 else 0)
    )
    if n_clusters < min_clusters_needed:
        warnings.warn(
            f"Only {n_clusters} cluster(s) found, but {min_clusters_needed} needed for "
            f"train/val/test split. Consider lowering the similarity threshold."
        )

    train_clusters = cluster_ids[:n_train]
    val_clusters = cluster_ids[n_train : n_train + n_val]
    test_clusters = cluster_ids[n_train + n_val :]

    train_items = [item for cid in train_clusters for item in cluster_to_items[cid]]
    val_items = [item for cid in val_clusters for item in cluster_to_items[cid]]
    test_items = [item for cid in test_clusters for item in cluster_to_items[cid]]

    return DataSplit(train=train_items, val=val_items, test=test_items, seed=seed)


def split_by_sequence_identity(
    paths: Sequence[str | Path],
    threshold: float = 0.5,
    train: float = 0.8,
    val: float = 0.1,
    test: float = 0.1,
    seed: int | None = 42,
    coverage: float = 0.8,
    threads: int = 4,
) -> DataSplit[Path]:
    """
    Split paths based on sequence identity clustering.

    Structures are first clustered by sequence identity using MMseqs2.
    Then clusters (not individual structures) are split into train/val/test.
    This ensures no homologous sequences appear in different splits.

    Args:
        paths: Sequence of CIF/PDB file paths.
        threshold: Sequence identity threshold (default: 0.5).
            0.3 = remote homologs, 0.5 = same family, 0.9 = near-identical.
        train: Fraction for training set (default: 0.8).
        val: Fraction for validation set (default: 0.1).
        test: Fraction for test set (default: 0.1).
        seed: Random seed for reproducibility (default: 42).
        coverage: Minimum alignment coverage for clustering (default: 0.8).
        threads: Number of threads for MMseqs2 (default: 4).

    Returns:
        DataSplit with Path objects, clustered to prevent homolog leakage.

    Raises:
        RuntimeError: If MMseqs2 is not installed.

    Example:
        >>> split = split_by_sequence_identity(paths, threshold=0.5)
        >>> print(f"Train: {len(split.train)}, Test: {len(split.test)}")
    """
    from ciffy.operations.cluster import cluster

    result = cluster(
        paths,
        threshold=threshold,
        threads=threads,
        coverage=coverage,
    )

    return split_by_clusters(
        result.paths,
        result.labels.tolist(),
        train=train,
        val=val,
        test=test,
        seed=seed,
    )


def split_to_directories(
    split: DataSplit[Path],
    output_dir: str | Path,
    symlink: bool = True,
    exist_ok: bool = False,
) -> dict[str, Path]:
    """
    Create train/val/test directories with symlinks or copies of files.

    Args:
        split: DataSplit containing Path objects.
        output_dir: Parent directory for split subdirectories.
            Creates output_dir/train/, output_dir/val/, output_dir/test/.
        symlink: If True (default), create symbolic links to original files.
            If False, copy files (uses more disk space but is portable).
        exist_ok: If True, skip files that already exist in the destination.
            If False (default), raise an error if destination exists.

    Returns:
        Dict mapping split names to directory paths:
        {"train": Path(...), "val": Path(...), "test": Path(...)}

    Raises:
        TypeError: If items are not Path objects.
        FileExistsError: If destination files exist and exist_ok=False.

    Example:
        >>> split = split_by_sequence_identity(paths, threshold=0.5)
        >>> dirs = split_to_directories(split, "data/splits/")
        >>> train_dataset = PolymerDataset(dirs["train"])
    """
    output_dir = Path(output_dir)
    result = {}

    for name, items in [("train", split.train), ("val", split.val), ("test", split.test)]:
        if not items:
            continue

        if not all(isinstance(item, Path) for item in items):
            raise TypeError(
                f"split_to_directories() requires Path items, got {type(items[0]).__name__}."
            )

        split_dir = output_dir / name
        split_dir.mkdir(parents=True, exist_ok=True)

        for item in items:
            dest = split_dir / item.name

            if dest.exists() or dest.is_symlink():
                if exist_ok:
                    continue
                raise FileExistsError(
                    f"Destination already exists: {dest}. "
                    f"Use exist_ok=True to skip existing files."
                )

            if symlink:
                dest.symlink_to(item.resolve())
            else:
                shutil.copy2(item, dest)

        result[name] = split_dir

    return result


# Shorter alias
split_by_sequence = split_by_sequence_identity

__all__ = [
    "DataSplit",
    "split_items",
    "split_train_test",
    "split_by_clusters",
    "split_by_sequence_identity",
    "split_by_sequence",
    "split_to_directories",
]
