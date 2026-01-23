"""Private splitting utilities for PolymerDataset."""

from __future__ import annotations

import random
import warnings
from pathlib import Path
from typing import Sequence, TypeVar

T = TypeVar("T")


def _split_items(
    items: Sequence[T],
    train: float,
    val: float,
    test: float,
    seed: int | None,
) -> tuple[list[T], list[T], list[T]]:
    """
    Random split keeping items intact.

    Args:
        items: Sequence of items to split.
        train: Fraction for training set.
        val: Fraction for validation set.
        test: Fraction for test set.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_items, val_items, test_items).
    """
    items_list = list(items)
    n = len(items_list)

    if n == 0:
        return [], [], []

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

    return train_items, val_items, test_items


def _split_by_clusters(
    items: Sequence[T],
    labels: Sequence[int],
    train: float,
    val: float,
    test: float,
    seed: int | None,
) -> tuple[list[T], list[T], list[T]]:
    """
    Split keeping all items with same label together.

    Args:
        items: Sequence of items to split.
        labels: Cluster label for each item.
        train: Fraction for training set.
        val: Fraction for validation set.
        test: Fraction for test set.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_items, val_items, test_items).
    """
    items_list = list(items)
    labels_list = list(labels)

    if len(items_list) == 0:
        return [], [], []

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

    return train_items, val_items, test_items


def _split_by_sequence_identity(
    paths: Sequence[Path],
    threshold: float,
    train: float,
    val: float,
    test: float,
    seed: int | None,
    coverage: float,
    threads: int,
) -> tuple[list[Path], list[Path], list[Path]]:
    """
    Cluster by sequence identity then split.

    Args:
        paths: Sequence of CIF/PDB file paths.
        threshold: Sequence identity threshold.
        train: Fraction for training set.
        val: Fraction for validation set.
        test: Fraction for test set.
        seed: Random seed for reproducibility.
        coverage: Minimum alignment coverage for clustering.
        threads: Number of threads for MMseqs2.

    Returns:
        Tuple of (train_paths, val_paths, test_paths).
    """
    from ciffy.operations.cluster import cluster

    result = cluster(
        paths,
        threshold=threshold,
        threads=threads,
        coverage=coverage,
    )

    return _split_by_clusters(
        result.paths,
        result.labels.tolist(),
        train=train,
        val=val,
        test=test,
        seed=seed,
    )
