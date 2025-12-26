"""
Reusable train/validation/test splitting for molecular structure data.

Provides utilities for splitting datasets by structure (file) to avoid data
leakage. Residues from the same structure are correlated and should not appear
in both training and test sets.

Example:
    >>> from ciffy.nn.split import DataSplit
    >>> from pathlib import Path
    >>>
    >>> # Get all CIF files
    >>> cif_files = list(Path("data/").glob("*.cif"))
    >>>
    >>> # Create 80/10/10 split
    >>> split = DataSplit.from_paths(cif_files, train=0.8, val=0.1, test=0.1, seed=42)
    >>> print(f"Train: {len(split.train)}, Val: {len(split.val)}, Test: {len(split.test)}")
    >>>
    >>> # Use in training
    >>> train_data = load_data(split.train)
    >>> val_data = load_data(split.val)
    >>> test_data = load_data(split.test)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar, Generic, Sequence, Hashable
import random

T = TypeVar("T")


@dataclass
class DataSplit(Generic[T]):
    """
    Container for train/validation/test splits.

    Supports splitting any sequence of items (paths, indices, IDs) while
    maintaining reproducibility via a seed.

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

    def __post_init__(self):
        """Validate no overlap between splits."""
        if self._has_overlap():
            raise ValueError("Splits have overlapping items")

    def _has_overlap(self) -> bool:
        """Check if any items appear in multiple splits."""
        # Convert to hashable if possible for set operations
        try:
            train_set = set(self.train)
            val_set = set(self.val)
            test_set = set(self.test)
            return bool(
                train_set & val_set or train_set & test_set or val_set & test_set
            )
        except TypeError:
            # Items not hashable, skip validation
            return False

    @classmethod
    def from_items(
        cls,
        items: Sequence[T],
        train: float = 0.8,
        val: float = 0.1,
        test: float = 0.1,
        seed: int | None = 42,
    ) -> "DataSplit[T]":
        """
        Create a split from a sequence of items.

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
            >>> split = DataSplit.from_items(paths, train=0.8, val=0.1, test=0.1)
            >>> len(split.train), len(split.val), len(split.test)
            (80, 10, 10)
        """
        # Validate ratios
        if any(r < 0 for r in [train, val, test]):
            raise ValueError("Split ratios must be non-negative")

        total = train + val + test
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")

        # Convert to list for shuffling
        items_list = list(items)
        n = len(items_list)

        if n == 0:
            return cls(train=[], val=[], test=[], seed=seed)

        # Shuffle with seed
        if seed is not None:
            rng = random.Random(seed)
            rng.shuffle(items_list)
        else:
            random.shuffle(items_list)

        # Compute split indices
        n_train = int(n * train)
        n_val = int(n * val)
        # Test gets the remainder to ensure all items are used

        train_items = items_list[:n_train]
        val_items = items_list[n_train : n_train + n_val]
        test_items = items_list[n_train + n_val :]

        return cls(train=train_items, val=val_items, test=test_items, seed=seed)

    @classmethod
    def from_paths(
        cls,
        paths: Sequence[str | Path],
        train: float = 0.8,
        val: float = 0.1,
        test: float = 0.1,
        seed: int | None = 42,
    ) -> "DataSplit[Path]":
        """
        Create a split from file paths.

        Convenience method that converts strings to Path objects.

        Args:
            paths: Sequence of file paths (as strings or Path objects).
            train: Fraction for training set.
            val: Fraction for validation set.
            test: Fraction for test set.
            seed: Random seed for reproducibility.

        Returns:
            DataSplit with Path objects.

        Example:
            >>> from glob import glob
            >>> cif_files = glob("data/*.cif")
            >>> split = DataSplit.from_paths(cif_files, train=0.8, val=0.1, test=0.1)
        """
        path_list = [Path(p) if isinstance(p, str) else p for p in paths]
        return cls.from_items(path_list, train=train, val=val, test=test, seed=seed)

    @classmethod
    def train_test(
        cls,
        items: Sequence[T],
        train: float = 0.8,
        seed: int | None = 42,
    ) -> "DataSplit[T]":
        """
        Create a simple train/test split (no validation set).

        Args:
            items: Sequence of items to split.
            train: Fraction for training set (default: 0.8).
            seed: Random seed for reproducibility.

        Returns:
            DataSplit with train and test lists (val is empty).

        Example:
            >>> split = DataSplit.train_test(paths, train=0.9)
            >>> len(split.val)  # No validation set
            0
        """
        return cls.from_items(items, train=train, val=0.0, test=1.0 - train, seed=seed)

    def __len__(self) -> int:
        """Total number of items across all splits."""
        return len(self.train) + len(self.val) + len(self.test)

    def summary(self) -> str:
        """Return a summary string of the split."""
        total = len(self)
        if total == 0:
            return "DataSplit(empty)"
        return (
            f"DataSplit(train={len(self.train)} ({len(self.train)/total:.1%}), "
            f"val={len(self.val)} ({len(self.val)/total:.1%}), "
            f"test={len(self.test)} ({len(self.test)/total:.1%}), "
            f"seed={self.seed})"
        )

    def __repr__(self) -> str:
        return self.summary()


def split_by_structure(
    paths: Sequence[str | Path],
    train: float = 0.8,
    val: float = 0.1,
    test: float = 0.1,
    seed: int | None = 42,
) -> DataSplit[Path]:
    """
    Split CIF files into train/val/test sets by structure.

    This is the recommended way to split molecular structure data to avoid
    data leakage. Residues from the same structure are correlated (same
    crystallographic conditions, similar conformations) and should not
    appear in both training and test sets.

    Args:
        paths: Sequence of CIF file paths.
        train: Fraction for training set (default: 0.8).
        val: Fraction for validation set (default: 0.1).
        test: Fraction for test set (default: 0.1).
        seed: Random seed for reproducibility (default: 42).

    Returns:
        DataSplit containing Path objects for each split.

    Example:
        >>> from ciffy.nn.split import split_by_structure
        >>> from glob import glob
        >>>
        >>> cif_files = glob("data/*.cif")
        >>> split = split_by_structure(cif_files, train=0.8, val=0.1, test=0.1)
        >>>
        >>> # Train only on training structures
        >>> trainer.train(split.train)
        >>>
        >>> # Evaluate on held-out test set
        >>> metrics = trainer.evaluate(split.test)
    """
    return DataSplit.from_paths(paths, train=train, val=val, test=test, seed=seed)


@dataclass
class DataScalingSplit(Generic[T]):
    """
    Container for data scaling experiments with consistent test set.

    For experiments comparing model performance across different training
    set sizes, this ensures all experiments use the same held-out test set.
    Training subsets are nested: smaller sizes are prefixes of the shuffled
    training pool.

    Attributes:
        train_pool: Shuffled training items (request any prefix via get_train).
        test: Test set items (same for all experiments).
        seed: Random seed used for splitting.

    Example:
        >>> scaling = DataScalingSplit.from_items(
        ...     items=cif_files,
        ...     test_fraction=0.2,
        ...     seed=42,
        ... )
        >>> # Get any training size up to max
        >>> train_50 = scaling.get_train(50)
        >>> train_200 = scaling.get_train(200)
        >>> # Smaller sets are always prefixes of larger ones
        >>> assert train_50 == train_200[:50]
    """

    train_pool: list[T]
    test: list[T]
    seed: int | None = None

    @classmethod
    def from_items(
        cls,
        items: Sequence[T],
        test_fraction: float = 0.2,
        seed: int | None = 42,
    ) -> "DataScalingSplit[T]":
        """
        Create a data scaling split with shuffled training pool and fixed test set.

        Args:
            items: Full dataset to split.
            test_fraction: Fraction of data for test set (default: 0.2).
            seed: Random seed for reproducibility.

        Returns:
            DataScalingSplit with shuffled train_pool and fixed test set.

        Example:
            >>> scaling = DataScalingSplit.from_items(
            ...     items=list(range(1000)),
            ...     test_fraction=0.2,
            ... )
            >>> len(scaling.test)  # 200 items
            200
            >>> len(scaling.train_pool)  # 800 items
            800
            >>> scaling.get_train(100)  # First 100 of shuffled pool
        """
        items_list = list(items)
        n_total = len(items_list)

        if n_total == 0:
            return cls(train_pool=[], test=[], seed=seed)

        # Shuffle deterministically
        if seed is not None:
            rng = random.Random(seed)
            rng.shuffle(items_list)
        else:
            random.shuffle(items_list)

        # Split into train pool and test set
        n_test = int(n_total * test_fraction)

        train_pool = items_list[:n_total - n_test]
        test_items = items_list[n_total - n_test:]

        return cls(
            train_pool=train_pool,
            test=test_items,
            seed=seed,
        )

    @classmethod
    def from_paths(
        cls,
        paths: Sequence[str | Path],
        test_fraction: float = 0.2,
        seed: int | None = 42,
    ) -> "DataScalingSplit[Path]":
        """
        Create a data scaling split from file paths.

        Args:
            paths: Sequence of file paths.
            test_fraction: Fraction of data for test set.
            seed: Random seed for reproducibility.

        Returns:
            DataScalingSplit with Path objects.
        """
        path_list = [Path(p) if isinstance(p, str) else p for p in paths]
        return cls.from_items(path_list, test_fraction, seed)

    def get_train(self, size: int) -> list[T]:
        """
        Get training subset of specified size.

        Returns the first `size` items from the shuffled training pool.
        Smaller sizes are always prefixes of larger sizes.

        Args:
            size: Number of training items to return.

        Returns:
            List of training items.

        Raises:
            ValueError: If size exceeds available training data.
        """
        if size > len(self.train_pool):
            raise ValueError(
                f"Requested size {size} exceeds available training data {len(self.train_pool)}"
            )
        return self.train_pool[:size]

    @property
    def max_train_size(self) -> int:
        """Maximum available training set size."""
        return len(self.train_pool)

    def summary(self) -> str:
        """Return a summary string."""
        return (
            f"DataScalingSplit(train_pool={len(self.train_pool)}, "
            f"test={len(self.test)}, seed={self.seed})"
        )

    def __repr__(self) -> str:
        return self.summary()


def create_scaling_split(
    paths: Sequence[str | Path],
    test_fraction: float = 0.2,
    seed: int | None = 42,
) -> DataScalingSplit[Path]:
    """
    Create a data scaling split for comparing performance across training set sizes.

    This is the recommended way to compare model performance across different
    training set sizes. The test set is fixed across all experiments, and
    smaller training sets are prefixes of the shuffled training pool.

    Args:
        paths: Sequence of file paths (e.g., CIF files).
        test_fraction: Fraction of data for test set (default: 0.2).
        seed: Random seed for reproducibility (default: 42).

    Returns:
        DataScalingSplit with shuffled training pool.

    Example:
        >>> from ciffy.nn.split import create_scaling_split
        >>> from glob import glob
        >>>
        >>> cif_files = glob("data/*.cif")
        >>> scaling = create_scaling_split(cif_files, test_fraction=0.2)
        >>>
        >>> # Run experiments with consistent test set
        >>> for size in [50, 100, 200, 500]:
        ...     train_paths = scaling.get_train(size)
        ...     model = train(train_paths)
        ...     metrics = evaluate(model, scaling.test)
        ...     print(f"Size {size}: {metrics}")
    """
    return DataScalingSplit.from_paths(paths, test_fraction, seed)


__all__ = [
    "DataSplit",
    "DataScalingSplit",
    "split_by_structure",
    "create_scaling_split",
]
