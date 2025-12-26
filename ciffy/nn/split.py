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


__all__ = [
    "DataSplit",
    "split_by_structure",
]
