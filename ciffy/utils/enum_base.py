"""
Base enum classes with array conversion capabilities.

Provides IndexEnum for enums that map to integer indices and PairEnum
for storing pairs of enum values with array conversion.
"""

from __future__ import annotations
from enum import Enum
import itertools
import numpy as np


class PairEnum(list):
    """
    Store a set of pairs of atom enums with array conversion capabilities.

    Useful for representing bonds or other pairwise relationships between
    enum values. Provides methods to convert pairs to array indices and
    to create pairwise lookup tables.

    Example:
        >>> bonds = PairEnum([(Atom.C, Atom.O), (Atom.C, Atom.N)])
        >>> bonds.indices()
        array([[6, 8], [6, 7]])
    """

    def __init__(
        self: PairEnum,
        bonds: list[tuple[Enum, Enum]],
    ) -> None:
        super().__init__(bonds)

    def __add__(
        self: PairEnum,
        other: list,
    ) -> PairEnum:
        return self.__class__(super().__add__(other))

    def indices(self: PairEnum) -> np.ndarray:
        """
        Convert pairs to an array of their integer values.

        Returns:
            Array of shape (N, 2) where N is the number of pairs.
        """
        return np.array([
            [atom1.value, atom2.value]
            for atom1, atom2 in self
        ], dtype=np.int64)

    def pairwise(self: PairEnum) -> np.ndarray:
        """
        Create a symmetric lookup table for pair indices.

        Returns:
            Square array where entry [i,j] contains the pair index
            for atoms with values i and j, or -1 if no such pair exists.
        """
        n = self.indices().max() + 1
        table = np.full((n, n), -1, dtype=np.int64)

        for ix, (x, y) in enumerate(self):
            table[x.value, y.value] = ix
            table[y.value, x.value] = ix

        return table


class IndexEnum(Enum):
    """
    An enum with array conversion capabilities.

    Extends standard Enum with methods to convert enum values to arrays,
    lists, and dictionaries. Useful for biochemistry constants where enum
    values represent atom indices.

    Example:
        >>> class Element(IndexEnum):
        ...     C = 6
        ...     N = 7
        ...     O = 8
        >>> Element.index()
        array([6, 7, 8])
        >>> Element.dict()
        {'C': 6, 'N': 7, 'O': 8}
    """

    @classmethod
    def index(cls: type[IndexEnum]) -> np.ndarray:
        """
        Return an array of all enum values.

        Returns:
            Integer array containing all values in the enum.
        """
        return np.array([
            atom.value for atom in cls
        ], dtype=np.int64)

    @classmethod
    def list(
        cls: type[IndexEnum],
        modifier: str = '',
    ) -> list[str]:
        """
        Return enum names as a list.

        Args:
            modifier: Optional prefix to add to each name.

        Returns:
            List of enum names, optionally with prefix.
        """
        return [
            modifier + field.name
            for field in cls
        ]

    @classmethod
    def dict(
        cls: type[IndexEnum],
        modifier: str = '',
    ) -> dict[str, int]:
        """
        Return the enum as a name-to-value dictionary.

        Args:
            modifier: Optional prefix to add to each name.

        Returns:
            Dictionary mapping names to integer values.
        """
        return {
            modifier + field.name: field.value
            for field in cls
        }

    @classmethod
    def revdict(
        cls: type[IndexEnum],
        modifier: str = '',
    ) -> dict[int, str]:
        """
        Return the enum as a value-to-name dictionary.

        Args:
            modifier: Optional prefix to add to each name.

        Returns:
            Dictionary mapping integer values to names.
        """
        return {
            field.value: modifier + field.name
            for field in cls
        }

    @classmethod
    def pairs(cls: type[IndexEnum]) -> PairEnum:
        """
        Return all unique pairs of enum values.

        Pairs are unordered, so (A, B) and (B, A) are considered the same
        and only one is included.

        Returns:
            PairEnum containing all unique pairs.
        """
        pairs = []
        for x, y in itertools.product(cls, cls):
            if (x, y) not in pairs and (y, x) not in pairs:
                pairs.append((x, y))

        return PairEnum(pairs)
