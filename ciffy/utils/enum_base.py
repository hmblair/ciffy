"""
Base enum classes with tensor conversion capabilities.

Provides IndexEnum for enums that map to integer indices and PairEnum
for storing pairs of enum values with tensor conversion.
"""

from __future__ import annotations
from enum import Enum
import itertools
import torch


class PairEnum(list):
    """
    Store a set of pairs of atom enums with tensor conversion capabilities.

    Useful for representing bonds or other pairwise relationships between
    enum values. Provides methods to convert pairs to tensor indices and
    to create pairwise lookup tables.

    Example:
        >>> bonds = PairEnum([(Atom.C, Atom.O), (Atom.C, Atom.N)])
        >>> bonds.indices()
        tensor([[6, 8], [6, 7]])
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

    def indices(self: PairEnum) -> torch.Tensor:
        """
        Convert pairs to a tensor of their integer values.

        Returns:
            Tensor of shape (N, 2) where N is the number of pairs.
        """
        return torch.tensor([
            [atom1.value, atom2.value]
            for atom1, atom2 in self
        ])

    def pairwise(self: PairEnum) -> torch.Tensor:
        """
        Create a symmetric lookup table for pair indices.

        Returns:
            Square tensor where entry [i,j] contains the pair index
            for atoms with values i and j, or -1 if no such pair exists.
        """
        n = self.indices().max() + 1
        table = torch.ones(n, n, dtype=torch.long) * -1

        for ix, (x, y) in enumerate(self):
            table[x.value, y.value] = ix
            table[y.value, x.value] = ix

        return table


class IndexEnum(Enum):
    """
    An enum with tensor conversion capabilities.

    Extends standard Enum with methods to convert enum values to tensors,
    lists, and dictionaries. Useful for biochemistry constants where enum
    values represent atom indices.

    Example:
        >>> class Element(IndexEnum):
        ...     C = 6
        ...     N = 7
        ...     O = 8
        >>> Element.index()
        tensor([6, 7, 8])
        >>> Element.dict()
        {'C': 6, 'N': 7, 'O': 8}
    """

    @classmethod
    def index(cls: type[IndexEnum]) -> torch.Tensor:
        """
        Return a tensor of all enum values.

        Returns:
            Long tensor containing all integer values in the enum.
        """
        return torch.tensor([
            atom.value for atom in cls
        ]).long()

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
