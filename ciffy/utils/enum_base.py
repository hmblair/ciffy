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
        seen = set()
        for x, y in itertools.product(cls, cls):
            # Use sorted values as canonical key for unordered pair
            key = (min(x.value, y.value), max(x.value, y.value))
            if key not in seen:
                seen.add(key)
                pairs.append((x, y))

        return PairEnum(pairs)


class ResidueType:
    """
    A residue definition with index, metadata, and atom access.

    Enables nested access pattern:
        Residue.A.value         # → residue index
        Residue.A.C3p.value     # → atom index
        Residue.A.molecule_type # → Molecule.RNA
        Residue.A.bonds         # → bond list

    Attributes:
        value: Residue index (for embedding layers).
        name: Residue name (e.g., 'A', 'ALA').
        atoms: The atom enum class for this residue.
        molecule_type: Molecule type (Molecule.RNA, etc.).
        abbrev: Single-letter abbreviation.
    """

    __slots__ = ('_name', '_index', '_atoms', '_molecule_type', '_abbrev')

    def __init__(
        self,
        name: str,
        index: int,
        atoms: type[IndexEnum],
        molecule_type: int,
        abbrev: str,
    ) -> None:
        self._name = name
        self._index = index
        self._atoms = atoms
        self._molecule_type = molecule_type
        self._abbrev = abbrev

    @property
    def value(self) -> int:
        """Residue index (for embedding layers)."""
        return self._index

    @property
    def name(self) -> str:
        """Residue name (e.g., 'A', 'ALA')."""
        return self._name

    @property
    def atoms(self) -> type[IndexEnum]:
        """The atom enum class for this residue."""
        return self._atoms

    @property
    def molecule_type(self) -> int:
        """Molecule type (Molecule.RNA, Molecule.PROTEIN, etc.)."""
        return self._molecule_type

    @property
    def abbrev(self) -> str:
        """Single-letter abbreviation."""
        return self._abbrev

    @property
    def bond_indices(self):
        """
        (M, 2) int32 array of bonded atom pairs (global indices).

        Returns None if no bonds are defined for this residue.
        """
        return getattr(self._atoms, 'bond_indices', None)

    @property
    def dihedral_patterns(self) -> dict:
        """
        Dict[int -> np.ndarray] mapping dihedral type index to local atom indices.

        Maps integer dihedral type (0-10) to (4,) int32 array of local atom indices.
        Local indices of -1 indicate atoms in adjacent residues.

        Returns empty dict if no dihedral patterns are defined.
        """
        return getattr(self._atoms, 'dihedral_patterns', {})

    def __getattr__(self, name: str):
        """Delegate attribute access to atom enum."""
        return getattr(self._atoms, name)

    def __iter__(self):
        """Iterate over atoms."""
        return iter(self._atoms)

    def __len__(self) -> int:
        """Number of atoms in this residue."""
        return len(list(self._atoms))

    def __repr__(self) -> str:
        return f"Residue.{self._name}"


class ResidueMeta(type):
    """
    Metaclass enabling iteration and reverse lookup on Residue class.

    Enables:
        list(Residue)     # → iterate over all residues
        Residue(0)        # → reverse lookup by index
        Residue["A"]      # → lookup by name
        len(Residue)      # → number of residues
        "A" in Residue    # → containment check
    """

    _members: dict[str, ResidueType]
    _by_index: dict[int, ResidueType]

    def __iter__(cls):
        """Iterate over all residue types."""
        return iter(cls._members.values())

    def __len__(cls) -> int:
        """Number of residue types."""
        return len(cls._members)

    def __call__(cls, index: int) -> ResidueType:
        """Reverse lookup by index."""
        if index not in cls._by_index:
            raise ValueError(f"No residue with index {index}")
        return cls._by_index[index]

    def __getitem__(cls, name: str) -> ResidueType:
        """Lookup by name."""
        if name not in cls._members:
            raise KeyError(f"No residue named '{name}'")
        return cls._members[name]

    def __contains__(cls, item) -> bool:
        """Check if name or ResidueType is in Residue."""
        if isinstance(item, str):
            return item in cls._members
        if isinstance(item, ResidueType):
            return item in cls._members.values()
        return False
