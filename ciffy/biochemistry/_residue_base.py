"""
Base class for Residue namespace with static methods.

The generated _generated_residues_v2.py imports this and adds the actual residue members.
"""

from typing import ClassVar, Iterator, TYPE_CHECKING

if TYPE_CHECKING:
    from .atom import AtomGroup


class ResidueBase:
    """
    Namespace for all residue types.

    Each residue is an AtomGroup with metadata:

    Usage:
        Residue.A.value         # -> residue index (0)
        Residue.A.P             # -> Atom(P, 2) - IS an int
        Residue.A.P.local       # -> local atom position
        Residue.A.molecule_type # -> 1 (Molecule.RNA.value)
        Residue.A.ideal         # -> (n, 3) coordinates
        Residue.A.bonds         # -> (n, 2) bond pairs

        list(Residue.all())     # -> iterate all residues
        Residue.from_index(0)   # -> lookup by index
        Residue.from_name("A")  # -> lookup by name

        # Filtering for ML workflows
        Residue.A.subset({2, 3, 5})  # -> filtered AtomGroup
    """

    _members: ClassVar[dict[str, "AtomGroup"]] = {}
    _by_index: ClassVar[dict[int, "AtomGroup"]] = {}

    @classmethod
    def from_index(cls, idx: int) -> "AtomGroup":
        """Lookup residue by index."""
        return cls._by_index[idx]

    @classmethod
    def from_name(cls, name: str) -> "AtomGroup":
        """Lookup residue by name."""
        return cls._members[name]

    @classmethod
    def all(cls) -> Iterator["AtomGroup"]:
        """Iterate over all residue types."""
        return iter(cls._members.values())

    @classmethod
    def count(cls) -> int:
        """Number of residue types."""
        return len(cls._members)

    @classmethod
    def _init_members(cls) -> None:
        """Populate lookup tables from class attributes. Called after generation."""
        from .atom import AtomGroup
        for name in dir(cls):
            if name.startswith("_"):
                continue
            attr = getattr(cls, name)
            if isinstance(attr, AtomGroup) and attr.value is not None:
                cls._members[name] = attr
                cls._by_index[attr.value] = attr
