"""
Atom and AtomGroup classes for biochemistry entity representation.

Minimal 2-class design:
- Atom: int subclass with name and local position (self-documenting, works directly in arrays)
- AtomGroup: unified container with optional geometry (handles residues, groups, and subsets)

Example:
    >>> from ciffy.biochemistry import Residue
    >>> Residue.A.P            # Atom(P, 2) - IS an int
    >>> Residue.A.P == 2       # True
    >>> array[Residue.A.P]     # Works directly, no .value needed
    >>> Residue.A.P.name       # "P" (for debugging)
    >>> Residue.A.P.local      # 0 (position in residue)

    >>> Residue.A.ideal        # (37, 3) float32 array
    >>> Residue.A.bonds        # (n, 2) int64 array

    >>> subset = Residue.A.subset({2, 3, 5})
    >>> subset.ideal           # (3, 3) filtered coords
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Iterator

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def atom_to_element(atom: Atom | str) -> int:
    """
    Get the element atomic number for an atom from its name.

    Parses the atom name to determine the element. For standard biological
    macromolecule atoms (proteins, RNA, DNA), the element is typically the
    first character of the name.

    Args:
        atom: An Atom object or atom name string (e.g., "CA", "P", "O3'").

    Returns:
        Atomic number of the element (e.g., 6 for carbon, 7 for nitrogen).
        Returns 0 if the element cannot be determined.

    Example:
        >>> atom_to_element("CA")  # Alpha carbon
        6
        >>> atom_to_element("P")   # Phosphorus
        15
        >>> atom_to_element(Residue.A.N1)  # Nitrogen
        7
    """
    # Import here to avoid circular dependency
    from ._generated_elements import Element

    name = atom.name if isinstance(atom, Atom) else atom
    if not name:
        return 0

    first_char = name[0].upper()

    # Map first character to element
    # These are the elements found in standard biological macromolecules
    element_map = {
        'H': Element.H,
        'C': Element.C,
        'N': Element.N,
        'O': Element.O,
        'P': Element.P,
        'S': Element.S,
    }

    element = element_map.get(first_char)
    return element.value if element is not None else 0


class Atom(int):
    """
    An int that knows its name and local position.

    Subclasses int so atoms work directly in array indexing and comparisons,
    while remaining self-documenting for debugging.

    Attributes:
        name: Atom name (e.g., "P", "C3p", "CA").
        local: 0-indexed position within the residue for array access.

    Example:
        >>> atom = Atom("P", 2, 0)
        >>> atom == 2           # True (IS an int)
        >>> array[atom]         # Works directly
        >>> int(atom)           # 2
        >>> atom.name           # "P"
        >>> atom.local          # 0
        >>> repr(atom)          # "Atom(P, 2)"
    """

    # Note: int subclasses can't use __slots__, attributes stored in __dict__

    def __new__(cls, name: str, value: int, local: int = 0) -> Atom:
        obj = super().__new__(cls, value)
        obj.name = name
        obj.local = local
        return obj

    def __repr__(self) -> str:
        return f"Atom({self.name}, {int(self)})"

    # Preserve name/local through arithmetic (though rarely needed)
    def __add__(self, other: int) -> int:
        return int.__add__(int(self), other)

    def __radd__(self, other: int) -> int:
        return int.__add__(other, int(self))

    def __sub__(self, other: int) -> int:
        return int.__sub__(int(self), other)

    def __rsub__(self, other: int) -> int:
        return int.__sub__(other, int(self))

    # For compatibility with code that uses .value
    @property
    def value(self) -> int:
        """The integer value (for compatibility). Prefer using the Atom directly."""
        return int(self)

    # Class-level cache for from_value lookup
    _registry: dict[int, Atom] | None = None

    @classmethod
    def from_value(cls, value: int) -> Atom:
        """
        Get an Atom by its unique integer value.

        Args:
            value: The atom's unique integer value.

        Returns:
            The Atom with that value.

        Raises:
            KeyError: If no atom with that value exists.

        Example:
            >>> Atom.from_value(2)
            Atom(P, 2)
        """
        if cls._registry is None:
            # Lazy-load registry from ATOM_NAMES
            from ._generated_atoms import ATOM_NAMES
            cls._registry = {v: cls(name, v) for v, name in ATOM_NAMES.items()}

        if value not in cls._registry:
            raise KeyError(f"No atom with value {value}")
        return cls._registry[value]

    @classmethod
    def count(cls) -> int:
        """
        Return the vocabulary size for atoms (max value + 1).

        Use this for embedding layer dimensions.
        """
        if cls._registry is None:
            from ._generated_atoms import ATOM_NAMES
            cls._registry = {v: cls(name, v) for v, name in ATOM_NAMES.items()}
        return max(cls._registry.keys()) + 1


class AtomGroup:
    """
    Container for atoms, optionally with geometry.

    Unified class that handles:
    - Residues: AtomGroup with value, molecule_type, ideal, bonds
    - Groups (Sugar, etc.): AtomGroup without geometry
    - Subsets: AtomGroup with filtered geometry

    Supports lazy loading: if `loader` is provided instead of `members`,
    data is loaded on first access.

    Attributes:
        name: Group name (e.g., "A", "Sugar", "A_subset").
        value: Residue index (None for plain groups).
        molecule_type: Molecule type index (None for plain groups).
        abbrev: Single-letter abbreviation (None for plain groups).
        ideal: Ideal coordinates, shape (n_atoms, 3) float32 (None if no geometry).
        bonds: Bond pairs, shape (n_bonds, 2) int64 (None if no geometry).

    Example:
        >>> # Residue with geometry
        >>> Residue.A.P          # Atom(P, 2)
        >>> Residue.A.value      # 0 (residue index)
        >>> Residue.A.ideal      # (37, 3) array
        >>> Residue.A.bonds      # (n, 2) array

        >>> # Hierarchical group without geometry
        >>> Sugar.C5p.A          # Atom(C5p, 6)
        >>> Sugar.C5p.index()    # array([6, 43, 152, ...])

        >>> # Subset with filtered geometry
        >>> subset = Residue.A.subset({2, 3, 5})
        >>> subset.ideal         # (3, 3) filtered
    """

    __slots__ = (
        'name', '_members', '_index_cache',
        'value', 'molecule_type', 'abbrev', '_ideal', '_bonds',
        '_loader',  # Lazy loader function
        '_orig_locals',  # For subset geometry filtering
    )

    def __init__(
        self,
        name: str,
        members: dict[str, Atom | AtomGroup] | None = None,
        *,
        value: int | None = None,
        molecule_type: int | None = None,
        abbrev: str | None = None,
        ideal: NDArray[np.float32] | None = None,
        bonds: NDArray[np.int64] | None = None,
        loader: Callable[[], tuple[dict, NDArray | None, NDArray | None]] | None = None,
        _orig_locals: dict[int, int] | None = None,
    ) -> None:
        """
        Create an AtomGroup.

        Args:
            name: Group name.
            members: Dict mapping names to Atom or nested AtomGroup.
                If None and loader is provided, will be loaded lazily.
            value: Residue index (for residues only).
            molecule_type: Molecule type (for residues only).
            abbrev: Single-letter abbreviation (for residues only).
            ideal: Ideal coordinates, (n_atoms, 3) float32.
            bonds: Bond pairs, (n_bonds, 2) int64.
            loader: Optional function returning (members, ideal, bonds).
                Used for lazy loading - data loaded on first access.
            _orig_locals: Internal - maps atom value to original local index.
        """
        self.name = name
        self._members = members
        self._index_cache: NDArray[np.int64] | None = None
        self.value = value
        self.molecule_type = molecule_type
        self.abbrev = abbrev
        self._ideal = ideal
        self._bonds = bonds
        self._loader = loader
        self._orig_locals = _orig_locals

    def _ensure_loaded(self) -> None:
        """Load data if using lazy loading."""
        if self._members is None and self._loader is not None:
            members, ideal, bonds = self._loader()
            self._members = members
            self._ideal = ideal
            self._bonds = bonds
            self._loader = None  # Clear loader after use

    @property
    def ideal(self) -> NDArray[np.float32] | None:
        """Ideal coordinates, shape (n_atoms, 3)."""
        self._ensure_loaded()
        return self._ideal

    @property
    def bonds(self) -> NDArray[np.int64] | None:
        """Bond pairs, shape (n_bonds, 2)."""
        self._ensure_loaded()
        return self._bonds

    def __getattr__(self, name: str) -> Atom | AtomGroup:
        """Access member by name: group.P -> Atom or nested AtomGroup."""
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
        # Ensure data is loaded before accessing members
        self._ensure_loaded()
        members = object.__getattribute__(self, '_members')
        if members is not None and name in members:
            return members[name]
        group_name = object.__getattribute__(self, 'name')
        raise AttributeError(f"'{group_name}' has no member '{name}'")

    def index(self) -> NDArray[np.int64]:
        """
        All atom values as a sorted array.

        Returns:
            Sorted array of unique atom values.
        """
        if self._index_cache is None:
            self._ensure_loaded()
            values = []
            for v in self._members.values():
                if isinstance(v, AtomGroup):
                    values.extend(v.index().tolist())
                elif isinstance(v, Atom):
                    values.append(int(v))
            self._index_cache = np.array(sorted(set(values)), dtype=np.int64)
        return self._index_cache

    def __iter__(self) -> Iterator[Atom]:
        """Iterate over Atom members (skips nested AtomGroups)."""
        self._ensure_loaded()
        for v in self._members.values():
            if isinstance(v, Atom):
                yield v

    def __len__(self) -> int:
        """Number of direct members."""
        self._ensure_loaded()
        return len(self._members)

    def __contains__(self, item: str | int | Atom) -> bool:
        """Check if name, value, or Atom is in group."""
        self._ensure_loaded()
        if isinstance(item, str):
            return item in self._members
        if isinstance(item, int):
            return item in set(self.index())
        if isinstance(item, Atom):
            return int(item) in set(self.index())
        return False

    def __getitem__(self, value: int) -> Atom:
        """
        Get atom by its integer value.

        Args:
            value: The atom's integer value.

        Returns:
            The Atom with that value.

        Raises:
            KeyError: If no atom with that value exists.

        Example:
            >>> Residue.A[2]  # Get atom with value 2
            Atom(P, 2)
        """
        self._ensure_loaded()
        for member in self._members.values():
            if isinstance(member, Atom) and int(member) == value:
                return member
            if isinstance(member, AtomGroup):
                try:
                    return member[value]
                except KeyError:
                    pass
        raise KeyError(f"No atom with value {value} in {self.name}")

    def __repr__(self) -> str:
        # Collect atoms for display
        atoms = [(name, int(atom)) for name, atom in self.items()]
        n_atoms = len(atoms)

        if self.value is not None:
            # Residue - show truncated atom list
            if n_atoms <= 6:
                atom_str = ", ".join(f"{name}={val}" for name, val in atoms)
            else:
                shown = atoms[:3] + atoms[-3:]
                atom_str = ", ".join(f"{name}={val}" for name, val in shown[:3])
                atom_str += f", ..., "
                atom_str += ", ".join(f"{name}={val}" for name, val in shown[3:])
            return f"Residue.{self.name}({atom_str})"

        # Plain AtomGroup
        self._ensure_loaded()
        n_groups = sum(1 for v in self._members.values() if isinstance(v, AtomGroup))
        if n_atoms <= 8:
            atom_str = ", ".join(f"{name}={val}" for name, val in atoms)
        else:
            atom_str = ", ".join(f"{name}={val}" for name, val in atoms[:4])
            atom_str += ", ..."

        if n_groups > 0:
            return f"AtomGroup({self.name}, {n_groups} subgroups, {{{atom_str}}})"
        return f"AtomGroup({self.name}, {{{atom_str}}})"

    def items(self) -> Iterator[tuple[str, Atom]]:
        """Iterate over (name, Atom) pairs (skips nested AtomGroups)."""
        self._ensure_loaded()
        for k, v in self._members.items():
            if isinstance(v, Atom):
                yield k, v

    def dict(self) -> dict[str, int]:
        """Name -> value mapping for leaf atoms."""
        return {k: int(v) for k, v in self.items()}

    def list(self) -> list[str]:
        """List of atom names."""
        self._ensure_loaded()
        return list(self._members.keys())

    @property
    def atoms(self) -> "AtomGroup":
        """Return self for backward compatibility with ResidueType.atoms pattern."""
        return self

    @property
    def n_atoms(self) -> int:
        """Number of atoms (leaf members only)."""
        return sum(1 for _ in self)

    @property
    def n_bonds(self) -> int:
        """Number of bonds (0 if no geometry)."""
        return 0 if self.bonds is None else len(self.bonds)

    def elements(self) -> NDArray[np.int64]:
        """
        Get element atomic numbers for all atoms.

        Returns:
            (n_atoms,) int64 array of element atomic numbers, sorted by atom value
            to match the order returned by index().

        Example:
            >>> Residue.A.elements()
            array([15, 8, 8, 8, 8, ...])  # P=15, O=8, C=6, N=7
        """
        result = []
        # Sort by value (int) to match index() order
        for atom in sorted(self, key=lambda a: int(a)):
            result.append(atom_to_element(atom))
        return np.array(result, dtype=np.int64)

    @property
    def bond_lengths(self) -> NDArray[np.float64] | None:
        """
        Compute ideal bond lengths from stored geometry.

        Returns:
            (n_bonds,) float64 array, or None if no geometry.
        """
        if self.ideal is None or self.bonds is None or len(self.bonds) == 0:
            return None
        lengths = np.linalg.norm(
            self.ideal[self.bonds[:, 0]] - self.ideal[self.bonds[:, 1]],
            axis=1
        )
        return lengths.astype(np.float64)

    @property
    def angles(self) -> NDArray[np.int64] | None:
        """
        Extract angle triplets (A-B-C) from bond connectivity.

        An angle is formed by any two bonds sharing a common atom (the vertex B).
        Returns triplets where B is the central/vertex atom.

        Returns:
            (n_angles, 3) int64 array of local indices [A, B, C], or None.
        """
        if self.bonds is None or len(self.bonds) < 2:
            return None

        bonds = self.bonds  # (n_bonds, 2)
        n_bonds = len(bonds)

        # Create all bond pairs (upper triangle to avoid duplicates)
        i, j = np.triu_indices(n_bonds, k=1)
        bonds_i = bonds[i]  # (n_pairs, 2)
        bonds_j = bonds[j]  # (n_pairs, 2)

        # Find shared atoms (4 possible matches per pair)
        # shared_XY means bonds_i[:, X] == bonds_j[:, Y]
        shared_00 = bonds_i[:, 0] == bonds_j[:, 0]
        shared_01 = bonds_i[:, 0] == bonds_j[:, 1]
        shared_10 = bonds_i[:, 1] == bonds_j[:, 0]
        shared_11 = bonds_i[:, 1] == bonds_j[:, 1]

        # Build angle triplets for each case: [other_i, shared, other_j]
        triplets = []

        # Case 00: shared at i[:,0] and j[:,0] -> [i[:,1], shared, j[:,1]]
        if shared_00.any():
            mask = shared_00
            triplets.append(np.stack([
                bonds_i[mask, 1],
                bonds_i[mask, 0],  # shared vertex
                bonds_j[mask, 1],
            ], axis=1))

        # Case 01: shared at i[:,0] and j[:,1] -> [i[:,1], shared, j[:,0]]
        if shared_01.any():
            mask = shared_01
            triplets.append(np.stack([
                bonds_i[mask, 1],
                bonds_i[mask, 0],  # shared vertex
                bonds_j[mask, 0],
            ], axis=1))

        # Case 10: shared at i[:,1] and j[:,0] -> [i[:,0], shared, j[:,1]]
        if shared_10.any():
            mask = shared_10
            triplets.append(np.stack([
                bonds_i[mask, 0],
                bonds_i[mask, 1],  # shared vertex
                bonds_j[mask, 1],
            ], axis=1))

        # Case 11: shared at i[:,1] and j[:,1] -> [i[:,0], shared, j[:,0]]
        if shared_11.any():
            mask = shared_11
            triplets.append(np.stack([
                bonds_i[mask, 0],
                bonds_i[mask, 1],  # shared vertex
                bonds_j[mask, 0],
            ], axis=1))

        if not triplets:
            return None

        return np.concatenate(triplets, axis=0).astype(np.int64)

    @property
    def n_angles(self) -> int:
        """Number of angles (0 if no geometry)."""
        angles = self.angles
        return 0 if angles is None else len(angles)

    @property
    def angle_values(self) -> NDArray[np.float64] | None:
        """
        Compute ideal angle values from stored geometry.

        Returns:
            (n_angles,) float64 array of angles in radians, or None.
        """
        angles = self.angles
        if angles is None or self.ideal is None:
            return None

        coords = self.ideal  # (n_atoms, 3)

        # Vectorized angle computation
        a_pos = coords[angles[:, 0]]  # (n_angles, 3)
        b_pos = coords[angles[:, 1]]  # (n_angles, 3) - vertex
        c_pos = coords[angles[:, 2]]  # (n_angles, 3)

        v1 = a_pos - b_pos  # B -> A
        v2 = c_pos - b_pos  # B -> C

        # Compute cos(angle) = (v1 · v2) / (|v1| |v2|)
        dot = (v1 * v2).sum(axis=1)
        norm1 = np.linalg.norm(v1, axis=1)
        norm2 = np.linalg.norm(v2, axis=1)
        cos_angles = dot / (norm1 * norm2 + 1e-8)

        return np.arccos(np.clip(cos_angles, -1, 1)).astype(np.float64)

    # =========================================================================
    # Filtering methods
    # =========================================================================

    def subset(self, indices: set[int]) -> AtomGroup:
        """
        Create subset containing only atoms with specified values.

        If this group has geometry (ideal, bonds), the subset will have
        filtered geometry with renumbered local indices.

        Args:
            indices: Set of global atom values to include.

        Returns:
            New AtomGroup with filtered atoms and geometry.
        """
        self._ensure_loaded()
        # Filter atoms with renumbered locals
        filtered = {}
        new_local = 0
        orig_locals: dict[int, int] = {}

        for k, v in self._members.items():
            if isinstance(v, Atom) and int(v) in indices:
                filtered[k] = Atom(v.name, int(v), new_local)
                # Track original local for geometry filtering
                if self._orig_locals is not None:
                    orig_locals[int(v)] = self._orig_locals[int(v)]
                else:
                    orig_locals[int(v)] = v.local
                new_local += 1

        # Filter geometry if present
        new_ideal = None
        new_bonds = None

        if self.ideal is not None and filtered:
            # Get original local indices in order of new locals
            ideal_rows = []
            for atom in sorted(filtered.values(), key=lambda a: a.local):
                ideal_rows.append(self.ideal[orig_locals[int(atom)]])
            new_ideal = np.array(ideal_rows, dtype=np.float32)

        if self.bonds is not None and filtered:
            # Bonds are stored as local indices - we need to:
            # 1. Map old local -> atom value (using orig_locals reverse)
            # 2. Check if both atoms are in filtered set
            # 3. Map to new local indices

            # Build old_local -> atom_value mapping
            old_local_to_value: dict[int, int] = {}
            for atom in self._members.values():
                if isinstance(atom, Atom):
                    if self._orig_locals is not None:
                        old_local_to_value[self._orig_locals[int(atom)]] = int(atom)
                    else:
                        old_local_to_value[atom.local] = int(atom)

            # Map atom values to new local indices
            val_to_new_local = {int(a): a.local for a in filtered.values()}

            new_bonds_list = []
            for old_local1, old_local2 in self.bonds:
                val1 = old_local_to_value.get(old_local1)
                val2 = old_local_to_value.get(old_local2)
                if val1 is not None and val2 is not None:
                    if val1 in val_to_new_local and val2 in val_to_new_local:
                        new_bonds_list.append([val_to_new_local[val1], val_to_new_local[val2]])
            if new_bonds_list:
                new_bonds = np.array(new_bonds_list, dtype=np.int64)

        return AtomGroup(
            f"{self.name}_subset",
            filtered,
            ideal=new_ideal,
            bonds=new_bonds,
            _orig_locals=orig_locals if self.ideal is not None else None,
        )

    def exclude(self, names: set[str]) -> AtomGroup:
        """
        Create subset excluding atoms with specified names.

        Args:
            names: Set of atom names to exclude.

        Returns:
            New AtomGroup with filtered atoms and geometry.
        """
        self._ensure_loaded()
        keep_indices = {int(v) for k, v in self._members.items()
                        if isinstance(v, Atom) and k not in names}
        return self.subset(keep_indices)

    def include(self, indices: set[int]) -> AtomGroup:
        """Alias for subset() for compatibility."""
        return self.subset(indices)

    def exclude_names(self, names: set[str]) -> AtomGroup:
        """Alias for exclude() for compatibility."""
        return self.exclude(names)

    def terminal(self, start: bool = True, end: bool = True) -> "AtomGroup":
        """
        Return AtomGroup with terminal atoms filtered.

        Args:
            start: Keep 5'/N-terminal atoms.
            end: Keep 3'/C-terminal atoms.

        Returns:
            Filtered AtomGroup, or self if no filtering needed.
        """
        if start and end:
            return self

        from ._generated_molecule import TERMINAL_ATOMS

        terminals = TERMINAL_ATOMS.get(self.molecule_type)
        if terminals is None:
            return self

        exclude = set()
        if not start:
            exclude |= terminals[0]
        if not end:
            exclude |= terminals[1]

        return self.exclude(exclude) if exclude else self

    def filter(self, predicate: Callable[[Atom], bool]) -> AtomGroup:
        """
        Create subset based on arbitrary predicate.

        Args:
            predicate: Function taking Atom, returning True to include.

        Returns:
            New AtomGroup with filtered atoms and geometry.
        """
        self._ensure_loaded()
        keep_indices = {int(v) for v in self._members.values()
                        if isinstance(v, Atom) and predicate(v)}
        return self.subset(keep_indices)

    # =========================================================================
    # Subset-specific properties (for ML workflows)
    # =========================================================================

    @property
    def atom_to_local(self) -> dict[int, int]:
        """Map global atom value -> local position."""
        return {int(a): a.local for a in self}

    def for_residue(self, residue: "AtomGroup") -> int:
        """
        Get the atom value for a specific residue type.

        For hierarchical AtomGroups (like Sugar.C1p or PurineBase.N9), this
        returns the atom value for the specified residue.

        Args:
            residue: A residue AtomGroup (e.g., Residue.A).

        Returns:
            The integer atom value for this residue.

        Raises:
            AttributeError: If this atom is not present in the residue.

        Example:
            >>> Sugar.C1p.for_residue(Residue.A)
            13  # C1' atom value for adenine
            >>> PurineBase.N9.for_residue(Residue.G)
            97  # N9 atom value for guanine
        """
        self._ensure_loaded()
        residue_name = residue.name
        if residue_name in self._members:
            member = self._members[residue_name]
            if isinstance(member, Atom):
                return int(member)
            elif isinstance(member, AtomGroup):
                # Nested group - shouldn't happen for atom positions
                raise TypeError(f"Expected Atom, got AtomGroup for {residue_name}")
        raise AttributeError(
            f"'{self.name}' has no atom for residue '{residue_name}'"
        )


def _to_python_name(pdb_name: str) -> str:
    """Convert PDB atom name to Python-safe identifier.

    Converts prime notation (O3', C5') to Python-safe names (O3p, C5p).
    This matches the naming used in AtomGroup dict keys for attribute access.
    """
    return pdb_name.replace("'", "p").replace('"', "").replace("*", "s")


def build_atom_group(
    name: str,
    sources: list[tuple[str, AtomGroup]],
    atom_filter: set[str] | None = None,
) -> AtomGroup:
    """
    Build a hierarchical AtomGroup grouping atoms by name across residue types.

    Creates an AtomGroup where each atom name maps to a nested AtomGroup containing
    all residues that have that atom.

    Args:
        name: Name for the created AtomGroup.
        sources: List of (residue_name, AtomGroup) pairs.
        atom_filter: Optional set of atom names to include (Python-safe names).

    Returns:
        AtomGroup with nested AtomGroups for each atom position.

    Example:
        >>> purines = [("A", Residue.A), ("G", Residue.G)]
        >>> PurineBase = build_atom_group("PurineBase", purines, {"N1", "N9"})
        >>> PurineBase.N1.A      # Atom(N1, 20)
        >>> PurineBase.N1.index()  # array of all N1 values
    """
    from collections import defaultdict

    atoms_by_name: dict[str, dict[str, Atom]] = defaultdict(dict)

    for residue_name, source in sources:
        for atom in source:
            # Use Python-safe name as dict key for attribute access
            py_name = _to_python_name(atom.name)
            if atom_filter is None or py_name in atom_filter:
                atoms_by_name[py_name][residue_name] = atom

    members: dict[str, AtomGroup] = {}
    for py_name, residue_atoms in sorted(atoms_by_name.items()):
        inner_members = {
            res_name: atom for res_name, atom in sorted(residue_atoms.items())
        }
        members[py_name] = AtomGroup(py_name, inner_members)

    return AtomGroup(name, members)
