"""
Chain-level operations for polymer manipulation.

Provides functions for combining and extending polymer chains:
- join(): Combine multiple polymers into one
- Helpers for polymer extension and residue positioning
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..backend import Array, get_backend, check_compatible
from ..backend import ops
from ..biochemistry import Scale
from ..polymer import Field

if TYPE_CHECKING:
    from ..polymer import Polymer


def _validate_poly_only(polymer: "Polymer", operation: str) -> None:
    """
    Validate that a polymer has no HETATM atoms.

    Args:
        polymer: Polymer to validate.
        operation: Name of operation for error message.

    Raises:
        ValueError: If polymer has non-polymer atoms.
    """
    if polymer.nonpoly() > 0:
        raise ValueError(
            f"{operation} requires poly-only polymers (no HETATM atoms). "
            f"Use polymer.poly() to get the polymer portion first."
        )


def _validate_same_backend(*polymers: "Polymer") -> str:
    """
    Validate that all polymers have the same backend.

    Args:
        *polymers: Polymers to check.

    Returns:
        The common backend name ('numpy' or 'torch').

    Raises:
        ValueError: If backends differ.
    """
    if not polymers:
        raise ValueError("At least one polymer required")

    backends = [p.backend for p in polymers]
    if len(set(backends)) > 1:
        raise ValueError(
            f"All polymers must have the same backend. Got: {backends}"
        )
    return backends[0]


def join(*polymers: "Polymer") -> "Polymer":
    """
    Combine multiple Polymer objects into a single Polymer.

    Creates a new Polymer containing all chains from all input polymers.
    Chains are concatenated in order, with each polymer's chains appearing
    in sequence.

    Args:
        *polymers: Polymers to combine. All must have the same backend
            and be poly-only (no HETATM atoms).

    Returns:
        A new Polymer containing all chains from all inputs.

    Raises:
        ValueError: If no polymers provided, backends differ, or any
            polymer has HETATM atoms.

    Example:
        >>> import ciffy
        >>> p1 = ciffy.load("chain_a.cif").poly()
        >>> p2 = ciffy.load("chain_b.cif").poly()
        >>> combined = ciffy.join(p1, p2)
        >>> combined.size(ciffy.CHAIN)
        2
    """
    from ..polymer import Polymer

    if not polymers:
        raise ValueError("join() requires at least one polymer")

    # Filter out empty polymers
    non_empty = [p for p in polymers if not p.empty()]
    if not non_empty:
        # All empty - return first polymer's empty copy
        empty = Polymer(pdb_id=polymers[0].pdb_id)
        return empty.torch() if polymers[0].backend == "torch" else empty

    # Validate all inputs
    backend = _validate_same_backend(*non_empty)
    for p in non_empty:
        _validate_poly_only(p, "join()")

    # Single polymer case - return a copy using _clone
    if len(non_empty) == 1:
        p = non_empty[0]
        mol_types_data = p._get_field_data('molecule_types')
        return p._clone(
            coordinates=ops.clone(p.coordinates),
            atoms=ops.clone(p.atoms),
            elements=ops.clone(p.elements),
            sequence=ops.clone(p.sequence),
            names=list(p.names),
            strands=list(p.strands),
            molecule_types=ops.clone(mol_types_data) if mol_types_data is not None else None,
            descriptions=list(p.descriptions) if p.descriptions else None,
        )

    # Concatenate arrays
    coordinates = ops.cat([p.coordinates for p in non_empty], axis=0)
    atoms = ops.cat([p.atoms for p in non_empty], axis=0)
    elements = ops.cat([p.elements for p in non_empty], axis=0)
    sequence = ops.cat([p.sequence for p in non_empty], axis=0)

    res_sizes = ops.cat([p._sizes[Scale.RESIDUE] for p in non_empty], axis=0)
    chn_sizes = ops.cat([p._sizes[Scale.CHAIN] for p in non_empty], axis=0)
    lengths = ops.cat([p.lengths for p in non_empty], axis=0)

    # Compute molecule size (total atoms)
    total_atoms = sum(p.size() for p in non_empty)
    mol_sizes = ops.array([total_atoms], like=coordinates)

    sizes = {
        Scale.RESIDUE: res_sizes,
        Scale.CHAIN: chn_sizes,
        Scale.MOLECULE: mol_sizes,
    }

    # Concatenate lists
    names = []
    strands = []
    for p in non_empty:
        names.extend(p.names)
        strands.extend(p.strands)

    # Handle molecule types - only include if all polymers have them
    molecule_types = None
    mol_types_list = [p._get_field_data('molecule_types') for p in non_empty]
    if all(mt is not None for mt in mol_types_list):
        molecule_types = ops.cat(mol_types_list, axis=0)

    # Handle descriptions
    descriptions = None
    if all(p.descriptions is not None for p in non_empty):
        descriptions = []
        for p in non_empty:
            descriptions.extend(p.descriptions)

    # Compute total polymer count
    polymer_count = sum(p.polymer_count for p in non_empty)

    # Determine pdb_id
    pdb_ids = set(p.pdb_id for p in non_empty)
    if len(pdb_ids) == 1:
        pdb_id = pdb_ids.pop()
    else:
        pdb_id = "joined"

    # Create hierarchy
    from ..polymer.hierarchy import _Hierarchy
    hierarchy = _Hierarchy.from_sizes_and_lengths(
        sizes=sizes,
        lengths=lengths,
        polymer_count=polymer_count,
        ref=coordinates,
    )

    # Build kwargs, only including molecule_types if present
    kwargs = dict(
        coordinates=Field(coordinates, Scale.ATOM),
        atoms=Field(atoms, Scale.ATOM),
        elements=Field(elements, Scale.ATOM),
        sequence=Field(sequence, Scale.RESIDUE),
        pdb_id=pdb_id,
        names=names,
        strands=strands,
        descriptions=descriptions,
    )
    if molecule_types is not None:
        kwargs['molecule_types'] = Field(molecule_types, Scale.CHAIN)

    return Polymer(hierarchy, **kwargs)
