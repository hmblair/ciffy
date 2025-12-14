"""
Extract conformational ensembles from polymer structures.

Provides functions to extract coordinates for all instances of a residue type
in a standardized dense array format suitable for statistical learning.

The key challenge this module solves is that different instances of the same
residue type can have varying numbers of atoms due to:

- Missing/unresolved atoms (crystallographic disorder)
- Terminal modifications (e.g., OP3 only at 5' end of RNA)
- Hydrogen atoms not resolved in X-ray structures

The solution is to find the intersection of atoms present in ALL instances,
guaranteeing a dense output array with no missing values.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from ..backend import is_torch, to_numpy, Array
from .reduction import Reduction

if TYPE_CHECKING:
    from ..polymer import Polymer
    from ..utils.enum_base import ResidueType


def _from_numpy(arr: np.ndarray, reference: Array) -> Array:
    """Convert numpy array to match reference backend/device."""
    if is_torch(reference):
        import torch
        return torch.from_numpy(arr).to(reference.device)
    return arr


def extract(
    poly: "Polymer",
    residue: "ResidueType",
    atoms: list | None = None,
    center: bool = False,
    align: bool = False,
    scale: bool = False,
) -> tuple[Array, list[int]]:
    """
    Extract coordinates for all instances of a residue type.

    Returns a dense array containing only atoms present in ALL instances,
    ensuring no missing values. Terminal atoms (e.g., OP3 on 5' residues)
    and unresolved atoms (e.g., hydrogens in X-ray structures) are
    automatically excluded since they don't appear universally.

    This is useful for statistical learning on molecular conformations,
    where you need a fixed-size feature vector for each residue instance.

    Args:
        poly: Polymer structure to extract from.
        residue: Residue type from the Residue enum (e.g., Residue.A for
            adenosine, Residue.ALA for alanine).
        atoms: Optional list of specific atom types to extract. Can be
            atom enum members (e.g., [A.C1p, A.N9]) or integer indices.
            If provided, only these atoms are included. Raises ValueError
            if any requested atom is missing from any instance.
        center: If True, center each residue at the origin by subtracting
            the centroid of all atoms in that residue.
        align: If True, align each residue to its principal axes using SVD.
            This removes rotational variance. Implies centering.
        scale: If True, scale each residue to unit standard deviation.
            Implies centering. Useful for normalizing coordinates before
            statistical learning.

    Returns:
        A tuple of (coords, atom_indices):

        - coords: Dense coordinate array of shape (n_instances, n_atoms, 3).
          Backend matches input (NumPy or PyTorch).
        - atom_indices: List of atom type indices in column order. These
          correspond to the residue's atom enum values (e.g., for adenosine,
          index 2 is A.P, index 9 is A.C3p, etc.).

    Raises:
        ValueError: If no residues of the given type are found.
        ValueError: If no atoms are common to all instances.
        ValueError: If requested atoms are not present in all instances.

    Examples:
        Basic extraction of all adenosines::

            from ciffy import load
            from ciffy.biochemistry import Residue
            from ciffy.operations import extract

            poly = load("ribosome.cif")
            coords, atoms = extract(poly, Residue.A)
            # coords.shape = (n_adenosines, n_common_atoms, 3)

        Centered and aligned for rotation-invariant learning::

            coords, atoms = extract(poly, Residue.A, center=True, align=True)

        Extract only nucleobase atoms::

            from ciffy.biochemistry._generated_atoms import A
            base_atoms = [A.N9, A.C8, A.N7, A.C5, A.C6, A.N6, A.N1, A.C2, A.N3, A.C4]
            coords, atoms = extract(poly, Residue.A, atoms=base_atoms, align=True)
            # coords.shape = (n_adenosines, 10, 3)

        Map returned indices to atom names::

            coords, atom_indices = extract(poly, Residue.A)
            atom_names = [A(idx).name for idx in atom_indices]
            # ['P', 'OP1', 'OP2', 'O5p', 'C5p', ...]
    """
    from ..types import Scale

    # Work with polymer atoms only (residues don't include HETATM)
    # Then filter to residues of the target type and remove empty ones
    sub = poly.poly().by_residue(residue.value).strip()

    # Apply transformations at residue level (each implies centering)
    if align:
        sub, _ = sub.align(Scale.RESIDUE)
    if scale:
        sub, _ = sub.scale(Scale.RESIDUE)
    if center and not align and not scale:
        sub, _ = sub.center(Scale.RESIDUE)

    n_residues = sub.size(Scale.RESIDUE)
    if n_residues == 0:
        raise ValueError(f"No residues of type {residue.name} found in polymer")

    # Collate atoms and coordinates per residue
    per_res_atoms = sub.reduce(sub.atoms, Scale.RESIDUE, Reduction.COLLATE)
    per_res_coords = sub.reduce(sub.coordinates, Scale.RESIDUE, Reduction.COLLATE)

    # Find atoms present in ALL residues (intersection)
    atom_sets = [set(to_numpy(a).tolist()) for a in per_res_atoms]
    common_atoms = set.intersection(*atom_sets)

    if len(common_atoms) == 0:
        raise ValueError(f"No atoms common to all {residue.name} residues")

    # If specific atoms requested, filter to those
    if atoms is not None:
        requested = set(a.value if hasattr(a, 'value') else a for a in atoms)
        missing = requested - common_atoms
        if missing:
            raise ValueError(
                f"Atoms {missing} not present in all {residue.name} instances"
            )
        common_atoms = requested

    # Sort by atom index for canonical ordering
    common_atoms = sorted(common_atoms)
    n_atoms = len(common_atoms)

    # Build dense output array
    # Create mapping from atom index to output column
    atom_to_col = {atom: col for col, atom in enumerate(common_atoms)}

    result = np.zeros((n_residues, n_atoms, 3), dtype=np.float32)

    for i, (res_atoms, res_coords) in enumerate(zip(per_res_atoms, per_res_coords)):
        res_atoms_np = to_numpy(res_atoms)
        res_coords_np = to_numpy(res_coords)

        for atom_idx, coord in zip(res_atoms_np, res_coords_np):
            if atom_idx in atom_to_col:
                result[i, atom_to_col[atom_idx]] = coord

    # Convert back to original backend if needed
    result = _from_numpy(result, poly.coordinates)

    return result, common_atoms
