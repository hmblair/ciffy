"""
Masking and specialized selection functions for Polymer objects.

Functions for creating masks, checking resolution, and selecting structural regions.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..polymer import Polymer

from ..backend import Array, ops
from ..biochemistry import Scale, Backbone, Nucleobase, Phosphate, Sidechain


def mask(
    polymer: Polymer,
    indices: Array | int,
    source: Scale,
    dest: Scale = Scale.ATOM,
) -> Array:
    """
    Create a boolean mask selecting specific units.

    Args:
        polymer: Source polymer.
        indices: Indices of units to select.
        source: Scale of the indices.
        dest: Scale of the output mask.

    Returns:
        Boolean array at dest scale.
    """
    counts = polymer.size(source)
    objects = ops.zeros(counts, like=polymer._hierarchy._ref, dtype='bool')
    objects[indices] = True
    return polymer.expand(objects, source, dest)


def resolved(polymer: Polymer, scale: Scale = Scale.RESIDUE) -> Array:
    """
    Get mask of resolved (non-empty) units.

    Args:
        polymer: Source polymer.
        scale: Scale to check.

    Returns:
        Boolean tensor where True indicates resolved units.
    """
    return polymer._hierarchy.resolved(scale)


def strip(polymer: Polymer, scale: Scale = Scale.RESIDUE) -> Polymer:
    """
    Remove unresolved units at a scale.

    Args:
        polymer: Source polymer.
        scale: Scale at which to strip.

    Returns:
        New Polymer without empty units.
    """
    resolved_mask = polymer._hierarchy.resolved(scale)
    return polymer.select(resolved_mask, scale)


# Specialized structural selections

def backbone(polymer: Polymer) -> Polymer:
    """
    Select backbone atoms (sugar-phosphate for RNA/DNA, N-CA-C-O for protein).

    Args:
        polymer: Source polymer.

    Returns:
        New Polymer with only backbone atoms.
    """
    from .filters import by_atom
    return by_atom(polymer, Backbone.index())


def nucleobase(polymer: Polymer) -> Polymer:
    """
    Select RNA nucleobase atoms.

    Args:
        polymer: Source polymer.

    Returns:
        New Polymer with only nucleobase atoms.
    """
    from .filters import by_atom
    return by_atom(polymer, Nucleobase.index())


def phosphate(polymer: Polymer) -> Polymer:
    """
    Select RNA/DNA phosphate atoms.

    Args:
        polymer: Source polymer.

    Returns:
        New Polymer with only phosphate atoms.
    """
    from .filters import by_atom
    return by_atom(polymer, Phosphate.index())


def sidechain(polymer: Polymer) -> Polymer:
    """
    Select protein sidechain atoms.

    Args:
        polymer: Source polymer.

    Returns:
        New Polymer with only sidechain atoms.
    """
    from .filters import by_atom
    return by_atom(polymer, Sidechain.index())
