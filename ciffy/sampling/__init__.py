"""
Sampling utilities for generating realistic polymer conformations.

This module provides functions for sampling backbone dihedrals from
empirical distributions fitted to PDB data.
"""

from .backbone import randomize_backbone, sample_protein_dihedrals

__all__ = [
    "randomize_backbone",
    "sample_protein_dihedrals",
]
