"""Centralized tolerance configuration for ciffy tests.

All numerical tolerances should be imported from here. This allows:
1. Single place to adjust tolerances for float32/float64 modes
2. Context-aware tolerances (stricter for small structures, looser for large)
3. Different profiles for CPU vs GPU testing
"""

from dataclasses import dataclass
from typing import Optional
import os


@dataclass
class ToleranceProfile:
    """A complete set of numerical tolerances for testing."""

    # Roundtrip RMSD tolerances (Angstroms)
    roundtrip_single_residue: float = 1e-5
    roundtrip_small: float = 1e-4  # < 10 residues
    roundtrip_medium: float = 1e-3  # < 100 residues
    roundtrip_large: float = 1e-2  # >= 100 residues
    roundtrip_real_structure: float = 1e-3  # PDB structures

    # Angular tolerances (radians)
    dihedral_match: float = 1e-5
    dihedral_biopython_match: float = 1e-4  # Cross-library comparison
    angle_range_epsilon: float = 1e-5  # For [0, pi] and [-pi, pi] checks

    # Coordinate tolerances (Angstroms)
    coord_roundtrip: float = 0.001  # CIF read/write (3 decimal places)
    bond_length: float = 0.01  # Bond length deviation from ideal

    # Gradient tolerances
    gradcheck_eps: float = 1e-4
    gradcheck_atol: float = 1e-3
    gradcheck_rtol: float = 1e-2

    # General tolerances
    allclose_atol: float = 1e-5
    allclose_rtol: float = 1e-5

    # Geometry tolerances
    orthogonality: float = 1e-4  # R @ R.T ≈ I
    rotation_determinant: float = 1e-5  # det(R) ≈ 1
    center_origin: float = 1e-4  # centered coords ≈ 0
    alignment_rmsd: float = 1e-4  # RMSD after Kabsch alignment
    symmetry: float = 1e-6  # Symmetric matrix checks

    # Metric tolerances
    score_self: float = 1e-6  # TM-score/lDDT of self should be ~1.0

    def roundtrip_for_size(self, n_residues: int) -> float:
        """Get appropriate roundtrip tolerance based on structure size."""
        if n_residues == 1:
            return self.roundtrip_single_residue
        elif n_residues < 10:
            return self.roundtrip_small
        elif n_residues < 100:
            return self.roundtrip_medium
        else:
            return self.roundtrip_large


# Default profile for CPU tests
DEFAULT = ToleranceProfile()

# Relaxed profile for GPU tests (accumulated float32 errors)
GPU = ToleranceProfile(
    roundtrip_small=1e-3,
    roundtrip_medium=1e-2,
    roundtrip_real_structure=1e-3,
    gradcheck_atol=1e-3,
    gradcheck_rtol=1e-2,
)

# Strict profile for numerical validation tests
STRICT = ToleranceProfile(
    roundtrip_single_residue=1e-6,
    roundtrip_small=1e-5,
    roundtrip_medium=1e-4,
    dihedral_match=1e-6,
    allclose_atol=1e-6,
    allclose_rtol=1e-6,
)


def get_tolerances(device: str = "cpu") -> ToleranceProfile:
    """Get appropriate tolerance profile for device."""
    if device in ("cuda", "mps"):
        return GPU
    return DEFAULT
