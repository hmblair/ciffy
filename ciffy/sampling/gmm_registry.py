"""
Registry for residue-type-specific Gaussian Mixture Models.

Provides a flexible system for organizing GMMs by residue type with fallback
hierarchy: specific residue → residue group → global default.

This enables extensibility (e.g., GLY-specific Ramachandran, purine vs
pyrimidine RNA dihedrals) while maintaining backward compatibility.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ciffy.biochemistry import Molecule, Residue
    from ciffy.utils.gmm import GaussianMixtureModel


class GMMRegistry:
    """
    Registry for residue-type-specific GMMs with fallback hierarchy.

    Lookup order when getting a GMM for a specific residue:
    1. Residue-specific GMM (e.g., "GLY.npz" for glycine)
    2. Residue group GMM (e.g., "purine.npz" for A/G)
    3. Global default GMM (e.g., "default.npz")

    Example:
        >>> from ciffy.biochemistry import Molecule, Residue
        >>> registry = GMMRegistry()
        >>> # Get RNA purine GMM (uses purine.npz)
        >>> gmm_a = registry.get_gmm(Molecule.RNA, Residue.A)
        >>> # Get protein alanine GMM (falls back to default.npz if ALA.npz not found)
        >>> gmm_ala = registry.get_gmm(Molecule.PROTEIN, Residue.ALA)
    """

    def __init__(self):
        """Initialize registry with molecule type to directory mapping."""
        self._cache: dict[tuple, GaussianMixtureModel] = {}
        self._molecule_dirs = {
            "PROTEIN": "gmm/protein",
            "RNA": "gmm/rna",
            "DNA": "gmm/dna",
        }

    def get_gmm(
        self,
        molecule_type: Molecule | str,
        residue_type: Residue | None = None,
    ) -> GaussianMixtureModel:
        """
        Get GMM for residue type with fallback hierarchy.

        Args:
            molecule_type: Molecule type (PROTEIN, RNA, or DNA) or string name
            residue_type: Specific residue type (optional). If None, returns global GMM.

        Returns:
            Loaded GaussianMixtureModel

        Raises:
            ValueError: If molecule type not supported or no GMM found
        """
        from ciffy.biochemistry import Molecule

        # Normalize molecule type to string for cache and lookup
        if isinstance(molecule_type, Molecule):
            mol_str = molecule_type.name
        else:
            mol_str = str(molecule_type).upper()

        # Check cache
        cache_key = (mol_str, residue_type)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Try to load GMM with fallback hierarchy
        gmm = None

        # Step 1: Try residue-specific GMM (e.g., GLY.npz, A.npz)
        if residue_type is not None:
            gmm = self._try_load_specific(mol_str, residue_type)
            if gmm is not None:
                self._cache[cache_key] = gmm
                return gmm

        # Step 2: Try residue group GMM (e.g., purine.npz, pyrimidine.npz)
        if residue_type is not None:
            group = self._get_residue_group(mol_str, residue_type)
            if group is not None:
                gmm = self._try_load_group(mol_str, group)
                if gmm is not None:
                    self._cache[cache_key] = gmm
                    return gmm

        # Step 3: Fall back to global GMM (default.npz)
        gmm = self._load_default(mol_str)
        if gmm is None:
            raise ValueError(
                f"Could not find GMM for {mol_str}"
                + (f" residue {residue_type}" if residue_type else "")
                + ". Make sure GMM files are installed in ciffy/data/gmm/"
            )

        self._cache[cache_key] = gmm
        return gmm

    def _try_load_specific(
        self,
        mol_str: str,
        residue_type,
    ) -> GaussianMixtureModel | None:
        """Try to load residue-specific GMM (e.g., GLY.npz)."""
        residue_name = residue_type.name if hasattr(residue_type, "name") else str(residue_type)
        mol_dir = self._molecule_dirs.get(mol_str)
        if mol_dir is None:
            return None

        path = self._get_data_path(f"{mol_dir}/{residue_name}.npz")
        return self._load_if_exists(path)

    def _try_load_group(
        self,
        mol_str: str,
        group_name: str,
    ) -> GaussianMixtureModel | None:
        """Try to load group GMM (e.g., purine.npz)."""
        mol_dir = self._molecule_dirs.get(mol_str)
        if mol_dir is None:
            return None

        path = self._get_data_path(f"{mol_dir}/{group_name}.npz")
        return self._load_if_exists(path)

    def _load_default(self, mol_str: str) -> GaussianMixtureModel | None:
        """Load global default GMM."""
        mol_dir = self._molecule_dirs.get(mol_str)
        if mol_dir is None:
            return None

        path = self._get_data_path(f"{mol_dir}/default.npz")
        return self._load_if_exists(path)

    def _get_residue_group(self, mol_str: str, residue_type) -> str | None:
        """
        Map residue to group name.

        Returns:
            Group name (e.g., 'purine', 'pyrimidine') or None if no group
        """
        from ciffy.biochemistry import Residue

        if mol_str == "RNA":
            # Purines: A, G (and their deoxy variants DA, DG)
            if residue_type in [Residue.A, Residue.G, Residue.DA, Residue.DG]:
                return "purine"
            # Pyrimidines: C, U (RNA) and DC, DT (DNA)
            elif residue_type in [Residue.C, Residue.U, Residue.DC, Residue.DT]:
                return "pyrimidine"
        elif mol_str == "DNA":
            # DNA is same as RNA for now
            if residue_type in [Residue.DA, Residue.DG]:
                return "purine"
            elif residue_type in [Residue.DC, Residue.DT]:
                return "pyrimidine"

        # Future: protein groups (e.g., aromatic, charged, etc.)
        return None

    @staticmethod
    def _get_data_path(relative_path: str) -> Path:
        """
        Get absolute path to data file in ciffy/data/.

        Args:
            relative_path: Path relative to ciffy/data (e.g., "gmm/rna/default.npz")

        Returns:
            Absolute Path object
        """
        data_dir = Path(__file__).parent.parent / "data"
        return data_dir / relative_path

    @staticmethod
    def _load_if_exists(path: Path) -> GaussianMixtureModel | None:
        """
        Load GMM from file if it exists.

        Args:
            path: Path to .npz file

        Returns:
            GaussianMixtureModel if file exists, None otherwise
        """
        if not path.exists():
            return None

        from ciffy.utils.gmm import GaussianMixtureModel

        return GaussianMixtureModel.load(path)

    def clear_cache(self):
        """Clear the GMM cache."""
        self._cache.clear()


# Global registry instance for convenient access
_global_registry = GMMRegistry()


def get_gmm(
    molecule_type: Molecule | str,
    residue_type: Residue | None = None,
) -> GaussianMixtureModel:
    """
    Convenience function to get a GMM from the global registry.

    Args:
        molecule_type: Molecule type (PROTEIN, RNA, or DNA)
        residue_type: Specific residue type (optional)

    Returns:
        GaussianMixtureModel
    """
    return _global_registry.get_gmm(molecule_type, residue_type)
