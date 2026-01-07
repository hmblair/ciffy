"""DataModule for consolidated VAE training.

Handles data for all residue types together, providing batches
grouped by residue type with atom types, coordinates, and masks.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

if TYPE_CHECKING:
    from ciffy.biochemistry import Residue


class ConsolidatedResidueDataset(Dataset):
    """Dataset that yields (atom_types, coords, mask, residue_idx) samples.

    Groups samples by residue type for efficient batching.
    """

    def __init__(
        self,
        residue_data: dict["Residue", dict],
        max_atoms: int,
    ):
        """Initialize dataset.

        Args:
            residue_data: Dict mapping Residue to dict with 'coords', 'transforms', 'atoms'.
            max_atoms: Maximum number of atoms across all residue types.
        """
        self.max_atoms = max_atoms
        self.residue_data = residue_data
        self.residues = list(residue_data.keys())

        # Build flat index: (residue_idx, sample_idx)
        self.indices = []
        for res_idx, res in enumerate(self.residues):
            n_samples = len(residue_data[res]["coords"])
            for i in range(n_samples):
                self.indices.append((res_idx, i))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Get a sample.

        Returns:
            atom_types: (max_atoms,) padded atom type indices.
            coords: (max_atoms, 3) padded coordinates.
            mask: (max_atoms,) boolean mask.
            transforms: (6,) SE(3) transform parameters.
            residue_idx: Index into self.residues.
        """
        res_idx, sample_idx = self.indices[idx]
        res = self.residues[res_idx]
        data = self.residue_data[res]

        # Get raw data
        coords = data["coords"][sample_idx]  # (n_atoms, 3)
        transforms = data["transforms"][sample_idx]  # (6,)
        atoms = data["atoms"]  # (n_atoms,) - same for all samples

        n_atoms = len(atoms)

        # Pad to max_atoms
        atom_types = torch.zeros(self.max_atoms, dtype=torch.long)
        atom_types[:n_atoms] = torch.from_numpy(atoms).long()

        coords_padded = torch.zeros(self.max_atoms, 3, dtype=torch.float32)
        coords_padded[:n_atoms] = torch.from_numpy(coords).float()

        mask = torch.zeros(self.max_atoms, dtype=torch.bool)
        mask[:n_atoms] = True

        transforms_t = torch.from_numpy(transforms).float()

        return atom_types, coords_padded, mask, transforms_t, res_idx


def collate_by_residue(batch):
    """Collate function that groups samples by residue type.

    Returns:
        atom_types: (B, max_atoms) padded atom type indices.
        coords: (B, max_atoms, 3) padded coordinates.
        masks: (B, max_atoms) boolean masks.
        transforms: (B, 6) SE(3) transform parameters.
        residue_idxs: (B,) residue type indices.
    """
    # Unpack batch - order matches __getitem__ return
    atom_types = torch.stack([b[0] for b in batch])
    coords = torch.stack([b[1] for b in batch])
    masks = torch.stack([b[2] for b in batch])
    transforms = torch.stack([b[3] for b in batch])
    residue_idxs = torch.tensor([b[4] for b in batch])

    return atom_types, coords, masks, transforms, residue_idxs


class ConsolidatedDataModule(LightningDataModule):
    """DataModule for consolidated VAE training.

    Extracts residue data for all specified types and provides batches
    with (atom_types, coords, mask, residue_idx).
    """

    def __init__(
        self,
        cif_paths: list[Path],
        residues: list["Residue"],
        train_split: float = 0.8,
        min_coverage: float = 0.9,
        batch_size: int = 256,
        seed: int = 42,
    ) -> None:
        """Initialize the consolidated data module.

        Args:
            cif_paths: List of paths to CIF files.
            residues: List of residue types to extract.
            train_split: Fraction of structures for training.
            min_coverage: Minimum atom coverage for extraction.
            batch_size: Training batch size.
            seed: Random seed for splitting.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["cif_paths", "residues"])

        self.cif_paths = cif_paths
        self.residues = residues
        self.train_split = train_split
        self.min_coverage = min_coverage
        self.batch_size = batch_size
        self.seed = seed

        # Set in setup()
        self.residue_atoms: dict["Residue", list[int]] | None = None
        self.max_atoms: int = 0
        self.train_dataset: ConsolidatedResidueDataset | None = None
        self.val_dataset: ConsolidatedResidueDataset | None = None

    def setup(self, stage: str) -> None:
        """Extract residue data and create datasets."""
        if stage not in ("fit", "validate") or self.train_dataset is not None:
            return

        from ciffy.nn.flow.residue.data import extract_residues_with_links

        # Split paths by structure
        np.random.seed(self.seed)
        n_train = int(len(self.cif_paths) * self.train_split)
        indices = np.random.permutation(len(self.cif_paths))
        train_paths = [self.cif_paths[i] for i in indices[:n_train]]
        test_paths = [self.cif_paths[i] for i in indices[n_train:]]

        # Extract data for each residue type
        train_data = {}
        test_data = {}
        self.residue_atoms = {}
        self.max_atoms = 0

        for res in self.residues:
            # Training data
            coords, transforms, atoms = extract_residues_with_links(
                train_paths,
                res,
                min_coverage=self.min_coverage,
                verbose=False,
            )

            if len(coords) == 0:
                continue

            train_data[res] = {
                "coords": coords,
                "transforms": transforms,
                "atoms": atoms,
            }
            self.residue_atoms[res] = atoms.tolist()
            self.max_atoms = max(self.max_atoms, len(atoms))

            # Test data
            if len(test_paths) > 0:
                test_coords, test_transforms, _ = extract_residues_with_links(
                    test_paths,
                    res,
                    min_coverage=self.min_coverage,
                    verbose=False,
                )
                if len(test_coords) > 0:
                    test_data[res] = {
                        "coords": test_coords,
                        "transforms": test_transforms,
                        "atoms": atoms,
                    }
                else:
                    test_data[res] = train_data[res]
            else:
                test_data[res] = train_data[res]

        # Create datasets
        self.train_dataset = ConsolidatedResidueDataset(train_data, self.max_atoms)
        self.val_dataset = ConsolidatedResidueDataset(test_data, self.max_atoms)

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            collate_fn=collate_by_residue,
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=collate_by_residue,
        )


__all__ = ["ConsolidatedDataModule", "ConsolidatedResidueDataset"]
