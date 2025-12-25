"""
Training infrastructure for ResidueFlowModel.

Provides a unified interface for training models across multiple residue types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from .model import ResidueFlowModel
from .data import extract_residues_with_links
from .train import train_pca_flow

if TYPE_CHECKING:
    from ciffy.biochemistry import Residue


@dataclass
class TrainingConfig:
    """Configuration for ResidueFlowModel training.

    Attributes:
        latent_dim: Number of latent dimensions (PCA components).
        n_layers: Number of normalizing flow layers.
        hidden_dim: Hidden dimension in coupling networks.
        bound: Tanh bound for decode (in std devs). None for unbounded.
        n_epochs: Number of training epochs.
        batch_size: Batch size for training.
        lr: Learning rate.
        min_coverage: Minimum fraction of instances an atom must appear in.
        max_bond_length: Maximum O3'-P distance to accept as connected.
        device: Device to train on ('cpu' or 'cuda').
    """

    latent_dim: int = 12
    n_layers: int = 8
    hidden_dim: int = 64
    bound: float | None = 3.0
    n_epochs: int = 200
    batch_size: int = 256
    lr: float = 1e-3
    min_coverage: float = 0.9
    max_bond_length: float = 2.0
    device: str = "cpu"


@dataclass
class TrainingResult:
    """Result of training a single ResidueFlowModel.

    Attributes:
        model: The trained ResidueFlowModel.
        residue: The residue type.
        n_instances: Number of training instances.
        n_atoms: Number of atoms per residue.
        pca_rmsd: PCA reconstruction RMSD.
        flow_rmsd: Flow reconstruction RMSD.
        var_explained: Variance explained by PCA.
        n_params: Number of model parameters.
    """

    model: ResidueFlowModel
    residue: "Residue"
    n_instances: int
    n_atoms: int
    pca_rmsd: float
    flow_rmsd: float
    var_explained: float
    n_params: int


class ResidueFlowTrainer:
    """
    Trainer for ResidueFlowModels across multiple residue types.

    Provides a unified interface for:
    - Training models for individual residue types
    - Training models for all specified residue types
    - Saving trained models to disk

    Example:
        >>> from ciffy.nn.flow.residue import ResidueFlowTrainer, TrainingConfig
        >>> from ciffy.biochemistry import Residue
        >>>
        >>> config = TrainingConfig(latent_dim=12, n_epochs=100)
        >>> trainer = ResidueFlowTrainer(config)
        >>>
        >>> # Train all RNA residue types
        >>> results = trainer.train_all(cif_paths, [Residue.A, Residue.C, Residue.G, Residue.U])
        >>>
        >>> # Save models
        >>> trainer.save(results, "models/rna_v1")
    """

    def __init__(self, config: TrainingConfig | None = None):
        """
        Initialize trainer with configuration.

        Args:
            config: Training configuration. If None, uses defaults.
        """
        self.config = config or TrainingConfig()

    def train_single(
        self,
        cif_paths: list[Path],
        residue: "Residue",
        verbose: bool = True,
    ) -> TrainingResult:
        """
        Train a ResidueFlowModel for a single residue type.

        Args:
            cif_paths: List of paths to CIF files.
            residue: Residue type to train on.
            verbose: Print progress information.

        Returns:
            TrainingResult with trained model and metrics.

        Raises:
            ValueError: If no residues of the specified type are found.
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Training {residue.name}")
            print(f"{'='*60}")

        # Extract data with link transforms
        coords, transforms, atoms = extract_residues_with_links(
            cif_paths,
            residue,
            min_coverage=self.config.min_coverage,
            max_bond_length=self.config.max_bond_length,
            verbose=verbose,
        )

        n_instances = len(coords)
        n_atoms = len(atoms)

        if verbose:
            print(f"\nExtracted {n_instances} instances with {n_atoms} atoms each")

        # Flatten coords and concatenate with transforms for extended representation
        coords_flat = coords.reshape(n_instances, -1)  # (N, n_atoms*3)
        import numpy as np
        extended_data = np.concatenate([coords_flat, transforms], axis=1)  # (N, n_atoms*3 + 6)

        if verbose:
            print(f"Extended representation: {extended_data.shape[1]} dimensions")

        # Train
        flow, info = train_pca_flow(
            extended_data,
            latent_dim=self.config.latent_dim,
            n_layers=self.config.n_layers,
            hidden_dim=self.config.hidden_dim,
            bound=self.config.bound,
            n_epochs=self.config.n_epochs,
            batch_size=self.config.batch_size,
            lr=self.config.lr,
            device=self.config.device,
            verbose=verbose,
        )

        # Create ResidueFlowModel wrapper
        model = ResidueFlowModel(
            flow=flow,
            residue=residue,
            atom_indices=atoms,
        )

        return TrainingResult(
            model=model,
            residue=residue,
            n_instances=n_instances,
            n_atoms=n_atoms,
            pca_rmsd=info["pca_rmsd"],
            flow_rmsd=info["flow_rmsd"],
            var_explained=info["var_explained"],
            n_params=info["n_params"],
        )

    def train_all(
        self,
        cif_paths: list[Path],
        residues: list["Residue"],
        verbose: bool = True,
    ) -> dict["Residue", TrainingResult]:
        """
        Train ResidueFlowModels for multiple residue types.

        Args:
            cif_paths: List of paths to CIF files.
            residues: List of residue types to train.
            verbose: Print progress information.

        Returns:
            Dict mapping residue type to TrainingResult.
        """
        results = {}

        for residue in residues:
            try:
                result = self.train_single(cif_paths, residue, verbose=verbose)
                results[residue] = result
            except ValueError as e:
                if verbose:
                    print(f"\nSkipping {residue.name}: {e}")

        if verbose:
            self._print_summary(results)

        return results

    def _print_summary(self, results: dict["Residue", TrainingResult]) -> None:
        """Print summary table of training results."""
        print(f"\n{'='*80}")
        print("TRAINING SUMMARY")
        print(f"{'='*80}")
        print(
            f"{'Residue':<10} {'N':<8} {'Atoms':<8} {'Var%':<8} "
            f"{'PCA RMSD':<12} {'Flow RMSD':<12} {'Params':<10}"
        )
        print("-" * 80)

        for residue, result in results.items():
            print(
                f"{residue.name:<10} "
                f"{result.n_instances:<8} "
                f"{result.n_atoms:<8} "
                f"{result.var_explained*100:.1f}%{'':<4} "
                f"{result.pca_rmsd:.4f}Å{'':<5} "
                f"{result.flow_rmsd:.4f}Å{'':<5} "
                f"{result.n_params:,}"
            )

    def save(
        self,
        results: dict["Residue", TrainingResult],
        path: str | Path,
    ) -> None:
        """
        Save trained models to directory.

        Creates a PolymerFlowModel-compatible directory structure with
        one subdirectory per residue type.

        Args:
            results: Training results from train_all().
            path: Directory to save models to.
        """
        import json

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save each residue model
        for residue, result in results.items():
            result.model.save(path / residue.name)

        # Save metadata
        config = {
            "residue_types": [r.name for r in results.keys()],
            "latent_dim": self.config.latent_dim,
            "training_config": {
                "latent_dim": self.config.latent_dim,
                "n_layers": self.config.n_layers,
                "hidden_dim": self.config.hidden_dim,
                "bound": self.config.bound,
                "n_epochs": self.config.n_epochs,
                "min_coverage": self.config.min_coverage,
            },
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        print(f"\nSaved {len(results)} models to {path}")

    def to_polymer_model(
        self,
        results: dict["Residue", TrainingResult],
    ) -> "PolymerFlowModel":
        """
        Convert training results to a PolymerFlowModel.

        Args:
            results: Training results from train_all().

        Returns:
            PolymerFlowModel with all trained residue models.
        """
        from ciffy.nn.flow import PolymerFlowModel

        residue_models = {r: res.model for r, res in results.items()}
        return PolymerFlowModel(residue_models)
