"""
Training infrastructure for ResidueFlowModel.

Provides a unified interface for training models across multiple residue types
with proper train/test splitting by structure to avoid data leakage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from glob import glob
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import torch

from .model import ResidueFlowModel
from .data import ResidueDataset
from .train import train_pca_flow
from ...split import split_by_structure
from ...trainer_registry import register_trainer
from ...base_trainer import BaseConfig, DataConfig, OutputConfig, TrainingConfig

if TYPE_CHECKING:
    from ciffy.biochemistry import Residue


@dataclass
class ResidueFlowModelConfig:
    """Model configuration for ResidueFlowModel.

    Attributes:
        latent_dim: Number of latent dimensions (PCA components).
        n_layers: Number of normalizing flow layers.
        hidden_dim: Hidden dimension in coupling networks.
        bound: Tanh bound for decode (in std devs). None for unbounded.
    """

    latent_dim: int = 12
    n_layers: int = 4
    hidden_dim: int = 56
    bound: float | None = None


@dataclass
class ResidueFlowDataConfig(DataConfig):
    """Data configuration for ResidueFlowModel training.

    Extends base DataConfig with flow-specific settings.

    Attributes:
        cif_patterns: Glob patterns for CIF files.
        residue: Residue type to train (e.g., "A", "C", "G", or "U").
        min_coverage: Minimum fraction of instances an atom must appear in.
        max_bond_length: Maximum O3'-P distance to accept as connected.
        train_split: Fraction of structures for training.
        test_split: Fraction of structures for testing.
        split_seed: Random seed for reproducible splits.
    """

    cif_patterns: list[str] | None = None
    residue: str | None = None
    min_coverage: float = 0.9
    max_bond_length: float = 2.0
    train_split: float = 0.8
    test_split: float = 0.2
    split_seed: int | None = 42


@dataclass
class ResidueFlowOutputConfig(OutputConfig):
    """Output configuration for ResidueFlowModel training.

    Extends base OutputConfig with report settings.

    Attributes:
        generate_report: Whether to generate training report.
        report_path: Path to save report. None for output_dir/training_report.html.
    """

    generate_report: bool = True
    report_path: str | None = None


@dataclass
class ResidueFlowTrainingConfig(BaseConfig):
    """Full configuration for ResidueFlowModel training.

    Uses nested dataclasses for consistency with other trainers.
    Inherits from BaseConfig to use shared from_dict() machinery.

    Example:
        >>> config = ResidueFlowTrainingConfig(
        ...     model=ResidueFlowModelConfig(latent_dim=16),
        ...     data=ResidueFlowDataConfig(data_dir="./cif"),
        ... )
    """

    model: ResidueFlowModelConfig = field(default_factory=ResidueFlowModelConfig)
    data: ResidueFlowDataConfig = field(default_factory=ResidueFlowDataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output: ResidueFlowOutputConfig = field(default_factory=ResidueFlowOutputConfig)


@dataclass
class TrainingResult:
    """Result of training a single ResidueFlowModel.

    Attributes:
        model: The trained ResidueFlowModel.
        residue: The residue type.
        n_train: Number of training instances.
        n_test: Number of test instances.
        n_atoms: Number of atoms per residue.
        pca_rmsd: PCA reconstruction RMSD (on train data).
        train_rmsd: Flow reconstruction RMSD on training data.
        test_rmsd: Flow reconstruction RMSD on held-out test data.
        flow_rmsd: Alias for test_rmsd (legacy compatibility).
        var_explained: Variance explained by PCA.
        n_params: Number of model parameters.
        train_nll: Negative log-likelihood on training data.
        test_nll: Negative log-likelihood on test data.
        train_gaussianity: Gaussianity score on training data (0-1).
        test_gaussianity: Gaussianity score on test data (0-1).
    """

    model: ResidueFlowModel
    residue: "Residue"
    n_train: int
    n_test: int
    n_atoms: int
    pca_rmsd: float
    train_rmsd: float
    test_rmsd: float
    var_explained: float
    n_params: int
    train_nll: float = 0.0
    test_nll: float = 0.0
    train_gaussianity: float = 0.0
    test_gaussianity: float = 0.0

    @property
    def flow_rmsd(self) -> float:
        """Legacy alias for test_rmsd."""
        return self.test_rmsd

    @property
    def n_instances(self) -> int:
        """Legacy alias for total instances (train + test)."""
        return self.n_train + self.n_test


@register_trainer("flow", ResidueFlowTrainingConfig)
class ResidueFlowTrainer:
    """
    Trainer for ResidueFlowModels across multiple residue types.

    Provides a unified interface for:
    - Training models for individual residue types
    - Training models for all specified residue types
    - Saving trained models to disk

    Example:
        >>> from ciffy.nn.flow.residue import ResidueFlowTrainer, ResidueFlowTrainingConfig
        >>> from ciffy.biochemistry import Residue
        >>>
        >>> config = ResidueFlowTrainingConfig(latent_dim=12, n_epochs=100)
        >>> trainer = ResidueFlowTrainer(config)
        >>>
        >>> # Train all RNA residue types
        >>> results = trainer.train_all(cif_paths, [Residue.A, Residue.C, Residue.G, Residue.U])
        >>>
        >>> # Save models
        >>> trainer.save(results, "models/rna_v1")
    """

    def __init__(self, config: ResidueFlowTrainingConfig | None = None, quiet: bool = False):
        """
        Initialize trainer with configuration.

        Args:
            config: Training configuration. If None, uses defaults.
            quiet: If True, suppress progress output.
        """
        self.config = config or ResidueFlowTrainingConfig()
        self.quiet = quiet
        self.train_dataset: ResidueDataset | None = None

    @property
    def train_dataset_size(self) -> int:
        """Return total training dataset size for progress reporting."""
        if self.train_dataset is not None:
            return len(self.train_dataset)
        return 0

    def _get_device(self) -> str:
        """Resolve device string, handling 'auto'."""
        device = self.config.training.device
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device

    def train(
        self,
        resume_path: str | None = None,
        progress_callback: Callable[[int, int, dict], None] | None = None,
    ) -> dict[str, Any]:
        """
        Unified training interface for CLI integration.

        Trains a single residue type from config and saves the model.

        Args:
            resume_path: Not used (flow models don't support resume).
            progress_callback: Optional callback for progress updates.
                Signature: callback(epoch, total_epochs, metrics)

        Returns:
            Dict with training results:
                - status: 'success' or 'failed'
                - epochs_trained: Number of epochs completed
                - total_epochs: Total configured epochs
                - checkpoint_path: Path to saved model

        Raises:
            ValueError: If no CIF paths or residue is configured.
        """
        from ...validation import validate_training_config, print_validation_result

        verbose = not self.quiet

        # Validate configuration before training
        if verbose:
            print("\nValidating training configuration...")

        validation = validate_training_config(
            data_dir=self.config.data.data_dir,
            cif_patterns=self.config.data.cif_patterns,
            residues=[self.config.data.residue] if self.config.data.residue else None,
            output_dir=self.config.output.checkpoint_dir,
            device=self.config.training.device,
        )

        if verbose:
            print_validation_result(validation, show_info=True, show_summary=True)

        if validation.has_errors:
            error_msgs = [i.message for i in validation.issues if i.level == "error"]
            raise ValueError(
                "Training configuration invalid:\n  " + "\n  ".join(error_msgs)
            )

        # Get CIF paths and residue from config
        cif_paths = self._get_cif_paths()
        residue = self._get_residue()

        if residue is None:
            raise ValueError("No residue configured. Set data.residue in config.")

        # Train single residue
        try:
            result = self.train_single(
                cif_paths, residue, verbose=verbose, progress_callback=progress_callback
            )
        except ValueError as e:
            return {
                "status": "failed",
                "epochs_trained": 0,
                "total_epochs": self.config.training.epochs,
                "error": str(e),
            }

        # Save model
        output_dir = Path(self.config.output.checkpoint_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        result.model.save(output_dir)

        if verbose:
            print(f"\nSaved model to {output_dir}")

        train_result = {
            "status": "success",
            "epochs_trained": self.config.training.epochs,
            "total_epochs": self.config.training.epochs,
            "n_samples": result.n_train,
            "checkpoint_path": str(output_dir),
            "extra_metrics": {
                "residue": residue.name,
                "n_train": result.n_train,
                "n_test": result.n_test,
                "n_atoms": result.n_atoms,
                "pca_rmsd": result.pca_rmsd,
                "train_rmsd": result.train_rmsd,
                "test_rmsd": result.test_rmsd,
                "var_explained": result.var_explained,
                "n_params": result.n_params,
            },
        }

        # Generate training report
        if self.config.output.generate_report:
            self._generate_report(train_result, verbose)

        return train_result

    def _get_cif_paths(self) -> list[Path]:
        """Get CIF file paths from config."""
        paths = []

        # From data_dir
        if self.config.data.data_dir:
            data_path = Path(self.config.data.data_dir)
            if data_path.is_dir():
                paths.extend(data_path.glob("*.cif"))
                paths.extend(data_path.glob("**/*.cif"))

        # From cif_patterns
        if self.config.data.cif_patterns:
            for pattern in self.config.data.cif_patterns:
                paths.extend(Path(p) for p in glob(pattern))

        # Remove duplicates while preserving order
        seen = set()
        unique_paths = []
        for p in paths:
            p_resolved = p.resolve()
            if p_resolved not in seen:
                seen.add(p_resolved)
                unique_paths.append(p)

        return unique_paths

    def _get_residue(self) -> "Residue | None":
        """Get residue type from config."""
        from ciffy.biochemistry import Residue

        if not self.config.data.residue:
            return None

        try:
            return getattr(Residue, self.config.data.residue)
        except AttributeError:
            if not self.quiet:
                print(f"Warning: Unknown residue '{self.config.data.residue}'")
            return None

    def _generate_report(
        self,
        train_result: dict[str, Any],
        verbose: bool = True,
    ) -> None:
        """Generate training report.

        Args:
            train_result: Training result dictionary.
            verbose: Print report path.
        """
        from ...report import TrainingReport

        # Determine report path
        if self.config.output.report_path:
            report_path = Path(self.config.output.report_path)
        else:
            report_path = Path(self.config.output.checkpoint_dir) / "training_report.html"

        # Build config dict for report
        config_dict = {
            "model": {
                "latent_dim": self.config.model.latent_dim,
                "n_layers": self.config.model.n_layers,
                "hidden_dim": self.config.model.hidden_dim,
                "bound": self.config.model.bound,
            },
            "training": {
                "n_epochs": self.config.training.epochs,
                "batch_size": self.config.data.batch_size,
                "lr": self.config.training.lr,
                "device": self.config.training.device,
            },
            "data": {
                "data_dir": self.config.data.data_dir,
                "residues": self.config.data.residue_names,
                "min_coverage": self.config.data.min_coverage,
                "train_split": self.config.data.train_split,
            },
        }

        # Create and save report
        report = TrainingReport(
            model_type="flow",
            config=config_dict,
            results=train_result,
        )

        saved_path = report.save(report_path)

        if verbose:
            print(f"\n📄 Training report saved to: {saved_path}")

    def train_single(
        self,
        cif_paths: list[Path],
        residue: "Residue",
        verbose: bool = True,
        progress_callback: Callable[[int, int, dict], None] | None = None,
    ) -> TrainingResult:
        """
        Train a ResidueFlowModel for a single residue type.

        Data is split by structure (CIF file) to avoid leakage. Residues from
        the same structure are correlated and should not appear in both
        training and test sets.

        Args:
            cif_paths: List of paths to CIF files.
            residue: Residue type to train on.
            verbose: Print progress information.
            progress_callback: Optional callback for progress updates.
                Signature: callback(epoch, total_epochs, metrics)

        Returns:
            TrainingResult with trained model and metrics on held-out test set.

        Raises:
            ValueError: If no residues of the specified type are found.
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Training {residue.name}")
            print(f"{'='*60}")

        # Split structures into train/test sets
        split = split_by_structure(
            cif_paths,
            train=self.config.data.train_split,
            val=0.0,  # No validation set for now
            test=self.config.data.test_split,
            seed=self.config.data.split_seed,
        )

        if verbose:
            print(f"Split: {len(split.train)} train, {len(split.test)} test structures")

        # Create training dataset
        self.train_dataset = ResidueDataset(
            split.train,
            residue,
            min_coverage=self.config.data.min_coverage,
            max_bond_length=self.config.data.max_bond_length,
            verbose=verbose,
        )

        n_train = len(self.train_dataset)
        n_atoms = self.train_dataset.n_atoms
        atoms = self.train_dataset.atoms

        if n_train == 0:
            raise ValueError(f"No {residue.name} residues found in training structures")

        train_extended = self.train_dataset.data

        # Create test dataset (if any test structures)
        test_extended = None
        n_test = 0
        if len(split.test) > 0:
            try:
                test_dataset = ResidueDataset(
                    split.test,
                    residue,
                    min_coverage=self.config.data.min_coverage,
                    max_bond_length=self.config.data.max_bond_length,
                    verbose=False,  # Quiet for test extraction
                )
                n_test = len(test_dataset)
                if n_test > 0:
                    test_extended = test_dataset.data
            except ValueError:
                # No test residues found - this is okay
                pass

        if verbose:
            print(f"Extracted {n_train} train, {n_test} test instances with {n_atoms} atoms each")
            print(f"Extended representation: {train_extended.shape[1]} dimensions")

        # Train with proper train/test split
        flow, info = train_pca_flow(
            train_data=train_extended,
            test_data=test_extended,
            n_atoms=n_atoms,
            latent_dim=self.config.model.latent_dim,
            n_layers=self.config.model.n_layers,
            hidden_dim=self.config.model.hidden_dim,
            bound=self.config.model.bound,
            n_epochs=self.config.training.epochs,
            batch_size=self.config.data.batch_size,
            lr=self.config.training.lr,
            device=self._get_device(),
            verbose=verbose,
            progress_callback=progress_callback,
        )

        # Create ResidueFlowModel wrapper
        model = ResidueFlowModel(
            flow=flow,
            residue=residue,
            atom_indices=atoms,
            n_atoms=n_atoms,
        )

        return TrainingResult(
            model=model,
            residue=residue,
            n_train=n_train,
            n_test=n_test,
            n_atoms=n_atoms,
            pca_rmsd=info["pca_rmsd"],
            train_rmsd=info["train_rmsd"],
            test_rmsd=info["test_rmsd"],
            var_explained=info["var_explained"],
            n_params=info["n_params"],
            train_nll=info["train_nll"],
            test_nll=info["test_nll"],
            train_gaussianity=info["train_gaussianity"],
            test_gaussianity=info["test_gaussianity"],
        )

    def train_single_presplit(
        self,
        train_paths: list[Path],
        test_paths: list[Path],
        residue: "Residue",
        verbose: bool = True,
        progress_callback: Callable[[int, int, dict], None] | None = None,
    ) -> TrainingResult:
        """
        Train a ResidueFlowModel with pre-split train/test paths.

        Use this for data scaling experiments where you want the same test set
        across experiments with different training set sizes.

        Args:
            train_paths: List of training structure paths.
            test_paths: List of test structure paths.
            residue: Residue type to train on.
            verbose: Print progress information.
            progress_callback: Optional callback for progress updates.

        Returns:
            TrainingResult with trained model and metrics on test set.
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Training {residue.name}")
            print(f"{'='*60}")
            print(f"Pre-split: {len(train_paths)} train, {len(test_paths)} test structures")

        # Create training dataset
        self.train_dataset = ResidueDataset(
            train_paths,
            residue,
            min_coverage=self.config.data.min_coverage,
            max_bond_length=self.config.data.max_bond_length,
            verbose=verbose,
        )

        n_train = len(self.train_dataset)
        n_atoms = self.train_dataset.n_atoms
        atoms = self.train_dataset.atoms

        if n_train == 0:
            raise ValueError(f"No {residue.name} residues found in training structures")

        train_extended = self.train_dataset.data

        # Create test dataset
        test_extended = None
        n_test = 0
        if len(test_paths) > 0:
            try:
                test_dataset = ResidueDataset(
                    test_paths,
                    residue,
                    min_coverage=self.config.data.min_coverage,
                    max_bond_length=self.config.data.max_bond_length,
                    verbose=False,
                )
                n_test = len(test_dataset)
                if n_test > 0:
                    test_extended = test_dataset.data
            except ValueError:
                pass

        if verbose:
            print(f"Extracted {n_train} train, {n_test} test instances with {n_atoms} atoms each")
            print(f"Extended representation: {train_extended.shape[1]} dimensions")

        # Train with provided split
        flow, info = train_pca_flow(
            train_data=train_extended,
            test_data=test_extended,
            n_atoms=n_atoms,
            latent_dim=self.config.model.latent_dim,
            n_layers=self.config.model.n_layers,
            hidden_dim=self.config.model.hidden_dim,
            bound=self.config.model.bound,
            n_epochs=self.config.training.epochs,
            batch_size=self.config.data.batch_size,
            lr=self.config.training.lr,
            device=self._get_device(),
            verbose=verbose,
            progress_callback=progress_callback,
        )

        # Create ResidueFlowModel wrapper
        model = ResidueFlowModel(
            flow=flow,
            residue=residue,
            atom_indices=atoms,
            n_atoms=n_atoms,
        )

        return TrainingResult(
            model=model,
            residue=residue,
            n_train=n_train,
            n_test=n_test,
            n_atoms=n_atoms,
            pca_rmsd=info["pca_rmsd"],
            train_rmsd=info["train_rmsd"],
            test_rmsd=info["test_rmsd"],
            var_explained=info["var_explained"],
            n_params=info["n_params"],
            train_nll=info["train_nll"],
            test_nll=info["test_nll"],
            train_gaussianity=info["train_gaussianity"],
            test_gaussianity=info["test_gaussianity"],
        )

