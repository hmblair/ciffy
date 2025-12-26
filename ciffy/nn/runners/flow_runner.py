"""
Runner for flow model hyperparameter experiments.

Provides utilities for comparing flow models with different hyperparameters,
especially PCA dimension (latent_dim), in parallel across CPUs.

Example:
    >>> from ciffy.nn.runners.flow_runner import run_flow_experiments
    >>> from ciffy.biochemistry import Residue
    >>>
    >>> configs = [
    ...     FlowExperimentConfig(name="dim8", latent_dim=8),
    ...     FlowExperimentConfig(name="dim12", latent_dim=12),
    ...     FlowExperimentConfig(name="dim16", latent_dim=16),
    ... ]
    >>> results = run_flow_experiments(
    ...     configs,
    ...     cif_paths=["data/*.cif"],
    ...     residues=[Residue.A, Residue.C, Residue.G, Residue.U],
    ...     parallel=True,
    ...     max_workers=4,
    ... )
    >>> print(format_flow_results_table(results))
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from glob import glob
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

try:
    from rich.table import Table

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from .utils import (
    format_duration,
    format_status,
    format_progress_bar,
    ParallelRunner,
)

if TYPE_CHECKING:
    from multiprocessing import Queue
    from ciffy.biochemistry import Residue


@dataclass
class FlowExperimentConfig:
    """Configuration for a single flow experiment.

    Attributes:
        name: Experiment identifier.
        latent_dim: Number of PCA components / latent dimensions.
        n_layers: Number of flow layers.
        hidden_dim: Hidden dimension in coupling networks.
        bound: Tanh bound for decode (in std devs).
        n_epochs: Number of training epochs.
        batch_size: Batch size for training.
        lr: Learning rate.
        min_coverage: Minimum fraction of instances an atom must appear in.
        max_bond_length: Maximum O3'-P distance for connected residues.
        train_split: Fraction of structures for training (default: 0.8).
        test_split: Fraction of structures for testing (default: 0.2).
        split_seed: Random seed for reproducible splits (default: 42).
        max_structures: Maximum training structures (None = use all).
    """

    name: str
    latent_dim: int = 12
    n_layers: int = 8
    hidden_dim: int = 64
    bound: float | None = None
    n_epochs: int = 200
    batch_size: int = 256
    lr: float = 1e-3
    min_coverage: float = 0.9
    max_bond_length: float = 2.0
    train_split: float = 0.8
    test_split: float = 0.2
    split_seed: int | None = 42
    max_structures: int | None = None


@dataclass
class FlowExperimentResult:
    """Result from a flow experiment.

    Attributes:
        name: Experiment identifier.
        config: The configuration used.
        status: 'success' or 'failed'.
        residue_results: Dict mapping residue to per-residue metrics.
        mean_pca_rmsd: Mean PCA RMSD across residues.
        mean_train_rmsd: Mean train RMSD across residues.
        mean_test_rmsd: Mean test RMSD across residues (held-out).
        mean_flow_rmsd: Alias for mean_test_rmsd (legacy).
        mean_var_explained: Mean variance explained across residues.
        mean_train_nll: Mean train NLL across residues.
        mean_test_nll: Mean test NLL across residues.
        mean_test_gaussianity: Mean gaussianity score on test data.
        total_params: Total parameters across all residue models.
        n_train_structures: Number of training structures used.
        duration_seconds: Training time.
        error: Error message if failed.
        output_dir: Directory where models were saved.
    """

    name: str
    config: FlowExperimentConfig
    status: str  # 'success', 'failed'
    residue_results: dict = field(default_factory=dict)
    mean_pca_rmsd: float = 0.0
    mean_train_rmsd: float = 0.0
    mean_test_rmsd: float = 0.0
    mean_var_explained: float = 0.0
    mean_train_nll: float = 0.0
    mean_test_nll: float = 0.0
    mean_test_gaussianity: float = 0.0
    total_params: int = 0
    n_train_structures: int = 0
    duration_seconds: float = 0.0
    error: Optional[str] = None
    output_dir: Optional[str] = None

    @property
    def mean_flow_rmsd(self) -> float:
        """Legacy alias for mean_test_rmsd."""
        return self.mean_test_rmsd


@dataclass
class _FlowJobConfig:
    """Internal config that bundles experiment config with shared data paths.

    This is what gets passed to the parallel runner. We bundle the experiment
    config with the CIF paths and residues so each worker has everything it needs.

    For data scaling experiments, train_paths and test_paths can be set directly
    to use a pre-computed split (e.g., from DataScalingSplit).
    """

    experiment: FlowExperimentConfig
    cif_paths: list[str]  # Strings for pickling (used if train/test_paths not set)
    residue_names: list[str]  # Names for pickling
    output_dir: Optional[str] = None
    # Pre-split paths (optional, for data scaling experiments)
    train_paths: Optional[list[str]] = None
    test_paths: Optional[list[str]] = None


def _run_flow_job(
    job_config: _FlowJobConfig,
    device: str,
    progress_queue: "Queue",
) -> FlowExperimentResult:
    """Run a single flow experiment. Module-level function for pickling."""
    from ciffy.biochemistry import Residue
    from ciffy.nn.flow.residue import (
        ResidueFlowTrainer,
        ResidueFlowTrainingConfig,
    )

    config = job_config.experiment
    cif_paths = [Path(p) for p in job_config.cif_paths]
    residues = [getattr(Residue, name) for name in job_config.residue_names]
    output_dir = Path(job_config.output_dir) if job_config.output_dir else None

    # Pre-split paths (for data scaling experiments)
    train_paths = [Path(p) for p in job_config.train_paths] if job_config.train_paths else None
    test_paths = [Path(p) for p in job_config.test_paths] if job_config.test_paths else None

    start_time = time.time()

    def send_progress(status: str, current: int = 0, total: int = 0):
        if progress_queue is not None:
            try:
                progress_queue.put({
                    "name": config.name,
                    "status": status,
                    "current": current,
                    "total": total,
                    "device": device,
                    "time": time.time() - start_time,
                })
            except Exception:
                pass

    try:
        send_progress("running", 0, len(residues))

        # Create training config
        training_config = ResidueFlowTrainingConfig(
            latent_dim=config.latent_dim,
            n_layers=config.n_layers,
            hidden_dim=config.hidden_dim,
            bound=config.bound,
            n_epochs=config.n_epochs,
            batch_size=config.batch_size,
            lr=config.lr,
            min_coverage=config.min_coverage,
            max_bond_length=config.max_bond_length,
            device=device,
            train_split=config.train_split,
            test_split=config.test_split,
            split_seed=config.split_seed,
        )

        trainer = ResidueFlowTrainer(training_config)

        # Train residues one at a time with progress updates
        results = {}
        for i, residue in enumerate(residues):
            send_progress("running", i, len(residues))
            if train_paths is not None and test_paths is not None:
                result = trainer.train_single_presplit(
                    train_paths, test_paths, residue, verbose=False
                )
            else:
                result = trainer.train_single(cif_paths, residue, verbose=False)
            results[residue] = result

        duration = time.time() - start_time

        # Aggregate metrics
        residue_results = {}
        pca_rmsds = []
        train_rmsds = []
        test_rmsds = []
        var_explaineds = []
        train_nlls = []
        test_nlls = []
        test_gaussianities = []
        total_params = 0

        for residue, result in results.items():
            residue_results[residue.name] = {
                "n_train": result.n_train,
                "n_test": result.n_test,
                "n_atoms": result.n_atoms,
                "pca_rmsd": result.pca_rmsd,
                "train_rmsd": result.train_rmsd,
                "test_rmsd": result.test_rmsd,
                "var_explained": result.var_explained,
                "n_params": result.n_params,
                "train_nll": result.train_nll,
                "test_nll": result.test_nll,
                "train_gaussianity": result.train_gaussianity,
                "test_gaussianity": result.test_gaussianity,
            }
            pca_rmsds.append(result.pca_rmsd)
            train_rmsds.append(result.train_rmsd)
            test_rmsds.append(result.test_rmsd)
            var_explaineds.append(result.var_explained)
            train_nlls.append(result.train_nll)
            test_nlls.append(result.test_nll)
            test_gaussianities.append(result.test_gaussianity)
            total_params += result.n_params

        # Save models if output directory specified
        save_dir = None
        if output_dir is not None:
            save_dir = output_dir / config.name
            trainer.save(results, save_dir)

        send_progress("complete", len(residues), len(residues))

        # Determine number of training structures
        if train_paths is not None:
            n_train_structures = len(train_paths)
        else:
            # Estimate from split ratio
            n_train_structures = int(len(cif_paths) * config.train_split)

        return FlowExperimentResult(
            name=config.name,
            config=config,
            status="success",
            residue_results=residue_results,
            mean_pca_rmsd=float(np.mean(pca_rmsds)) if pca_rmsds else 0.0,
            mean_train_rmsd=float(np.mean(train_rmsds)) if train_rmsds else 0.0,
            mean_test_rmsd=float(np.mean(test_rmsds)) if test_rmsds else 0.0,
            mean_var_explained=float(np.mean(var_explaineds)) if var_explaineds else 0.0,
            mean_train_nll=float(np.mean(train_nlls)) if train_nlls else 0.0,
            mean_test_nll=float(np.mean(test_nlls)) if test_nlls else 0.0,
            mean_test_gaussianity=float(np.mean(test_gaussianities)) if test_gaussianities else 0.0,
            total_params=total_params,
            n_train_structures=n_train_structures,
            duration_seconds=duration,
            output_dir=str(save_dir) if save_dir else None,
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        send_progress("failed")
        return FlowExperimentResult(
            name=config.name,
            config=config,
            status="failed",
            duration_seconds=time.time() - start_time,
            error=str(e),
        )


def _create_flow_progress_table(states: dict[str, dict]) -> "Table":
    """Create progress table for flow experiments."""
    table = Table(title="Flow Experiments", show_header=True, header_style="bold")
    table.add_column("Experiment", style="cyan", width=20)
    table.add_column("Status", width=10)
    table.add_column("Progress", width=15)
    table.add_column("Device", width=10)
    table.add_column("Time", width=10)

    for name, state in states.items():
        status = state.get("status", "pending")
        current = state.get("current", 0)
        total = state.get("total", 0)
        device = state.get("device", "")
        elapsed = state.get("time", 0)

        status_str = format_status(status)
        progress_str = format_progress_bar(current, total)
        time_str = format_duration(elapsed)

        table.add_row(name, status_str, progress_str, device, time_str)

    return table


class FlowExperimentRunner(ParallelRunner[_FlowJobConfig, FlowExperimentResult]):
    """Runner for flow model experiments.

    Runs multiple flow experiment configurations in parallel across CPUs/GPUs.

    Example:
        >>> runner = FlowExperimentRunner(parallel=True, max_workers=4)
        >>> job_configs = runner.prepare_jobs(
        ...     configs=[FlowExperimentConfig("dim8", latent_dim=8)],
        ...     cif_paths=["data/*.cif"],
        ...     residues=[Residue.A, Residue.C, Residue.G, Residue.U],
        ... )
        >>> results = runner.run(job_configs)
    """

    def run_job(
        self,
        config: _FlowJobConfig,
        device: str,
        progress_queue: "Queue",
    ) -> FlowExperimentResult:
        """Run a single flow experiment."""
        return _run_flow_job(config, device, progress_queue)

    def get_job_name(self, config: _FlowJobConfig) -> str:
        """Get experiment name from config."""
        return config.experiment.name

    def create_progress_table(self, states: dict[str, dict]) -> "Table":
        """Create flow experiment progress table."""
        return _create_flow_progress_table(states)

    def create_failed_result(
        self,
        config: _FlowJobConfig,
        device: str,
        error: str,
    ) -> FlowExperimentResult:
        """Create a failed result."""
        return FlowExperimentResult(
            name=config.experiment.name,
            config=config.experiment,
            status="failed",
            error=error,
        )

    def prepare_jobs(
        self,
        configs: list[FlowExperimentConfig],
        cif_paths: list[str | Path],
        residues: list["Residue"],
        output_dir: Optional[str | Path] = None,
    ) -> list[_FlowJobConfig]:
        """Prepare job configs by bundling experiment configs with data paths.

        Args:
            configs: List of experiment configurations.
            cif_paths: List of CIF file paths or glob patterns.
            residues: List of residue types to train.
            output_dir: Optional base directory to save models.

        Returns:
            List of _FlowJobConfig ready for runner.run().
        """
        # Expand glob patterns
        expanded_paths = []
        for path in cif_paths:
            path_str = str(path)
            if "*" in path_str:
                expanded_paths.extend(glob(path_str))
            else:
                expanded_paths.append(path_str)

        if not expanded_paths:
            raise ValueError("No CIF files found")

        # Convert to strings for pickling
        cif_path_strs = [str(p) for p in expanded_paths]
        residue_names = [r.name for r in residues]
        output_str = str(output_dir) if output_dir else None

        return [
            _FlowJobConfig(
                experiment=config,
                cif_paths=cif_path_strs,
                residue_names=residue_names,
                output_dir=output_str,
            )
            for config in configs
        ]


def run_flow_experiments(
    configs: list[FlowExperimentConfig],
    cif_paths: list[str | Path],
    residues: list["Residue"],
    parallel: bool = True,
    max_workers: int | None = None,
    device: str = "cpu",
    output_dir: Optional[str | Path] = None,
    verbose: bool = True,
) -> list[FlowExperimentResult]:
    """
    Run multiple flow experiments with different configurations.

    Experiments can run in parallel across CPUs for faster comparison.

    Args:
        configs: List of experiment configurations to compare.
        cif_paths: List of CIF file paths or glob patterns.
        residues: List of residue types to train.
        parallel: If True, run experiments in parallel.
        max_workers: Maximum parallel workers. None for auto (uses CPU count).
        device: Device to train on ('cpu', 'cuda', etc.).
        output_dir: Optional base directory to save models.
        verbose: Print progress and results.

    Returns:
        List of FlowExperimentResult in the same order as configs.

    Example:
        >>> configs = [
        ...     FlowExperimentConfig("pca8", latent_dim=8),
        ...     FlowExperimentConfig("pca12", latent_dim=12),
        ...     FlowExperimentConfig("pca16", latent_dim=16),
        ... ]
        >>> results = run_flow_experiments(
        ...     configs,
        ...     cif_paths=["data/*.cif"],
        ...     residues=[Residue.A, Residue.C, Residue.G, Residue.U],
        ...     parallel=True,
        ...     max_workers=4,
        ... )
    """
    runner = FlowExperimentRunner(
        parallel=parallel,
        max_workers=max_workers,
        device=device,
    )

    job_configs = runner.prepare_jobs(
        configs=configs,
        cif_paths=cif_paths,
        residues=residues,
        output_dir=output_dir,
    )

    results = runner.run(job_configs)

    if verbose:
        print("\n" + format_flow_results_table(results))

    return results


def format_flow_results_table(
    results: list[FlowExperimentResult], show_residues: bool = False
) -> str:
    """
    Format flow experiment results as an ASCII table.

    Args:
        results: List of FlowExperimentResult objects.
        show_residues: If True, show per-residue breakdown.

    Returns:
        Formatted table string suitable for terminal output.
    """
    if not results:
        return "No results to display."

    lines = []

    # Summary table
    lines.append("=" * 80)
    lines.append("FLOW EXPERIMENT COMPARISON")
    lines.append("=" * 80)

    columns = [
        ("Experiment", 15),
        ("Latent", 7),
        ("Var%", 7),
        ("Test RMSD", 10),
        ("Train NLL", 10),
        ("Test NLL", 10),
        ("Gauss", 7),
        ("Params", 10),
        ("Time", 10),
    ]

    header = "  ".join(f"{name:<{width}}" for name, width in columns)
    separator = "  ".join("-" * width for _, width in columns)

    lines.append(header)
    lines.append(separator)

    for r in results:
        if r.status == "success":
            row_values = [
                r.name[:15],
                str(r.config.latent_dim),
                f"{r.mean_var_explained*100:.1f}%",
                f"{r.mean_test_rmsd:.4f}",
                f"{r.mean_train_nll:.2f}",
                f"{r.mean_test_nll:.2f}",
                f"{r.mean_test_gaussianity:.3f}",
                f"{r.total_params:,}",
                format_duration(r.duration_seconds),
            ]
        else:
            row_values = [
                r.name[:15],
                str(r.config.latent_dim),
                "N/A",
                "N/A",
                "N/A",
                "N/A",
                "N/A",
                "N/A",
                format_duration(r.duration_seconds),
            ]

        row = "  ".join(
            f"{val:<{width}}" for val, (_, width) in zip(row_values, columns)
        )
        lines.append(row)

    lines.append(separator)

    # Show errors
    failed = [r for r in results if r.status == "failed"]
    if failed:
        lines.append("")
        lines.append("Errors:")
        for r in failed:
            lines.append(f"  {r.name}: {r.error}")

    # Per-residue breakdown if requested
    if show_residues:
        successful = [r for r in results if r.status == "success"]
        if successful:
            lines.append("")
            lines.append("Per-Residue Breakdown:")
            lines.append("-" * 80)

            # Collect all residue names
            all_residues = set()
            for r in successful:
                all_residues.update(r.residue_results.keys())

            for residue_name in sorted(all_residues):
                lines.append(f"\n{residue_name}:")
                res_header = "  Experiment       Train   Test    Var%    PCA RMSD  Train     Test"
                lines.append(res_header)
                lines.append("  " + "-" * 70)

                for r in successful:
                    if residue_name in r.residue_results:
                        res = r.residue_results[residue_name]
                        lines.append(
                            f"  {r.name:<15} "
                            f"{res['n_train']:<7} "
                            f"{res['n_test']:<7} "
                            f"{res['var_explained']*100:5.1f}%   "
                            f"{res['pca_rmsd']:.4f}    "
                            f"{res['train_rmsd']:.4f}    "
                            f"{res['test_rmsd']:.4f}"
                        )

    return "\n".join(lines)


def create_latent_dim_sweep(
    base_name: str = "pca",
    latent_dims: list[int] | None = None,
    **kwargs,
) -> list[FlowExperimentConfig]:
    """
    Create configurations for a latent dimension sweep.

    Args:
        base_name: Base name for experiments.
        latent_dims: List of latent dimensions to test. Default [4, 8, 12, 16, 20].
        **kwargs: Additional config parameters to pass to all experiments.

    Returns:
        List of FlowExperimentConfig objects.

    Example:
        >>> configs = create_latent_dim_sweep("pca", [8, 12, 16], n_epochs=100)
        >>> # Creates configs: pca_dim8, pca_dim12, pca_dim16
    """
    if latent_dims is None:
        latent_dims = [4, 8, 12, 16, 20]

    return [
        FlowExperimentConfig(
            name=f"{base_name}_dim{dim}",
            latent_dim=dim,
            **kwargs,
        )
        for dim in latent_dims
    ]


def create_layer_sweep(
    base_name: str = "layers",
    n_layers_list: list[int] | None = None,
    latent_dim: int = 12,
    **kwargs,
) -> list[FlowExperimentConfig]:
    """
    Create configurations for a layer count sweep.

    Args:
        base_name: Base name for experiments.
        n_layers_list: List of layer counts to test. Default [4, 8, 12, 16].
        latent_dim: Latent dimension for all experiments.
        **kwargs: Additional config parameters.

    Returns:
        List of FlowExperimentConfig objects.
    """
    if n_layers_list is None:
        n_layers_list = [4, 8, 12, 16]

    return [
        FlowExperimentConfig(
            name=f"{base_name}_{n}L",
            latent_dim=latent_dim,
            n_layers=n,
            **kwargs,
        )
        for n in n_layers_list
    ]


def create_data_scaling_sweep(
    base_name: str = "scale",
    train_sizes: list[int] | None = None,
    latent_dims: list[int] | None = None,
    **kwargs,
) -> list[FlowExperimentConfig]:
    """
    Create configurations for a data scaling sweep (train_size × latent_dim grid).

    Creates a grid of experiments varying training set size and latent dimension.
    Use with `prepare_scaling_jobs` to ensure consistent test sets across experiments.

    Args:
        base_name: Base name for experiments.
        train_sizes: List of training set sizes. Default [50, 100, 200, 500].
        latent_dims: List of latent dimensions. Default [12] (single value for data scaling).
        **kwargs: Additional config parameters.

    Returns:
        List of FlowExperimentConfig objects with names like "scale_n50_dim12".

    Example:
        >>> configs = create_data_scaling_sweep(
        ...     "scale", train_sizes=[50, 100, 200], latent_dims=[8, 12]
        ... )
        >>> # Creates: scale_n50_dim8, scale_n50_dim12, scale_n100_dim8, ...
    """
    if train_sizes is None:
        train_sizes = [50, 100, 200, 500]
    if latent_dims is None:
        latent_dims = [12]

    configs = []
    for n_train in train_sizes:
        for dim in latent_dims:
            configs.append(
                FlowExperimentConfig(
                    name=f"{base_name}_n{n_train}_dim{dim}",
                    latent_dim=dim,
                    max_structures=n_train,
                    **kwargs,
                )
            )

    return configs


def prepare_scaling_jobs(
    configs: list[FlowExperimentConfig],
    cif_paths: list[str | Path],
    residues: list["Residue"],
    test_fraction: float = 0.2,
    seed: int | None = 42,
    output_dir: Optional[str | Path] = None,
) -> list[_FlowJobConfig]:
    """
    Prepare job configs for data scaling experiments with consistent test set.

    Uses DataScalingSplit to create a shuffled training pool where smaller sizes
    are prefixes of larger sizes. All experiments use the same held-out test set.

    Args:
        configs: List of experiment configurations (should have max_structures set).
        cif_paths: List of CIF file paths or glob patterns.
        residues: List of residue types to train.
        test_fraction: Fraction of data for test set (default: 0.2).
        seed: Random seed for reproducibility.
        output_dir: Optional base directory to save models.

    Returns:
        List of _FlowJobConfig with pre-split train/test paths.

    Example:
        >>> configs = create_data_scaling_sweep("scale", [50, 100, 200])
        >>> jobs = prepare_scaling_jobs(
        ...     configs, cif_paths, residues, test_fraction=0.2
        ... )
        >>> runner = FlowExperimentRunner()
        >>> results = runner.run(jobs)
    """
    from ..split import create_scaling_split

    # Expand glob patterns
    expanded_paths = []
    for path in cif_paths:
        path_str = str(path)
        if "*" in path_str:
            expanded_paths.extend(glob(path_str))
        else:
            expanded_paths.append(path_str)

    if not expanded_paths:
        raise ValueError("No CIF files found")

    # Create data scaling split with shuffled training pool
    scaling_split = create_scaling_split(
        expanded_paths,
        test_fraction=test_fraction,
        seed=seed,
    )

    # Convert residues to names for pickling
    residue_names = [r.name for r in residues]
    output_str = str(output_dir) if output_dir else None

    # Create job configs with pre-split paths
    job_configs = []
    for config in configs:
        train_size = config.max_structures
        if train_size is None:
            # Use full training pool
            train_size = scaling_split.max_train_size

        train_paths = [str(p) for p in scaling_split.get_train(train_size)]
        test_paths = [str(p) for p in scaling_split.test]

        job_configs.append(
            _FlowJobConfig(
                experiment=config,
                cif_paths=expanded_paths,  # Keep for compatibility
                residue_names=residue_names,
                output_dir=output_str,
                train_paths=train_paths,
                test_paths=test_paths,
            )
        )

    return job_configs


def run_data_scaling_experiments(
    configs: list[FlowExperimentConfig],
    cif_paths: list[str | Path],
    residues: list["Residue"],
    test_fraction: float = 0.2,
    seed: int | None = 42,
    parallel: bool = True,
    max_workers: int | None = None,
    device: str = "cpu",
    output_dir: Optional[str | Path] = None,
    verbose: bool = True,
) -> list[FlowExperimentResult]:
    """
    Run data scaling experiments with consistent test set across all sizes.

    This is a convenience wrapper that combines create_data_scaling_sweep,
    prepare_scaling_jobs, and FlowExperimentRunner.run().

    Args:
        configs: Experiment configs (should have max_structures set).
        cif_paths: List of CIF file paths or glob patterns.
        residues: List of residue types to train.
        test_fraction: Fraction of data for test set (default: 0.2).
        seed: Random seed for reproducibility.
        parallel: If True, run experiments in parallel.
        max_workers: Maximum parallel workers.
        device: Device to train on.
        output_dir: Optional directory to save models.
        verbose: Print progress and results.

    Returns:
        List of FlowExperimentResult in the same order as configs.

    Example:
        >>> configs = create_data_scaling_sweep("scale", [50, 100, 200], [8, 12])
        >>> results = run_data_scaling_experiments(
        ...     configs,
        ...     cif_paths=["data/*.cif"],
        ...     residues=[Residue.A, Residue.C, Residue.G, Residue.U],
        ...     test_fraction=0.2,
        ... )
    """
    job_configs = prepare_scaling_jobs(
        configs=configs,
        cif_paths=cif_paths,
        residues=residues,
        test_fraction=test_fraction,
        seed=seed,
        output_dir=output_dir,
    )

    runner = FlowExperimentRunner(
        parallel=parallel,
        max_workers=max_workers,
        device=device,
    )

    results = runner.run(job_configs)

    if verbose:
        print("\n" + format_scaling_results_table(results))

    return results


def format_scaling_results_table(results: list[FlowExperimentResult]) -> str:
    """
    Format data scaling experiment results as an ASCII table.

    Shows training set size, latent dimension, and all metrics.

    Args:
        results: List of FlowExperimentResult objects.

    Returns:
        Formatted table string.
    """
    if not results:
        return "No results to display."

    lines = []

    lines.append("=" * 100)
    lines.append("DATA SCALING EXPERIMENT RESULTS")
    lines.append("=" * 100)

    columns = [
        ("Experiment", 18),
        ("N Train", 8),
        ("Dim", 5),
        ("Var%", 7),
        ("Test RMSD", 10),
        ("Train NLL", 10),
        ("Test NLL", 10),
        ("Gauss", 7),
        ("Params", 10),
        ("Time", 8),
    ]

    header = "  ".join(f"{name:<{width}}" for name, width in columns)
    separator = "  ".join("-" * width for _, width in columns)

    lines.append(header)
    lines.append(separator)

    for r in results:
        if r.status == "success":
            row_values = [
                r.name[:18],
                str(r.n_train_structures),
                str(r.config.latent_dim),
                f"{r.mean_var_explained*100:.1f}%",
                f"{r.mean_test_rmsd:.4f}",
                f"{r.mean_train_nll:.2f}",
                f"{r.mean_test_nll:.2f}",
                f"{r.mean_test_gaussianity:.3f}",
                f"{r.total_params:,}",
                format_duration(r.duration_seconds),
            ]
        else:
            row_values = [
                r.name[:18],
                "N/A",
                str(r.config.latent_dim),
                "N/A",
                "N/A",
                "N/A",
                "N/A",
                "N/A",
                "N/A",
                format_duration(r.duration_seconds),
            ]

        row = "  ".join(
            f"{val:<{width}}" for val, (_, width) in zip(row_values, columns)
        )
        lines.append(row)

    lines.append(separator)

    # Show errors
    failed = [r for r in results if r.status == "failed"]
    if failed:
        lines.append("")
        lines.append("Errors:")
        for r in failed:
            lines.append(f"  {r.name}: {r.error}")

    return "\n".join(lines)


__all__ = [
    "FlowExperimentConfig",
    "FlowExperimentResult",
    "FlowExperimentRunner",
    "run_flow_experiments",
    "format_flow_results_table",
    "create_latent_dim_sweep",
    "create_layer_sweep",
    # Data scaling experiments
    "create_data_scaling_sweep",
    "prepare_scaling_jobs",
    "run_data_scaling_experiments",
    "format_scaling_results_table",
]
