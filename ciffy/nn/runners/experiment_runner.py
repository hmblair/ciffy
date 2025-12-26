"""
Multi-experiment runner for parallel training across GPUs.

Provides utilities for running multiple training configurations in parallel
with automatic GPU assignment and result comparison.

Example:
    >>> from ciffy.nn.runners import run_experiments, format_results_table
    >>>
    >>> results = run_experiments(
    ...     ["config1.yaml", "config2.yaml"],
    ...     parallel=True,
    ...     device="auto",
    ... )
    >>> print(format_results_table(results))
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import TYPE_CHECKING

try:
    from rich.table import Table

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from ..training import ExperimentResult
from .utils import (
    format_duration,
    format_status,
    format_progress_bar,
    ParallelRunner,
)

if TYPE_CHECKING:
    from multiprocessing import Queue


def _run_vae_experiment(
    config_path: Path,
    device: str,
    progress_queue: "Queue",
    scripts_dir: str,
) -> ExperimentResult:
    """
    Run a single training experiment in a subprocess with progress reporting.

    This is a module-level function for pickling compatibility.
    """
    experiment_name = config_path.stem
    start_time = time.time()

    # Create temp log file
    log_fd, log_file = tempfile.mkstemp(
        prefix=f"ciffy_{experiment_name}_", suffix=".log"
    )
    os.close(log_fd)

    def send_progress(
        status: str, epoch: int = 0, total_epochs: int = 0, loss: float | None = None
    ):
        if progress_queue is not None:
            try:
                progress_queue.put({
                    "name": experiment_name,
                    "status": status,
                    "epoch": epoch,
                    "total_epochs": total_epochs,
                    "loss": loss,
                    "device": device,
                    "time": time.time() - start_time,
                })
            except Exception:
                pass

    # Capture stdout/stderr to log file
    log_buffer = io.StringIO()
    old_stdout, old_stderr = sys.stdout, sys.stderr

    try:
        sys.stdout = log_buffer
        sys.stderr = log_buffer

        # Add scripts directory to path for importing train_vae
        if scripts_dir and scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)

        from train_vae import train_vae, load_config

        # Load config to get total epochs
        config = load_config(str(config_path))
        total_epochs = config.training.epochs

        send_progress("running", 0, total_epochs)

        # Create progress callback
        def progress_callback(epoch: int, total: int, metrics: dict):
            send_progress("running", epoch, total, metrics.get("loss"))

        # Run training
        result = train_vae(
            config_path=str(config_path),
            device_override=device,
            experiment_name=experiment_name,
            quiet=True,
            progress_callback=progress_callback,
        )

        duration = time.time() - start_time

        if result.get("error"):
            send_progress("failed", 0, total_epochs)
            return ExperimentResult(
                name=experiment_name,
                config_path=str(config_path),
                status="failed",
                device=device or "unknown",
                duration_seconds=duration,
                error=result["error"],
                total_epochs=total_epochs,
                log_file=log_file,
            )

        send_progress("complete", total_epochs, total_epochs, result.get("best_loss"))
        return ExperimentResult(
            name=experiment_name,
            config_path=str(config_path),
            status="success",
            final_loss=result.get("final_loss"),
            best_loss=result.get("best_loss"),
            recon_loss=result.get("final_recon_loss"),
            kl_loss=result.get("final_kl_loss"),
            epochs_trained=result.get("epochs_trained", total_epochs),
            total_epochs=total_epochs,
            n_samples=result.get("n_samples", 0),
            device=device or result.get("device", "unknown"),
            duration_seconds=duration,
            checkpoint_path=result.get("checkpoint_path"),
            log_file=log_file,
        )

    except Exception as e:
        traceback.print_exc()
        duration = time.time() - start_time
        send_progress("failed", 0, 0)
        return ExperimentResult(
            name=experiment_name,
            config_path=str(config_path),
            status="failed",
            device=device or "unknown",
            duration_seconds=duration,
            error=str(e),
            log_file=log_file,
        )

    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr

        if log_file:
            with open(log_file, "w") as f:
                f.write(log_buffer.getvalue())


def _create_experiment_progress_table(states: dict[str, dict]) -> "Table":
    """Create a rich Table showing experiment progress."""
    table = Table(title="Experiment Progress", show_header=True, header_style="bold")
    table.add_column("Experiment", style="cyan", width=20)
    table.add_column("Status", width=10)
    table.add_column("Progress", width=15)
    table.add_column("Loss", width=12)
    table.add_column("Device", width=10)
    table.add_column("Time", width=10)

    for name, state in states.items():
        status = state.get("status", "pending")
        epoch = state.get("epoch", 0)
        total = state.get("total_epochs", 0)
        loss = state.get("loss")
        device = state.get("device", "")
        elapsed = state.get("time", 0)

        status_str = format_status(status)
        progress_str = format_progress_bar(epoch, total)
        loss_str = f"{loss:.4f}" if loss is not None else "-"
        time_str = format_duration(elapsed)

        table.add_row(name, status_str, progress_str, loss_str, device, time_str)

    return table


class ExperimentRunner(ParallelRunner[Path, ExperimentResult]):
    """Runner for VAE training experiments.

    Runs multiple training configurations in parallel across GPUs.

    Example:
        >>> runner = ExperimentRunner(parallel=True, device="auto")
        >>> results = runner.run([Path("config1.yaml"), Path("config2.yaml")])
    """

    def __init__(
        self,
        parallel: bool = True,
        max_workers: int | None = None,
        device: str = "auto",
    ):
        super().__init__(parallel=parallel, max_workers=max_workers, device=device)
        self.scripts_dir = str(Path(__file__).parent.parent.parent / "scripts")

    def run_job(
        self,
        config: Path,
        device: str,
        progress_queue: "Queue",
    ) -> ExperimentResult:
        """Run a single training experiment."""
        return _run_vae_experiment(config, device, progress_queue, self.scripts_dir)

    def get_job_name(self, config: Path) -> str:
        """Get experiment name from config path."""
        return config.stem

    def create_progress_table(self, states: dict[str, dict]) -> "Table":
        """Create experiment progress table."""
        return _create_experiment_progress_table(states)

    def create_failed_result(
        self,
        config: Path,
        device: str,
        error: str,
    ) -> ExperimentResult:
        """Create a failed experiment result."""
        return ExperimentResult(
            name=config.stem,
            config_path=str(config),
            status="failed",
            device=device,
            error=error,
        )

    def validate_config(self, config: Path) -> None:
        """Validate config file exists."""
        if not config.exists():
            raise FileNotFoundError(f"Config not found: {config}")


def run_experiments(
    config_paths: list[str | Path],
    parallel: bool = True,
    max_workers: int | None = None,
    device: str = "auto",
) -> list[ExperimentResult]:
    """
    Run multiple training experiments, optionally in parallel.

    Experiments are distributed across available GPUs in a round-robin
    fashion. Each experiment runs in a separate process for memory isolation.

    A live progress table is displayed showing the status of each experiment
    (requires the `rich` library).

    Args:
        config_paths: List of paths to YAML config files.
        parallel: If True, run experiments in parallel across GPUs.
        max_workers: Maximum parallel experiments. If None, uses GPU count
            (for CUDA) or 1 (for MPS/CPU).
        device: Device strategy:
            - "auto": Use CUDA if available, else MPS, else CPU
            - "cuda": Distribute across CUDA GPUs
            - "cpu": Run all on CPU (parallel via processes)
            - "mps": Run all on MPS (sequential - MPS doesn't multiprocess well)

    Returns:
        List of ExperimentResult for each config, in the same order as input.

    Raises:
        ImportError: If PyTorch or rich is not available.
        FileNotFoundError: If any config file does not exist.

    Example:
        >>> results = run_experiments(
        ...     ["small.yaml", "medium.yaml", "large.yaml"],
        ...     parallel=True,
        ...     device="auto",
        ... )
        >>> for r in results:
        ...     print(f"{r.name}: {r.best_loss:.4f}")
    """
    runner = ExperimentRunner(
        parallel=parallel,
        max_workers=max_workers,
        device=device,
    )
    configs = [Path(p) for p in config_paths]
    return runner.run(configs)


def format_results_table(results: list[ExperimentResult], show_errors: bool = True) -> str:
    """
    Format experiment results as an ASCII table.

    Args:
        results: List of ExperimentResult objects.
        show_errors: If True, show error messages for failed experiments.

    Returns:
        Formatted table string suitable for terminal output.

    Example:
        >>> print(format_results_table(results))
        Experiment            Status    Best Loss   Recon       ...
        --------------------  --------  ----------  ----------  ...
        vae_small             success   0.1234      0.0812      ...
    """
    if not results:
        return "No results to display."

    columns = [
        ("Experiment", 20),
        ("Status", 8),
        ("Best Loss", 10),
        ("Recon", 10),
        ("KL", 10),
        ("Epochs", 10),
        ("Device", 8),
        ("Time", 10),
    ]

    lines = []
    header = "  ".join(f"{name:<{width}}" for name, width in columns)
    separator = "  ".join("-" * width for _, width in columns)

    lines.append(header)
    lines.append(separator)

    for r in results:
        row_values = [
            r.name[:20],
            r.status[:8],
            f"{r.best_loss:.4f}" if r.best_loss is not None else "N/A",
            f"{r.recon_loss:.4f}" if r.recon_loss is not None else "N/A",
            f"{r.kl_loss:.4f}" if r.kl_loss is not None else "N/A",
            f"{r.epochs_trained}/{r.total_epochs}" if r.total_epochs else str(r.epochs_trained),
            r.device[:8],
            format_duration(r.duration_seconds),
        ]

        row = "  ".join(f"{val:<{width}}" for val, (_, width) in zip(row_values, columns))
        lines.append(row)

    lines.append(separator)
    successful = sum(1 for r in results if r.status == "success")
    total_time = sum(r.duration_seconds for r in results)
    lines.append(f"Total: {successful}/{len(results)} succeeded in {format_duration(total_time)}")

    if show_errors:
        failed = [r for r in results if r.status == "failed"]
        if failed:
            lines.append("")
            lines.append("Errors:")
            for r in failed:
                error_msg = r.error or "Unknown error"
                lines.append(f"  {r.name}: {error_msg}")
                if r.log_file:
                    lines.append(f"    Log: {r.log_file}")

    return "\n".join(lines)


__all__ = [
    "ExperimentRunner",
    "run_experiments",
    "format_results_table",
    "ExperimentResult",
]
