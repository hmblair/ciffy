"""
Multi-config training runner for unified `ciffy train` command.

Provides utilities for running multiple training configurations in parallel
with automatic GPU assignment and progress tracking.

Example:
    >>> from ciffy.nn.runners import run_training_jobs
    >>>
    >>> results = run_training_jobs(
    ...     ["config1.yaml", "config2.yaml"],
    ...     parallel=True,
    ...     device="auto",
    ... )
    >>> for result in results:
    ...     if result.status == "success":
    ...         print(f"{result.name}: {result.epochs_trained} epochs")
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

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


@dataclass
class TrainingResult:
    """Result from a training job.

    Unified result structure across all trainer types (flow, diffusion, etc.).

    Attributes:
        name: Job identifier (config filename without extension).
        config_path: Path to the YAML configuration file.
        trainer_type: Type of trainer used (e.g., 'latent_diffusion', 'flow').
        status: One of 'success' or 'failed'.
        final_loss: Loss from the final epoch (if applicable).
        best_loss: Best loss achieved during training (if applicable).
        epochs_trained: Number of epochs completed.
        total_epochs: Total epochs configured.
        n_samples: Number of training samples.
        device: Device used (e.g., 'cuda:0', 'cpu').
        duration_seconds: Total training time in seconds.
        checkpoint_path: Path to saved checkpoint.
        extra_metrics: Trainer-specific additional metrics.
        error: Error message if status is 'failed'.
        log_file: Path to log file containing stdout/stderr.
    """

    name: str
    config_path: str
    trainer_type: str
    status: str  # 'success' | 'failed'

    # Common metrics
    final_loss: Optional[float] = None
    best_loss: Optional[float] = None
    epochs_trained: int = 0
    total_epochs: int = 0
    n_samples: int = 0
    device: str = ""
    duration_seconds: float = 0.0
    checkpoint_path: Optional[str] = None

    # Trainer-specific metrics
    extra_metrics: dict = field(default_factory=dict)

    error: Optional[str] = None
    log_file: Optional[str] = None


def _load_config(config_path: Path) -> dict[str, Any]:
    """Load YAML config and extract trainer type."""
    if not YAML_AVAILABLE:
        raise ImportError("PyYAML is required for training. Install with: pip install pyyaml")

    with open(config_path) as f:
        config = yaml.safe_load(f) or {}

    if "trainer" not in config:
        raise ValueError(
            f"Config {config_path} missing 'trainer' field. "
            f"Valid options: latent_diffusion, flow, diffusion"
        )

    return config


def _import_trainer(trainer_type: str) -> tuple[type, type]:
    """Import trainer module and return (trainer_cls, config_cls) from registry.

    This function imports the appropriate trainer module, which triggers
    registration via the @register_trainer decorator. Then it retrieves
    the trainer and config classes from the registry.

    Args:
        trainer_type: Trainer type name (e.g., 'flow', 'latent_diffusion').

    Returns:
        Tuple of (trainer_class, config_class).

    Raises:
        ValueError: If trainer type is unknown.
    """
    from ..trainer_registry import get_trainer

    # Import the module to trigger registration
    if trainer_type == "latent_diffusion":
        from ..diffusion import latent_trainer  # noqa: F401
    elif trainer_type == "flow":
        from ..flow.residue import trainer  # noqa: F401
    elif trainer_type == "diffusion":
        from ..diffusion import trainer  # noqa: F401
    else:
        raise ValueError(
            f"Unknown trainer type: {trainer_type}. "
            f"Valid options: latent_diffusion, flow, diffusion"
        )

    return get_trainer(trainer_type)


def _run_training_job(
    config_path: Path,
    device: str,
    progress_queue: "Queue",
) -> TrainingResult:
    """Run a single training job. Module-level function for pickling."""
    job_name = config_path.stem
    start_time = time.time()

    # Create temp log file
    log_fd, log_file = tempfile.mkstemp(
        prefix=f"ciffy_train_{job_name}_", suffix=".log"
    )
    os.close(log_fd)

    def send_progress(
        status: str,
        epoch: int = 0,
        total: int = 0,
        loss: Optional[float] = None,
        trainer_type: str = "",
        extra_metrics: Optional[dict] = None,
    ):
        if progress_queue is not None:
            try:
                msg = {
                    "name": job_name,
                    "status": status,
                    "epoch": epoch,
                    "total_epochs": total,
                    "loss": loss,
                    "device": device,
                    "trainer_type": trainer_type,
                    "time": time.time() - start_time,
                }
                if extra_metrics:
                    msg.update(extra_metrics)
                progress_queue.put(msg)
            except Exception:
                pass

    # Capture stdout/stderr to log file
    log_buffer = io.StringIO()
    old_stdout, old_stderr = sys.stdout, sys.stderr

    try:
        sys.stdout = log_buffer
        sys.stderr = log_buffer

        # Load config
        config = _load_config(config_path)
        trainer_type = config["trainer"]

        send_progress("running", 0, 0, trainer_type=trainer_type)

        # Create progress callback
        def progress_callback(epoch: int, total: int, metrics: dict):
            # Extract loss, pass rest as extra metrics
            loss = metrics.pop("loss", None)
            send_progress(
                "running",
                epoch,
                total,
                loss,
                trainer_type,
                extra_metrics=metrics if metrics else None,
            )

        # Import trainer and config class from registry
        # Each trainer module registers itself when imported
        trainer_cls, config_cls = _import_trainer(trainer_type)

        # Create config from dict with device override
        trainer_config = config_cls.from_dict(config, device=device)

        # Instantiate and run trainer
        trainer = trainer_cls(trainer_config, quiet=True)

        # Get initial dataset size if available (for progress display)
        n_samples = getattr(trainer, 'train_dataset_size', 0)
        if n_samples > 0:
            send_progress("running", 0, 0, trainer_type=trainer_type,
                          extra_metrics={"n_samples": n_samples})

        # Run training
        result = trainer.train(progress_callback=progress_callback)

        duration = time.time() - start_time
        send_progress(
            "complete",
            result.get("epochs_trained", 0),
            result.get("total_epochs", 0),
            result.get("final_loss"),
            trainer_type,
        )

        return TrainingResult(
            name=job_name,
            config_path=str(config_path),
            trainer_type=trainer_type,
            status="success" if result.get("status", "success") == "success" else "failed",
            final_loss=result.get("final_loss"),
            best_loss=result.get("best_loss"),
            epochs_trained=result.get("epochs_trained", 0),
            total_epochs=result.get("total_epochs", 0),
            n_samples=result.get("n_samples", 0),
            device=device,
            duration_seconds=duration,
            checkpoint_path=result.get("checkpoint_path"),
            extra_metrics=result.get("extra_metrics", {}),
            log_file=log_file,
        )

    except Exception as e:
        traceback.print_exc()
        duration = time.time() - start_time
        send_progress("failed", 0, 0)
        return TrainingResult(
            name=job_name,
            config_path=str(config_path),
            trainer_type=config.get("trainer", "unknown") if "config" in dir() else "unknown",
            status="failed",
            device=device,
            duration_seconds=duration,
            error=str(e),
            log_file=log_file,
        )

    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr

        if log_file:
            with open(log_file, "w") as f:
                f.write(log_buffer.getvalue())


def _format_count(n: int | None) -> str:
    """Format large numbers with K/M suffix."""
    if n is None:
        return "-"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _create_training_progress_table(states: dict[str, dict]) -> "Table":
    """Create a rich Table showing training progress."""
    table = Table(title="Training Progress", show_header=True, header_style="bold")
    table.add_column("Config", style="cyan", width=18)
    table.add_column("Trainer", width=12)
    table.add_column("Status", width=10)
    table.add_column("Progress", width=15)
    table.add_column("Loss", width=10)
    table.add_column("Params", width=8)
    table.add_column("Samples", width=8)
    table.add_column("Device", width=8)
    table.add_column("Time", width=8)

    for name, state in states.items():
        status = state.get("status", "pending")
        trainer_type = state.get("trainer_type", "")
        epoch = state.get("epoch", 0)
        total = state.get("total_epochs", 0)
        loss = state.get("loss")
        n_params = state.get("n_params")
        n_samples = state.get("n_samples")
        dev = state.get("device", "")
        elapsed = state.get("time", 0)

        table.add_row(
            name,
            trainer_type,
            format_status(status),
            format_progress_bar(epoch, total) if total > 0 else "...",
            f"{loss:.4f}" if loss is not None else "-",
            _format_count(n_params),
            _format_count(n_samples),
            dev,
            format_duration(elapsed),
        )

    return table


class TrainingRunner(ParallelRunner[Path, TrainingResult]):
    """Runner for unified training jobs.

    Runs multiple training configurations in parallel across GPUs.
    Supports all registered trainer types (latent_diffusion, flow, diffusion).

    Example:
        >>> runner = TrainingRunner(parallel=True, device="auto")
        >>> results = runner.run([Path("config1.yaml"), Path("config2.yaml")])
    """

    def run_job(
        self,
        config: Path,
        device: str,
        progress_queue: "Queue",
    ) -> TrainingResult:
        """Run a single training job."""
        return _run_training_job(config, device, progress_queue)

    def get_job_name(self, config: Path) -> str:
        """Get job name from config path."""
        return config.stem

    def create_progress_table(self, states: dict[str, dict]) -> "Table":
        """Create training progress table."""
        return _create_training_progress_table(states)

    def create_failed_result(
        self,
        config: Path,
        device: str,
        error: str,
    ) -> TrainingResult:
        """Create a failed training result."""
        return TrainingResult(
            name=config.stem,
            config_path=str(config),
            trainer_type="unknown",
            status="failed",
            device=device,
            error=error,
        )

    def validate_config(self, config: Path) -> None:
        """Validate config file exists and has required fields."""
        if not config.exists():
            raise FileNotFoundError(f"Config not found: {config}")

        # Validate trainer field exists
        try:
            cfg = _load_config(config)
        except Exception as e:
            raise ValueError(f"Invalid config {config}: {e}")


def run_training_jobs(
    config_paths: list[str | Path],
    parallel: bool = True,
    max_workers: int | None = None,
    device: str = "auto",
) -> list[TrainingResult]:
    """
    Run multiple training jobs from config files.

    Jobs are distributed across available GPUs in a round-robin fashion.
    Each job runs in a separate process for memory isolation.

    Args:
        config_paths: List of paths to YAML config files.
        parallel: If True, run jobs in parallel across GPUs.
        max_workers: Maximum parallel jobs. If None, uses GPU count.
        device: Device strategy ('auto', 'cuda', 'cpu', 'mps').

    Returns:
        List of TrainingResult for each config, in the same order as input.

    Raises:
        ImportError: If PyTorch, PyYAML, or rich is not available.
        FileNotFoundError: If any config file does not exist.
        ValueError: If any config is invalid.

    Example:
        >>> results = run_training_jobs(
        ...     ["configs/latent_diffusion.yaml", "configs/flow.yaml"],
        ...     parallel=True,
        ...     device="auto",
        ... )
        >>> for r in results:
        ...     print(f"{r.name}: {r.status}, {r.epochs_trained} epochs")
    """
    runner = TrainingRunner(
        parallel=parallel,
        max_workers=max_workers,
        device=device,
    )
    configs = [Path(p) for p in config_paths]
    return runner.run(configs)


def format_training_results_table(
    results: list[TrainingResult],
    show_errors: bool = True,
) -> str:
    """
    Format training results as an ASCII table.

    Args:
        results: List of TrainingResult objects.
        show_errors: If True, show error messages for failed jobs.

    Returns:
        Formatted table string suitable for terminal output.
    """
    if not results:
        return "No results to display."

    columns = [
        ("Config", 20),
        ("Trainer", 15),
        ("Status", 8),
        ("Best Loss", 10),
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
        epochs_str = f"{r.epochs_trained}/{r.total_epochs}" if r.total_epochs else str(r.epochs_trained)
        row_values = [
            r.name[:20],
            r.trainer_type[:15],
            r.status[:8],
            f"{r.best_loss:.4f}" if r.best_loss is not None else "N/A",
            epochs_str[:10],
            r.device[:8],
            format_duration(r.duration_seconds),
        ]

        row = "  ".join(
            f"{val:<{width}}" for val, (_, width) in zip(row_values, columns)
        )
        lines.append(row)

    lines.append(separator)

    # Summary
    successful = sum(1 for r in results if r.status == "success")
    total_time = sum(r.duration_seconds for r in results)
    lines.append(f"Total: {successful}/{len(results)} succeeded in {format_duration(total_time)}")

    # Errors
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
    "TrainingResult",
    "TrainingRunner",
    "run_training_jobs",
    "format_training_results_table",
]
