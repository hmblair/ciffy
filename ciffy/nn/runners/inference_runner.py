"""
Multi-job inference runner for parallel structure generation.

Provides utilities for running multiple inference configurations in parallel
with automatic GPU assignment and progress tracking, mirroring the
experiment_runner infrastructure.

Example:
    >>> from ciffy.nn.runners import run_inference_jobs
    >>>
    >>> results = run_inference_jobs(
    ...     ["config1.yaml", "config2.yaml"],
    ...     parallel=True,
    ...     device="auto",
    ... )
    >>> for result in results:
    ...     if result.status == "success":
    ...         print(f"{result.name}: {result.n_structures} structures")
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

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
class InferenceResult:
    """Result from an inference job.

    Attributes:
        name: Job identifier (config filename without extension).
        config_path: Path to the YAML configuration file.
        status: One of 'success', 'failed', or 'running'.
        n_structures: Number of structures generated.
        n_sequences: Number of input sequences processed.
        device: Device used (e.g., 'cuda:0', 'cpu').
        duration_seconds: Total inference time in seconds.
        output_dir: Directory containing output .cif files.
        error: Error message if status is 'failed', None otherwise.
        log_file: Path to log file containing stdout/stderr.
    """

    name: str
    config_path: str
    status: str  # 'success', 'failed', 'running'
    n_structures: int = 0
    n_sequences: int = 0
    device: str = ""
    duration_seconds: float = 0.0
    output_dir: Optional[str] = None
    error: Optional[str] = None
    log_file: Optional[str] = None


def _run_inference_job(
    config_path: Path,
    device: str,
    progress_queue: "Queue",
    scripts_dir: str,
) -> InferenceResult:
    """Run a single inference job. Module-level function for pickling."""
    job_name = config_path.stem
    start_time = time.time()

    # Create temp log file
    log_fd, log_file = tempfile.mkstemp(
        prefix=f"ciffy_inference_{job_name}_", suffix=".log"
    )
    os.close(log_fd)

    def send_progress(status: str, n_done: int = 0, n_total: int = 0):
        if progress_queue is not None:
            try:
                progress_queue.put({
                    "name": job_name,
                    "status": status,
                    "n_done": n_done,
                    "n_total": n_total,
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

        # Add scripts directory to path
        if scripts_dir and scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)

        from run_inference import run_inference

        send_progress("running", 0, 0)

        # Run inference
        result = run_inference(
            config_path=str(config_path),
            device_override=device,
            job_name=job_name,
            quiet=True,
        )

        duration = time.time() - start_time

        if result.get("error"):
            send_progress("failed", 0, 0)
            return InferenceResult(
                name=job_name,
                config_path=str(config_path),
                status="failed",
                device=device or "unknown",
                duration_seconds=duration,
                error=result["error"],
                log_file=log_file,
            )

        send_progress(
            "complete",
            result["n_structures"],
            result["n_structures"],
        )

        return InferenceResult(
            name=job_name,
            config_path=str(config_path),
            status="success",
            n_structures=result["n_structures"],
            n_sequences=result["n_sequences"],
            device=device or result.get("device", "unknown"),
            duration_seconds=duration,
            output_dir=result.get("output_dir"),
            log_file=log_file,
        )

    except Exception as e:
        traceback.print_exc()
        duration = time.time() - start_time
        send_progress("failed", 0, 0)
        return InferenceResult(
            name=job_name,
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


def _create_inference_progress_table(states: dict[str, dict]) -> "Table":
    """Create a rich Table showing inference progress."""
    table = Table(title="Inference Progress", show_header=True, header_style="bold")
    table.add_column("Job", style="cyan", width=20)
    table.add_column("Status", width=10)
    table.add_column("Progress", width=20)
    table.add_column("Device", width=10)
    table.add_column("Time", width=10)

    for name, state in states.items():
        status = state.get("status", "pending")
        n_done = state.get("n_done", 0)
        n_total = state.get("n_total", 0)
        device = state.get("device", "")
        elapsed = state.get("time", 0)

        status_str = format_status(status)
        progress_str = format_progress_bar(n_done, n_total)
        time_str = format_duration(elapsed)

        table.add_row(name, status_str, progress_str, device, time_str)

    return table


class InferenceRunner(ParallelRunner[Path, InferenceResult]):
    """Runner for inference jobs.

    Runs multiple inference configurations in parallel across GPUs.

    Example:
        >>> runner = InferenceRunner(parallel=True, device="auto")
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
    ) -> InferenceResult:
        """Run a single inference job."""
        return _run_inference_job(config, device, progress_queue, self.scripts_dir)

    def get_job_name(self, config: Path) -> str:
        """Get job name from config path."""
        return config.stem

    def create_progress_table(self, states: dict[str, dict]) -> "Table":
        """Create inference progress table."""
        return _create_inference_progress_table(states)

    def create_failed_result(
        self,
        config: Path,
        device: str,
        error: str,
    ) -> InferenceResult:
        """Create a failed inference result."""
        return InferenceResult(
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


def run_inference_jobs(
    config_paths: list[str | Path],
    parallel: bool = True,
    max_workers: int | None = None,
    device: str = "auto",
) -> list[InferenceResult]:
    """
    Run multiple inference jobs, optionally in parallel.

    Jobs are distributed across available GPUs in a round-robin fashion.
    Each job runs in a separate process for memory isolation.

    Args:
        config_paths: List of paths to YAML config files.
        parallel: If True, run jobs in parallel across GPUs.
        max_workers: Maximum parallel jobs. If None, uses GPU count.
        device: Device strategy ('auto', 'cuda', 'cpu', 'mps').

    Returns:
        List of InferenceResult for each config, in the same order as input.

    Raises:
        ImportError: If PyTorch or rich is not available.
        FileNotFoundError: If any config file does not exist.

    Example:
        >>> results = run_inference_jobs(
        ...     ["config1.yaml", "config2.yaml"],
        ...     parallel=True,
        ...     device="auto",
        ... )
        >>> for r in results:
        ...     print(f"{r.name}: {r.status}")
    """
    runner = InferenceRunner(
        parallel=parallel,
        max_workers=max_workers,
        device=device,
    )
    configs = [Path(p) for p in config_paths]
    return runner.run(configs)


def format_inference_results_table(
    results: list[InferenceResult], show_errors: bool = True
) -> str:
    """
    Format inference results as an ASCII table.

    Args:
        results: List of InferenceResult objects.
        show_errors: If True, show error messages for failed jobs.

    Returns:
        Formatted table string suitable for terminal output.
    """
    if not results:
        return "No results to display."

    columns = [
        ("Job", 20),
        ("Status", 8),
        ("Structures", 12),
        ("Sequences", 10),
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
            str(r.n_structures) if r.n_structures > 0 else "N/A",
            str(r.n_sequences) if r.n_sequences > 0 else "N/A",
            r.device[:8],
            format_duration(r.duration_seconds),
        ]

        row = "  ".join(
            f"{val:<{width}}" for val, (_, width) in zip(row_values, columns)
        )
        lines.append(row)

    lines.append(separator)
    successful = sum(1 for r in results if r.status == "success")
    total_structures = sum(r.n_structures for r in results)
    total_time = sum(r.duration_seconds for r in results)
    lines.append(
        f"Total: {successful}/{len(results)} succeeded, {total_structures} structures in {format_duration(total_time)}"
    )

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
    "InferenceResult",
    "InferenceRunner",
    "run_inference_jobs",
    "format_inference_results_table",
]
