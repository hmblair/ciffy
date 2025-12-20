"""
Multi-job inference runner for parallel structure generation.

Provides utilities for running multiple inference configurations in parallel
with automatic GPU assignment and progress tracking, mirroring the
experiment_runner infrastructure.

Example:
    >>> from ciffy.nn.inference_runner import run_inference_jobs
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

import logging
import multiprocessing as mp
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


logger = logging.getLogger(__name__)


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


def _get_num_gpus() -> int:
    """Get number of available CUDA GPUs."""
    if not TORCH_AVAILABLE:
        return 0
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.device_count()


def _run_single_inference_with_progress(args: tuple) -> InferenceResult:
    """
    Run a single inference job in a subprocess with progress reporting.

    Args:
        args: Tuple of (config_path, job_name, device, scripts_dir, progress_queue)

    Returns:
        InferenceResult with inference outcomes.
    """
    import io
    import sys
    import tempfile
    import traceback

    config_path, job_name, device, scripts_dir, progress_queue = args

    start_time = time.time()

    # Create temp log file
    log_fd, log_file = tempfile.mkstemp(
        prefix=f"ciffy_inference_{job_name}_", suffix=".log"
    )
    import os

    os.close(log_fd)

    def send_progress(status: str, n_done: int = 0, n_total: int = 0):
        """Send progress update to queue."""
        if progress_queue is not None:
            try:
                progress_queue.put(
                    {
                        "name": job_name,
                        "status": status,
                        "n_done": n_done,
                        "n_total": n_total,
                        "device": device,
                        "time": time.time() - start_time,
                    }
                )
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
            config_path=config_path,
            device_override=device,
            job_name=job_name,
            quiet=True,
        )

        duration = time.time() - start_time

        if result.get("error"):
            send_progress("failed", 0, 0)
            return InferenceResult(
                name=job_name,
                config_path=config_path,
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
            config_path=config_path,
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
            config_path=config_path,
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


def _create_progress_table(job_states: dict[str, dict]) -> Table:
    """Create a rich Table showing inference progress."""
    table = Table(title="Inference Progress", show_header=True, header_style="bold")
    table.add_column("Job", style="cyan", width=20)
    table.add_column("Status", width=10)
    table.add_column("Progress", width=20)
    table.add_column("Device", width=10)
    table.add_column("Time", width=10)

    for name, state in job_states.items():
        status = state.get("status", "pending")
        n_done = state.get("n_done", 0)
        n_total = state.get("n_total", 0)
        device = state.get("device", "")
        elapsed = state.get("time", 0)

        # Format status with color
        if status == "complete":
            status_str = "[green]complete[/green]"
        elif status == "failed":
            status_str = "[red]failed[/red]"
        elif status == "running":
            status_str = "[yellow]running[/yellow]"
        else:
            status_str = "[dim]pending[/dim]"

        # Format progress bar
        if n_total > 0:
            pct = n_done / n_total
            filled = int(pct * 10)
            bar = "█" * filled + "░" * (10 - filled)
            progress_str = f"{bar} {n_done}/{n_total}"
        else:
            progress_str = "..."

        # Format time
        time_str = _format_duration(elapsed)

        table.add_row(name, status_str, progress_str, device, time_str)

    return table


def _progress_display_thread(
    progress_queue,
    job_names: list[str],
    stop_event: threading.Event,
) -> None:
    """Thread that reads from progress queue and updates the live display."""
    if not RICH_AVAILABLE:
        return

    console = Console()

    # Initialize job states
    job_states = {name: {"status": "pending"} for name in job_names}

    with Live(
        _create_progress_table(job_states), console=console, refresh_per_second=4
    ) as live:
        while not stop_event.is_set():
            while True:
                try:
                    msg = progress_queue.get(timeout=0.1)
                    name = msg.get("name")
                    if name in job_states:
                        job_states[name].update(msg)
                except Exception:
                    break

            live.update(_create_progress_table(job_states))

        # Final update
        live.update(_create_progress_table(job_states))


def run_inference_jobs(
    config_paths: list[str | Path],
    parallel: bool = True,
    max_workers: Optional[int] = None,
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
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for run_inference_jobs")

    config_paths = [Path(p) for p in config_paths]

    # Validate all configs exist
    for path in config_paths:
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")

    # Determine device strategy and worker count
    num_gpus = _get_num_gpus()

    if device == "auto":
        if num_gpus > 0:
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    # Determine max workers
    if max_workers is None:
        if device == "cuda":
            max_workers = num_gpus if num_gpus > 0 else 1
        elif device == "mps":
            max_workers = 1
        else:
            max_workers = min(len(config_paths), mp.cpu_count())

    # Find the scripts directory
    scripts_dir = str(Path(__file__).parent.parent.parent / "scripts")

    # Prepare job arguments
    job_names = []
    jobs = []

    manager = mp.Manager()
    progress_queue = manager.Queue()

    for i, config_path in enumerate(config_paths):
        job_name = config_path.stem
        job_names.append(job_name)

        # Assign device based on strategy
        if device == "cuda" and num_gpus > 0:
            job_device = f"cuda:{i % num_gpus}"
        else:
            job_device = device

        jobs.append((str(config_path), job_name, job_device, scripts_dir, progress_queue))

    results: list[InferenceResult] = []

    # Start progress display thread
    stop_event = threading.Event()
    display_thread = threading.Thread(
        target=_progress_display_thread,
        args=(progress_queue, job_names, stop_event),
        daemon=True,
    )
    display_thread.start()

    try:
        if not parallel or max_workers == 1:
            # Sequential execution
            for job_args in jobs:
                result = _run_single_inference_with_progress(job_args)
                results.append(result)
        else:
            # Parallel execution
            ctx = mp.get_context("spawn")

            with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
                future_to_job = {
                    executor.submit(_run_single_inference_with_progress, job): job
                    for job in jobs
                }

                for future in as_completed(future_to_job):
                    job_args = future_to_job[future]
                    job_name = job_args[1]

                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        result = InferenceResult(
                            name=job_name,
                            config_path=job_args[0],
                            status="failed",
                            device=job_args[2],
                            error=f"Executor error: {e}",
                        )
                        results.append(result)
    finally:
        stop_event.set()
        display_thread.join(timeout=2.0)

    # Sort results to match input order
    name_to_result = {r.name: r for r in results}
    ordered_results = []
    for config_path in config_paths:
        job_name = config_path.stem
        if job_name in name_to_result:
            ordered_results.append(name_to_result[job_name])

    return ordered_results


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
            _format_duration(r.duration_seconds),
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
        f"Total: {successful}/{len(results)} succeeded, {total_structures} structures in {_format_duration(total_time)}"
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


def _format_duration(seconds: float) -> str:
    """Format duration as human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m{secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h{mins}m"


__all__ = [
    "InferenceResult",
    "run_inference_jobs",
    "format_inference_results_table",
]
