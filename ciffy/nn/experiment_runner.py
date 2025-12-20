"""
Multi-experiment runner for parallel training across GPUs.

Provides utilities for running multiple training configurations in parallel
with automatic GPU assignment and result comparison.

Example:
    >>> from ciffy.nn.experiment_runner import run_experiments, format_results_table
    >>>
    >>> results = run_experiments(
    ...     ["config1.yaml", "config2.yaml"],
    ...     parallel=True,
    ...     device="auto",
    ... )
    >>> print(format_results_table(results))
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Check for rich library
try:
    from rich.live import Live
    from rich.table import Table
    from rich.console import Console

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from .training import ExperimentResult

if TYPE_CHECKING:
    from multiprocessing import Queue

logger = logging.getLogger(__name__)


def _get_num_gpus() -> int:
    """Get number of available CUDA GPUs."""
    if not TORCH_AVAILABLE:
        return 0
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.device_count()


def _run_single_experiment_with_progress(args: tuple) -> ExperimentResult:
    """
    Run a single training experiment in a subprocess with progress reporting.

    This function is designed to be called by ProcessPoolExecutor.
    It imports the training script and runs training with the given config,
    sending progress updates to a queue. All stdout/stderr is captured to a temp log file.

    Args:
        args: Tuple of (config_path, experiment_name, device, scripts_dir, progress_queue)

    Returns:
        ExperimentResult with training outcomes.
    """
    import io
    import sys
    import tempfile
    import traceback

    config_path, experiment_name, device, scripts_dir, progress_queue = args

    start_time = time.time()

    # Create temp log file (deleted=False so it persists for user to inspect)
    log_fd, log_file = tempfile.mkstemp(prefix=f"ciffy_{experiment_name}_", suffix=".log")
    import os
    os.close(log_fd)  # Close the file descriptor, we'll write via StringIO

    def send_progress(status: str, epoch: int = 0, total_epochs: int = 0, loss: float | None = None):
        """Send progress update to queue."""
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
                pass  # Ignore queue errors

    # Capture stdout/stderr to log file
    log_buffer = io.StringIO()
    old_stdout, old_stderr = sys.stdout, sys.stderr

    try:
        # Redirect output to buffer
        sys.stdout = log_buffer
        sys.stderr = log_buffer

        # Add scripts directory to path for importing train_vae
        if scripts_dir and scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)

        from train_vae import train_vae, load_config

        # Load config to get total epochs
        config = load_config(config_path)
        total_epochs = config.training.epochs

        # Send initial status
        send_progress("running", 0, total_epochs)

        # Create progress callback
        def progress_callback(epoch: int, total: int, metrics: dict):
            send_progress("running", epoch, total, metrics.get("loss"))

        # Run training
        result = train_vae(
            config_path=config_path,
            device_override=device,
            experiment_name=experiment_name,
            quiet=True,  # Suppress per-experiment progress bars
            progress_callback=progress_callback,
        )

        duration = time.time() - start_time

        if result.get("error"):
            send_progress("failed", 0, total_epochs)
            return ExperimentResult(
                name=experiment_name,
                config_path=config_path,
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
            config_path=config_path,
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
        # Capture the full traceback
        traceback.print_exc()
        duration = time.time() - start_time
        send_progress("failed", 0, 0)
        return ExperimentResult(
            name=experiment_name,
            config_path=config_path,
            status="failed",
            device=device or "unknown",
            duration_seconds=duration,
            error=str(e),
            log_file=log_file,
        )

    finally:
        # Restore stdout/stderr
        sys.stdout, sys.stderr = old_stdout, old_stderr

        # Write captured output to log file
        if log_file:
            with open(log_file, "w") as f:
                f.write(log_buffer.getvalue())


def _create_progress_table(experiment_states: dict[str, dict]) -> "Table":
    """Create a rich Table showing experiment progress."""
    table = Table(title="Experiment Progress", show_header=True, header_style="bold")
    table.add_column("Experiment", style="cyan", width=20)
    table.add_column("Status", width=10)
    table.add_column("Progress", width=15)
    table.add_column("Loss", width=12)
    table.add_column("Device", width=10)
    table.add_column("Time", width=10)

    for name, state in experiment_states.items():
        status = state.get("status", "pending")
        epoch = state.get("epoch", 0)
        total = state.get("total_epochs", 0)
        loss = state.get("loss")
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
        if total > 0:
            pct = epoch / total
            filled = int(pct * 10)
            bar = "█" * filled + "░" * (10 - filled)
            progress_str = f"{bar} {epoch}/{total}"
        else:
            progress_str = "..."

        # Format loss
        loss_str = f"{loss:.4f}" if loss is not None else "-"

        # Format time
        time_str = _format_duration(elapsed)

        table.add_row(name, status_str, progress_str, loss_str, device, time_str)

    return table


def _progress_display_thread(
    progress_queue: "Queue",
    experiment_names: list[str],
    stop_event: threading.Event,
) -> None:
    """Thread that reads from progress queue and updates the live display."""
    console = Console()

    # Initialize experiment states
    experiment_states = {name: {"status": "pending"} for name in experiment_names}

    with Live(_create_progress_table(experiment_states), console=console, refresh_per_second=4) as live:
        while not stop_event.is_set():
            # Process all available messages
            while True:
                try:
                    # Non-blocking get with timeout
                    msg = progress_queue.get(timeout=0.1)
                    name = msg.get("name")
                    if name in experiment_states:
                        experiment_states[name].update(msg)
                except Exception:
                    break  # Queue empty or error

            # Update display
            live.update(_create_progress_table(experiment_states))

        # Final update
        live.update(_create_progress_table(experiment_states))


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
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for run_experiments")

    if not RICH_AVAILABLE:
        raise ImportError(
            "rich is required for experiment progress display. "
            "Install with: pip install rich"
        )

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
            max_workers = 1  # MPS doesn't support multiprocessing well
        else:
            max_workers = min(len(config_paths), mp.cpu_count())

    # Find the scripts directory for importing train_vae
    scripts_dir = str(Path(__file__).parent.parent.parent / "scripts")

    # Prepare experiment arguments
    experiment_names = []
    experiments = []

    # Create a multiprocessing manager for the queue
    manager = mp.Manager()
    progress_queue = manager.Queue()

    for i, config_path in enumerate(config_paths):
        # Extract experiment name from filename
        exp_name = config_path.stem
        experiment_names.append(exp_name)

        # Assign device based on strategy
        if device == "cuda" and num_gpus > 0:
            exp_device = f"cuda:{i % num_gpus}"
        else:
            exp_device = device

        experiments.append((str(config_path), exp_name, exp_device, scripts_dir, progress_queue))

    results: list[ExperimentResult] = []

    # Start progress display thread
    stop_event = threading.Event()
    display_thread = threading.Thread(
        target=_progress_display_thread,
        args=(progress_queue, experiment_names, stop_event),
        daemon=True,
    )
    display_thread.start()

    try:
        if not parallel or max_workers == 1:
            # Sequential execution
            for exp_args in experiments:
                result = _run_single_experiment_with_progress(exp_args)
                results.append(result)
        else:
            # Parallel execution with ProcessPoolExecutor
            # Use 'spawn' context for CUDA compatibility
            ctx = mp.get_context("spawn")

            with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
                # Submit all experiments
                future_to_exp = {
                    executor.submit(_run_single_experiment_with_progress, exp): exp
                    for exp in experiments
                }

                # Collect results as they complete
                for future in as_completed(future_to_exp):
                    exp_args = future_to_exp[future]
                    exp_name = exp_args[1]

                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        # Handle unexpected executor errors (log file may not exist)
                        result = ExperimentResult(
                            name=exp_name,
                            config_path=exp_args[0],
                            status="failed",
                            device=exp_args[2],
                            error=f"Executor error: {e}",
                        )
                        results.append(result)
    finally:
        # Stop display thread
        stop_event.set()
        display_thread.join(timeout=2.0)

    # Sort results to match input order
    name_to_result = {r.name: r for r in results}
    ordered_results = []
    for config_path in config_paths:
        exp_name = config_path.stem
        if exp_name in name_to_result:
            ordered_results.append(name_to_result[exp_name])

    return ordered_results


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

    # Define columns
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

    # Build header
    lines = []
    header = "  ".join(f"{name:<{width}}" for name, width in columns)
    separator = "  ".join("-" * width for _, width in columns)

    lines.append(header)
    lines.append(separator)

    # Build rows
    for r in results:
        row_values = [
            r.name[:20],  # Truncate long names
            r.status[:8],
            f"{r.best_loss:.4f}" if r.best_loss is not None else "N/A",
            f"{r.recon_loss:.4f}" if r.recon_loss is not None else "N/A",
            f"{r.kl_loss:.4f}" if r.kl_loss is not None else "N/A",
            f"{r.epochs_trained}/{r.total_epochs}" if r.total_epochs else str(r.epochs_trained),
            r.device[:8],
            _format_duration(r.duration_seconds),
        ]

        row = "  ".join(f"{val:<{width}}" for val, (_, width) in zip(row_values, columns))
        lines.append(row)

    # Summary line
    lines.append(separator)
    successful = sum(1 for r in results if r.status == "success")
    total_time = sum(r.duration_seconds for r in results)
    lines.append(f"Total: {successful}/{len(results)} succeeded in {_format_duration(total_time)}")

    # Show errors for failed experiments
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
    "run_experiments",
    "format_results_table",
    "ExperimentResult",
]
