"""Shared utilities for experiment and inference runners.

Provides common functionality to eliminate code duplication between
experiment_runner.py and inference_runner.py.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Generic, TypeVar

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

if TYPE_CHECKING:
    from multiprocessing import Queue

logger = logging.getLogger(__name__)

# Type variable for result types
T = TypeVar("T")


def get_num_gpus() -> int:
    """Get number of available CUDA GPUs.

    Returns:
        Number of CUDA GPUs, or 0 if CUDA is not available.
    """
    if not TORCH_AVAILABLE:
        return 0
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.device_count()


def format_duration(seconds: float) -> str:
    """Format duration as human-readable string.

    Args:
        seconds: Duration in seconds.

    Returns:
        Formatted string like "45.2s", "5m30s", or "2h15m".
    """
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


def get_device_strategy(device: str) -> str:
    """Determine device strategy based on available hardware.

    Args:
        device: Requested device ('auto', 'cuda', 'mps', 'cpu').

    Returns:
        Resolved device string.
    """
    if not TORCH_AVAILABLE:
        return "cpu"

    if device == "auto":
        num_gpus = get_num_gpus()
        if num_gpus > 0:
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    return device


def get_max_workers(device: str, num_configs: int, max_workers: int | None = None) -> int:
    """Determine maximum parallel workers based on device.

    Args:
        device: Device string ('cuda', 'mps', 'cpu').
        num_configs: Number of configs to process.
        max_workers: User-specified max workers, or None for auto.

    Returns:
        Maximum number of parallel workers.
    """
    import multiprocessing as mp

    if max_workers is not None:
        return max_workers

    if device == "cuda":
        num_gpus = get_num_gpus()
        return num_gpus if num_gpus > 0 else 1
    elif device == "mps":
        return 1  # MPS doesn't support multiprocessing well
    else:
        return min(num_configs, mp.cpu_count())


def format_status(status: str) -> str:
    """Format status string with rich color markup.

    Args:
        status: Status string ('complete', 'failed', 'running', or other).

    Returns:
        Status string with rich color markup.
    """
    if status == "complete":
        return "[green]complete[/green]"
    elif status == "failed":
        return "[red]failed[/red]"
    elif status == "running":
        return "[yellow]running[/yellow]"
    else:
        return "[dim]pending[/dim]"


def format_progress_bar(current: int, total: int) -> str:
    """Format a progress bar string.

    Args:
        current: Current progress value.
        total: Total value.

    Returns:
        Progress bar string like "████████░░ 8/10".
    """
    if total > 0:
        pct = current / total
        filled = int(pct * 10)
        bar = "█" * filled + "░" * (10 - filled)
        return f"{bar} {current}/{total}"
    else:
        return "..."


class ProgressDisplayThread:
    """Thread that reads from progress queue and updates a live display.

    Encapsulates the common pattern of reading progress updates from a
    multiprocessing queue and updating a rich Live display.

    Args:
        progress_queue: Multiprocessing queue with progress updates.
        names: List of job/experiment names to track.
        create_table_fn: Function that creates a rich Table from states dict.

    Example:
        >>> def create_table(states):
        ...     table = Table(title="Progress")
        ...     # ... build table from states ...
        ...     return table
        >>>
        >>> display = ProgressDisplayThread(queue, ["exp1", "exp2"], create_table)
        >>> display.start()
        >>> # ... run jobs ...
        >>> display.stop()
    """

    def __init__(
        self,
        progress_queue: "Queue",
        names: list[str],
        create_table_fn: Callable[[dict[str, dict]], "Table"],
    ):
        self.progress_queue = progress_queue
        self.names = names
        self.create_table_fn = create_table_fn

        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.states = {name: {"status": "pending"} for name in names}

    def start(self) -> None:
        """Start the display thread."""
        if not RICH_AVAILABLE:
            return

        self.thread = threading.Thread(
            target=self._run,
            daemon=True,
        )
        self.thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        """Stop the display thread.

        Args:
            timeout: Maximum time to wait for thread to finish.
        """
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=timeout)

    def _run(self) -> None:
        """Main loop for the display thread."""
        console = Console()

        with Live(
            self.create_table_fn(self.states),
            console=console,
            refresh_per_second=4,
        ) as live:
            while not self.stop_event.is_set():
                # Process all available messages
                while True:
                    try:
                        msg = self.progress_queue.get(timeout=0.1)
                        name = msg.get("name")
                        if name in self.states:
                            self.states[name].update(msg)
                    except Exception:
                        break

                # Update display
                live.update(self.create_table_fn(self.states))

            # Final update
            live.update(self.create_table_fn(self.states))


# Type variables for generic runners
TConfig = TypeVar("TConfig")  # Config type (Path, dataclass, etc.)
TResult = TypeVar("TResult")  # Result type


class ParallelRunner(ABC, Generic[TConfig, TResult]):
    """Generic base class for running jobs in parallel.

    Provides common infrastructure for parallel execution across CPUs/GPUs
    with progress tracking. Subclasses implement job-specific logic.

    Type parameters:
        TConfig: Configuration type (Path, dataclass, dict, etc.)
        TResult: Result type (must have a 'name' attribute)

    Example:
        >>> @dataclass
        ... class MyConfig:
        ...     name: str
        ...     param: int
        ...
        >>> @dataclass
        ... class MyResult:
        ...     name: str
        ...     value: float
        ...
        >>> class MyRunner(ParallelRunner[MyConfig, MyResult]):
        ...     def run_job(self, config, device, queue):
        ...         # Job logic here
        ...         return MyResult(name=config.name, value=42.0)
        ...
        ...     def get_job_name(self, config):
        ...         return config.name
        ...
        ...     # ... implement other abstract methods
        >>>
        >>> runner = MyRunner()
        >>> results = runner.run([MyConfig("exp1", 10), MyConfig("exp2", 20)])
    """

    def __init__(
        self,
        parallel: bool = True,
        max_workers: int | None = None,
        device: str = "auto",
    ):
        """Initialize runner.

        Args:
            parallel: If True, run jobs in parallel.
            max_workers: Maximum parallel workers. None for auto-detect.
            device: Device strategy ('auto', 'cuda', 'mps', 'cpu').
        """
        self.parallel = parallel
        self.max_workers = max_workers
        self.device = device

    @abstractmethod
    def run_job(
        self,
        config: TConfig,
        device: str,
        progress_queue: "Queue",
    ) -> TResult:
        """Run a single job. Must be implemented by subclass.

        This method may run in a subprocess, so it must be picklable.

        Args:
            config: Job configuration.
            device: Device to run on (e.g., 'cpu', 'cuda:0').
            progress_queue: Queue for sending progress updates.
                Send dicts with at least {'name': str, 'status': str}.

        Returns:
            Result object of type TResult.
        """
        pass

    @abstractmethod
    def get_job_name(self, config: TConfig) -> str:
        """Get job name from config for progress display.

        Args:
            config: Job configuration.

        Returns:
            Job name string.
        """
        pass

    @abstractmethod
    def create_progress_table(self, states: dict[str, dict]) -> "Table":
        """Create progress table for live display.

        Args:
            states: Dict mapping job name to state dict.

        Returns:
            Rich Table for display.
        """
        pass

    @abstractmethod
    def create_failed_result(
        self,
        config: TConfig,
        device: str,
        error: str,
    ) -> TResult:
        """Create a result object for a failed job.

        Args:
            config: Job configuration.
            device: Device string.
            error: Error message.

        Returns:
            Result object of type TResult with failed status.
        """
        pass

    def get_result_name(self, result: TResult) -> str:
        """Get job name from result for ordering. Override if needed.

        Args:
            result: Result object.

        Returns:
            Job name.
        """
        return result.name  # type: ignore

    def validate_config(self, config: TConfig) -> None:
        """Validate a config before running. Override to add validation.

        Args:
            config: Job configuration.

        Raises:
            ValueError: If config is invalid.
        """
        pass

    def run(self, configs: list[TConfig]) -> list[TResult]:
        """Run jobs for all configs.

        Args:
            configs: List of job configurations.

        Returns:
            List of results in the same order as input configs.

        Raises:
            ImportError: If PyTorch or rich is not available.
            ValueError: If any config is invalid.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")

        if not RICH_AVAILABLE:
            raise ImportError(
                "rich is required for progress display. "
                "Install with: pip install rich"
            )

        if not configs:
            return []

        # Validate all configs
        for config in configs:
            self.validate_config(config)

        # Determine device strategy
        device = get_device_strategy(self.device)
        num_gpus = get_num_gpus()

        # Determine worker count
        max_workers = get_max_workers(device, len(configs), self.max_workers)

        # Create progress queue
        manager = mp.Manager()
        progress_queue = manager.Queue()

        # Get job names and prepare job args
        job_names = [self.get_job_name(c) for c in configs]

        # Assign devices (round-robin for multi-GPU)
        job_devices = []
        for i in range(len(configs)):
            if device == "cuda" and num_gpus > 0:
                job_devices.append(f"cuda:{i % num_gpus}")
            else:
                job_devices.append(device)

        results: list[TResult] = []

        # Start progress display
        progress_display = ProgressDisplayThread(
            progress_queue, job_names, self.create_progress_table
        )
        progress_display.start()

        try:
            if not self.parallel or max_workers == 1:
                # Sequential execution
                for config, job_device in zip(configs, job_devices):
                    result = self.run_job(config, job_device, progress_queue)
                    results.append(result)
            else:
                # Parallel execution with spawn context for CUDA
                ctx = mp.get_context("spawn")

                with ProcessPoolExecutor(
                    max_workers=max_workers, mp_context=ctx
                ) as executor:
                    future_to_config = {
                        executor.submit(
                            self.run_job, config, job_device, progress_queue
                        ): (config, job_device)
                        for config, job_device in zip(configs, job_devices)
                    }

                    for future in as_completed(future_to_config):
                        config, job_device = future_to_config[future]

                        try:
                            result = future.result()
                            results.append(result)
                        except Exception as e:
                            result = self.create_failed_result(
                                config, job_device, f"Executor error: {e}"
                            )
                            results.append(result)
        finally:
            progress_display.stop()

        # Reorder results to match input order
        name_to_result = {self.get_result_name(r): r for r in results}
        ordered_results = []
        for config in configs:
            name = self.get_job_name(config)
            if name in name_to_result:
                ordered_results.append(name_to_result[name])

        return ordered_results


# Legacy alias for backwards compatibility
BaseJobRunner = ParallelRunner


__all__ = [
    "get_num_gpus",
    "format_duration",
    "get_device_strategy",
    "get_max_workers",
    "format_status",
    "format_progress_bar",
    "ProgressDisplayThread",
    "ParallelRunner",
    "BaseJobRunner",  # Legacy alias
    "TORCH_AVAILABLE",
    "RICH_AVAILABLE",
]
