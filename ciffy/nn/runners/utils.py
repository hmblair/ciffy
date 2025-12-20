"""Shared utilities for experiment and inference runners.

Provides common functionality to eliminate code duplication between
experiment_runner.py and inference_runner.py.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Callable

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


__all__ = [
    "get_num_gpus",
    "format_duration",
    "get_device_strategy",
    "get_max_workers",
    "format_status",
    "format_progress_bar",
    "ProgressDisplayThread",
    "TORCH_AVAILABLE",
    "RICH_AVAILABLE",
]
