"""
Unified timing utilities for benchmarking.

Provides consistent timing infrastructure with proper CUDA/MPS synchronization
for accurate GPU benchmarking.

Usage:
    from tests.profiling.timing import Timer, TimingResult, get_sync_fn

    # Simple benchmark
    result = Timer.benchmark(my_function, warmup=3, runs=10)
    print(f"Mean: {result.mean*1000:.2f}ms")

    # With CUDA synchronization
    sync = get_sync_fn("cuda")
    result = Timer.benchmark(my_function, sync=sync)

    # As context manager
    with Timer(sync=sync) as t:
        my_function()
    print(f"Elapsed: {t.elapsed:.3f}s")
"""

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np


# Default benchmark parameters
DEFAULT_WARMUP = 3
DEFAULT_RUNS = 10


@dataclass
class TimingResult:
    """Statistics from a benchmark run."""
    mean: float
    std: float
    min: float
    max: float
    runs: int

    def __str__(self) -> str:
        return f"{self.mean*1000:.2f}ms ± {self.std*1000:.2f}ms"

    def to_dict(self) -> dict:
        """Convert to dictionary for compatibility with older code."""
        return {
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "runs": self.runs,
        }


class Timer:
    """
    Context manager and utility for benchmarking functions.

    Handles device synchronization for accurate GPU timing. For CUDA/MPS,
    synchronization is performed before starting the timer (to ensure
    previous operations are complete) and after the operation (to ensure
    the timed operation is complete before stopping the timer).

    Usage:
        # As context manager
        with Timer() as t:
            result = some_function()
        print(f"Elapsed: {t.elapsed:.3f}s")

        # For repeated benchmarks
        result = Timer.benchmark(some_function, warmup=3, runs=10)
        print(f"Mean: {result.mean*1000:.2f}ms")

        # With CUDA synchronization
        sync = get_sync_fn("cuda")
        result = Timer.benchmark(some_function, sync=sync)
    """

    def __init__(self, sync: Callable[[], None] | None = None):
        """
        Args:
            sync: Optional synchronization function (e.g., torch.cuda.synchronize).
                  Called before and after timing to ensure accurate measurements.
        """
        self._sync = sync or (lambda: None)
        self._start: float = 0
        self._end: float = 0

    def __enter__(self) -> "Timer":
        self._sync()
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        self._sync()
        self._end = time.perf_counter()

    @property
    def elapsed(self) -> float:
        """Elapsed time in seconds."""
        return self._end - self._start

    @classmethod
    def benchmark(
        cls,
        func: Callable,
        warmup: int = DEFAULT_WARMUP,
        runs: int = DEFAULT_RUNS,
        sync: Callable[[], None] | None = None,
    ) -> TimingResult:
        """
        Run a function multiple times and return timing statistics.

        Args:
            func: Function to benchmark (called with no arguments).
            warmup: Number of warmup runs (not timed).
            runs: Number of timed runs.
            sync: Optional device synchronization function.

        Returns:
            TimingResult with mean, std, min, max times in seconds.
        """
        sync = sync or (lambda: None)

        # Warmup runs (not timed, but still sync to ensure completion)
        for _ in range(warmup):
            func()
            sync()

        # Timed runs
        times = []
        for _ in range(runs):
            sync()  # Ensure previous work is done before starting timer
            start = time.perf_counter()
            func()
            sync()  # Ensure this work is done before stopping timer
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        return TimingResult(
            mean=float(np.mean(times)),
            std=float(np.std(times)),
            min=float(np.min(times)),
            max=float(np.max(times)),
            runs=runs,
        )


def get_sync_fn(device: str) -> Callable[[], None]:
    """
    Get the appropriate synchronization function for a device.

    For CUDA and MPS devices, returns the synchronization function that
    blocks until all operations on the device are complete. For CPU,
    returns a no-op.

    Args:
        device: Device string ('cpu', 'cuda', 'cuda:0', 'mps', etc.)

    Returns:
        Synchronization function.
    """
    # Normalize device string
    device_type = device.split(":")[0].lower()

    if device_type == "cuda":
        try:
            import torch
            return torch.cuda.synchronize
        except ImportError:
            return lambda: None
    elif device_type == "mps":
        try:
            import torch
            return torch.mps.synchronize
        except (ImportError, AttributeError):
            return lambda: None

    return lambda: None


def get_available_devices() -> list[str]:
    """
    Get list of available devices for benchmarking.

    Returns:
        List of device strings (e.g., ['cpu', 'cuda', 'mps']).
    """
    devices = ["cpu"]

    try:
        import torch

        if torch.cuda.is_available():
            devices.append("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            devices.append("mps")
    except ImportError:
        pass

    return devices
