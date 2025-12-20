"""Multi-job runners for parallel training and inference.

Provides utilities for running multiple experiments or inference jobs
in parallel with GPU distribution and progress tracking.

Example:
    >>> from ciffy.nn.runners import run_experiments, run_inference_jobs
    >>>
    >>> results = run_experiments(["config1.yaml", "config2.yaml"])
    >>> inference_results = run_inference_jobs(["inference1.yaml"])
"""

from .utils import (
    get_num_gpus,
    format_duration,
    format_status,
    format_progress_bar,
    ProgressDisplayThread,
)
from .experiment_runner import (
    run_experiments,
    format_results_table,
)
from .inference_runner import (
    InferenceResult,
    run_inference_jobs,
    format_inference_results_table,
)

# Re-export ExperimentResult from training module for convenience
from ..training import ExperimentResult

__all__ = [
    # Shared utilities
    "get_num_gpus",
    "format_duration",
    "format_status",
    "format_progress_bar",
    "ProgressDisplayThread",
    # Experiment running
    "ExperimentResult",
    "run_experiments",
    "format_results_table",
    # Inference running
    "InferenceResult",
    "run_inference_jobs",
    "format_inference_results_table",
]
