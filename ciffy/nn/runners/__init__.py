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
    ParallelRunner,
    BaseJobRunner,  # Legacy alias for ParallelRunner
)
from .experiment_runner import (
    ExperimentRunner,
    run_experiments,
    format_results_table,
)
from .inference_runner import (
    InferenceResult,
    InferenceRunner,
    run_inference_jobs,
    format_inference_results_table,
)
from .flow_runner import (
    FlowExperimentConfig,
    FlowExperimentResult,
    FlowExperimentRunner,
    run_flow_experiments,
    format_flow_results_table,
    create_latent_dim_sweep,
    create_layer_sweep,
    # Data scaling experiments
    create_data_scaling_sweep,
    prepare_scaling_jobs,
    run_data_scaling_experiments,
    format_scaling_results_table,
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
    "ParallelRunner",
    "BaseJobRunner",  # Legacy alias
    # Experiment running
    "ExperimentResult",
    "ExperimentRunner",
    "run_experiments",
    "format_results_table",
    # Inference running
    "InferenceResult",
    "InferenceRunner",
    "run_inference_jobs",
    "format_inference_results_table",
    # Flow experiments
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
