"""Registry-based backend consistency testing.

This module provides a registry system for backend functions that automatically
generates consistency tests between numpy and torch implementations.

To add a new backend function to the test suite:
1. Create an input generator function
2. Register the function with @register_backend_function or BACKEND_FUNCTIONS.append()

Example:
    @register_backend_function(
        name="my_function",
        input_generator=generate_my_function_inputs,
    )
    def my_function(arr):
        ...

Or manually:
    BACKEND_FUNCTIONS.append(BackendFunctionSpec(
        name="my_function",
        input_generator=generate_my_function_inputs,
    ))
"""

from dataclasses import dataclass, field
from typing import Callable, Any
import numpy as np


@dataclass
class BackendFunctionSpec:
    """Specification for a backend function to test.

    Attributes:
        name: Function name (must match the name in ciffy.backend.ops)
        input_generator: Callable that returns dict of {arg_name: numpy_array}
        comparator: Optional custom comparison function(numpy_result, torch_result) -> bool
        atol: Absolute tolerance for comparison (default 1e-5)
        rtol: Relative tolerance for comparison (default 1e-5)
        output_processor: Optional function to extract comparable values from output
        skip_reason: If set, skip this test with this reason
    """

    name: str
    input_generator: Callable[[], dict[str, Any]]
    comparator: Callable[[Any, Any], bool] | None = None
    atol: float = 1e-5
    rtol: float = 1e-5
    output_processor: Callable[[Any], Any] | None = None
    skip_reason: str | None = None


# Global registry of backend functions to test
BACKEND_FUNCTIONS: list[BackendFunctionSpec] = []


def register_backend_function(
    name: str,
    input_generator: Callable[[], dict[str, Any]],
    **kwargs,
) -> None:
    """Register a backend function for consistency testing.

    Args:
        name: Function name in ciffy.backend.ops
        input_generator: Function returning dict of numpy inputs
        **kwargs: Additional arguments for BackendFunctionSpec
    """
    BACKEND_FUNCTIONS.append(BackendFunctionSpec(
        name=name,
        input_generator=input_generator,
        **kwargs,
    ))


# =============================================================================
# Input Generators
# =============================================================================


def _random_seed():
    """Get a fixed seed for reproducible tests."""
    return 42


def generate_scatter_sum_inputs() -> dict[str, Any]:
    """Generate inputs for scatter_sum."""
    np.random.seed(_random_seed())
    n, dim_size = 100, 10
    return {
        "src": np.random.randn(n, 3).astype(np.float32),
        "index": np.random.randint(0, dim_size, size=n).astype(np.int64),
        "dim_size": dim_size,
    }


def generate_scatter_mean_inputs() -> dict[str, Any]:
    """Generate inputs for scatter_mean."""
    np.random.seed(_random_seed())
    n, dim_size = 100, 10
    return {
        "src": np.random.randn(n, 3).astype(np.float32),
        "index": np.random.randint(0, dim_size, size=n).astype(np.int64),
        "dim_size": dim_size,
    }


def generate_scatter_max_inputs() -> dict[str, Any]:
    """Generate inputs for scatter_max."""
    np.random.seed(_random_seed())
    n, dim_size = 100, 10
    return {
        "src": np.random.randn(n, 3).astype(np.float32),
        "index": np.random.randint(0, dim_size, size=n).astype(np.int64),
        "dim_size": dim_size,
    }


def generate_scatter_min_inputs() -> dict[str, Any]:
    """Generate inputs for scatter_min."""
    np.random.seed(_random_seed())
    n, dim_size = 100, 10
    return {
        "src": np.random.randn(n, 3).astype(np.float32),
        "index": np.random.randint(0, dim_size, size=n).astype(np.int64),
        "dim_size": dim_size,
    }


def generate_cdist_inputs() -> dict[str, Any]:
    """Generate inputs for cdist."""
    np.random.seed(_random_seed())
    return {
        "x1": np.random.randn(20, 3).astype(np.float32),
        "x2": np.random.randn(30, 3).astype(np.float32),
    }


def generate_repeat_interleave_inputs() -> dict[str, Any]:
    """Generate inputs for repeat_interleave."""
    np.random.seed(_random_seed())
    n = 10
    return {
        "arr": np.random.randn(n, 3).astype(np.float32),
        "repeats": np.random.randint(1, 5, size=n).astype(np.int64),
    }


def generate_cat_inputs() -> dict[str, Any]:
    """Generate inputs for cat."""
    np.random.seed(_random_seed())
    return {
        "arrays": [
            np.random.randn(5, 3).astype(np.float32),
            np.random.randn(7, 3).astype(np.float32),
            np.random.randn(3, 3).astype(np.float32),
        ],
        "axis": 0,
    }


def generate_multiply_inputs() -> dict[str, Any]:
    """Generate inputs for multiply."""
    np.random.seed(_random_seed())
    return {
        "a": np.random.randn(10, 3).astype(np.float32),
        "b": np.random.randn(10, 3).astype(np.float32),
    }


def generate_sign_inputs() -> dict[str, Any]:
    """Generate inputs for sign."""
    np.random.seed(_random_seed())
    return {
        "arr": np.random.randn(50).astype(np.float32),
    }


def generate_sqrt_inputs() -> dict[str, Any]:
    """Generate inputs for sqrt."""
    np.random.seed(_random_seed())
    # Ensure positive values for sqrt
    return {
        "arr": np.abs(np.random.randn(50).astype(np.float32)) + 0.1,
    }


def generate_clamp_inputs() -> dict[str, Any]:
    """Generate inputs for clamp."""
    np.random.seed(_random_seed())
    return {
        "arr": np.random.randn(50).astype(np.float32) * 5,
        "min_val": -2.0,
        "max_val": 2.0,
    }


def generate_det_inputs() -> dict[str, Any]:
    """Generate inputs for det."""
    np.random.seed(_random_seed())
    # Generate a random square matrix
    return {
        "arr": np.random.randn(4, 4).astype(np.float32),
    }


def generate_svd_inputs() -> dict[str, Any]:
    """Generate inputs for svd."""
    np.random.seed(_random_seed())
    return {
        "arr": np.random.randn(5, 3).astype(np.float32),
    }


def generate_svdvals_inputs() -> dict[str, Any]:
    """Generate inputs for svdvals."""
    np.random.seed(_random_seed())
    return {
        "arr": np.random.randn(5, 3).astype(np.float32),
    }


def generate_eigh_inputs() -> dict[str, Any]:
    """Generate inputs for eigh (symmetric matrix)."""
    np.random.seed(_random_seed())
    # Create a symmetric matrix
    a = np.random.randn(4, 4).astype(np.float32)
    symmetric = (a + a.T) / 2
    return {
        "arr": symmetric,
    }


def generate_argsort_inputs() -> dict[str, Any]:
    """Generate inputs for argsort."""
    np.random.seed(_random_seed())
    return {
        "arr": np.random.randn(50).astype(np.float32),
    }


def generate_diff_inputs() -> dict[str, Any]:
    """Generate inputs for diff."""
    np.random.seed(_random_seed())
    return {
        "arr": np.random.randn(50).astype(np.float32),
    }


def generate_nonzero_1d_inputs() -> dict[str, Any]:
    """Generate inputs for nonzero_1d."""
    np.random.seed(_random_seed())
    # Create array with some zeros
    arr = np.random.randn(50).astype(np.float32)
    arr[arr < 0] = 0  # Set negative values to zero
    return {
        "arr": arr,
    }


def generate_topk_inputs() -> dict[str, Any]:
    """Generate inputs for topk."""
    np.random.seed(_random_seed())
    return {
        "arr": np.random.randn(50).astype(np.float32),
        "k": 5,
    }


def generate_isin_inputs() -> dict[str, Any]:
    """Generate inputs for isin."""
    np.random.seed(_random_seed())
    return {
        "arr": np.random.randint(0, 20, size=50).astype(np.int64),
        "values": [1, 5, 10, 15],
    }


# =============================================================================
# Output Processors (for functions with complex outputs)
# =============================================================================


def process_scatter_max_output(result: tuple) -> Any:
    """Extract just the values from scatter_max (ignore argmax)."""
    return result[0]


def process_scatter_min_output(result: tuple) -> Any:
    """Extract just the values from scatter_min (ignore argmin)."""
    return result[0]


def process_svd_output(result: tuple) -> tuple:
    """Process SVD output - handle sign ambiguity in U and Vh."""
    U, S, Vh = result
    # SVD has sign ambiguity - U and Vh columns can be negated together
    # We just compare singular values which are unique
    return S


def process_eigh_output(result: tuple) -> tuple:
    """Process eigh output - eigenvalues only (eigenvectors have sign ambiguity)."""
    eigenvalues, eigenvectors = result
    return eigenvalues


def process_topk_output(result: tuple) -> Any:
    """Extract just the values from topk (indices may differ for equal values)."""
    values, indices = result
    return values


# =============================================================================
# Custom Comparators
# =============================================================================


def compare_sorted_indices(np_result: Any, torch_result: Any) -> bool:
    """Compare results that are index arrays (order matters)."""
    import torch
    if isinstance(torch_result, torch.Tensor):
        torch_result = torch_result.cpu().numpy()
    return np.array_equal(np_result, torch_result)


# =============================================================================
# Register All Backend Functions
# =============================================================================

# Scatter operations
register_backend_function("scatter_sum", generate_scatter_sum_inputs)
register_backend_function("scatter_mean", generate_scatter_mean_inputs)
register_backend_function(
    "scatter_max",
    generate_scatter_max_inputs,
    output_processor=process_scatter_max_output,
)
register_backend_function(
    "scatter_min",
    generate_scatter_min_inputs,
    output_processor=process_scatter_min_output,
)

# Array operations
register_backend_function("cdist", generate_cdist_inputs)
register_backend_function("repeat_interleave", generate_repeat_interleave_inputs)
register_backend_function("cat", generate_cat_inputs)
register_backend_function("multiply", generate_multiply_inputs)
register_backend_function("sign", generate_sign_inputs)

# Math operations
register_backend_function("sqrt", generate_sqrt_inputs)
register_backend_function("clamp", generate_clamp_inputs)
register_backend_function("argsort", generate_argsort_inputs, comparator=compare_sorted_indices)
register_backend_function("diff", generate_diff_inputs)

# Linear algebra
register_backend_function("det", generate_det_inputs, atol=1e-4)  # det can accumulate error
register_backend_function(
    "svd",
    generate_svd_inputs,
    output_processor=process_svd_output,
    atol=1e-4,
)
register_backend_function("svdvals", generate_svdvals_inputs, atol=1e-4)
register_backend_function(
    "eigh",
    generate_eigh_inputs,
    output_processor=process_eigh_output,
    atol=1e-4,
)

# Utility operations
register_backend_function("nonzero_1d", generate_nonzero_1d_inputs, comparator=compare_sorted_indices)
register_backend_function(
    "topk",
    generate_topk_inputs,
    output_processor=process_topk_output,
    atol=1e-5,
)
register_backend_function("isin", generate_isin_inputs, comparator=compare_sorted_indices)


def get_registered_function_names() -> list[str]:
    """Get list of all registered function names."""
    return [spec.name for spec in BACKEND_FUNCTIONS]
