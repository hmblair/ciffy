"""
Backend abstraction for NumPy and PyTorch array operations.

This module provides a unified interface for array operations that can work
with either NumPy arrays or PyTorch tensors. The backend is automatically
detected from the array type.

Submodules
----------
autograd
    PyTorch autograd functions for differentiable internal coordinate
    conversions. See :mod:`ciffy.backend.autograd` for details.

dispatch
    Device-agnostic dispatch for internal coordinate operations.
    Automatically selects CUDA, CPU, or autograd implementations.
    See :mod:`ciffy.backend.dispatch` for details.

cuda_ops
    CUDA kernel wrappers for GPU-accelerated operations.
    See :mod:`ciffy.backend.cuda_ops` for details.
"""

from .core import (
    Array,
    Backend,
    get_backend,
    is_torch,
    is_numpy,
    to_numpy,
    to_torch,
    size,
    get_device,
    check_compatible,
    has_nan,
    has_inf,
    any_abs_greater_than,
)

from .ops import (
    svd,
    svdvals,
    det,
    eigh,
    multiply,
    sqrt,
    clamp,
    # Array operations
    scatter_sum,
    scatter_mean,
    cdist,
    cat,
    repeat_interleave,
)

__all__ = [
    # Core
    "Array",
    "Backend",
    "get_backend",
    "is_torch",
    "is_numpy",
    "to_numpy",
    "to_torch",
    "size",
    "get_device",
    "check_compatible",
    "has_nan",
    "has_inf",
    "any_abs_greater_than",
    # Linear algebra
    "svd",
    "svdvals",
    "det",
    "eigh",
    "multiply",
    # Math
    "sqrt",
    "clamp",
    # Array operations
    "scatter_sum",
    "scatter_mean",
    "cdist",
    "cat",
    "repeat_interleave",
]
