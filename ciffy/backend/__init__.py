"""
Backend abstraction for NumPy and PyTorch array operations.

This module provides a unified interface for array operations that can work
with either NumPy arrays or PyTorch tensors. The backend is automatically
detected from the array type.

Submodules
----------
graph
    Bond graph construction and topology data structures.
    See :mod:`ciffy.backend.graph` for details.

cuda_ops
    CUDA kernel wrappers for GPU-accelerated operations.
    See :mod:`ciffy.backend.cuda_ops` for details.
"""

from .core import (
    Array,
    Backend,
    Dtype,
    get_backend,
    is_torch,
    is_numpy,
    to_numpy,
    to_torch,
    as_dtype,
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
    pinv,
    diag,
    diag_embed,
    diagonal,
    multiply,
    sqrt,
    sin,
    cos,
    acos,
    log,
    exp,
    norm,
    outer,
    clamp,
    where,
    bmm,
    unsqueeze,
    expand,
    transpose,
    sigmoid,
    prod,
    mean,
    # Array operations
    scatter_sum,
    scatter_mean,
    cdist,
    cat,
    stack,
    repeat_interleave,
    # Array creation/manipulation
    clone,
    empty,
    empty_like,
    zeros,
    zeros_like,
    zeros_nd,
    ones,
    ones_like,
    eye,
    fill_diagonal,
    full,
    randn,
    allclose,
    to_backend,
    convert_backend,
    # Dtype conversion
    to_float64,
    to_dtype_of,
)

__all__ = [
    # Core
    "Array",
    "Backend",
    "Dtype",
    "get_backend",
    "is_torch",
    "is_numpy",
    "to_numpy",
    "to_torch",
    "as_dtype",
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
    "pinv",
    "diag",
    "diagonal",
    "multiply",
    "bmm",
    # Math
    "sqrt",
    "sin",
    "cos",
    "acos",
    "log",
    "exp",
    "norm",
    "outer",
    "clamp",
    "where",
    "unsqueeze",
    "expand",
    "transpose",
    "sigmoid",
    "prod",
    "mean",
    # Array operations
    "scatter_sum",
    "scatter_mean",
    "cdist",
    "cat",
    "stack",
    "repeat_interleave",
    # Array creation/manipulation
    "clone",
    "empty",
    "empty_like",
    "zeros",
    "zeros_like",
    "zeros_nd",
    "ones",
    "ones_like",
    "eye",
    "fill_diagonal",
    "full",
    "randn",
    "allclose",
    "to_backend",
    "convert_backend",
    # Dtype conversion
    "to_float64",
    "to_dtype_of",
]
