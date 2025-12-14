"""
PyTorch implementations of array operations.

Provides pure PyTorch implementations without torch_scatter dependency.
"""

from __future__ import annotations

import torch


def _ensure_same_device(index: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    """Move index to same device as src if needed."""
    if index.device != src.device:
        return index.to(src.device)
    return index


def scatter_sum(
    src: torch.Tensor,
    index: torch.Tensor,
    dim_size: int,
) -> torch.Tensor:
    """
    Sum values into an output tensor at specified indices.

    Args:
        src: Source tensor of shape (N, ...).
        index: Index tensor of shape (N,) with values in [0, dim_size).
        dim_size: Size of output first dimension.

    Returns:
        Tensor of shape (dim_size, ...) with summed values.
    """
    index = _ensure_same_device(index, src)
    result = torch.zeros(
        (dim_size, *src.shape[1:]),
        dtype=src.dtype,
        device=src.device
    )
    # Expand index to match src shape for scatter_add_
    expanded_index = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
    result.scatter_add_(0, expanded_index, src)
    return result


def scatter_mean(
    src: torch.Tensor,
    index: torch.Tensor,
    dim_size: int,
) -> torch.Tensor:
    """
    Average values into an output tensor at specified indices.

    Args:
        src: Source tensor of shape (N, ...).
        index: Index tensor of shape (N,) with values in [0, dim_size).
        dim_size: Size of output first dimension.

    Returns:
        Tensor of shape (dim_size, ...) with averaged values.
    """
    index = _ensure_same_device(index, src)
    sums = scatter_sum(src, index, dim_size)
    counts = torch.zeros(dim_size, dtype=torch.long, device=src.device)
    counts.scatter_add_(0, index, torch.ones_like(index))
    counts = counts.clamp(min=1)
    # Reshape counts for broadcasting
    shape = (-1,) + (1,) * (sums.dim() - 1)
    return sums / counts.view(shape).float()


def scatter_max(
    src: torch.Tensor,
    index: torch.Tensor,
    dim_size: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Maximum values at specified indices.

    Args:
        src: Source tensor of shape (N, ...).
        index: Index tensor of shape (N,) with values in [0, dim_size).
        dim_size: Size of output first dimension.

    Returns:
        Tuple of (max_values, argmax_indices). argmax_indices may be None.
    """
    index = _ensure_same_device(index, src)
    # Use appropriate min value based on dtype
    if src.dtype.is_floating_point:
        fill_value = float('-inf')
    else:
        fill_value = torch.iinfo(src.dtype).min

    result = torch.full(
        (dim_size, *src.shape[1:]),
        fill_value,
        dtype=src.dtype,
        device=src.device
    )
    expanded_index = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
    result.scatter_reduce_(0, expanded_index, src, reduce='amax', include_self=False)

    # Replace sentinel with 0 for empty bins
    result = torch.where(result == fill_value, torch.zeros_like(result), result)

    return result, None


def scatter_min(
    src: torch.Tensor,
    index: torch.Tensor,
    dim_size: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Minimum values at specified indices.

    Args:
        src: Source tensor of shape (N, ...).
        index: Index tensor of shape (N,) with values in [0, dim_size).
        dim_size: Size of output first dimension.

    Returns:
        Tuple of (min_values, argmin_indices). argmin_indices may be None.
    """
    index = _ensure_same_device(index, src)
    # Use appropriate max value based on dtype
    if src.dtype.is_floating_point:
        fill_value = float('inf')
    else:
        fill_value = torch.iinfo(src.dtype).max

    result = torch.full(
        (dim_size, *src.shape[1:]),
        fill_value,
        dtype=src.dtype,
        device=src.device
    )
    expanded_index = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
    result.scatter_reduce_(0, expanded_index, src, reduce='amin', include_self=False)

    # Replace sentinel with 0 for empty bins
    result = torch.where(result == fill_value, torch.zeros_like(result), result)

    return result, None


def repeat_interleave(arr: torch.Tensor, repeats: torch.Tensor) -> torch.Tensor:
    """
    Repeat elements of a tensor along the first axis.

    Args:
        arr: Tensor to repeat elements from.
        repeats: Number of times to repeat each element.

    Returns:
        Tensor with repeated elements.
    """
    # Ensure repeats is on same device as arr
    if repeats.device != arr.device:
        repeats = repeats.to(arr.device)
    return arr.repeat_interleave(repeats, dim=0)


def cdist(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise Euclidean distances.

    Args:
        x1: Tensor of shape (M, D).
        x2: Tensor of shape (N, D).

    Returns:
        Distance matrix of shape (M, N).
    """
    return torch.cdist(x1, x2)


def zeros(
    shape: tuple[int, ...],
    dtype: torch.dtype = torch.float32,
    device: str = 'cpu'
) -> torch.Tensor:
    """Create a zeros tensor."""
    return torch.zeros(shape, dtype=dtype, device=device)


def ones(
    shape: tuple[int, ...],
    dtype: torch.dtype = torch.float32,
    device: str = 'cpu'
) -> torch.Tensor:
    """Create a ones tensor."""
    return torch.ones(shape, dtype=dtype, device=device)


def arange(n: int, dtype: torch.dtype = torch.long, device: str = 'cpu') -> torch.Tensor:
    """Create a range tensor."""
    return torch.arange(n, dtype=dtype, device=device)


def cat(tensors: list, dim: int = 0) -> torch.Tensor:
    """Concatenate tensors."""
    return torch.cat(tensors, dim=dim)


def multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Element-wise multiplication."""
    return torch.multiply(a, b)


def sign(arr: torch.Tensor) -> torch.Tensor:
    """Element-wise sign (-1, 0, or 1)."""
    return torch.sign(arr)
