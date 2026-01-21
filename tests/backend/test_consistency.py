"""Tests for numpy/torch backend consistency.

This module automatically tests that all registered backend functions produce
identical results between numpy and torch implementations.

To add a new backend function to these tests:
1. Add an input generator to tests/testing/backend_registry.py
2. Register the function with register_backend_function()

The tests will automatically pick up any newly registered functions.
"""

import numpy as np
import pytest

from tests.utils import TORCH_AVAILABLE
from tests.testing import get_tolerances
from tests.testing.backend_registry import BACKEND_FUNCTIONS, BackendFunctionSpec


def _convert_inputs_to_torch(inputs: dict, torch_module):
    """Convert numpy inputs to torch tensors."""
    converted = {}
    for key, value in inputs.items():
        if isinstance(value, np.ndarray):
            converted[key] = torch_module.from_numpy(value.copy())
        elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.ndarray):
            # List of arrays (e.g., for cat)
            converted[key] = [torch_module.from_numpy(arr.copy()) for arr in value]
        else:
            # Scalars, tuples, etc. pass through unchanged
            converted[key] = value
    return converted


def _to_numpy(result):
    """Convert result to numpy for comparison."""
    try:
        import torch
        if isinstance(result, torch.Tensor):
            return result.detach().cpu().numpy()
    except ImportError:
        pass

    if isinstance(result, np.ndarray):
        return result
    if isinstance(result, (int, float)):
        return np.array(result)
    if isinstance(result, tuple):
        return tuple(_to_numpy(r) for r in result)
    if isinstance(result, list):
        return [_to_numpy(r) for r in result]
    return result


def _compare_results(
    np_result,
    torch_result,
    spec: BackendFunctionSpec,
) -> bool:
    """Compare numpy and torch results using spec's comparator or default."""
    # Apply output processor if specified
    if spec.output_processor is not None:
        np_result = spec.output_processor(np_result)
        torch_result = spec.output_processor(torch_result)

    # Convert torch to numpy
    torch_result = _to_numpy(torch_result)
    np_result = _to_numpy(np_result)

    # Use custom comparator if provided
    if spec.comparator is not None:
        return spec.comparator(np_result, torch_result)

    # Default: numpy allclose
    if isinstance(np_result, tuple):
        return all(
            np.allclose(n, t, atol=spec.atol, rtol=spec.rtol)
            for n, t in zip(np_result, torch_result)
        )

    return np.allclose(np_result, torch_result, atol=spec.atol, rtol=spec.rtol)


# Generate test IDs from function names
def _get_test_id(spec: BackendFunctionSpec) -> str:
    return spec.name


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestBackendConsistency:
    """Test that numpy and torch backends produce identical results."""

    @pytest.mark.parametrize(
        "spec",
        BACKEND_FUNCTIONS,
        ids=[_get_test_id(s) for s in BACKEND_FUNCTIONS],
    )
    def test_backend_consistency(self, spec: BackendFunctionSpec):
        """Test that function produces same results with numpy and torch."""
        import torch
        from ciffy.backend import ops

        # Skip if requested
        if spec.skip_reason:
            pytest.skip(spec.skip_reason)

        # Get the function from ops module
        func = getattr(ops, spec.name, None)
        if func is None:
            pytest.fail(f"Function {spec.name} not found in ciffy.backend.ops")

        # Generate inputs
        np_inputs = spec.input_generator()
        torch_inputs = _convert_inputs_to_torch(np_inputs, torch)

        # Run with numpy backend
        np_result = func(**np_inputs)

        # Run with torch backend
        torch_result = func(**torch_inputs)

        # Compare results
        assert _compare_results(np_result, torch_result, spec), (
            f"Backend mismatch for {spec.name}:\n"
            f"  numpy result: {_to_numpy(np_result)}\n"
            f"  torch result: {_to_numpy(torch_result)}"
        )


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestBackendConsistencyEdgeCases:
    """Test edge cases for backend consistency."""

    def test_scatter_sum_empty_bins(self):
        """Test scatter_sum when some bins receive no values."""
        import torch
        from ciffy.backend import ops

        # Create inputs where some bins are empty
        np_src = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        np_index = np.array([0, 0, 2], dtype=np.int64)  # bin 1 is empty
        dim_size = 3

        torch_src = torch.from_numpy(np_src)
        torch_index = torch.from_numpy(np_index)

        np_result = ops.scatter_sum(np_src, np_index, dim_size)
        torch_result = ops.scatter_sum(torch_src, torch_index, dim_size)

        assert np.allclose(np_result, torch_result.numpy())
        assert np_result[1].sum() == 0  # Empty bin should be zero

    def test_scatter_mean_empty_bins(self):
        """Test scatter_mean when some bins receive no values."""
        import torch
        from ciffy.backend import ops

        np_src = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        np_index = np.array([0, 0], dtype=np.int64)  # bins 1, 2 are empty
        dim_size = 3

        torch_src = torch.from_numpy(np_src)
        torch_index = torch.from_numpy(np_index)

        np_result = ops.scatter_mean(np_src, np_index, dim_size)
        torch_result = ops.scatter_mean(torch_src, torch_index, dim_size)

        assert np.allclose(np_result, torch_result.numpy())
        # Empty bins should be zero (not NaN)
        assert np.all(np.isfinite(np_result))

    def test_cdist_single_point(self):
        """Test cdist with single point inputs."""
        import torch
        from ciffy.backend import ops

        np_x1 = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        np_x2 = np.array([[4.0, 5.0, 6.0]], dtype=np.float32)

        torch_x1 = torch.from_numpy(np_x1)
        torch_x2 = torch.from_numpy(np_x2)

        np_result = ops.cdist(np_x1, np_x2)
        torch_result = ops.cdist(torch_x1, torch_x2)

        assert np.allclose(np_result, torch_result.numpy())
        assert np_result.shape == (1, 1)

    def test_cat_single_array(self):
        """Test cat with a single array."""
        import torch
        from ciffy.backend import ops

        np_arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        torch_arr = torch.from_numpy(np_arr)

        np_result = ops.cat([np_arr], axis=0)
        torch_result = ops.cat([torch_arr], axis=0)

        assert np.allclose(np_result, torch_result.numpy())

    def test_repeat_interleave_zeros(self):
        """Test repeat_interleave with some zero repeats."""
        import torch
        from ciffy.backend import ops

        np_arr = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
        np_repeats = np.array([2, 0, 1], dtype=np.int64)  # Middle element repeated 0 times

        torch_arr = torch.from_numpy(np_arr)
        torch_repeats = torch.from_numpy(np_repeats)

        np_result = ops.repeat_interleave(np_arr, np_repeats)
        torch_result = ops.repeat_interleave(torch_arr, torch_repeats)

        assert np.allclose(np_result, torch_result.numpy())
        assert np_result.shape[0] == 3  # 2 + 0 + 1

    def test_clamp_no_bounds(self):
        """Test clamp with None bounds."""
        import torch
        from ciffy.backend import ops

        np_arr = np.array([-5.0, 0.0, 5.0], dtype=np.float32)
        torch_arr = torch.from_numpy(np_arr)

        # Only min bound
        np_result = ops.clamp(np_arr, min_val=-2.0, max_val=None)
        torch_result = ops.clamp(torch_arr, min_val=-2.0, max_val=None)
        assert np.allclose(np_result, torch_result.numpy())

        # Only max bound
        np_result = ops.clamp(np_arr, min_val=None, max_val=2.0)
        torch_result = ops.clamp(torch_arr, min_val=None, max_val=2.0)
        assert np.allclose(np_result, torch_result.numpy())


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestRegistryCompleteness:
    """Test that all ops functions are registered."""

    def test_all_dispatched_functions_registered(self):
        """Verify all functions using _get_ops dispatch are in the registry."""
        from ciffy.backend import ops
        from tests.testing.backend_registry import get_registered_function_names

        # Functions that use _get_ops dispatch (from reading the source)
        dispatched_functions = {
            "scatter_sum",
            "scatter_mean",
            "scatter_max",
            "scatter_min",
            "repeat_interleave",
            "cdist",
            "cat",
            "multiply",
            "sign",
        }

        # Functions with inline dispatch
        inline_dispatch_functions = {
            "eigh",
            "det",
            "svd",
            "svdvals",
            "sqrt",
            "clamp",
            "argsort",
            "diff",
            "nonzero_1d",
            "topk",
            "isin",
        }

        all_backend_functions = dispatched_functions | inline_dispatch_functions
        registered = set(get_registered_function_names())

        missing = all_backend_functions - registered
        if missing:
            pytest.fail(
                f"Backend functions not registered for testing: {missing}\n"
                f"Add them to tests/testing/backend_registry.py"
            )

    def test_registry_has_valid_functions(self):
        """Verify all registered functions exist in ops module."""
        from ciffy.backend import ops
        from tests.testing.backend_registry import BACKEND_FUNCTIONS

        for spec in BACKEND_FUNCTIONS:
            assert hasattr(ops, spec.name), (
                f"Registered function {spec.name} not found in ciffy.backend.ops"
            )
