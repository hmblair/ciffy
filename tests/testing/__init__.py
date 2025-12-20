"""Testing infrastructure for ciffy tests.

This package provides centralized:
- Tolerance configuration for numerical comparisons
- Reusable test patterns (roundtrip, gradient flow, etc.)
- Domain-specific assertion helpers
- Expected value derivation (replacing hardcoded enum values)
- Backend consistency testing registry
"""

from .tolerances import ToleranceProfile, DEFAULT, GPU, STRICT, get_tolerances
from .patterns import (
    assert_roundtrip_preserves_structure,
    assert_gradient_flows,
    assert_cif_roundtrip,
)
from .assertions import (
    assert_valid_angles,
    assert_valid_dihedrals,
    assert_positive_distances,
    assert_coordinates_finite,
)
from .expectations import expected_sequence_values, assert_sequence_matches
from .backend_registry import (
    BackendFunctionSpec,
    BACKEND_FUNCTIONS,
    register_backend_function,
    get_registered_function_names,
)

__all__ = [
    # Tolerances
    "ToleranceProfile",
    "DEFAULT",
    "GPU",
    "STRICT",
    "get_tolerances",
    # Patterns
    "assert_roundtrip_preserves_structure",
    "assert_gradient_flows",
    "assert_cif_roundtrip",
    # Assertions
    "assert_valid_angles",
    "assert_valid_dihedrals",
    "assert_positive_distances",
    "assert_coordinates_finite",
    # Expectations
    "expected_sequence_values",
    "assert_sequence_matches",
    # Backend registry
    "BackendFunctionSpec",
    "BACKEND_FUNCTIONS",
    "register_backend_function",
    "get_registered_function_names",
]
