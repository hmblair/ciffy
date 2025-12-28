"""
Pytest configuration for ciffy.nn tests.

Skips all tests in this directory if ciffy.nn is not installed.
The nn module is excluded from the PyPI distribution (research code).
"""

# Check if ciffy.nn is available BEFORE pytest collects tests
try:
    import ciffy.nn  # noqa: F401
    NN_AVAILABLE = True
except ImportError:
    NN_AVAILABLE = False

# Tell pytest to ignore all test files if nn module is not available
if not NN_AVAILABLE:
    collect_ignore = [
        "test_polymer_flow.py",
        "test_residue_flow.py",
        "test_trainer_conformance.py",
        "test_training_runner.py",
    ]
