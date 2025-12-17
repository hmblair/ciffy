"""
ciffy-cuda: CUDA acceleration for ciffy.

This package provides GPU-accelerated coordinate conversion operations
for ciffy. Installing this package adds the ciffy._cuda extension module.

The CUDA extension is automatically used when:
1. This package is installed
2. PyTorch tensors are on a CUDA device
3. The operation supports GPU acceleration

Example:
    >>> import ciffy
    >>> polymer = ciffy.load("structure.cif", backend="torch")
    >>> polymer = polymer.to("cuda")  # Move to GPU
    >>> # Internal coordinate operations now use CUDA kernels
    >>> internals = polymer.distances  # GPU-accelerated
"""

__all__ = []

# Verify the extension was installed correctly
def _check_extension():
    """Check that ciffy._cuda extension is available."""
    try:
        import ciffy._cuda
        return True
    except ImportError:
        return False

HAS_CUDA_EXTENSION = _check_extension()
