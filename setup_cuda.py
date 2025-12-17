#!/usr/bin/env python3
"""
Legacy wrapper for CUDA extension build.

DEPRECATED: Use `pip install ./cuda` or `pip install ciffy-cuda` instead.

This script is kept for backwards compatibility. It delegates to cuda/setup.py.

Usage (legacy):
    python setup_cuda.py build_ext --inplace

Recommended (new):
    pip install -e ./cuda                    # Development
    pip install ciffy-cuda                   # From PyPI
"""

import os
import sys
import warnings

warnings.warn(
    "setup_cuda.py is deprecated. Use 'pip install ./cuda' or 'pip install ciffy-cuda' instead.",
    DeprecationWarning,
    stacklevel=1
)

# Change to cuda directory and run its setup.py
cuda_dir = os.path.join(os.path.dirname(__file__), 'cuda')
os.chdir(cuda_dir)

# Import and run the cuda setup
sys.path.insert(0, cuda_dir)
from setup import main
main()
