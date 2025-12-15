"""
Separate setup script for ciffy CUDA extension.

This file builds the optional CUDA extension for GPU-accelerated coordinate
conversions. It uses PyTorch's build system (BuildExtension) which requires
nvcc and a CUDA-capable PyTorch installation.

Usage:
    pip install -e . --no-build-isolation  # First, install the main package
    python setup_cuda.py build_ext --inplace  # Then, build CUDA extension

Environment variables:
    CIFFY_CUDA_ARCH: Comma-separated list of GPU architectures (e.g., "86" or "70,75,80,86")
                     If not set, auto-detects from available GPU or uses common defaults.
    CIFFY_CUDA_DEBUG: Set to "1" to build with debug symbols (-g -G)

Requirements:
    - PyTorch with CUDA support
    - CUDA toolkit (nvcc)
"""

import os
import sys


def get_cuda_arch_flags():
    """
    Determine which GPU architectures to compile for.

    Priority:
    1. CIFFY_CUDA_ARCH environment variable
    2. Auto-detect from available GPU
    3. Fall back to common architectures for distribution

    Returns:
        List of nvcc gencode flags
    """
    # Check environment variable first
    env_arch = os.environ.get('CIFFY_CUDA_ARCH', '').strip()
    if env_arch:
        archs = [a.strip() for a in env_arch.split(',') if a.strip()]
        print(f"Using architectures from CIFFY_CUDA_ARCH: {archs}")
    else:
        # Try to auto-detect from GPU
        archs = []
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    cap = torch.cuda.get_device_capability(i)
                    arch = f"{cap[0]}{cap[1]}"
                    if arch not in archs:
                        archs.append(arch)
                        device_name = torch.cuda.get_device_name(i)
                        print(f"Detected GPU {i}: {device_name} (sm_{arch})")
        except Exception as e:
            print(f"Warning: Could not detect GPU architecture: {e}")

        if not archs:
            # Fall back to common architectures for distribution builds
            archs = ['70', '75', '80', '86', '89', '90']
            print(f"No GPU detected, using common architectures: {archs}")
            print("Tip: Set CIFFY_CUDA_ARCH=XX for faster builds (e.g., CIFFY_CUDA_ARCH=86)")

    # Convert to gencode flags
    flags = []
    for arch in archs:
        # Ensure arch is just the number (e.g., "86" not "sm_86")
        arch = arch.replace('sm_', '').replace('compute_', '')
        flags.append(f'-gencode=arch=compute_{arch},code=sm_{arch}')

    return flags


def check_cuda_compatibility():
    """
    Check CUDA toolkit and PyTorch compatibility.

    Returns:
        Tuple of (pytorch_cuda_version, toolkit_version) or exits on error
    """
    import torch

    pytorch_cuda = torch.version.cuda
    print(f"PyTorch CUDA version: {pytorch_cuda}")

    # Try to get toolkit version from nvcc
    import subprocess
    try:
        result = subprocess.run(
            ['nvcc', '--version'],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            # Parse version from output like "Cuda compilation tools, release 12.1, V12.1.66"
            import re
            match = re.search(r'release (\d+\.\d+)', result.stdout)
            if match:
                toolkit_version = match.group(1)
                print(f"CUDA toolkit version: {toolkit_version}")

                # Warn if major versions differ
                pt_major = pytorch_cuda.split('.')[0]
                tk_major = toolkit_version.split('.')[0]
                if pt_major != tk_major:
                    print(f"WARNING: PyTorch was built with CUDA {pytorch_cuda}, "
                          f"but toolkit is {toolkit_version}")
                    print("This may cause issues. Consider matching versions.")

                return pytorch_cuda, toolkit_version
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"Warning: Could not determine CUDA toolkit version: {e}")

    return pytorch_cuda, None


def main():
    # Check for PyTorch CUDA
    try:
        import torch
        if not torch.cuda.is_available():
            print("ERROR: PyTorch CUDA not available")
            print()
            print("Possible fixes:")
            print("  1. Install PyTorch with CUDA:")
            print("     pip install torch --index-url https://download.pytorch.org/whl/cu121")
            print("  2. Check that CUDA drivers are installed:")
            print("     nvidia-smi")
            print("  3. Verify CUDA toolkit is in PATH:")
            print("     nvcc --version")
            sys.exit(1)
        print(f"PyTorch version: {torch.__version__}")
    except ImportError:
        print("ERROR: PyTorch not found")
        print("Install PyTorch with CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu121")
        sys.exit(1)

    # Check compatibility
    check_cuda_compatibility()

    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    from setuptools import setup

    cuda_sources = [
        'ciffy/src/internal/batch.cu',
        'ciffy/src/internal/cuda_module.cu',
    ]

    # Check if source files exist
    missing = [src for src in cuda_sources if not os.path.exists(src)]
    if missing:
        print(f"ERROR: Missing CUDA source files: {missing}")
        print("Make sure you're running from the ciffy root directory.")
        sys.exit(1)

    print("Building CUDA extension for coordinate conversions...")

    # Build nvcc flags
    nvcc_flags = ['-O3', '--expt-relaxed-constexpr']

    # Add debug flags if requested
    if os.environ.get('CIFFY_CUDA_DEBUG', '').lower() in ('1', 'true', 'yes'):
        nvcc_flags.extend(['-g', '-G', '-lineinfo'])
        print("Debug build enabled")

    # Add architecture flags
    nvcc_flags.extend(get_cuda_arch_flags())

    cuda_ext = CUDAExtension(
        name='ciffy._cuda',
        sources=cuda_sources,
        include_dirs=['ciffy/src'],
        extra_compile_args={
            'cxx': ['-O3', '-std=c++17'],
            'nvcc': nvcc_flags,
        }
    )

    setup(
        name='ciffy-cuda',
        ext_modules=[cuda_ext],
        cmdclass={'build_ext': BuildExtension},
    )

    print()
    print("CUDA extension built successfully!")
    print("Verify with: python -c \"import ciffy._cuda; print('OK')\"")


if __name__ == '__main__':
    main()
