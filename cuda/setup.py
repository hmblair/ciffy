"""
Setup script for ciffy-cuda extension.

Builds the optional CUDA extension for GPU-accelerated coordinate conversions.
Uses PyTorch's build system (BuildExtension) which requires nvcc and a
CUDA-capable PyTorch installation.

This extension installs into the ciffy namespace as ciffy._cuda, enabling
GPU acceleration when both ciffy and ciffy-cuda are installed.

Usage:
    pip install .                    # Standard install
    pip install -e . --no-build-isolation  # Editable install

Environment variables:
    CIFFY_CUDA_ARCH: Comma-separated GPU architectures (e.g., "86" or "70,75,80,86")
                    Auto-detects from available GPU if not set.
    CIFFY_CUDA_THREADS: Number of threads for parallel nvcc compilation (default: nproc)
    CIFFY_CUDA_DEBUG: Set to "1" for debug builds (-g -G)
"""

import os
import re
import subprocess
import sys
from pathlib import Path

# ============================================================================
# CUDA Utilities
# ============================================================================

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
    env_arch = os.environ.get('CIFFY_CUDA_ARCH', '').strip()
    if env_arch:
        archs = [a.strip() for a in env_arch.split(',') if a.strip()]
        print(f"Using architectures from CIFFY_CUDA_ARCH: {archs}")
    else:
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
            archs = ['70', '75', '80', '86', '89', '90']
            print(f"No GPU detected, using common architectures: {archs}")
            print("Tip: Set CIFFY_CUDA_ARCH=XX for faster builds (e.g., CIFFY_CUDA_ARCH=86)")

    flags = []
    for arch in archs:
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

    try:
        result = subprocess.run(
            ['nvcc', '--version'],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            match = re.search(r'release (\d+\.\d+)', result.stdout)
            if match:
                toolkit_version = match.group(1)
                print(f"CUDA toolkit version: {toolkit_version}")

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


def check_pytorch_cuda():
    """Verify PyTorch CUDA is available, exit with helpful message if not."""
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
        return torch
    except ImportError:
        print("ERROR: PyTorch not found")
        print("Install PyTorch with CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu121")
        sys.exit(1)


# ============================================================================
# Build Configuration
# ============================================================================

def main():
    torch = check_pytorch_cuda()
    check_cuda_compatibility()

    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    from setuptools import setup

    # Source files are in ../ciffy/src/internal/ (relative to cuda/)
    cuda_dir = Path(__file__).parent
    project_root = cuda_dir.parent
    src_dir = project_root / 'ciffy' / 'src' / 'internal'

    # Use relative paths for setuptools compatibility
    cuda_sources = [
        '../ciffy/src/internal/batch.cu',
        '../ciffy/src/internal/cuda_module.cu',
    ]

    # Verify source files exist (using absolute paths for check)
    missing = [src for src in cuda_sources if not (cuda_dir / src).exists()]
    if missing:
        print(f"ERROR: Missing CUDA source files: {missing}")
        print(f"Expected to find sources in: {src_dir}")
        print("Make sure you're building from the cuda/ directory.")
        sys.exit(1)

    print("Building CUDA extension for coordinate conversions...")

    # Build nvcc flags
    nvcc_flags = ['-O3', '--expt-relaxed-constexpr']

    # Parallel compilation: use multiple threads for nvcc
    # Can be overridden with CIFFY_CUDA_THREADS environment variable
    n_threads = os.environ.get('CIFFY_CUDA_THREADS', '').strip()
    if not n_threads:
        import multiprocessing
        n_threads = str(multiprocessing.cpu_count())
    nvcc_flags.append(f'--threads={n_threads}')
    print(f"Using {n_threads} threads for CUDA compilation")

    if os.environ.get('CIFFY_CUDA_DEBUG', '').lower() in ('1', 'true', 'yes'):
        nvcc_flags.extend(['-g', '-G', '-lineinfo'])
        print("Debug build enabled")

    nvcc_flags.extend(get_cuda_arch_flags())

    cuda_ext = CUDAExtension(
        name='ciffy._cuda',  # Installs into ciffy namespace
        sources=cuda_sources,
        include_dirs=['../ciffy/src'],
        extra_compile_args={
            'cxx': ['-O3', '-std=c++17'],
            'nvcc': nvcc_flags,
        }
    )

    setup(
        ext_modules=[cuda_ext],
        cmdclass={'build_ext': BuildExtension},
    )

    print()
    print("CUDA extension built successfully!")
    print("Verify with: python -c \"import ciffy._cuda; print('OK')\"")


if __name__ == '__main__':
    main()
