# ciffy-cuda

CUDA acceleration for [ciffy](https://github.com/hmblair/ciffy) - GPU-accelerated coordinate conversions for molecular structures.

## Installation

```bash
pip install ciffy-cuda
```

This automatically installs `ciffy` as a dependency.

### Requirements

- NVIDIA GPU with CUDA support
- PyTorch with CUDA (e.g., `pip install torch --index-url https://download.pytorch.org/whl/cu121`)
- CUDA toolkit matching your PyTorch version

## Usage

Once installed, CUDA acceleration is used automatically when tensors are on GPU:

```python
import ciffy

# Load structure with PyTorch backend
polymer = ciffy.load("structure.cif", backend="torch")

# Move to GPU
polymer = polymer.to("cuda")

# Operations now use CUDA kernels
distances = polymer.distances    # GPU-accelerated
angles = polymer.angles          # GPU-accelerated
dihedrals = polymer.dihedrals    # GPU-accelerated
```

## Building from Source

```bash
git clone https://github.com/hmblair/ciffy.git
cd ciffy
pip install -e .           # Install base package
pip install -e ./cuda      # Install CUDA extension
```

### Build Options

Environment variables:

- `CIFFY_CUDA_ARCH`: GPU architectures to compile for (e.g., `"86"` or `"70,75,80,86"`)
- `CIFFY_CUDA_DEBUG`: Set to `"1"` for debug builds

## Verifying Installation

```python
>>> import ciffy._cuda
>>> print("CUDA extension loaded!")
```

Or check via ciffy:

```python
>>> from ciffy.backend.cuda_ops import HAS_CUDA_EXTENSION
>>> print(f"CUDA extension available: {HAS_CUDA_EXTENSION}")
```
