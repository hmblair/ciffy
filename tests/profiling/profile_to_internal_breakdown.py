"""Profile breakdown of to_internal forward pass."""
import time
import torch
import ciffy

def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def time_fn(fn, warmup=2, runs=10):
    """Time a function with warmup."""
    for _ in range(warmup):
        fn()
    sync()

    times = []
    for _ in range(runs):
        sync()
        start = time.perf_counter()
        fn()
        sync()
        end = time.perf_counter()
        times.append(end - start)
    return sum(times) / len(times) * 1000  # ms

# Load structure
filepath = "tests/data/9MDS.cif"
polymer = ciffy.load(filepath, backend="torch").poly()

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    polymer = polymer.to(device)

print(f"Device: {device}")
print(f"Atoms: {polymer.size():,}")

# Warmup and get z-matrix built
_ = polymer.dihedrals
original_coords = polymer.coordinates.clone()

# Get components info
mgr = polymer._coord_manager
n_components = mgr._components.n_components
print(f"Components: {n_components}")

# ============================================================================
# Profile the full forward pass
# ============================================================================
print("\n" + "="*60)
print("Full to_internal forward pass (via polymer.dihedrals)")
print("="*60)

def full_forward():
    polymer.coordinates = original_coords.clone()
    return polymer.dihedrals

full_time = time_fn(full_forward)
print(f"Total: {full_time:.2f}ms")

# ============================================================================
# Profile individual components
# ============================================================================
print("\n" + "="*60)
print("Breakdown of _recompute_internal")
print("="*60)

# 1. Just the clone
def just_clone():
    return original_coords.clone()

clone_time = time_fn(just_clone)
print(f"coords.clone(): {clone_time:.2f}ms")

# 2. Setting coordinates (includes validation)
def set_coords():
    polymer._coord_manager._coordinates = original_coords.clone()
    polymer._coord_manager._cartesian_valid = True
    polymer._coord_manager._internal_valid = False

set_coords_time = time_fn(set_coords)
print(f"Setting coordinates: {set_coords_time:.2f}ms")

# 3. get_anchor_coords (vectorized gather - the optimized version)
polymer.coordinates = original_coords.clone()  # Reset state
_ = polymer.dihedrals  # Ensure z-matrix exists

def get_anchor_coords_only():
    return mgr._components.get_anchor_coords(original_coords)

anchor_time = time_fn(get_anchor_coords_only)
print(f"get_anchor_coords (vectorized): {anchor_time:.2f}ms")

# 4. cartesian_to_internal alone (the actual CUDA kernel)
from ciffy.backend.dispatch import cartesian_to_internal
zmatrix_indices = polymer._coord_manager._zmatrix.indices

# Ensure indices are on correct device
if device == "cuda" and not zmatrix_indices.is_cuda:
    zmatrix_indices = torch.from_numpy(zmatrix_indices).to(device)

def c2i_only():
    return cartesian_to_internal(original_coords, zmatrix_indices)

c2i_time = time_fn(c2i_only)
print(f"cartesian_to_internal (CUDA kernel): {c2i_time:.2f}ms")

# 5. Direct CUDA kernel call (bypassing dispatch)
cuda_time = None
if device == "cuda":
    from ciffy.backend.cuda_ops import cuda_cartesian_to_internal, HAS_CUDA_EXTENSION
    if HAS_CUDA_EXTENSION:
        coords_f32 = original_coords.to(torch.float32).contiguous()
        indices_i64 = zmatrix_indices.to(torch.int64).contiguous()

        def cuda_kernel_only():
            return cuda_cartesian_to_internal(coords_f32, indices_i64)

        cuda_time = time_fn(cuda_kernel_only)
        print(f"cuda_cartesian_to_internal (direct): {cuda_time:.2f}ms")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*60)
print("Summary")
print("="*60)
print(f"Total measured: {full_time:.2f}ms")
print(f"  - clone: {clone_time:.2f}ms ({clone_time/full_time*100:.1f}%)")
print(f"  - get_anchor_coords: {anchor_time:.2f}ms ({anchor_time/full_time*100:.1f}%)")
print(f"  - cartesian_to_internal: {c2i_time:.2f}ms ({c2i_time/full_time*100:.1f}%)")
if cuda_time is not None:
    print(f"  - cuda kernel direct: {cuda_time:.2f}ms ({cuda_time/full_time*100:.1f}%)")
