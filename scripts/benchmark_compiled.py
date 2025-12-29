"""Benchmark chain assembly with torch.compile."""
import torch
import numpy as np
import time

torch.set_float32_matmul_precision('high')

def timeit(fn, n_warmup=3, n_runs=10):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    return np.mean(times), np.std(times)


def cumulative_matmul(M):
    """Sequential cumulative matrix multiplication."""
    n = len(M)
    result = torch.zeros_like(M)
    result[0] = M[0]
    for i in range(1, n):
        result[i] = result[i-1] @ M[i]
    return result


def build_se3_matrices(transforms):
    """Convert (n, 6) axis-angle+translation to (n, 4, 4) SE(3) matrices."""
    n = len(transforms)
    device = transforms.device

    axis_angles = transforms[:, :3]
    translations = transforms[:, 3:]

    angles = torch.norm(axis_angles, dim=1, keepdim=True)
    safe_angles = torch.where(angles < 1e-8, torch.ones_like(angles), angles)
    axes = axis_angles / safe_angles

    K = torch.zeros(n, 3, 3, device=device)
    K[:, 0, 1] = -axes[:, 2]
    K[:, 0, 2] = axes[:, 1]
    K[:, 1, 0] = axes[:, 2]
    K[:, 1, 2] = -axes[:, 0]
    K[:, 2, 0] = -axes[:, 1]
    K[:, 2, 1] = axes[:, 0]

    eye = torch.eye(3, device=device).unsqueeze(0).expand(n, -1, -1)
    sin_a = torch.sin(angles).unsqueeze(-1)
    cos_a = torch.cos(angles).unsqueeze(-1)
    Rs = eye + sin_a * K + (1 - cos_a) * (K @ K)

    T = torch.zeros(n, 4, 4, device=device)
    T[:, :3, :3] = Rs
    T[:, :3, 3] = translations
    T[:, 3, 3] = 1.0

    return T


def assemble_chain(coords, transforms):
    """
    Assemble chain from per-residue coordinates and transforms.

    Args:
        coords: (n_residues, n_atoms, 3) canonical coordinates
        transforms: (n_residues, 6) SE(3) transforms [axis-angle, translation]

    Returns:
        (n_residues, n_atoms, 3) positioned coordinates
    """
    n_residues, n_atoms, _ = coords.shape
    device = coords.device

    # Build SE(3) matrices
    T = build_se3_matrices(transforms)

    # Cumulative product
    T_cumul = cumulative_matmul(T)

    # Apply to all coordinates
    ones = torch.ones(n_residues, n_atoms, 1, device=device)
    coords_h = torch.cat([coords, ones], dim=2)  # (n, n_atoms, 4)

    # Batch transform: (n, 4, 4) @ (n, 4, n_atoms) -> (n, 4, n_atoms)
    result_h = torch.bmm(T_cumul, coords_h.transpose(1, 2)).transpose(1, 2)

    return result_h[:, :, :3]


# Compile the full assembly
assemble_chain_compiled = torch.compile(assemble_chain, mode="reduce-overhead")

print("=" * 70)
print("CHAIN ASSEMBLY WITH torch.compile")
print("=" * 70)
print(f"GPU: {torch.cuda.get_device_name()}")
print()

chain_lengths = [10, 50, 100, 200, 500, 1000]
n_atoms = 22

print(f"{'Residues':>10} | {'Eager (ms)':>12} | {'Compiled (ms)':>12} | {'Speedup':>10}")
print("-" * 60)

for n_res in chain_lengths:
    coords = torch.randn(n_res, n_atoms, 3, device='cuda')
    transforms = torch.randn(n_res, 6, device='cuda') * 0.1
    transforms[:, 3:] *= 3

    def run_eager():
        return assemble_chain(coords, transforms)

    def run_compiled():
        return assemble_chain_compiled(coords, transforms)

    eager_mean, eager_std = timeit(run_eager)
    compiled_mean, compiled_std = timeit(run_compiled)

    speedup = eager_mean / compiled_mean

    print(f"{n_res:>10} | {eager_mean:>8.2f}±{eager_std:>3.1f} | {compiled_mean:>8.2f}±{compiled_std:>3.1f} | {speedup:>8.1f}x")

# Compare with CPU
print()
print("=" * 70)
print("CPU vs GPU COMPILED")
print("=" * 70)
print(f"{'Residues':>10} | {'CPU (ms)':>12} | {'GPU Compiled':>12} | {'GPU Speedup':>10}")
print("-" * 60)

for n_res in chain_lengths:
    coords_cpu = torch.randn(n_res, n_atoms, 3)
    coords_gpu = coords_cpu.to('cuda')
    transforms_cpu = torch.randn(n_res, 6) * 0.1
    transforms_cpu[:, 3:] *= 3
    transforms_gpu = transforms_cpu.to('cuda')

    def run_cpu():
        return assemble_chain(coords_cpu, transforms_cpu)

    def run_gpu():
        return assemble_chain_compiled(coords_gpu, transforms_gpu)

    cpu_mean, cpu_std = timeit(run_cpu, n_warmup=2, n_runs=5)
    gpu_mean, gpu_std = timeit(run_gpu)

    speedup = cpu_mean / gpu_mean

    print(f"{n_res:>10} | {cpu_mean:>8.2f}±{cpu_std:>3.1f} | {gpu_mean:>8.2f}±{gpu_std:>3.1f} | {speedup:>8.1f}x")

# Batch processing (multiple chains)
print()
print("=" * 70)
print("BATCH PROCESSING (Multiple chains)")
print("=" * 70)

n_res = 100
batch_sizes = [1, 10, 50, 100]

print(f"Chain length: {n_res} residues")
print(f"{'Batch':>10} | {'CPU (ms)':>12} | {'GPU Compiled':>12} | {'GPU Speedup':>10}")
print("-" * 60)

for batch in batch_sizes:
    coords_cpu = torch.randn(batch, n_res, n_atoms, 3)
    coords_gpu = coords_cpu.to('cuda')
    transforms_cpu = torch.randn(batch, n_res, 6) * 0.1
    transforms_cpu[:, :, 3:] *= 3
    transforms_gpu = transforms_cpu.to('cuda')

    def run_batch_cpu():
        results = []
        for i in range(batch):
            results.append(assemble_chain(coords_cpu[i], transforms_cpu[i]))
        return torch.stack(results)

    def run_batch_gpu():
        results = []
        for i in range(batch):
            results.append(assemble_chain_compiled(coords_gpu[i], transforms_gpu[i]))
        return torch.stack(results)

    cpu_mean, cpu_std = timeit(run_batch_cpu, n_warmup=2, n_runs=5)
    gpu_mean, gpu_std = timeit(run_batch_gpu, n_warmup=2, n_runs=5)

    speedup = cpu_mean / gpu_mean

    print(f"{batch:>10} | {cpu_mean:>8.2f}±{cpu_std:>3.1f} | {gpu_mean:>8.2f}±{gpu_std:>3.1f} | {speedup:>8.1f}x")

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print("""
torch.compile with mode='reduce-overhead':
- Fuses kernel launches in the sequential loop
- Eliminates GPU overhead that made eager mode slower than CPU
- Provides significant speedup for chain assembly

Recommendation: Use torch.compile for GPU chain assembly
""")
