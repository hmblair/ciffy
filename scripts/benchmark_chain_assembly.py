"""
Benchmark chain assembly optimizations.

Compares:
1. Current sequential assembly (Python loop)
2. Cumulative transforms approach (batched)
3. CPU vs GPU performance

Usage:
    python scripts/benchmark_chain_assembly.py
    # Or with rpy for GPU:
    rpy gpu scripts/benchmark_chain_assembly.py
"""

import time
import numpy as np
import torch
from typing import Callable

# Ensure we can import ciffy
import sys
sys.path.insert(0, '.')


def timeit(fn: Callable, n_warmup: int = 3, n_runs: int = 10) -> tuple[float, float]:
    """Time a function, returning mean and std in milliseconds."""
    # Warmup
    for _ in range(n_warmup):
        fn()

    # Timed runs
    times = []
    for _ in range(n_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)

    return np.mean(times), np.std(times)


# =============================================================================
# CURRENT IMPLEMENTATION (for reference)
# =============================================================================

def current_assemble_chain_numpy(
    residue_coords: list[np.ndarray],
    transforms: list[np.ndarray],
    frame_cols: np.ndarray,  # (n_residues, 3) prev frame columns
    next_frame_cols: np.ndarray,  # (n_residues, 3) next frame columns
) -> np.ndarray:
    """Current sequential assembly - Python loop over residues."""
    from ciffy.geometry.transforms import (
        compute_frame_from_indices,
        apply_relative_transform,
    )

    all_coords = []
    prev_coords = None
    prev_origin = None
    prev_R = None

    for i, (coords, transform) in enumerate(zip(residue_coords, transforms)):
        if i == 0:
            positioned = coords
        else:
            # Compute outgoing frame from prev
            prev_origin, prev_R = compute_frame_from_indices(
                prev_coords, frame_cols[i-1], z_toward_origin=True
            )

            # Apply transform
            target_origin, target_R = apply_relative_transform(
                prev_origin, prev_R, transform
            )

            # Compute incoming frame
            current_origin, current_R = compute_frame_from_indices(
                coords, next_frame_cols[i], z_toward_origin=True
            )

            # Align
            R_correction = target_R @ current_R.T
            t_correction = target_origin - R_correction @ current_origin
            positioned = (R_correction @ coords.T).T + t_correction

        all_coords.append(positioned)
        prev_coords = positioned

    return np.concatenate(all_coords, axis=0)


# =============================================================================
# OPTIMIZED: CUMULATIVE TRANSFORMS
# =============================================================================

def rodrigues_numpy(axis_angles: np.ndarray) -> np.ndarray:
    """Convert (n, 3) axis-angle vectors to (n, 3, 3) rotation matrices using Rodrigues formula."""
    n = len(axis_angles)
    angles = np.linalg.norm(axis_angles, axis=1, keepdims=True)
    safe_angles = np.where(angles < 1e-8, 1.0, angles)
    axes = axis_angles / safe_angles

    # Skew-symmetric matrices K
    K = np.zeros((n, 3, 3))
    K[:, 0, 1] = -axes[:, 2]
    K[:, 0, 2] = axes[:, 1]
    K[:, 1, 0] = axes[:, 2]
    K[:, 1, 2] = -axes[:, 0]
    K[:, 2, 0] = -axes[:, 1]
    K[:, 2, 1] = axes[:, 0]

    eye = np.eye(3)[None, :, :].repeat(n, axis=0)
    sin_a = np.sin(angles)[:, :, None]
    cos_a = np.cos(angles)[:, :, None]

    Rs = eye + sin_a * K + (1 - cos_a) * (K @ K)
    return Rs


def se3_to_matrix_batch(transforms: np.ndarray) -> np.ndarray:
    """Convert (n, 6) transforms to (n, 4, 4) matrices."""
    n = len(transforms)
    axis_angles = transforms[:, :3]
    translations = transforms[:, 3:]

    Rs = rodrigues_numpy(axis_angles)

    # Build homogeneous matrices
    T = np.zeros((n, 4, 4))
    T[:, :3, :3] = Rs
    T[:, :3, 3] = translations
    T[:, 3, 3] = 1.0

    return T


def cumulative_matmul_numpy(matrices: np.ndarray) -> np.ndarray:
    """Compute cumulative product of (n, 4, 4) matrices."""
    n = len(matrices)
    result = np.zeros_like(matrices)
    result[0] = matrices[0]
    for i in range(1, n):
        result[i] = result[i-1] @ matrices[i]
    return result


def cumulative_matmul_torch(matrices: torch.Tensor) -> torch.Tensor:
    """Compute cumulative product of (n, 4, 4) matrices using torch - sequential."""
    n = len(matrices)
    result = torch.zeros_like(matrices)
    result[0] = matrices[0]
    for i in range(1, n):
        result[i] = result[i-1] @ matrices[i]
    return result


def cumulative_matmul_parallel_scan(matrices: torch.Tensor) -> torch.Tensor:
    """
    Compute cumulative product using parallel scan (Blelloch algorithm).

    This has O(log n) depth instead of O(n), enabling GPU parallelization.
    Uses the associativity of matrix multiplication.
    """
    n = len(matrices)
    if n == 0:
        return matrices
    if n == 1:
        return matrices.clone()

    # Pad to power of 2
    log2_n = int(np.ceil(np.log2(n)))
    padded_n = 2 ** log2_n

    # Pad with identity matrices
    if padded_n > n:
        eye = torch.eye(4, device=matrices.device, dtype=matrices.dtype)
        padding = eye.unsqueeze(0).expand(padded_n - n, -1, -1)
        work = torch.cat([matrices, padding], dim=0)
    else:
        work = matrices.clone()

    # Up-sweep (reduce) phase
    for d in range(log2_n):
        step = 2 ** (d + 1)
        # Indices for parallel computation
        indices = torch.arange(step - 1, padded_n, step, device=matrices.device)
        prev_indices = indices - 2**d

        # Batch matrix multiply
        work[indices] = work[prev_indices] @ work[indices]

    # Down-sweep phase
    work[padded_n - 1] = torch.eye(4, device=matrices.device, dtype=matrices.dtype)

    for d in range(log2_n - 1, -1, -1):
        step = 2 ** (d + 1)
        indices = torch.arange(step - 1, padded_n, step, device=matrices.device)
        prev_indices = indices - 2**d

        # Save left child
        temp = work[prev_indices].clone()
        work[prev_indices] = work[indices].clone()
        work[indices] = temp @ work[indices]

    # The result is now exclusive prefix. Convert to inclusive by shifting and multiplying.
    result = torch.zeros_like(work)
    result[0] = matrices[0]
    result[1:] = work[:-1] @ matrices[1:]

    return result[:n]


def cumulative_matmul_chunked(matrices: torch.Tensor, chunk_size: int = 32) -> torch.Tensor:
    """
    Hybrid approach: sequential within chunks, parallel across chunks.

    More practical than full parallel scan for typical chain lengths.
    """
    n = len(matrices)
    if n <= chunk_size:
        return cumulative_matmul_torch(matrices)

    n_chunks = (n + chunk_size - 1) // chunk_size

    # Process each chunk sequentially to get chunk-level cumulative products
    chunk_products = []
    chunk_results = []

    for c in range(n_chunks):
        start = c * chunk_size
        end = min(start + chunk_size, n)
        chunk = matrices[start:end]

        # Sequential within chunk
        chunk_cumul = cumulative_matmul_torch(chunk)
        chunk_results.append(chunk_cumul)
        chunk_products.append(chunk_cumul[-1])

    # Combine chunk products (small sequential operation)
    chunk_products = torch.stack(chunk_products)
    chunk_cumul_products = cumulative_matmul_torch(chunk_products)

    # Apply chunk prefix to each chunk's results
    result_chunks = []
    result_chunks.append(chunk_results[0])  # First chunk unchanged

    for c in range(1, n_chunks):
        prefix = chunk_cumul_products[c - 1]
        # Batch multiply: prefix @ each matrix in chunk
        chunk = chunk_results[c]
        transformed = prefix.unsqueeze(0) @ chunk  # Broadcasting
        result_chunks.append(transformed.squeeze(1) if transformed.dim() > 3 else transformed)

    return torch.cat(result_chunks, dim=0)


def optimized_assemble_chain_numpy(
    residue_coords: list[np.ndarray],
    transforms: np.ndarray,  # (n, 6) all transforms
    canonical_origins: np.ndarray,  # (n, 3) origin in canonical frame
) -> np.ndarray:
    """
    Optimized assembly using cumulative transforms.

    Key insight: Instead of iteratively positioning, we:
    1. Compute cumulative transforms T_0, T_0@T_1, T_0@T_1@T_2, ...
    2. Apply each cumulative transform to the canonical coordinates
    """
    n = len(residue_coords)

    # Convert to homogeneous matrices
    T_matrices = se3_to_matrix_batch(transforms)

    # Compute cumulative transforms
    T_cumulative = cumulative_matmul_numpy(T_matrices)

    # Apply to each residue
    all_coords = []
    for i, coords in enumerate(residue_coords):
        # coords is (n_atoms, 3)
        # Add homogeneous coordinate
        ones = np.ones((len(coords), 1))
        coords_h = np.concatenate([coords, ones], axis=1)  # (n_atoms, 4)

        # Transform
        transformed_h = (T_cumulative[i] @ coords_h.T).T  # (n_atoms, 4)
        all_coords.append(transformed_h[:, :3])

    return np.concatenate(all_coords, axis=0)


def optimized_assemble_chain_torch(
    residue_coords: torch.Tensor,  # (n_residues, n_atoms, 3) padded
    transforms: torch.Tensor,  # (n, 6)
    mask: torch.Tensor,  # (n_residues, n_atoms) bool
) -> torch.Tensor:
    """
    Fully batched assembly for GPU.

    Assumes all residues have same atom count (padded).
    """
    n_residues, n_atoms, _ = residue_coords.shape
    device = residue_coords.device

    # Convert transforms to matrices
    axis_angles = transforms[:, :3]
    translations = transforms[:, 3:]

    # Rodrigues formula for batch rotation
    angles = torch.norm(axis_angles, dim=1, keepdim=True)
    safe_angles = torch.where(angles < 1e-8, torch.ones_like(angles), angles)
    axes = axis_angles / safe_angles

    # Build rotation matrices using Rodrigues
    K = torch.zeros(n_residues, 3, 3, device=device)
    K[:, 0, 1] = -axes[:, 2]
    K[:, 0, 2] = axes[:, 1]
    K[:, 1, 0] = axes[:, 2]
    K[:, 1, 2] = -axes[:, 0]
    K[:, 2, 0] = -axes[:, 1]
    K[:, 2, 1] = axes[:, 0]

    eye = torch.eye(3, device=device).unsqueeze(0).expand(n_residues, -1, -1)
    sin_a = torch.sin(angles).unsqueeze(-1)
    cos_a = torch.cos(angles).unsqueeze(-1)

    Rs = eye + sin_a * K + (1 - cos_a) * (K @ K)

    # Build homogeneous matrices
    T = torch.zeros(n_residues, 4, 4, device=device)
    T[:, :3, :3] = Rs
    T[:, :3, 3] = translations
    T[:, 3, 3] = 1.0

    # Cumulative product (still sequential but on GPU)
    T_cumulative = torch.zeros_like(T)
    T_cumulative[0] = T[0]
    for i in range(1, n_residues):
        T_cumulative[i] = T_cumulative[i-1] @ T[i]

    # Apply to all coordinates at once
    ones = torch.ones(n_residues, n_atoms, 1, device=device)
    coords_h = torch.cat([residue_coords, ones], dim=2)  # (n, n_atoms, 4)

    # Batch matrix multiply: (n, 4, 4) @ (n, 4, n_atoms) -> (n, 4, n_atoms)
    transformed_h = torch.bmm(T_cumulative, coords_h.transpose(1, 2)).transpose(1, 2)

    return transformed_h[:, :, :3]


# =============================================================================
# BENCHMARK
# =============================================================================

def generate_test_data(n_residues: int, n_atoms_per_res: int = 22):
    """Generate random test data for benchmarking."""
    np.random.seed(42)

    # Random coordinates for each residue
    residue_coords = [
        np.random.randn(n_atoms_per_res, 3).astype(np.float32)
        for _ in range(n_residues)
    ]

    # Random transforms (small rotations and translations)
    transforms = np.random.randn(n_residues, 6).astype(np.float32) * 0.1
    transforms[:, 3:] *= 3  # Larger translations

    # Frame columns (pretend all residues have same layout)
    frame_cols = np.array([[0, 1, 2]] * n_residues, dtype=np.int32)
    next_frame_cols = np.array([[3, 4, 5]] * n_residues, dtype=np.int32)

    # Canonical origins (for optimized version)
    canonical_origins = np.zeros((n_residues, 3), dtype=np.float32)

    return residue_coords, transforms, frame_cols, next_frame_cols, canonical_origins


def run_benchmarks():
    """Run all benchmarks."""
    print("=" * 70)
    print("CHAIN ASSEMBLY BENCHMARK")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    chain_lengths = [10, 50, 100, 200, 500, 1000]
    n_atoms = 22  # Typical nucleotide

    print(f"\nAtoms per residue: {n_atoms}")
    print("-" * 80)
    print(f"{'Residues':>10} | {'Current (ms)':>12} | {'Numpy Opt':>12} | {'Speedup':>8}")
    print("-" * 80)

    for n_res in chain_lengths:
        residue_coords, transforms, frame_cols, next_frame_cols, origins = generate_test_data(n_res, n_atoms)

        # Current implementation
        def run_current():
            return current_assemble_chain_numpy(residue_coords, transforms, frame_cols, next_frame_cols)

        # Optimized numpy
        def run_optimized():
            return optimized_assemble_chain_numpy(residue_coords, transforms, origins)

        try:
            current_mean, current_std = timeit(run_current, n_warmup=2, n_runs=5)
        except Exception as e:
            current_mean, current_std = float('nan'), float('nan')

        try:
            opt_mean, opt_std = timeit(run_optimized, n_warmup=2, n_runs=5)
        except Exception as e:
            opt_mean, opt_std = float('nan'), float('nan')

        speedup = current_mean / opt_mean if opt_mean > 0 and not np.isnan(opt_mean) else float('nan')

        print(f"{n_res:>10} | {current_mean:>8.2f}±{current_std:>3.1f} | {opt_mean:>8.2f}±{opt_std:>3.1f} | {speedup:>8.2f}x")

    # GPU benchmark if available
    if device == "cuda":
        print("\n" + "=" * 70)
        print("CUMULATIVE MATMUL STRATEGIES (GPU)")
        print("=" * 70)
        print(f"{'n':>6} | {'Sequential':>12} | {'Chunked-32':>12} | {'Chunked-64':>12} | {'Best':>8}")
        print("-" * 70)

        for n_res in chain_lengths:
            residue_coords, transforms, _, _, _ = generate_test_data(n_res, n_atoms)
            transforms_gpu = torch.from_numpy(transforms).to('cuda')

            # Build SE3 matrices on GPU
            axis_angles = transforms_gpu[:, :3]
            translations = transforms_gpu[:, 3:]

            angles = torch.norm(axis_angles, dim=1, keepdim=True)
            safe_angles = torch.where(angles < 1e-8, torch.ones_like(angles), angles)
            axes = axis_angles / safe_angles

            K = torch.zeros(n_res, 3, 3, device='cuda')
            K[:, 0, 1] = -axes[:, 2]
            K[:, 0, 2] = axes[:, 1]
            K[:, 1, 0] = axes[:, 2]
            K[:, 1, 2] = -axes[:, 0]
            K[:, 2, 0] = -axes[:, 1]
            K[:, 2, 1] = axes[:, 0]

            eye = torch.eye(3, device='cuda').unsqueeze(0).expand(n_res, -1, -1)
            sin_a = torch.sin(angles).unsqueeze(-1)
            cos_a = torch.cos(angles).unsqueeze(-1)
            Rs = eye + sin_a * K + (1 - cos_a) * (K @ K)

            T = torch.zeros(n_res, 4, 4, device='cuda')
            T[:, :3, :3] = Rs
            T[:, :3, 3] = translations
            T[:, 3, 3] = 1.0

            def run_seq():
                return cumulative_matmul_torch(T)

            def run_chunked_32():
                return cumulative_matmul_chunked(T, chunk_size=32)

            def run_chunked_64():
                return cumulative_matmul_chunked(T, chunk_size=64)

            seq_mean, seq_std = timeit(run_seq, n_warmup=3, n_runs=10)
            c32_mean, c32_std = timeit(run_chunked_32, n_warmup=3, n_runs=10)
            c64_mean, c64_std = timeit(run_chunked_64, n_warmup=3, n_runs=10)

            best = min(seq_mean, c32_mean, c64_mean)
            best_name = "seq" if best == seq_mean else ("c32" if best == c32_mean else "c64")

            print(f"{n_res:>6} | {seq_mean:>8.3f}±{seq_std:>2.1f} | {c32_mean:>8.3f}±{c32_std:>2.1f} | {c64_mean:>8.3f}±{c64_std:>2.1f} | {best_name:>8}")

        # Full assembly comparison
        print("\n" + "=" * 70)
        print("FULL ASSEMBLY: CPU vs GPU")
        print("=" * 70)
        print(f"{'Residues':>10} | {'CPU (ms)':>12} | {'GPU (ms)':>12} | {'Speedup':>10}")
        print("-" * 70)

        for n_res in chain_lengths:
            residue_coords, transforms, _, _, _ = generate_test_data(n_res, n_atoms)

            # Stack into tensors
            coords_np = np.stack(residue_coords)  # (n_res, n_atoms, 3)
            coords_cpu = torch.from_numpy(coords_np)
            coords_gpu = coords_cpu.to('cuda')

            transforms_cpu = torch.from_numpy(transforms)
            transforms_gpu = transforms_cpu.to('cuda')

            mask = torch.ones(n_res, n_atoms, dtype=torch.bool)
            mask_gpu = mask.to('cuda')

            def run_cpu():
                return optimized_assemble_chain_torch(coords_cpu, transforms_cpu, mask)

            def run_gpu():
                return optimized_assemble_chain_torch(coords_gpu, transforms_gpu, mask_gpu)

            cpu_mean, cpu_std = timeit(run_cpu, n_warmup=3, n_runs=10)
            gpu_mean, gpu_std = timeit(run_gpu, n_warmup=3, n_runs=10)

            speedup = cpu_mean / gpu_mean if gpu_mean > 0 else 0

            print(f"{n_res:>10} | {cpu_mean:>8.2f}±{cpu_std:>3.1f} | {gpu_mean:>8.2f}±{gpu_std:>3.1f} | {speedup:>10.2f}x")

    # Batch size scaling (many chains at once)
    print("\n" + "=" * 70)
    print("BATCH SCALING (Multiple chains simultaneously)")
    print("=" * 70)

    n_res = 100  # Fixed chain length
    batch_sizes = [1, 10, 50, 100, 200]

    print(f"Chain length: {n_res} residues, {n_atoms} atoms each")
    print("-" * 70)
    print(f"{'Batch Size':>10} | {'CPU (ms)':>15} | {'GPU (ms)':>15} | {'GPU Speedup':>12}")
    print("-" * 70)

    for batch_size in batch_sizes:
        residue_coords, transforms, _, _, _ = generate_test_data(n_res, n_atoms)
        coords_np = np.stack(residue_coords)

        # Batch = repeat the same chain
        coords_batch = np.tile(coords_np, (batch_size, 1, 1, 1))  # (batch, n_res, n_atoms, 3)
        transforms_batch = np.tile(transforms, (batch_size, 1, 1))  # (batch, n_res, 6)

        coords_cpu = torch.from_numpy(coords_batch)
        transforms_cpu = torch.from_numpy(transforms_batch)

        def run_batch_cpu():
            results = []
            for i in range(batch_size):
                mask = torch.ones(n_res, n_atoms, dtype=torch.bool)
                r = optimized_assemble_chain_torch(coords_cpu[i], transforms_cpu[i], mask)
                results.append(r)
            return torch.stack(results)

        cpu_mean, cpu_std = timeit(run_batch_cpu, n_warmup=2, n_runs=5)

        if device == "cuda":
            coords_gpu = coords_cpu.to('cuda')
            transforms_gpu = transforms_cpu.to('cuda')

            def run_batch_gpu():
                results = []
                for i in range(batch_size):
                    mask = torch.ones(n_res, n_atoms, dtype=torch.bool, device='cuda')
                    r = optimized_assemble_chain_torch(coords_gpu[i], transforms_gpu[i], mask)
                    results.append(r)
                return torch.stack(results)

            gpu_mean, gpu_std = timeit(run_batch_gpu, n_warmup=2, n_runs=5)
            speedup = cpu_mean / gpu_mean
            print(f"{batch_size:>10} | {cpu_mean:>10.2f} ± {cpu_std:>3.1f} | {gpu_mean:>10.2f} ± {gpu_std:>3.1f} | {speedup:>10.2f}x")
        else:
            print(f"{batch_size:>10} | {cpu_mean:>10.2f} ± {cpu_std:>3.1f} | {'N/A':>15} | {'N/A':>12}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
Key findings:
1. Current implementation: Python loop + per-residue matrix ops
2. Optimized: Cumulative transforms (still sequential for matmul)
3. GPU benefit: Matrix operations faster, but cumulative loop limits speedup

Bottleneck: Cumulative matrix multiplication is inherently sequential.
Possible solutions:
- Parallel scan algorithm (log(n) depth)
- Chunked computation with inter-chunk parallelism
- Approximate methods for very long chains
""")


if __name__ == "__main__":
    run_benchmarks()
