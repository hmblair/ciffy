"""Diagnose GPU overhead in chain assembly."""
import torch
import time

torch.set_float32_matmul_precision('high')

print("=" * 70)
print("GPU OVERHEAD DIAGNOSIS")
print("=" * 70)
print(f"GPU: {torch.cuda.get_device_name()}")
print()

n = 100  # 100 matrices

# Create data on GPU (no transfer)
matrices_gpu = torch.randn(n, 4, 4, device='cuda')
matrices_cpu = matrices_gpu.cpu()

# ============================================================
# Test 1: Measure kernel launch overhead
# ============================================================
print("TEST 1: Kernel launch overhead")
print("-" * 50)

# Single matmul
a = torch.randn(4, 4, device='cuda')
b = torch.randn(4, 4, device='cuda')

torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(1000):
    c = a @ b
torch.cuda.synchronize()
time_nosync = (time.perf_counter() - start) / 1000 * 1000  # ms

torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(1000):
    c = a @ b
    torch.cuda.synchronize()  # Force sync each iteration
time_sync = (time.perf_counter() - start) / 1000 * 1000  # ms

print(f"Single 4x4 matmul (no sync): {time_nosync*1000:.3f} μs")
print(f"Single 4x4 matmul (with sync): {time_sync*1000:.3f} μs")
print(f"Sync overhead per op: {(time_sync - time_nosync)*1000:.3f} μs")
print()

# ============================================================
# Test 2: Sequential loop overhead
# ============================================================
print("TEST 2: Sequential cumulative matmul")
print("-" * 50)

def cumul_loop(M):
    n = len(M)
    result = torch.zeros_like(M)
    result[0] = M[0]
    for i in range(1, n):
        result[i] = result[i-1] @ M[i]
    return result

# GPU
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(10):
    result = cumul_loop(matrices_gpu)
torch.cuda.synchronize()
gpu_time = (time.perf_counter() - start) / 10 * 1000

# CPU
start = time.perf_counter()
for _ in range(10):
    result = cumul_loop(matrices_cpu)
cpu_time = (time.perf_counter() - start) / 10 * 1000

print(f"GPU eager:  {gpu_time:.3f} ms ({n} iterations)")
print(f"CPU:        {cpu_time:.3f} ms")
print(f"Per-iteration overhead: {(gpu_time - cpu_time) / n * 1000:.1f} μs")
print()

# ============================================================
# Test 3: Is it memory access or compute?
# ============================================================
print("TEST 3: Isolate memory vs compute")
print("-" * 50)

# Just indexing (memory access)
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(10):
    result = torch.zeros_like(matrices_gpu)
    result[0] = matrices_gpu[0]
    for i in range(1, n):
        result[i] = matrices_gpu[i]  # Just copy, no matmul
torch.cuda.synchronize()
copy_time = (time.perf_counter() - start) / 10 * 1000

# Matmul without dependency
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(10):
    result = torch.zeros_like(matrices_gpu)
    for i in range(n):
        result[i] = matrices_gpu[i] @ matrices_gpu[i]  # No dependency
torch.cuda.synchronize()
indep_time = (time.perf_counter() - start) / 10 * 1000

print(f"Just copying (no compute): {copy_time:.3f} ms")
print(f"Independent matmuls:       {indep_time:.3f} ms")
print(f"Dependent cumulative:      {gpu_time:.3f} ms")
print()

# ============================================================
# Test 4: Batch matmul (no loop)
# ============================================================
print("TEST 4: Batched operations (no Python loop)")
print("-" * 50)

# bmm - all at once
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(100):
    # This doesn't compute cumulative, but shows batched perf
    result = torch.bmm(matrices_gpu, matrices_gpu)
torch.cuda.synchronize()
bmm_time = (time.perf_counter() - start) / 100 * 1000

print(f"torch.bmm ({n} matrices): {bmm_time:.3f} ms")
print(f"Per-matrix: {bmm_time/n*1000:.3f} μs")
print()

# ============================================================
# Test 5: Python loop overhead itself
# ============================================================
print("TEST 5: Python loop overhead")
print("-" * 50)

# Empty loop
start = time.perf_counter()
for _ in range(10):
    for i in range(n):
        pass
empty_loop = (time.perf_counter() - start) / 10 * 1000

# Loop with tensor indexing
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(10):
    for i in range(n):
        x = matrices_gpu[i]
torch.cuda.synchronize()
index_loop = (time.perf_counter() - start) / 10 * 1000

print(f"Empty Python loop ({n} iters): {empty_loop*1000:.1f} μs")
print(f"Loop with GPU indexing:        {index_loop:.3f} ms")
print(f"Per-index overhead:            {(index_loop - empty_loop)/n*1000:.1f} μs")
print()

# ============================================================
# Test 6: torch.compile effect
# ============================================================
print("TEST 6: torch.compile effect")
print("-" * 50)

cumul_compiled = torch.compile(cumul_loop, mode="reduce-overhead")

# Warmup
for _ in range(3):
    _ = cumul_compiled(matrices_gpu)
torch.cuda.synchronize()

torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(10):
    result = cumul_compiled(matrices_gpu)
torch.cuda.synchronize()
compiled_time = (time.perf_counter() - start) / 10 * 1000

print(f"GPU eager:    {gpu_time:.3f} ms")
print(f"GPU compiled: {compiled_time:.3f} ms")
print(f"Speedup:      {gpu_time/compiled_time:.1f}x")
print()

# ============================================================
print("=" * 70)
print("DIAGNOSIS SUMMARY")
print("=" * 70)
print(f"""
The GPU is slower in eager mode because:

1. Each Python loop iteration triggers:
   - Tensor indexing (result[i-1]) → CUDA kernel
   - Matrix multiply (@) → CUDA kernel
   - Tensor assignment (result[i] = ...) → CUDA kernel

2. Each kernel has ~{(gpu_time - cpu_time) / n * 1000:.0f}μs overhead from:
   - Python → CUDA dispatch
   - Kernel launch latency
   - Implicit synchronization (data dependency)

3. For {n} iterations: {n} × {(gpu_time - cpu_time) / n * 1000:.0f}μs = {gpu_time - cpu_time:.1f}ms overhead

4. The actual 4×4 matmul computation is trivial (~1μs)
   but we're paying ~{(gpu_time - cpu_time) / n * 1000:.0f}μs overhead per iteration

torch.compile fixes this by:
- Tracing the loop and fusing operations
- Generating a single optimized CUDA kernel
- Eliminating per-iteration Python overhead
""")
