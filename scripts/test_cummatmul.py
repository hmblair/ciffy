"""Test cumulative matmul implementations."""
import torch
import numpy as np
import time

# Check if JAX is available
try:
    import jax
    import jax.numpy as jnp
    from jax.lax import associative_scan
    HAS_JAX = True
    print(f"JAX available: {jax.__version__}")
    print(f"JAX devices: {jax.devices()}")
except ImportError:
    HAS_JAX = False
    print("JAX not available")

print()

# Test cumulative matmul implementations
n = 500
matrices = torch.randn(n, 4, 4, device='cuda')

def sequential_cummatmul(M):
    result = torch.zeros_like(M)
    result[0] = M[0]
    for i in range(1, len(M)):
        result[i] = result[i-1] @ M[i]
    return result

# Benchmark sequential
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(10):
    result_seq = sequential_cummatmul(matrices)
    torch.cuda.synchronize()
seq_time = (time.perf_counter() - start) / 10 * 1000
print(f"Sequential cummatmul ({n} matrices): {seq_time:.2f} ms")

# Try torch.compile
try:
    compiled_cummatmul = torch.compile(sequential_cummatmul, mode="reduce-overhead")
    # Warmup
    for _ in range(3):
        _ = compiled_cummatmul(matrices)
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(10):
        result_compiled = compiled_cummatmul(matrices)
        torch.cuda.synchronize()
    compiled_time = (time.perf_counter() - start) / 10 * 1000
    print(f"torch.compile cummatmul: {compiled_time:.2f} ms")
    print(f"Speedup: {seq_time/compiled_time:.2f}x")

    # Verify correctness
    diff = (result_seq - result_compiled).abs().max().item()
    print(f"Max diff vs sequential: {diff:.2e}")
except Exception as e:
    print(f"torch.compile failed: {e}")

print()

# JAX associative_scan if available
if HAS_JAX:
    print("Testing JAX associative_scan...")
    matrices_jax = jnp.array(matrices.cpu().numpy())

    @jax.jit
    def jax_cummatmul(M):
        return associative_scan(jnp.matmul, M)

    # Warmup
    _ = jax_cummatmul(matrices_jax).block_until_ready()

    start = time.perf_counter()
    for _ in range(10):
        result_jax = jax_cummatmul(matrices_jax).block_until_ready()
    jax_time = (time.perf_counter() - start) / 10 * 1000
    print(f"JAX associative_scan: {jax_time:.2f} ms")
    print(f"Speedup vs sequential: {seq_time/jax_time:.2f}x")

    # Verify correctness
    result_jax_torch = torch.from_numpy(np.array(result_jax)).to('cuda')
    diff = (result_seq - result_jax_torch).abs().max().item()
    print(f"Max diff vs sequential: {diff:.2e}")

print()
print("=" * 60)
print("CONCLUSION")
print("=" * 60)
print("""
Matrix cumulative product options:
1. torch.cumprod - element-wise only, NOT what we need
2. Sequential loop - O(n) depth, GPU overhead
3. torch.compile - may fuse kernels, reduce overhead
4. JAX associative_scan - O(log n) depth, true parallelism

For chain assembly with 100-1000 residues, the key question is
whether the parallel scan overhead is worth it.
""")
