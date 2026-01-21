#!/usr/bin/env python
"""Benchmark torch.compile improvements for EquivariantTransformer.

Compares:
- No compilation (baseline)
- torch.compile with default mode
- torch.compile with reduce-overhead (CUDA graphs)
- torch.compile with max-autotune

Usage:
    python scripts/benchmark_compile.py
"""

import torch
import time
import argparse
from pathlib import Path
import gc

Path("outputs").mkdir(exist_ok=True)


def create_model_and_inputs(
    n_points: int = 500,
    hidden_mult: int = 32,
    hidden_lvals: list = [0, 1],
    num_layers: int = 4,
    k_neighbors: int = 16,
    device: str = "cuda",
):
    """Create model and synthetic inputs."""
    from ciffy.nn.geometric import EquivariantTransformer
    from ciffy.nn.geometric.representations import Repr

    in_repr = Repr(lvals=[0], mult=hidden_mult)
    out_repr = Repr(lvals=[0], mult=2)
    hidden_repr = Repr(lvals=hidden_lvals, mult=hidden_mult)

    model = EquivariantTransformer(
        in_repr=in_repr,
        out_repr=out_repr,
        hidden_repr=hidden_repr,
        hidden_layers=num_layers,
        edge_dim=16,
        edge_hidden_dim=64,
        k_neighbors=k_neighbors,
        nheads=4,
        dropout=0.0,
        transition=True,
        seq_pos_dim=8,
    ).to(device)

    coordinates = torch.randn(n_points, 3, device=device)
    node_features = torch.randn(n_points, hidden_mult, in_repr.dim(), device=device)
    seq_pos = torch.arange(n_points, device=device)

    return model, coordinates, node_features, seq_pos


def warmup(model, coords, features, seq_pos, n_warmup: int = 10):
    """Warmup to compile kernels."""
    for _ in range(n_warmup):
        with torch.no_grad():
            _ = model(coords, features, seq_pos=seq_pos)
    torch.cuda.synchronize()


def benchmark_forward(model, coords, features, seq_pos, n_iters: int = 100):
    """Benchmark forward pass, return average time in ms."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iters):
        with torch.no_grad():
            _ = model(coords, features, seq_pos=seq_pos)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / n_iters * 1000
    return elapsed


def benchmark_forward_backward(model, coords, features, seq_pos, n_iters: int = 50):
    """Benchmark forward + backward pass, return average time in ms."""
    features = features.clone().requires_grad_(True)

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iters):
        output = model(coords, features, seq_pos=seq_pos)
        loss = output.sum()
        loss.backward()
        model.zero_grad()
        features.grad = None
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / n_iters * 1000
    return elapsed


def run_benchmark(n_points: int, hidden_mult: int, num_layers: int, k_neighbors: int):
    """Run full benchmark suite for given configuration."""
    device = "cuda"
    results = {}

    print(f"\n{'='*60}")
    print(f"Configuration: N={n_points}, mult={hidden_mult}, layers={num_layers}, k={k_neighbors}")
    print(f"{'='*60}")

    # 1. Baseline (no compile)
    print("\n[1/4] Baseline (no compile)...")
    model, coords, features, seq_pos = create_model_and_inputs(
        n_points, hidden_mult, [0, 1], num_layers, k_neighbors, device
    )
    warmup(model, coords, features, seq_pos, n_warmup=5)

    fwd_time = benchmark_forward(model, coords, features, seq_pos)
    fwd_bwd_time = benchmark_forward_backward(model, coords, features, seq_pos)
    results["baseline"] = {"forward": fwd_time, "forward_backward": fwd_bwd_time}
    print(f"  Forward: {fwd_time:.2f} ms")
    print(f"  Forward+Backward: {fwd_bwd_time:.2f} ms")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # 2. torch.compile default mode
    print("\n[2/4] torch.compile (default mode)...")
    model, coords, features, seq_pos = create_model_and_inputs(
        n_points, hidden_mult, [0, 1], num_layers, k_neighbors, device
    )
    model = torch.compile(model, mode="default")

    print("  Compiling (first few runs)...")
    warmup(model, coords, features, seq_pos, n_warmup=10)

    fwd_time = benchmark_forward(model, coords, features, seq_pos)
    fwd_bwd_time = benchmark_forward_backward(model, coords, features, seq_pos)
    results["compile_default"] = {"forward": fwd_time, "forward_backward": fwd_bwd_time}
    print(f"  Forward: {fwd_time:.2f} ms")
    print(f"  Forward+Backward: {fwd_bwd_time:.2f} ms")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # 3. torch.compile reduce-overhead (CUDA graphs)
    print("\n[3/4] torch.compile (reduce-overhead / CUDA graphs)...")
    model, coords, features, seq_pos = create_model_and_inputs(
        n_points, hidden_mult, [0, 1], num_layers, k_neighbors, device
    )
    model = torch.compile(model, mode="reduce-overhead")

    print("  Compiling (first few runs)...")
    warmup(model, coords, features, seq_pos, n_warmup=10)

    fwd_time = benchmark_forward(model, coords, features, seq_pos)
    fwd_bwd_time = benchmark_forward_backward(model, coords, features, seq_pos)
    results["compile_reduce_overhead"] = {"forward": fwd_time, "forward_backward": fwd_bwd_time}
    print(f"  Forward: {fwd_time:.2f} ms")
    print(f"  Forward+Backward: {fwd_bwd_time:.2f} ms")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # 4. torch.compile max-autotune
    print("\n[4/4] torch.compile (max-autotune)...")
    model, coords, features, seq_pos = create_model_and_inputs(
        n_points, hidden_mult, [0, 1], num_layers, k_neighbors, device
    )
    model = torch.compile(model, mode="max-autotune")

    print("  Compiling and autotuning (this takes longer)...")
    warmup(model, coords, features, seq_pos, n_warmup=10)

    fwd_time = benchmark_forward(model, coords, features, seq_pos)
    fwd_bwd_time = benchmark_forward_backward(model, coords, features, seq_pos)
    results["compile_max_autotune"] = {"forward": fwd_time, "forward_backward": fwd_bwd_time}
    print(f"  Forward: {fwd_time:.2f} ms")
    print(f"  Forward+Backward: {fwd_bwd_time:.2f} ms")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return results


def print_summary(all_results: dict):
    """Print summary table."""
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    # Header
    print(f"\n{'Configuration':<30} {'Baseline':>10} {'Default':>10} {'Reduce-OH':>10} {'Max-AT':>10}")
    print("-"*80)

    for config, results in all_results.items():
        baseline_fwd = results["baseline"]["forward"]

        def speedup(mode):
            if mode in results:
                return baseline_fwd / results[mode]["forward"]
            return 0

        print(f"\n{config}")
        print(f"  {'Forward (ms)':<28} {baseline_fwd:>10.2f} {results['compile_default']['forward']:>10.2f} {results['compile_reduce_overhead']['forward']:>10.2f} {results['compile_max_autotune']['forward']:>10.2f}")
        print(f"  {'Speedup':<28} {'1.00x':>10} {speedup('compile_default'):>9.2f}x {speedup('compile_reduce_overhead'):>9.2f}x {speedup('compile_max_autotune'):>9.2f}x")

        baseline_bwd = results["baseline"]["forward_backward"]
        def speedup_bwd(mode):
            if mode in results:
                return baseline_bwd / results[mode]["forward_backward"]
            return 0

        print(f"  {'Fwd+Bwd (ms)':<28} {baseline_bwd:>10.2f} {results['compile_default']['forward_backward']:>10.2f} {results['compile_reduce_overhead']['forward_backward']:>10.2f} {results['compile_max_autotune']['forward_backward']:>10.2f}")
        print(f"  {'Speedup':<28} {'1.00x':>10} {speedup_bwd('compile_default'):>9.2f}x {speedup_bwd('compile_reduce_overhead'):>9.2f}x {speedup_bwd('compile_max_autotune'):>9.2f}x")


def main():
    parser = argparse.ArgumentParser(description="Benchmark torch.compile")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark (fewer configs)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA version: {torch.version.cuda}")

    all_results = {}

    if args.quick:
        # Quick benchmark - single configuration
        configs = [
            (500, 32, 4, 16),
        ]
    else:
        # Full benchmark - multiple configurations
        configs = [
            (500, 32, 4, 16),    # Small
            (1000, 32, 4, 16),   # Medium
            (2000, 32, 4, 16),   # Large
        ]

    for n_points, hidden_mult, num_layers, k_neighbors in configs:
        config_name = f"N={n_points}, mult={hidden_mult}, L={num_layers}"
        results = run_benchmark(n_points, hidden_mult, num_layers, k_neighbors)
        all_results[config_name] = results

    print_summary(all_results)

    # Save results
    import json
    with open("outputs/benchmark_compile_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nResults saved to outputs/benchmark_compile_results.json")


if __name__ == "__main__":
    main()
