#!/usr/bin/env python
"""Benchmark torch.compile with fixed input size (no recompilation).

This avoids the recompilation overhead that skewed previous results.
"""

import torch
import time
import argparse
from pathlib import Path
import gc

Path("outputs").mkdir(exist_ok=True)


def create_model_and_inputs(
    n_points: int,
    hidden_mult: int,
    num_layers: int,
    k_neighbors: int,
    device: str,
):
    """Create model and synthetic inputs."""
    from ciffy.nn.geometric import EquivariantTransformer
    from ciffy.nn.geometric.representations import Repr

    in_repr = Repr(lvals=[0], mult=hidden_mult)
    out_repr = Repr(lvals=[0], mult=2)
    hidden_repr = Repr(lvals=[0, 1], mult=hidden_mult)

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


def benchmark(model, coords, features, seq_pos, n_warmup=10, n_iters=100, backward=False):
    """Run benchmark with proper warmup."""
    if backward:
        features = features.clone().requires_grad_(True)

    # Warmup
    for _ in range(n_warmup):
        if backward:
            output = model(coords, features, seq_pos=seq_pos)
            output.sum().backward()
            model.zero_grad()
            features.grad = None
        else:
            with torch.no_grad():
                _ = model(coords, features, seq_pos=seq_pos)
    torch.cuda.synchronize()

    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iters):
        if backward:
            output = model(coords, features, seq_pos=seq_pos)
            output.sum().backward()
            model.zero_grad()
            features.grad = None
        else:
            with torch.no_grad():
                _ = model(coords, features, seq_pos=seq_pos)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / n_iters * 1000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-points", type=int, default=1000)
    parser.add_argument("--hidden-mult", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--k-neighbors", type=int, default=16)
    args = parser.parse_args()

    device = "cuda"
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"\nConfig: N={args.n_points}, mult={args.hidden_mult}, layers={args.num_layers}, k={args.k_neighbors}")

    results = {}

    # ============================================
    # 1. Baseline
    # ============================================
    print("\n" + "="*60)
    print("1. BASELINE (no compile)")
    print("="*60)

    model, coords, features, seq_pos = create_model_and_inputs(
        args.n_points, args.hidden_mult, args.num_layers, args.k_neighbors, device
    )

    fwd = benchmark(model, coords, features, seq_pos, backward=False)
    fwd_bwd = benchmark(model, coords, features, seq_pos, backward=True, n_iters=50)
    results["baseline"] = {"forward": fwd, "forward_backward": fwd_bwd}
    print(f"Forward:     {fwd:.2f} ms")
    print(f"Fwd+Bwd:     {fwd_bwd:.2f} ms")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ============================================
    # 2. torch.compile default
    # ============================================
    print("\n" + "="*60)
    print("2. torch.compile (default)")
    print("="*60)

    model, coords, features, seq_pos = create_model_and_inputs(
        args.n_points, args.hidden_mult, args.num_layers, args.k_neighbors, device
    )
    model = torch.compile(model, mode="default")

    print("Compiling...")
    fwd = benchmark(model, coords, features, seq_pos, backward=False, n_warmup=15)
    fwd_bwd = benchmark(model, coords, features, seq_pos, backward=True, n_warmup=15, n_iters=50)
    results["compile_default"] = {"forward": fwd, "forward_backward": fwd_bwd}
    print(f"Forward:     {fwd:.2f} ms  ({results['baseline']['forward']/fwd:.2f}x)")
    print(f"Fwd+Bwd:     {fwd_bwd:.2f} ms  ({results['baseline']['forward_backward']/fwd_bwd:.2f}x)")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ============================================
    # 3. torch.compile reduce-overhead
    # ============================================
    print("\n" + "="*60)
    print("3. torch.compile (reduce-overhead / CUDA graphs)")
    print("="*60)

    model, coords, features, seq_pos = create_model_and_inputs(
        args.n_points, args.hidden_mult, args.num_layers, args.k_neighbors, device
    )
    model = torch.compile(model, mode="reduce-overhead")

    print("Compiling...")
    fwd = benchmark(model, coords, features, seq_pos, backward=False, n_warmup=15)
    fwd_bwd = benchmark(model, coords, features, seq_pos, backward=True, n_warmup=15, n_iters=50)
    results["compile_reduce_overhead"] = {"forward": fwd, "forward_backward": fwd_bwd}
    print(f"Forward:     {fwd:.2f} ms  ({results['baseline']['forward']/fwd:.2f}x)")
    print(f"Fwd+Bwd:     {fwd_bwd:.2f} ms  ({results['baseline']['forward_backward']/fwd_bwd:.2f}x)")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ============================================
    # 4. torch.compile max-autotune
    # ============================================
    print("\n" + "="*60)
    print("4. torch.compile (max-autotune)")
    print("="*60)

    model, coords, features, seq_pos = create_model_and_inputs(
        args.n_points, args.hidden_mult, args.num_layers, args.k_neighbors, device
    )
    model = torch.compile(model, mode="max-autotune")

    print("Compiling and autotuning (takes a while)...")
    fwd = benchmark(model, coords, features, seq_pos, backward=False, n_warmup=15)
    fwd_bwd = benchmark(model, coords, features, seq_pos, backward=True, n_warmup=15, n_iters=50)
    results["compile_max_autotune"] = {"forward": fwd, "forward_backward": fwd_bwd}
    print(f"Forward:     {fwd:.2f} ms  ({results['baseline']['forward']/fwd:.2f}x)")
    print(f"Fwd+Bwd:     {fwd_bwd:.2f} ms  ({results['baseline']['forward_backward']/fwd_bwd:.2f}x)")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ============================================
    # 5. BF16 + compile
    # ============================================
    print("\n" + "="*60)
    print("5. BF16 + torch.compile (default)")
    print("="*60)

    model, coords, features, seq_pos = create_model_and_inputs(
        args.n_points, args.hidden_mult, args.num_layers, args.k_neighbors, device
    )
    model = model.to(torch.bfloat16)
    coords = coords.to(torch.bfloat16)
    features = features.to(torch.bfloat16)
    model = torch.compile(model, mode="default")

    print("Compiling...")
    fwd = benchmark(model, coords, features, seq_pos, backward=False, n_warmup=15)
    fwd_bwd = benchmark(model, coords, features, seq_pos, backward=True, n_warmup=15, n_iters=50)
    results["bf16_compile"] = {"forward": fwd, "forward_backward": fwd_bwd}
    print(f"Forward:     {fwd:.2f} ms  ({results['baseline']['forward']/fwd:.2f}x)")
    print(f"Fwd+Bwd:     {fwd_bwd:.2f} ms  ({results['baseline']['forward_backward']/fwd_bwd:.2f}x)")

    # ============================================
    # Summary
    # ============================================
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\n{'Mode':<30} {'Forward':>12} {'Fwd+Bwd':>12}")
    print("-"*60)
    for mode, data in results.items():
        fwd_speedup = results['baseline']['forward'] / data['forward']
        bwd_speedup = results['baseline']['forward_backward'] / data['forward_backward']
        print(f"{mode:<30} {data['forward']:>8.2f} ms ({fwd_speedup:.2f}x) {data['forward_backward']:>8.2f} ms ({bwd_speedup:.2f}x)")


if __name__ == "__main__":
    main()
