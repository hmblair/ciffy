#!/usr/bin/env python
"""Profile EquivariantTransformer to identify bottlenecks.

Usage:
    python scripts/profile_transformer.py

Outputs:
    - outputs/profile_transformer_trace.json (Chrome trace)
    - outputs/profile_transformer_summary.txt (text summary)
"""

import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity
import time
import argparse
from pathlib import Path

# Ensure outputs directory exists
Path("outputs").mkdir(exist_ok=True)


def create_model_and_inputs(
    n_points: int = 500,
    hidden_mult: int = 32,
    hidden_lvals: list = [0, 1],
    num_layers: int = 4,
    k_neighbors: int = 16,
    device: str = "cuda",
):
    """Create model and synthetic inputs for profiling."""
    from ciffy.nn.geometric import EquivariantTransformer
    from ciffy.nn.geometric.representations import Repr

    # Define representations
    in_repr = Repr(lvals=[0], mult=hidden_mult)  # Scalar input
    out_repr = Repr(lvals=[0], mult=2)  # Scalar output
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

    # Create synthetic inputs
    coordinates = torch.randn(n_points, 3, device=device)
    node_features = torch.randn(n_points, hidden_mult, in_repr.dim(), device=device)
    seq_pos = torch.arange(n_points, device=device)

    return model, coordinates, node_features, seq_pos


def warmup(model, coordinates, node_features, seq_pos, n_warmup: int = 5):
    """Warmup the model to ensure CUDA kernels are compiled."""
    print(f"Warming up with {n_warmup} iterations...")
    for _ in range(n_warmup):
        with torch.no_grad():
            _ = model(coordinates, node_features, seq_pos=seq_pos)
    torch.cuda.synchronize()
    print("Warmup complete.")


def profile_forward(model, coordinates, node_features, seq_pos, n_iters: int = 10):
    """Profile forward pass."""
    print(f"\nProfiling forward pass ({n_iters} iterations)...")

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    ) as prof:
        for _ in range(n_iters):
            with record_function("forward"):
                with torch.no_grad():
                    output = model(coordinates, node_features, seq_pos=seq_pos)
            torch.cuda.synchronize()

    return prof


def profile_backward(model, coordinates, node_features, seq_pos, n_iters: int = 10):
    """Profile forward + backward pass."""
    print(f"\nProfiling forward+backward pass ({n_iters} iterations)...")

    # Need gradients for backward
    coordinates = coordinates.clone().requires_grad_(False)
    node_features = node_features.clone().requires_grad_(True)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    ) as prof:
        for _ in range(n_iters):
            with record_function("forward"):
                output = model(coordinates, node_features, seq_pos=seq_pos)

            with record_function("backward"):
                loss = output.sum()
                loss.backward()

            # Clear gradients for next iteration
            model.zero_grad()
            node_features.grad = None
            torch.cuda.synchronize()

    return prof


def profile_components(model, coordinates, node_features, seq_pos):
    """Profile individual components with manual timing."""
    from ciffy.nn.geometric.layers import build_knn_graph

    print("\n" + "="*60)
    print("Component-level profiling")
    print("="*60)

    N = coordinates.size(0)
    k = model.k_neighbors
    device = coordinates.device

    results = {}
    n_runs = 20

    # 1. k-NN graph construction
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_runs):
        neighbor_idx = build_knn_graph(coordinates, k)
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / n_runs * 1000
    results["1. k-NN graph"] = elapsed
    print(f"k-NN graph construction: {elapsed:.3f} ms")

    # 2. Displacement computation
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_runs):
        neighbor_coords = coordinates[neighbor_idx]
        displacements = coordinates.unsqueeze(1) - neighbor_coords
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / n_runs * 1000
    results["2. Displacements"] = elapsed
    print(f"Displacement computation: {elapsed:.3f} ms")

    # 3. RBF computation
    distances = displacements.norm(dim=-1)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_runs):
        edge_features = model.rbf(distances)
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / n_runs * 1000
    results["3. RBF"] = elapsed
    print(f"RBF computation: {elapsed:.3f} ms")

    # 4. Sequence position encoding
    if model.seq_pos_enc is not None:
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(n_runs):
            seq_features = model.seq_pos_enc(seq_pos, neighbor_idx)
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / n_runs * 1000
        results["4. Seq pos encoding"] = elapsed
        print(f"Sequence position encoding: {elapsed:.3f} ms")
        edge_features = torch.cat([edge_features, seq_features], dim=-1)

    # 5. Equivariant basis computation
    displacements_flat = displacements.view(N * k, 3)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_runs):
        all_bases = model.bases(displacements_flat)
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / n_runs * 1000
    results["5. Equivariant basis"] = elapsed
    print(f"Equivariant basis computation: {elapsed:.3f} ms")

    # 6. Single transformer layer
    layer = model.layers[0]
    basis_flat = all_bases[0]
    basis = basis_flat.view(N, k, *basis_flat.shape[1:])

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_runs):
        with torch.no_grad():
            out = layer(basis, node_features, edge_features, neighbor_idx, seq_pos=seq_pos)
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / n_runs * 1000
    results["6. Single transformer layer"] = elapsed
    print(f"Single transformer layer: {elapsed:.3f} ms")

    # 7. Breakdown of transformer layer components
    print("\n  Transformer layer breakdown:")

    # 7a. Layer norm
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_runs):
        normed = layer.ln1(node_features)
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / n_runs * 1000
    results["  7a. LayerNorm"] = elapsed
    print(f"    LayerNorm: {elapsed:.3f} ms")

    # 7b. Attention
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_runs):
        with torch.no_grad():
            attn_out = layer.attn(basis, edge_features, normed, neighbor_idx, seq_pos=seq_pos)
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / n_runs * 1000
    results["  7b. Attention"] = elapsed
    print(f"    Attention: {elapsed:.3f} ms")

    # 7c. Transition (if present)
    if layer.transition is not None:
        residual = node_features + attn_out
        normed2 = layer.ln2(residual)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(n_runs):
            with torch.no_grad():
                trans_out = layer.transition(normed2)
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / n_runs * 1000
        results["  7c. Transition"] = elapsed
        print(f"    Transition: {elapsed:.3f} ms")

    # 8. Further breakdown of attention
    print("\n  Attention breakdown:")
    attn = layer.attn

    # 8a. QKV convolution
    src_idx = neighbor_idx.flatten()
    basis_flat = basis.view(N * k, *basis.shape[2:])
    edge_feats_flat = edge_features.view(N * k, -1)

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_runs):
        with torch.no_grad():
            qkv = attn.conv_qkv(basis_flat, edge_feats_flat, normed, src_idx)
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / n_runs * 1000
    results["    8a. QKV conv"] = elapsed
    print(f"      QKV convolution: {elapsed:.3f} ms")

    # 8b. Attention scores and softmax
    qkv = qkv.view(N, k, 3 * attn.out_mult, attn.out_dim)
    queries, keys, values = qkv.chunk(3, dim=2)
    q_heads = queries.flatten(-2, -1).view(N, k, attn.nheads, attn.head_dim)
    k_heads = keys.flatten(-2, -1).view(N, k, attn.nheads, attn.head_dim)
    v_heads = values.flatten(-2, -1).view(N, k, attn.nheads, attn.head_dim)
    q_heads = q_heads.transpose(1, 2).contiguous()
    k_heads = k_heads.transpose(1, 2).contiguous()
    v_heads = v_heads.transpose(1, 2).contiguous()

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_runs):
        scores = (q_heads * k_heads).sum(dim=-1) * attn.scale
        weights = torch.softmax(scores, dim=-1)
        weighted = weights.unsqueeze(-1) * v_heads
        output = weighted.sum(dim=2)
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / n_runs * 1000
    results["    8b. Attn scores"] = elapsed
    print(f"      Attention scores + softmax: {elapsed:.3f} ms")

    # 8c. Output projection
    output = output.view(N, attn.out_mult, attn.out_dim)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_runs):
        with torch.no_grad():
            proj_out = attn.out_proj(output)
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / n_runs * 1000
    results["    8c. Output proj"] = elapsed
    print(f"      Output projection: {elapsed:.3f} ms")

    # 9. QKV conv breakdown (the expensive part)
    print("\n  QKV convolution breakdown:")
    conv = attn.conv_qkv

    # 9a. Radial weights
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_runs):
        with torch.no_grad():
            weights = conv.rwlin(edge_feats_flat)
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / n_runs * 1000
    results["      9a. Radial weights"] = elapsed
    print(f"        Radial weight network: {elapsed:.3f} ms")

    # 9b. Feature gather
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_runs):
        f_src = normed[src_idx]
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / n_runs * 1000
    results["      9b. Feature gather"] = elapsed
    print(f"        Feature gather: {elapsed:.3f} ms")

    # 9c. Tensor contractions
    f_src = normed[src_idx]
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_runs):
        contracted = f_src.unsqueeze(1) @ basis_flat
        output = (weights @ contracted).sum(dim=1)
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / n_runs * 1000
    results["      9c. Tensor contract"] = elapsed
    print(f"        Tensor contractions: {elapsed:.3f} ms")

    return results


def main():
    parser = argparse.ArgumentParser(description="Profile EquivariantTransformer")
    parser.add_argument("--n-points", type=int, default=500, help="Number of points")
    parser.add_argument("--hidden-mult", type=int, default=32, help="Hidden multiplicity")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--k-neighbors", type=int, default=16, help="k for k-NN")
    parser.add_argument("--n-iters", type=int, default=10, help="Profiling iterations")
    parser.add_argument("--no-backward", action="store_true", help="Skip backward profiling")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA version: {torch.version.cuda}")

    # Create model and inputs
    print(f"\nCreating model with N={args.n_points}, mult={args.hidden_mult}, "
          f"layers={args.num_layers}, k={args.k_neighbors}")

    model, coordinates, node_features, seq_pos = create_model_and_inputs(
        n_points=args.n_points,
        hidden_mult=args.hidden_mult,
        num_layers=args.num_layers,
        k_neighbors=args.k_neighbors,
        device=device,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Warmup
    warmup(model, coordinates, node_features, seq_pos)

    # Component-level profiling (manual timing)
    results = profile_components(model, coordinates, node_features, seq_pos)

    # PyTorch profiler - forward only
    prof_fwd = profile_forward(model, coordinates, node_features, seq_pos, args.n_iters)

    # Save forward trace
    prof_fwd.export_chrome_trace("outputs/profile_transformer_trace.json")
    print("\nChrome trace saved to outputs/profile_transformer_trace.json")

    # Print summary tables
    print("\n" + "="*60)
    print("CUDA Kernel Summary (Forward Pass)")
    print("="*60)
    print(prof_fwd.key_averages().table(
        sort_by="cuda_time_total", row_limit=20
    ))

    # Backward profiling
    if not args.no_backward:
        prof_bwd = profile_backward(model, coordinates, node_features, seq_pos, args.n_iters)

        print("\n" + "="*60)
        print("CUDA Kernel Summary (Forward + Backward)")
        print("="*60)
        print(prof_bwd.key_averages().table(
            sort_by="cuda_time_total", row_limit=20
        ))

    # Save text summary
    with open("outputs/profile_transformer_summary.txt", "w") as f:
        f.write(f"EquivariantTransformer Profiling Results\n")
        f.write(f"="*60 + "\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  N points: {args.n_points}\n")
        f.write(f"  Hidden mult: {args.hidden_mult}\n")
        f.write(f"  Num layers: {args.num_layers}\n")
        f.write(f"  k neighbors: {args.k_neighbors}\n")
        f.write(f"  Parameters: {n_params:,}\n\n")

        f.write(f"Component Timing (ms):\n")
        for name, time_ms in results.items():
            f.write(f"  {name}: {time_ms:.3f}\n")

        f.write(f"\n\nForward Pass Kernel Summary:\n")
        f.write(prof_fwd.key_averages().table(sort_by="cuda_time_total", row_limit=30))

        if not args.no_backward:
            f.write(f"\n\nForward+Backward Kernel Summary:\n")
            f.write(prof_bwd.key_averages().table(sort_by="cuda_time_total", row_limit=30))

    print("\nSummary saved to outputs/profile_transformer_summary.txt")

    # Print total forward time
    print("\n" + "="*60)
    print("Total Forward Pass Time")
    print("="*60)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(20):
        with torch.no_grad():
            _ = model(coordinates, node_features, seq_pos=seq_pos)
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / 20 * 1000
    print(f"Average forward pass: {elapsed:.2f} ms")


if __name__ == "__main__":
    main()
