"""
Performance profiling for ResidueFlowModel and PolymerFlowModel.

Benchmarks encode, decode, and sample operations across different
sequence lengths, batch sizes, and devices.

Usage:
    python tests/profiling/profile_flow.py
    python tests/profiling/profile_flow.py --device cuda
    python tests/profiling/profile_flow.py --markdown
"""

import os
import sys

import numpy as np
import torch

# Handle both direct execution and module import
try:
    from .timing import Timer, TimingResult, get_sync_fn, get_available_devices
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
    from timing import Timer, TimingResult, get_sync_fn, get_available_devices


# Default benchmark parameters
DEFAULT_WARMUP = 3
DEFAULT_RUNS = 10


# =============================================================================
# Mock Models for Profiling (no trained weights needed)
# =============================================================================


def create_mock_residue_model(residue, n_atoms: int = 20, latent_dim: int = 8):
    """
    Create a ResidueFlowModel-like object for profiling.

    Uses random PCA components - not trained, but exercises the same code paths.
    """
    from ciffy.nn.flow import PCAFlow, ResidueFlowModel
    from ciffy.nn.flow.residue.data import compute_pca

    # Generate synthetic data for PCA
    np.random.seed(42)
    n_samples = 100
    extended_dim = n_atoms * 3 + 6  # coords + transform

    data = np.random.randn(n_samples, extended_dim).astype(np.float32)
    V, mean, _, var_explained = compute_pca(data, n_components=latent_dim)

    flow = PCAFlow(
        torch.from_numpy(V).float(),
        torch.from_numpy(mean).float(),
        n_layers=4,
        hidden_dim=64,
        bound=3.0,
    )

    # Get atom indices from residue
    atom_indices = list(residue.index()[:n_atoms])

    return ResidueFlowModel(
        flow=flow,
        residue=residue,
        atom_indices=atom_indices,
        n_atoms=n_atoms,
    )


def create_mock_polymer_model(residue_types: list, n_atoms: int = 20, latent_dim: int = 8):
    """
    Create a PolymerFlowModel for profiling.

    Args:
        residue_types: List of Residue enums to include in the model.
        n_atoms: Number of atoms per residue.
        latent_dim: Latent dimension for each residue model.

    Returns:
        PolymerFlowModel with mock residue models.
    """
    from ciffy.nn.flow import PolymerFlowModel

    residue_models = {
        res: create_mock_residue_model(res, n_atoms, latent_dim)
        for res in residue_types
    }

    return PolymerFlowModel(residue_models)


# =============================================================================
# ResidueFlowModel Benchmarks
# =============================================================================


def benchmark_residue_encode(
    model,
    batch_sizes: list[int],
    device: str = "cpu",
    runs: int = DEFAULT_RUNS,
) -> dict[int, TimingResult]:
    """Benchmark ResidueFlowModel.encode() for various batch sizes."""
    sync = get_sync_fn(device)
    model.flow.to(device)
    model.flow.eval()

    results = {}
    n_atoms = model.n_atoms

    for batch_size in batch_sizes:
        # Create random coordinates
        coords = torch.randn(batch_size, n_atoms, 3, device=device)
        transforms = torch.zeros(batch_size, 6, device=device)

        def run():
            with torch.no_grad():
                return model.encode(coords, transforms)

        results[batch_size] = Timer.benchmark(run, runs=runs, sync=sync)

    return results


def benchmark_residue_decode(
    model,
    batch_sizes: list[int],
    device: str = "cpu",
    runs: int = DEFAULT_RUNS,
) -> dict[int, TimingResult]:
    """Benchmark ResidueFlowModel.decode() for various batch sizes."""
    sync = get_sync_fn(device)
    model.flow.to(device)
    model.flow.eval()

    results = {}
    latent_dim = model.latent_dim

    for batch_size in batch_sizes:
        # Create random latents
        z = torch.randn(batch_size, latent_dim, device=device)

        def run():
            with torch.no_grad():
                return model.decode(z)

        results[batch_size] = Timer.benchmark(run, runs=runs, sync=sync)

    return results


def benchmark_residue_sample(
    model,
    sample_counts: list[int],
    device: str = "cpu",
    runs: int = DEFAULT_RUNS,
) -> dict[int, TimingResult]:
    """Benchmark ResidueFlowModel.sample() for various sample counts."""
    sync = get_sync_fn(device)
    model.flow.to(device)
    model.flow.eval()

    results = {}

    for n_samples in sample_counts:
        def run():
            with torch.no_grad():
                return model.sample(n_samples)

        results[n_samples] = Timer.benchmark(run, runs=runs, sync=sync)

    return results


# =============================================================================
# PolymerFlowModel Benchmarks
# =============================================================================


def _move_polymer_model_to_device(model, device: str) -> None:
    """Move all residue flows to the specified device."""
    for residue_model in model.residue_models.values():
        residue_model.flow.to(device)
        residue_model.flow.eval()


def benchmark_polymer_encode(
    model,
    sequence_lengths: list[int],
    device: str = "cpu",
    runs: int = DEFAULT_RUNS,
) -> dict[int, TimingResult]:
    """Benchmark PolymerFlowModel.encode() for various sequence lengths."""
    sync = get_sync_fn(device)
    _move_polymer_model_to_device(model, device)

    results = {}
    # Keys are already int after normalization
    residue_types = list(model.residue_models.keys())

    for seq_len in sequence_lengths:
        # Create random sequence using available residue types (already ints)
        sequence = np.array(
            [residue_types[i % len(residue_types)] for i in range(seq_len)],
            dtype=np.int64
        )

        # Create random coordinates (sum of atoms for sequence)
        n_atoms = sum(model.residue_models[r].n_atoms for r in sequence)
        coords = torch.randn(n_atoms, 3, device=device)

        def run():
            with torch.no_grad():
                return model.encode(coords, sequence)

        results[seq_len] = Timer.benchmark(run, runs=runs, sync=sync)

    return results


def benchmark_polymer_decode(
    model,
    sequence_lengths: list[int],
    device: str = "cpu",
    runs: int = DEFAULT_RUNS,
) -> dict[int, TimingResult]:
    """Benchmark PolymerFlowModel.decode() for various sequence lengths."""
    sync = get_sync_fn(device)
    _move_polymer_model_to_device(model, device)

    results = {}
    residue_types = list(model.residue_models.keys())
    latent_dim = model.latent_dim

    for seq_len in sequence_lengths:
        # Create random sequence (keys are already ints)
        sequence = np.array(
            [residue_types[i % len(residue_types)] for i in range(seq_len)],
            dtype=np.int64
        )

        # Create random latents
        latents = torch.randn(seq_len, latent_dim, device=device)

        def run():
            with torch.no_grad():
                return model.decode(latents, sequence)

        results[seq_len] = Timer.benchmark(run, runs=runs, sync=sync)

    return results


def benchmark_polymer_sample(
    model,
    sequence_lengths: list[int],
    device: str = "cpu",
    runs: int = DEFAULT_RUNS,
) -> dict[int, TimingResult]:
    """Benchmark PolymerFlowModel.sample() for various sequence lengths."""
    sync = get_sync_fn(device)
    _move_polymer_model_to_device(model, device)

    results = {}
    residue_types = list(model.residue_models.keys())

    for seq_len in sequence_lengths:
        # Create random sequence (keys are already ints)
        sequence = np.array(
            [residue_types[i % len(residue_types)] for i in range(seq_len)],
            dtype=np.int64
        )

        def run():
            with torch.no_grad():
                return model.sample(sequence)

        results[seq_len] = Timer.benchmark(run, runs=runs, sync=sync)

    return results


# =============================================================================
# Frame Computation Benchmarks
# =============================================================================


def benchmark_frame_computation(
    batch_sizes: list[int],
    device: str = "cpu",
    runs: int = DEFAULT_RUNS,
) -> dict[str, dict[int, TimingResult]]:
    """
    Benchmark frame computation functions.

    Compares:
    - compute_frame_from_indices (fast path with pre-resolved indices)
    - compute_prev_frame (convenience wrapper)
    """
    from ciffy.geometry import compute_frame_from_indices, compute_prev_frame
    from ciffy.biochemistry import Residue

    sync = get_sync_fn(device)
    residue = Residue.A

    # Pre-resolve frame indices (simulating what ResidueFlowModel does at init)
    from ciffy.biochemistry.linking import LINKING_BY_TYPE
    atom_indices = list(residue.index()[:20])
    atom_to_col = {a: i for i, a in enumerate(atom_indices)}
    link_def = LINKING_BY_TYPE[residue.molecule_type]
    prev_frame_cols = link_def.prev_frame.resolve(residue, atom_to_col)
    z_toward_origin = link_def.prev_frame.z_toward_origin

    results = {"fast_path": {}, "wrapper": {}}

    for batch_size in batch_sizes:
        # Create random coordinates
        coords = torch.randn(batch_size, 20, 3, device=device)

        # Benchmark fast path (pre-resolved indices)
        def run_fast():
            for i in range(batch_size):
                compute_frame_from_indices(coords[i], prev_frame_cols, z_toward_origin)

        results["fast_path"][batch_size] = Timer.benchmark(run_fast, runs=runs, sync=sync)

        # Benchmark wrapper (resolves indices each call)
        def run_wrapper():
            for i in range(batch_size):
                compute_prev_frame(coords[i], atom_to_col, residue)

        results["wrapper"][batch_size] = Timer.benchmark(run_wrapper, runs=runs, sync=sync)

    return results


# =============================================================================
# Output Formatting
# =============================================================================


def print_residue_results(
    encode_results: dict,
    decode_results: dict,
    sample_results: dict,
    device: str,
) -> None:
    """Print ResidueFlowModel benchmark results."""
    print(f"\n{'='*60}")
    print(f"ResidueFlowModel Benchmarks (device: {device})")
    print(f"{'='*60}")

    print("\nEncode (batch_size -> time):")
    print(f"  {'Batch':<10} {'Mean':<15} {'Std':<15}")
    print(f"  {'-'*40}")
    for batch_size, result in sorted(encode_results.items()):
        print(f"  {batch_size:<10} {result.mean*1000:>10.3f} ms   ±{result.std*1000:>8.3f} ms")

    print("\nDecode (batch_size -> time):")
    print(f"  {'Batch':<10} {'Mean':<15} {'Std':<15}")
    print(f"  {'-'*40}")
    for batch_size, result in sorted(decode_results.items()):
        print(f"  {batch_size:<10} {result.mean*1000:>10.3f} ms   ±{result.std*1000:>8.3f} ms")

    print("\nSample (n_samples -> time):")
    print(f"  {'Samples':<10} {'Mean':<15} {'Std':<15}")
    print(f"  {'-'*40}")
    for n_samples, result in sorted(sample_results.items()):
        print(f"  {n_samples:<10} {result.mean*1000:>10.3f} ms   ±{result.std*1000:>8.3f} ms")


def print_polymer_results(
    encode_results: dict,
    decode_results: dict,
    sample_results: dict,
    device: str,
) -> None:
    """Print PolymerFlowModel benchmark results."""
    print(f"\n{'='*60}")
    print(f"PolymerFlowModel Benchmarks (device: {device})")
    print(f"{'='*60}")

    print("\nEncode (sequence_length -> time):")
    print(f"  {'SeqLen':<10} {'Mean':<15} {'Std':<15} {'Per-residue':<15}")
    print(f"  {'-'*55}")
    for seq_len, result in sorted(encode_results.items()):
        per_res = result.mean * 1000 / seq_len if seq_len > 0 else 0
        print(f"  {seq_len:<10} {result.mean*1000:>10.3f} ms   ±{result.std*1000:>8.3f} ms   {per_res:>8.3f} ms/res")

    print("\nDecode (sequence_length -> time):")
    print(f"  {'SeqLen':<10} {'Mean':<15} {'Std':<15} {'Per-residue':<15}")
    print(f"  {'-'*55}")
    for seq_len, result in sorted(decode_results.items()):
        per_res = result.mean * 1000 / seq_len if seq_len > 0 else 0
        print(f"  {seq_len:<10} {result.mean*1000:>10.3f} ms   ±{result.std*1000:>8.3f} ms   {per_res:>8.3f} ms/res")

    print("\nSample (sequence_length -> time):")
    print(f"  {'SeqLen':<10} {'Mean':<15} {'Std':<15} {'Per-residue':<15}")
    print(f"  {'-'*55}")
    for seq_len, result in sorted(sample_results.items()):
        per_res = result.mean * 1000 / seq_len if seq_len > 0 else 0
        print(f"  {seq_len:<10} {result.mean*1000:>10.3f} ms   ±{result.std*1000:>8.3f} ms   {per_res:>8.3f} ms/res")


def print_frame_results(results: dict, device: str) -> None:
    """Print frame computation benchmark results."""
    print(f"\n{'='*60}")
    print(f"Frame Computation Benchmarks (device: {device})")
    print(f"{'='*60}")

    print("\ncompute_frame_from_indices (fast path) vs compute_prev_frame (wrapper):")
    print(f"  {'Batch':<10} {'Fast Path':<18} {'Wrapper':<18} {'Speedup':<10}")
    print(f"  {'-'*55}")

    for batch_size in sorted(results["fast_path"].keys()):
        fast = results["fast_path"][batch_size]
        wrapper = results["wrapper"][batch_size]
        speedup = wrapper.mean / fast.mean if fast.mean > 0 else 0
        print(f"  {batch_size:<10} {fast.mean*1000:>10.3f} ms      {wrapper.mean*1000:>10.3f} ms      {speedup:>5.2f}x")


def generate_markdown_table(
    residue_encode: dict,
    residue_decode: dict,
    polymer_encode: dict,
    polymer_decode: dict,
    device: str,
) -> str:
    """Generate markdown tables for benchmark results."""
    lines = [
        f"## Flow Model Benchmarks (device: {device})",
        "",
        "### ResidueFlowModel",
        "",
        "| Batch Size | Encode | Decode |",
        "|------------|-------:|-------:|",
    ]

    for batch_size in sorted(residue_encode.keys()):
        enc = residue_encode[batch_size]
        dec = residue_decode[batch_size]
        lines.append(f"| {batch_size} | {enc.mean*1000:.3f} ms | {dec.mean*1000:.3f} ms |")

    lines.extend([
        "",
        "### PolymerFlowModel",
        "",
        "| Sequence Length | Encode | Decode | Per-residue Decode |",
        "|-----------------|-------:|-------:|-------------------:|",
    ])

    for seq_len in sorted(polymer_encode.keys()):
        enc = polymer_encode[seq_len]
        dec = polymer_decode[seq_len]
        per_res = dec.mean * 1000 / seq_len if seq_len > 0 else 0
        lines.append(f"| {seq_len} | {enc.mean*1000:.3f} ms | {dec.mean*1000:.3f} ms | {per_res:.3f} ms/res |")

    return "\n".join(lines)


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Flow model performance benchmark")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to benchmark on (cpu, cuda, mps)")
    parser.add_argument("--markdown", action="store_true",
                        help="Output markdown tables")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS,
                        help="Number of benchmark runs")
    parser.add_argument("--frame-only", action="store_true",
                        help="Only benchmark frame computation")
    args = parser.parse_args()

    # Check device availability
    available = get_available_devices()
    device = args.device
    if device not in available and device != "cpu":
        print(f"Warning: {device} not available, falling back to cpu")
        device = "cpu"

    print("Flow Model Performance Benchmark")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {device}")
    print(f"Available devices: {', '.join(available)}")

    # Benchmark parameters
    batch_sizes = [1, 8, 32, 128]
    sequence_lengths = [4, 16, 64, 256]

    # Frame computation benchmark
    if args.frame_only:
        frame_results = benchmark_frame_computation(batch_sizes, device, args.runs)
        print_frame_results(frame_results, device)
        return

    # Create mock models
    from ciffy.biochemistry import Residue

    print("\nCreating mock models...")
    residue_model = create_mock_residue_model(Residue.A, n_atoms=20, latent_dim=8)
    polymer_model = create_mock_polymer_model(
        [Residue.A, Residue.C, Residue.G, Residue.U],
        n_atoms=20,
        latent_dim=8,
    )

    # Run ResidueFlowModel benchmarks
    print("\nBenchmarking ResidueFlowModel...")
    residue_encode = benchmark_residue_encode(residue_model, batch_sizes, device, args.runs)
    residue_decode = benchmark_residue_decode(residue_model, batch_sizes, device, args.runs)
    residue_sample = benchmark_residue_sample(residue_model, batch_sizes, device, args.runs)

    # Run PolymerFlowModel benchmarks
    print("Benchmarking PolymerFlowModel...")
    polymer_encode = benchmark_polymer_encode(polymer_model, sequence_lengths, device, args.runs)
    polymer_decode = benchmark_polymer_decode(polymer_model, sequence_lengths, device, args.runs)
    polymer_sample = benchmark_polymer_sample(polymer_model, sequence_lengths, device, args.runs)

    # Run frame computation benchmarks
    print("Benchmarking frame computation...")
    frame_results = benchmark_frame_computation(batch_sizes, device, args.runs)

    # Output results
    if args.markdown:
        print(generate_markdown_table(
            residue_encode, residue_decode,
            polymer_encode, polymer_decode,
            device,
        ))
    else:
        print_residue_results(residue_encode, residue_decode, residue_sample, device)
        print_polymer_results(polymer_encode, polymer_decode, polymer_sample, device)
        print_frame_results(frame_results, device)


if __name__ == "__main__":
    main()
