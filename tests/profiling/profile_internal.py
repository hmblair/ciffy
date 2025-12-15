"""
Performance profiling for internal coordinate conversions.

Benchmarks conversion between Cartesian and internal coordinates
(Z-matrix representation) for both NumPy and PyTorch backends.

Usage:
    python tests/profiling/profile_internal.py
    python tests/profiling/profile_internal.py --structure 1ZEW
    python tests/profiling/profile_internal.py --torch-only
"""

import os
import time
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Get test data directory
TEST_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_DIR = os.path.join(TEST_DIR, "data")

# Default benchmark parameters
WARMUP_RUNS = 3
BENCHMARK_RUNS = 10


def _benchmark(func, warmup: int = WARMUP_RUNS, runs: int = BENCHMARK_RUNS) -> dict:
    """
    Run a function multiple times and return timing statistics.

    Returns:
        Dict with mean, std, min, max times in seconds.
    """
    # Warmup
    for _ in range(warmup):
        func()

    # Benchmark
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        func()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return {
        "mean": np.mean(times),
        "std": np.std(times),
        "min": np.min(times),
        "max": np.max(times),
    }


def benchmark_internal_coords(filepath: str, backend: str = "numpy",
                               runs: int = BENCHMARK_RUNS) -> dict:
    """
    Benchmark internal coordinate conversions for a structure.

    Args:
        filepath: Path to CIF file.
        backend: 'numpy' or 'torch'.
        runs: Number of benchmark runs.

    Returns:
        Dict with timing results for each operation.
    """
    import ciffy

    # Load structure
    polymer = ciffy.load(filepath, backend=backend).poly()

    results = {
        "file": os.path.basename(filepath),
        "backend": backend,
        "atoms": polymer.size(),
        "residues": polymer.size(ciffy.Scale.RESIDUE),
        "chains": polymer.size(ciffy.Scale.CHAIN),
    }

    # Benchmark Cartesian -> Internal (accessing dihedrals triggers computation)
    def to_internal():
        # Create fresh polymer to reset internal coords
        p = ciffy.load(filepath, backend=backend).poly()
        return p.dihedrals

    results["to_internal"] = _benchmark(to_internal, runs=runs)

    # Get Z-matrix size after initial computation
    _ = polymer.dihedrals  # Trigger computation
    zmatrix = polymer._coord_manager.zmatrix
    results["zmatrix_size"] = len(zmatrix)

    # Count orphan atoms (single-atom components)
    mgr = polymer._coord_manager
    n_components = mgr._components.n_components
    orphan_count = sum(
        1 for i in range(n_components)
        if mgr._components.get_component_size(i) == 1
    )
    results["orphan_atoms"] = orphan_count

    # Benchmark Internal -> Cartesian (setting dihedrals triggers reconstruction)
    orig_coords = polymer.coordinates.copy() if backend == "numpy" else polymer.coordinates.clone()

    def to_cartesian():
        # Copy dihedrals and set them back to trigger reconstruction
        if backend == "numpy":
            dihedrals = polymer.dihedrals.copy()
        else:
            dihedrals = polymer.dihedrals.clone()
        polymer.dihedrals = dihedrals
        return polymer.coordinates

    results["to_cartesian"] = _benchmark(to_cartesian, runs=runs)

    # Benchmark round-trip
    def round_trip():
        p = ciffy.load(filepath, backend=backend).poly()
        if backend == "numpy":
            dihedrals = p.dihedrals.copy()
        else:
            dihedrals = p.dihedrals.clone()
        p.dihedrals = dihedrals
        return p.coordinates

    results["round_trip"] = _benchmark(round_trip, runs=runs)

    return results


def benchmark_torch_gpu(filepath: str, runs: int = BENCHMARK_RUNS) -> dict | None:
    """
    Benchmark internal coordinate conversions on GPU.

    Returns:
        Dict with timing results, or None if CUDA not available.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return None
    except ImportError:
        return None

    import ciffy

    # Load and move to GPU
    polymer = ciffy.load(filepath, backend="torch").poly().to("cuda")

    results = {
        "file": os.path.basename(filepath),
        "backend": "torch+cuda",
        "atoms": polymer.size(),
        "device": str(polymer.coordinates.device),
    }

    # Benchmark Cartesian -> Internal on GPU
    def to_internal():
        torch.cuda.synchronize()
        dihedrals = polymer.dihedrals
        torch.cuda.synchronize()
        return dihedrals

    # Reset internal coords for fresh benchmark
    polymer._coord_manager._internal_valid = False
    polymer._coord_manager._zmatrix = None

    results["to_internal"] = _benchmark(to_internal, runs=runs)

    # Benchmark Internal -> Cartesian on GPU
    def to_cartesian():
        torch.cuda.synchronize()
        dihedrals = polymer.dihedrals.clone()
        polymer.dihedrals = dihedrals
        coords = polymer.coordinates
        torch.cuda.synchronize()
        return coords

    results["to_cartesian"] = _benchmark(to_cartesian, runs=runs)

    return results


def print_results(results: dict) -> None:
    """Pretty-print benchmark results."""
    print(f"\n{'='*70}")
    print(f"Structure: {results['file']} | Backend: {results['backend']}")
    print(f"{'='*70}")
    print(f"  Atoms: {results['atoms']:,} | Residues: {results.get('residues', '?'):,} | "
          f"Chains: {results.get('chains', '?')}")
    if 'zmatrix_size' in results:
        print(f"  Z-matrix entries: {results['zmatrix_size']:,} | "
              f"Orphan atoms: {results['orphan_atoms']:,}")
    print()

    # Print timing table
    print(f"  {'Operation':<20} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
    print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")

    for op in ["to_internal", "to_cartesian", "round_trip"]:
        if op in results:
            t = results[op]
            print(f"  {op:<20} {t['mean']*1000:>10.2f}ms {t['std']*1000:>10.2f}ms "
                  f"{t['min']*1000:>10.2f}ms {t['max']*1000:>10.2f}ms")

    # Print throughput
    if "to_internal" in results and results["atoms"] > 0:
        atoms = results["atoms"]
        to_int_ms = results["to_internal"]["mean"] * 1000
        to_cart_ms = results["to_cartesian"]["mean"] * 1000
        print()
        print(f"  Throughput:")
        print(f"    to_internal:   {atoms / to_int_ms * 1000:,.0f} atoms/sec")
        print(f"    to_cartesian:  {atoms / to_cart_ms * 1000:,.0f} atoms/sec")


def print_comparison(numpy_results: dict, torch_results: dict,
                     cuda_results: dict | None = None) -> None:
    """Print comparison between backends."""
    print(f"\n{'='*70}")
    print("Backend Comparison")
    print(f"{'='*70}")

    print(f"\n  {'Operation':<20} {'NumPy':>12} {'PyTorch':>12}", end="")
    if cuda_results:
        print(f" {'CUDA':>12}", end="")
    print()
    print(f"  {'-'*20} {'-'*12} {'-'*12}", end="")
    if cuda_results:
        print(f" {'-'*12}", end="")
    print()

    for op in ["to_internal", "to_cartesian", "round_trip"]:
        np_ms = numpy_results[op]["mean"] * 1000
        torch_ms = torch_results[op]["mean"] * 1000
        print(f"  {op:<20} {np_ms:>10.2f}ms {torch_ms:>10.2f}ms", end="")
        if cuda_results and op in cuda_results:
            cuda_ms = cuda_results[op]["mean"] * 1000
            print(f" {cuda_ms:>10.2f}ms", end="")
        print()

    # Speedup ratios
    print()
    print(f"  PyTorch vs NumPy:")
    for op in ["to_internal", "to_cartesian"]:
        np_ms = numpy_results[op]["mean"]
        torch_ms = torch_results[op]["mean"]
        ratio = np_ms / torch_ms if torch_ms > 0 else 0
        faster = "faster" if ratio > 1 else "slower"
        print(f"    {op}: {abs(ratio):.2f}x {faster}")

    if cuda_results:
        print(f"  CUDA vs NumPy:")
        for op in ["to_internal", "to_cartesian"]:
            if op in cuda_results:
                np_ms = numpy_results[op]["mean"]
                cuda_ms = cuda_results[op]["mean"]
                ratio = np_ms / cuda_ms if cuda_ms > 0 else 0
                faster = "faster" if ratio > 1 else "slower"
                print(f"    {op}: {abs(ratio):.2f}x {faster}")


def get_test_file(name: str) -> str:
    """Get path to test CIF file."""
    path = os.path.join(DATA_DIR, f"{name}.cif")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Test file not found: {path}")
    return path


if __name__ == "__main__":
    import argparse
    import ciffy

    parser = argparse.ArgumentParser(
        description="Benchmark internal coordinate conversions"
    )
    parser.add_argument(
        "--structure", "-s", type=str, default="9MDS",
        help="PDB ID to benchmark (default: 9MDS)"
    )
    parser.add_argument(
        "--runs", "-r", type=int, default=BENCHMARK_RUNS,
        help=f"Number of benchmark runs (default: {BENCHMARK_RUNS})"
    )
    parser.add_argument(
        "--numpy-only", action="store_true",
        help="Only benchmark NumPy backend"
    )
    parser.add_argument(
        "--torch-only", action="store_true",
        help="Only benchmark PyTorch backend"
    )
    parser.add_argument(
        "--cuda", action="store_true",
        help="Include CUDA GPU benchmark"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Benchmark all test structures"
    )
    args = parser.parse_args()

    print("Internal Coordinates Benchmark")
    print("=" * 70)
    print(f"ciffy version: {ciffy.__version__}")
    print(f"Benchmark runs: {args.runs}")

    # Determine which structures to benchmark
    if args.all:
        import glob
        structures = [
            os.path.splitext(os.path.basename(f))[0]
            for f in sorted(glob.glob(os.path.join(DATA_DIR, "*.cif")))
        ]
    else:
        structures = [args.structure]

    for structure in structures:
        try:
            filepath = get_test_file(structure)
        except FileNotFoundError as e:
            print(f"\nSkipping {structure}: {e}")
            continue

        numpy_results = None
        torch_results = None
        cuda_results = None

        # NumPy benchmark
        if not args.torch_only:
            numpy_results = benchmark_internal_coords(filepath, "numpy", args.runs)
            print_results(numpy_results)

        # PyTorch benchmark
        if not args.numpy_only:
            try:
                torch_results = benchmark_internal_coords(filepath, "torch", args.runs)
                print_results(torch_results)
            except ImportError:
                print("\nPyTorch not available, skipping torch benchmark")

        # CUDA benchmark
        if args.cuda and not args.numpy_only:
            cuda_results = benchmark_torch_gpu(filepath, args.runs)
            if cuda_results:
                print_results(cuda_results)
            else:
                print("\nCUDA not available, skipping GPU benchmark")

        # Print comparison if we have multiple backends
        if numpy_results and torch_results:
            print_comparison(numpy_results, torch_results, cuda_results)

    print()
