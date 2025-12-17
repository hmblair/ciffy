"""
Performance profiling for internal coordinate conversions.

Benchmarks conversion between Cartesian and internal coordinates
(Z-matrix representation) on CPU and any available GPU.

Usage:
    python tests/profiling/profile_internal.py
    python tests/profiling/profile_internal.py --structure 1ZEW
    python tests/profiling/profile_internal.py --all
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


def get_available_devices() -> list[str]:
    """
    Get list of available devices for benchmarking.

    Returns:
        List of device strings (e.g., ['cpu', 'cuda', 'mps']).
    """
    devices = ["cpu"]

    try:
        import torch

        if torch.cuda.is_available():
            devices.append("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            devices.append("mps")
    except ImportError:
        pass

    return devices


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


def benchmark_device(filepath: str, device: str, runs: int = BENCHMARK_RUNS) -> dict:
    """
    Benchmark internal coordinate conversions on a specific device.

    Args:
        filepath: Path to CIF file.
        device: Device string ('cpu', 'cuda', 'mps').
        runs: Number of benchmark runs.

    Returns:
        Dict with timing results for each operation.
    """
    import torch
    import ciffy

    # Load structure
    polymer = ciffy.load(filepath, backend="torch").poly()

    # Move to device
    if device != "cpu":
        polymer = polymer.to(device)

    results = {
        "file": os.path.basename(filepath),
        "device": device,
        "atoms": polymer.size(),
        "residues": polymer.size(ciffy.Scale.RESIDUE),
        "chains": polymer.size(ciffy.Scale.CHAIN),
    }

    def sync():
        """Synchronize device if needed."""
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()

    # Initialize Z-matrix by triggering first computation
    _ = polymer.dihedrals
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

    # Cache original coordinates for dirtying
    original_coords = polymer.coordinates.clone()

    # Benchmark Cartesian -> Internal (dirty coords, then access dihedrals)
    def to_internal():
        sync()
        # Dirty coordinates to force recomputation of internal coords
        polymer.coordinates = original_coords.clone()
        result = polymer.dihedrals
        sync()
        return result

    results["to_internal"] = _benchmark(to_internal, runs=runs)

    # Benchmark Internal -> Cartesian (set dihedrals, then access coordinates)
    def to_cartesian():
        sync()
        dihedrals = polymer.dihedrals.clone()
        polymer.dihedrals = dihedrals
        coords = polymer.coordinates
        sync()
        return coords

    results["to_cartesian"] = _benchmark(to_cartesian, runs=runs)

    # Benchmark round-trip (dirty coords -> internal -> cartesian)
    def round_trip():
        sync()
        # Dirty coordinates
        polymer.coordinates = original_coords.clone()
        # Get internal coords
        dihedrals = polymer.dihedrals.clone()
        # Set internal coords to trigger reconstruction
        polymer.dihedrals = dihedrals
        result = polymer.coordinates
        sync()
        return result

    results["round_trip"] = _benchmark(round_trip, runs=runs)

    return results


def print_results(results: dict) -> None:
    """Pretty-print benchmark results."""
    print(f"\n{'='*70}")
    print(f"Structure: {results['file']} | Device: {results['device']}")
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


def print_device_comparison(all_results: list[dict]) -> None:
    """Print comparison between devices."""
    if len(all_results) < 2:
        return

    print(f"\n{'='*70}")
    print("Device Comparison")
    print(f"{'='*70}")

    # Header
    devices = [r["device"] for r in all_results]
    header = f"  {'Operation':<20}"
    for device in devices:
        header += f" {device:>12}"
    print(header)

    divider = f"  {'-'*20}"
    for _ in devices:
        divider += f" {'-'*12}"
    print(divider)

    # Timing rows
    for op in ["to_internal", "to_cartesian", "round_trip"]:
        row = f"  {op:<20}"
        for r in all_results:
            if op in r:
                ms = r[op]["mean"] * 1000
                row += f" {ms:>10.2f}ms"
            else:
                row += f" {'N/A':>12}"
        print(row)

    # Speedup ratios (relative to CPU)
    cpu_results = next((r for r in all_results if r["device"] == "cpu"), None)
    if cpu_results:
        print()
        print("  Speedup vs CPU:")
        for r in all_results:
            if r["device"] == "cpu":
                continue
            for op in ["to_internal", "to_cartesian"]:
                if op in r and op in cpu_results:
                    cpu_ms = cpu_results[op]["mean"]
                    device_ms = r[op]["mean"]
                    ratio = cpu_ms / device_ms if device_ms > 0 else 0
                    faster = "faster" if ratio > 1 else "slower"
                    print(f"    {r['device']} {op}: {abs(ratio):.2f}x {faster}")


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
        "--all", action="store_true",
        help="Benchmark all test structures"
    )
    args = parser.parse_args()

    # Detect available devices
    devices = get_available_devices()

    print("Internal Coordinates Benchmark")
    print("=" * 70)
    print(f"ciffy version: {ciffy.__version__}")
    print(f"Benchmark runs: {args.runs}")
    print(f"Available devices: {', '.join(devices)}")

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

        all_results = []

        # Benchmark each device
        for device in devices:
            try:
                results = benchmark_device(filepath, device, args.runs)
                print_results(results)
                all_results.append(results)
            except Exception as e:
                print(f"\nFailed to benchmark on {device}: {e}")

        # Print comparison if we have multiple devices
        if len(all_results) > 1:
            print_device_comparison(all_results)

    print()
