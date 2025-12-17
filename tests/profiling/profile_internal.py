"""
Performance profiling for internal coordinate conversions.

Benchmarks forward and backward passes for conversion between Cartesian
and internal coordinates (Z-matrix representation) on CPU and GPU.

Usage:
    python tests/profiling/profile_internal.py
    python tests/profiling/profile_internal.py --structure 1ZEW
    python tests/profiling/profile_internal.py --all
"""

import os
import sys
import warnings
from dataclasses import dataclass
from typing import Callable

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Handle both direct execution and module import
try:
    from .timing import (
        Timer,
        TimingResult,
        get_sync_fn,
        get_available_devices,
        DEFAULT_RUNS,
    )
except ImportError:
    # Add parent directory to path for direct execution
    sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
    from timing import (
        Timer,
        TimingResult,
        get_sync_fn,
        get_available_devices,
        DEFAULT_RUNS,
    )

# Get test data directory
TEST_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_DIR = os.path.join(TEST_DIR, "data")

# Default benchmark parameters
BENCHMARK_RUNS = DEFAULT_RUNS


@dataclass
class BenchmarkResult:
    """Results for a single operation (forward + optional backward)."""
    name: str
    forward: TimingResult
    backward: TimingResult | None = None


# =============================================================================
# Benchmark Functions
# =============================================================================

def benchmark_to_internal(
    polymer,
    original_coords,
    sync: Callable[[], None],
    runs: int,
    include_backward: bool = True,
) -> BenchmarkResult:
    """
    Benchmark Cartesian -> Internal conversion (forward and backward).

    Args:
        polymer: Polymer object.
        original_coords: Original coordinates tensor.
        sync: Device synchronization function.
        runs: Number of benchmark runs.
        include_backward: Whether to benchmark backward pass.

    Returns:
        BenchmarkResult with forward and optionally backward timings.
    """
    import torch

    # Forward pass benchmark
    def forward():
        polymer.coordinates = original_coords.clone()
        return polymer.dihedrals

    forward_result = Timer.benchmark(forward, sync=sync, runs=runs)

    # Backward pass benchmark (if requested)
    backward_result = None
    if include_backward:
        # Measure forward+backward together, then subtract forward time
        # This avoids graph caching issues
        def forward_backward():
            coords = original_coords.clone().requires_grad_(True)
            polymer.coordinates = coords
            dihedrals = polymer.dihedrals
            loss = dihedrals.sum()
            loss.backward()
            return coords.grad

        fwd_bwd_result = Timer.benchmark(forward_backward, sync=sync, runs=runs)

        # Backward time = total - forward
        backward_result = TimingResult(
            mean=max(0, fwd_bwd_result.mean - forward_result.mean),
            std=(fwd_bwd_result.std**2 + forward_result.std**2)**0.5,
            min=max(0, fwd_bwd_result.min - forward_result.max),
            max=fwd_bwd_result.max - forward_result.min,
            runs=runs,
        )

    return BenchmarkResult("to_internal", forward_result, backward_result)


def benchmark_to_cartesian(
    polymer,
    original_coords,
    sync: Callable[[], None],
    runs: int,
    include_backward: bool = True,
) -> BenchmarkResult:
    """
    Benchmark Internal -> Cartesian conversion (forward and backward).

    Args:
        polymer: Polymer object.
        original_coords: Original coordinates (to reset state between runs).
        sync: Device synchronization function.
        runs: Number of benchmark runs.
        include_backward: Whether to benchmark backward pass.

    Returns:
        BenchmarkResult with forward and optionally backward timings.
    """
    import torch

    # Get base internal coordinates (all three are needed for reconstruction)
    # Use .detach() to ensure no orphaned grad_fn from previous computations
    base_distances = polymer.distances.detach().clone()
    base_angles = polymer.angles.detach().clone()
    base_dihedrals = polymer.dihedrals.detach().clone()

    # Forward pass benchmark: set internal coords then get cartesian
    def forward():
        # Setting dihedrals marks cartesian as dirty; accessing coordinates triggers NERF
        polymer.dihedrals = base_dihedrals.clone()
        return polymer.coordinates

    forward_result = Timer.benchmark(forward, sync=sync, runs=runs)

    # Backward pass benchmark (if requested)
    backward_result = None
    if include_backward:
        # Measure forward+backward together, then subtract forward time
        def forward_backward():
            # detach() ensures no connection to previous computation graphs
            dihedrals = base_dihedrals.detach().clone().requires_grad_(True)
            polymer.dihedrals = dihedrals
            coords = polymer.coordinates
            loss = coords.sum()
            loss.backward()
            return dihedrals.grad

        fwd_bwd_result = Timer.benchmark(forward_backward, sync=sync, runs=runs)

        # Backward time = total - forward
        backward_result = TimingResult(
            mean=max(0, fwd_bwd_result.mean - forward_result.mean),
            std=(fwd_bwd_result.std**2 + forward_result.std**2)**0.5,
            min=max(0, fwd_bwd_result.min - forward_result.max),
            max=fwd_bwd_result.max - forward_result.min,
            runs=runs,
        )

    return BenchmarkResult("to_cartesian", forward_result, backward_result)


def benchmark_round_trip(
    polymer,
    original_coords,
    sync: Callable[[], None],
    runs: int,
    include_backward: bool = True,
) -> BenchmarkResult:
    """
    Benchmark full round-trip: Cartesian -> Internal -> Cartesian.

    Args:
        polymer: Polymer object.
        original_coords: Original coordinates tensor.
        sync: Device synchronization function.
        runs: Number of benchmark runs.
        include_backward: Whether to benchmark backward pass.

    Returns:
        BenchmarkResult with forward and optionally backward timings.
    """
    def forward():
        # Cartesian -> Internal
        polymer.coordinates = original_coords.clone()
        dihedrals = polymer.dihedrals.clone()
        # Internal -> Cartesian
        polymer.dihedrals = dihedrals
        return polymer.coordinates

    forward_result = Timer.benchmark(forward, sync=sync, runs=runs)

    # Backward pass benchmark (if requested)
    backward_result = None
    if include_backward:
        def forward_backward():
            coords = original_coords.detach().clone().requires_grad_(True)
            polymer.coordinates = coords          # Dirty internal
            dihedrals = polymer.dihedrals         # Recompute (grad flows from coords)
            polymer.dihedrals = dihedrals         # Dirty cartesian
            new_coords = polymer.coordinates      # NERF (grad flows from dihedrals)
            loss = new_coords.sum()
            loss.backward()                       # Flows: NERF → cartesian_to_internal
            return coords.grad

        fwd_bwd_result = Timer.benchmark(forward_backward, sync=sync, runs=runs)

        # Backward time = total - forward
        backward_result = TimingResult(
            mean=max(0, fwd_bwd_result.mean - forward_result.mean),
            std=(fwd_bwd_result.std**2 + forward_result.std**2)**0.5,
            min=max(0, fwd_bwd_result.min - forward_result.max),
            max=fwd_bwd_result.max - forward_result.min,
            runs=runs,
        )

    return BenchmarkResult("round_trip", forward_result, backward_result)


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

    sync = get_sync_fn(device)

    # Initialize Z-matrix by triggering first computation
    _ = polymer.dihedrals
    zmatrix = polymer._coord_manager.zmatrix
    results["zmatrix_size"] = len(zmatrix)
    results["n_components"] = polymer._coord_manager._components.n_components

    # Cache original coordinates
    original_coords = polymer.coordinates.clone()

    # Run benchmarks
    results["to_internal"] = benchmark_to_internal(
        polymer, original_coords, sync, runs, include_backward=True
    )

    # Detach cached tensors after to_internal backward to prevent orphaned grad_fn
    # from affecting to_cartesian benchmark
    polymer.detach()

    results["to_cartesian"] = benchmark_to_cartesian(
        polymer, original_coords, sync, runs, include_backward=True
    )

    # Detach before round_trip benchmark
    polymer.detach()

    results["round_trip"] = benchmark_round_trip(
        polymer, original_coords, sync, runs, include_backward=True
    )

    return results


# =============================================================================
# Output Formatting
# =============================================================================

def print_results(results: dict) -> None:
    """Pretty-print benchmark results."""
    print(f"\n{'='*80}")
    print(f"Structure: {results['file']} | Device: {results['device']}")
    print(f"{'='*80}")
    print(f"  Atoms: {results['atoms']:,} | Residues: {results.get('residues', '?'):,} | "
          f"Chains: {results.get('chains', '?')}")
    if 'zmatrix_size' in results:
        print(f"  Z-matrix entries: {results['zmatrix_size']:,} | "
              f"Components: {results['n_components']:,}")
    print()

    # Print timing table
    print(f"  {'Operation':<16} {'Forward':>14} {'Backward':>14}")
    print(f"  {'-'*16} {'-'*14} {'-'*14}")

    for op_name in ["to_internal", "to_cartesian", "round_trip"]:
        if op_name not in results:
            continue
        bench: BenchmarkResult = results[op_name]
        fwd_str = f"{bench.forward.mean*1000:>10.2f}ms"
        if bench.backward:
            bwd_str = f"{bench.backward.mean*1000:>10.2f}ms"
        else:
            bwd_str = f"{'N/A':>12}"
        print(f"  {bench.name:<16} {fwd_str:>14} {bwd_str:>14}")

    # Print throughput
    if "to_internal" in results and results["atoms"] > 0:
        atoms = results["atoms"]
        print()
        print(f"  Throughput (forward):")
        for op_name in ["to_internal", "to_cartesian"]:
            if op_name in results:
                bench: BenchmarkResult = results[op_name]
                ms = bench.forward.mean * 1000
                print(f"    {op_name:<16} {atoms / ms * 1000:>12,.0f} atoms/sec")

        # Backward throughput
        print(f"  Throughput (backward):")
        for op_name in ["to_internal", "to_cartesian"]:
            if op_name in results:
                bench: BenchmarkResult = results[op_name]
                if bench.backward:
                    ms = bench.backward.mean * 1000
                    print(f"    {op_name:<16} {atoms / ms * 1000:>12,.0f} atoms/sec")


def print_device_comparison(all_results: list[dict]) -> None:
    """Print comparison between devices."""
    if len(all_results) < 2:
        return

    print(f"\n{'='*80}")
    print("Device Comparison (Forward Pass)")
    print(f"{'='*80}")

    devices = [r["device"] for r in all_results]

    # Header
    header = f"  {'Operation':<16}"
    for device in devices:
        header += f" {device:>12}"
    print(header)

    divider = f"  {'-'*16}"
    for _ in devices:
        divider += f" {'-'*12}"
    print(divider)

    # Forward pass rows
    for op_name in ["to_internal", "to_cartesian", "round_trip"]:
        row = f"  {op_name:<16}"
        for r in all_results:
            if op_name in r:
                bench: BenchmarkResult = r[op_name]
                ms = bench.forward.mean * 1000
                row += f" {ms:>10.2f}ms"
            else:
                row += f" {'N/A':>12}"
        print(row)

    # Backward pass comparison
    print()
    print("Backward Pass:")
    print(divider)
    for op_name in ["to_internal", "to_cartesian", "round_trip"]:
        row = f"  {op_name:<16}"
        for r in all_results:
            if op_name in r:
                bench: BenchmarkResult = r[op_name]
                if bench.backward:
                    ms = bench.backward.mean * 1000
                    row += f" {ms:>10.2f}ms"
                else:
                    row += f" {'N/A':>12}"
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
            print(f"    {r['device']}:")
            for op_name in ["to_internal", "to_cartesian", "round_trip"]:
                if op_name not in r or op_name not in cpu_results:
                    continue
                cpu_bench: BenchmarkResult = cpu_results[op_name]
                dev_bench: BenchmarkResult = r[op_name]

                # Forward speedup
                cpu_fwd = cpu_bench.forward.mean
                dev_fwd = dev_bench.forward.mean
                ratio_fwd = cpu_fwd / dev_fwd if dev_fwd > 0 else 0
                faster_fwd = "faster" if ratio_fwd > 1 else "slower"

                # Backward speedup
                if cpu_bench.backward and dev_bench.backward:
                    cpu_bwd = cpu_bench.backward.mean
                    dev_bwd = dev_bench.backward.mean
                    ratio_bwd = cpu_bwd / dev_bwd if dev_bwd > 0 else 0
                    faster_bwd = "faster" if ratio_bwd > 1 else "slower"
                    print(f"      {op_name}: fwd {abs(ratio_fwd):.2f}x {faster_fwd}, "
                          f"bwd {abs(ratio_bwd):.2f}x {faster_bwd}")
                else:
                    print(f"      {op_name}: fwd {abs(ratio_fwd):.2f}x {faster_fwd}")


def get_test_file(name: str) -> str:
    """Get path to test CIF file."""
    path = os.path.join(DATA_DIR, f"{name}.cif")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Test file not found: {path}")
    return path


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import argparse
    import ciffy

    parser = argparse.ArgumentParser(
        description="Benchmark internal coordinate conversions (forward and backward)"
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
    print("=" * 80)
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
                import traceback
                traceback.print_exc()

        # Print comparison if we have multiple devices
        if len(all_results) > 1:
            print_device_comparison(all_results)

    print()
