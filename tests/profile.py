"""
Performance profiling for ciffy CIF parser.

Compares single-threaded, multi-threaded (OpenMP), and BioPython parsing.

Usage:
    python -m pytest tests/profile.py -v -s
    python tests/profile.py  # Direct execution
"""

import glob
import os
import time
import warnings
import numpy as np
import pytest

# Suppress deprecation warnings during benchmarking
warnings.filterwarnings("ignore", category=DeprecationWarning, module="ciffy")

# Get test directory
TEST_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(TEST_DIR, "data")

# Find all CIF files in data directory
TEST_FILES = [
    (os.path.splitext(os.path.basename(f))[0], f)
    for f in sorted(glob.glob(os.path.join(DATA_DIR, "*.cif")))
]

# Number of iterations for benchmarking
BENCHMARK_RUNS = 10


def _set_omp_threads(n: int) -> None:
    """Set OpenMP thread count via environment variable."""
    os.environ["OMP_NUM_THREADS"] = str(n)


def _get_omp_threads() -> int:
    """Get current OpenMP thread count."""
    return int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 1))


def _bio_get_coords(iden: str, file: str) -> np.ndarray:
    """Load coordinates using BioPython's FastMMCIFParser."""
    from Bio.PDB.MMCIFParser import FastMMCIFParser

    parser = FastMMCIFParser(QUIET=True)
    stru = parser.get_structure(iden, file)
    coords = []

    for model in stru:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    coords.append(atom.get_vector()._ar)

    return np.stack(coords, axis=0) if coords else np.array([])


def _benchmark(func, runs: int = BENCHMARK_RUNS) -> tuple[float, float]:
    """
    Run a function multiple times and return timing statistics.

    Returns:
        Tuple of (mean_time, std_time) in seconds.
    """
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        func()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return np.mean(times), np.std(times)


def benchmark_file(pdb_id: str, filepath: str, runs: int = BENCHMARK_RUNS) -> dict:
    """
    Benchmark parsing a single file with all methods.

    Returns:
        Dict with timing results for each method.
    """
    import ciffy

    results = {"pdb_id": pdb_id, "file": filepath}

    # Warmup: multiple runs to populate file cache and stabilize performance
    for _ in range(3):
        ciffy.load(filepath, backend="numpy")

    # Single-threaded ciffy
    _set_omp_threads(1)
    mean, std = _benchmark(lambda: ciffy.load(filepath, backend="numpy"), runs)
    results["ciffy_1thread"] = {"mean": mean, "std": std}

    # Multi-threaded ciffy (use all cores)
    num_cores = os.cpu_count() or 4
    _set_omp_threads(num_cores)
    mean, std = _benchmark(lambda: ciffy.load(filepath, backend="numpy"), runs)
    results["ciffy_multithread"] = {"mean": mean, "std": std, "threads": num_cores}

    # BioPython
    try:
        # Warmup BioPython too
        _bio_get_coords(pdb_id, filepath)
        mean, std = _benchmark(lambda: _bio_get_coords(pdb_id, filepath), runs)
        results["biopython"] = {"mean": mean, "std": std}
    except ImportError:
        results["biopython"] = None

    # Load once to get atom count
    poly = ciffy.load(filepath, backend="numpy")
    results["atoms"] = poly.size()

    return results


def print_results(results: dict) -> None:
    """Pretty-print benchmark results."""
    print(f"\n{'='*60}")
    print(f"PDB: {results['pdb_id']} ({results['atoms']} atoms)")
    print(f"{'='*60}")

    c1 = results["ciffy_1thread"]
    print(f"ciffy (1 thread):    {c1['mean']*1000:7.2f} ms ± {c1['std']*1000:.2f} ms")

    cm = results["ciffy_multithread"]
    print(f"ciffy ({cm['threads']} threads):   {cm['mean']*1000:7.2f} ms ± {cm['std']*1000:.2f} ms")

    # Speedup from parallelization
    parallel_speedup = c1["mean"] / cm["mean"]
    print(f"  → Parallel speedup: {parallel_speedup:.2f}x")

    if results["biopython"]:
        bp = results["biopython"]
        print(f"BioPython:           {bp['mean']*1000:7.2f} ms ± {bp['std']*1000:.2f} ms")

        # Speedup vs BioPython
        bp_speedup_1t = bp["mean"] / c1["mean"]
        bp_speedup_mt = bp["mean"] / cm["mean"]
        print(f"  → vs BioPython (1T): {bp_speedup_1t:.2f}x faster")
        print(f"  → vs BioPython (MT): {bp_speedup_mt:.2f}x faster")
    else:
        print("BioPython:           (not installed)")


# ─────────────────────────────────────────────────────────────────────────────
# Pytest Integration
# ─────────────────────────────────────────────────────────────────────────────

class TestBenchmark:
    """Benchmark tests for ciffy performance."""

    @pytest.mark.parametrize("pdb_id,filepath", TEST_FILES)
    def test_benchmark(self, pdb_id: str, filepath: str) -> None:
        """Run benchmark and verify ciffy is faster than BioPython."""
        if not os.path.exists(filepath):
            pytest.skip(f"Test file not found: {filepath}")

        results = benchmark_file(pdb_id, filepath, runs=5)
        print_results(results)

        # Basic sanity checks
        assert results["ciffy_1thread"]["mean"] > 0
        assert results["ciffy_multithread"]["mean"] > 0

        # Parallel should be at least as fast as single-threaded
        # (on single-core machines they might be equal)
        assert results["ciffy_multithread"]["mean"] <= results["ciffy_1thread"]["mean"] * 1.1

        # If BioPython is available, ciffy should be faster
        if results["biopython"]:
            assert results["ciffy_1thread"]["mean"] < results["biopython"]["mean"], \
                "ciffy should be faster than BioPython"


# ─────────────────────────────────────────────────────────────────────────────
# Direct Execution
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("ciffy Performance Benchmark")
    print("="*60)

    for pdb_id, filepath in TEST_FILES:
        if os.path.exists(filepath):
            results = benchmark_file(pdb_id, filepath)
            print_results(results)
        else:
            print(f"\nSkipping {pdb_id}: file not found")

    print()
