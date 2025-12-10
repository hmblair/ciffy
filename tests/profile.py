"""
Performance profiling for ciffy CIF parser.

Compares ciffy vs BioPython and Biotite parsing performance.

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


def _biotite_load(file: str):
    """Load structure using Biotite."""
    from biotite.structure.io import load_structure
    return load_structure(file)


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

    # Define loader functions
    def load_ciffy():
        return ciffy.load(filepath, backend="numpy")

    def load_biopython():
        return _bio_get_coords(pdb_id, filepath)

    def load_biotite():
        return _biotite_load(filepath)

    # Check which libraries are available
    has_biopython = True
    has_biotite = True
    try:
        load_biopython()
    except ImportError:
        has_biopython = False
    try:
        load_biotite()
    except ImportError:
        has_biotite = False

    # Equal warmup for all: 3 runs each to stabilize file cache and JIT
    for _ in range(3):
        load_ciffy()
    if has_biopython:
        for _ in range(3):
            load_biopython()
    if has_biotite:
        for _ in range(3):
            load_biotite()

    # Benchmark each (file is now equally cached for all)
    mean, std = _benchmark(load_ciffy, runs)
    results["ciffy"] = {"mean": mean, "std": std}

    if has_biopython:
        mean, std = _benchmark(load_biopython, runs)
        results["biopython"] = {"mean": mean, "std": std}
    else:
        results["biopython"] = None

    if has_biotite:
        mean, std = _benchmark(load_biotite, runs)
        results["biotite"] = {"mean": mean, "std": std}
    else:
        results["biotite"] = None

    # Load once to get atom count
    poly = load_ciffy()
    results["atoms"] = poly.size()

    return results


def print_results(results: dict) -> None:
    """Pretty-print benchmark results."""
    print(f"\n{'='*60}")
    print(f"PDB: {results['pdb_id']} ({results['atoms']} atoms)")
    print(f"{'='*60}")

    c = results["ciffy"]
    print(f"ciffy:       {c['mean']*1000:7.2f} ms ± {c['std']*1000:.2f} ms")

    if results["biopython"]:
        bp = results["biopython"]
        print(f"BioPython:   {bp['mean']*1000:7.2f} ms ± {bp['std']*1000:.2f} ms")
        speedup = bp["mean"] / c["mean"]
        print(f"  → {speedup:.1f}x faster than BioPython")
    else:
        print("BioPython:   (not installed)")

    if results["biotite"]:
        bt = results["biotite"]
        print(f"Biotite:     {bt['mean']*1000:7.2f} ms ± {bt['std']*1000:.2f} ms")
        speedup = bt["mean"] / c["mean"]
        print(f"  → {speedup:.1f}x faster than Biotite")
    else:
        print("Biotite:     (not installed)")


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
        assert results["ciffy"]["mean"] > 0

        # If BioPython is available, ciffy should be faster
        if results["biopython"]:
            assert results["ciffy"]["mean"] < results["biopython"]["mean"], \
                "ciffy should be faster than BioPython"


# ─────────────────────────────────────────────────────────────────────────────
# Direct Execution
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import ciffy
    print("ciffy Performance Benchmark")
    print("="*60)
    print(f"ciffy version: {ciffy.__version__}")

    for pdb_id, filepath in TEST_FILES:
        if os.path.exists(filepath):
            results = benchmark_file(pdb_id, filepath)
            print_results(results)
        else:
            print(f"\nSkipping {pdb_id}: file not found")

    print()
