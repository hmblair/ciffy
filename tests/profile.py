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


def benchmark_file(pdb_id: str, filepath: str, runs: int = BENCHMARK_RUNS,
                   ciffy_only: bool = False) -> tuple[dict, dict | None]:
    """
    Benchmark parsing a single file with all methods.

    Returns:
        Tuple of (results dict, profile dict or None if profiling disabled).
    """
    import ciffy

    results = {"pdb_id": pdb_id, "file": filepath}

    # Check if C profiling is available
    has_profiling = hasattr(ciffy, '_get_profile')

    # Define loader functions
    def load_ciffy():
        return ciffy.load(filepath, backend="numpy")

    def load_biopython():
        return _bio_get_coords(pdb_id, filepath)

    def load_biotite():
        return _biotite_load(filepath)

    # Check which libraries are available
    has_biopython = not ciffy_only
    has_biotite = not ciffy_only
    if has_biopython:
        try:
            load_biopython()
        except ImportError:
            has_biopython = False
    if has_biotite:
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

    # Load once to get atom count and profile data
    poly = load_ciffy()
    results["atoms"] = poly.size()

    # Get profile data if available (from the last load)
    profile = None
    if has_profiling:
        profile = ciffy._get_profile()

    return results, profile


def print_profile_breakdown(profile: dict, total_ms: float) -> None:
    """Print per-phase timing breakdown from C profiling."""
    print("  Phase breakdown:")
    phases = [
        ("file_load", "File I/O"),
        ("block_parse", "Block parsing"),
        ("line_precomp", "Line precompute"),
        ("metadata", "Metadata"),
        ("batch_parse", "Batch parsing"),
        ("residue_count", "Residue count"),
        ("py_convert", "Python convert"),
    ]
    accounted = 0.0
    for key, label in phases:
        ms = profile.get(key, 0) * 1000
        pct = (ms / total_ms * 100) if total_ms > 0 else 0
        accounted += ms
        print(f"    {label:16s}: {ms:7.2f} ms ({pct:5.1f}%)")

    # Show batch_parse sub-phases if available
    batch_ms = profile.get("batch_parse", 0) * 1000
    if batch_ms > 0.1:  # Only show sub-phases if batch_parse is significant
        sub_phases = [
            ("batch_coords", "  -> coords"),
            ("batch_elements", "  -> elements"),
            ("batch_types", "  -> types"),
        ]
        sub_total = 0.0
        for key, label in sub_phases:
            ms = profile.get(key, 0) * 1000
            pct = (ms / batch_ms * 100) if batch_ms > 0 else 0
            sub_total += ms
            print(f"    {label:16s}: {ms:7.2f} ms ({pct:5.1f}% of batch)")
        # Show profiling overhead within batch
        batch_overhead = batch_ms - sub_total
        if batch_overhead > 0.01:
            pct = (batch_overhead / batch_ms * 100) if batch_ms > 0 else 0
            print(f"    {'  -> overhead':16s}: {batch_overhead:7.2f} ms ({pct:5.1f}% of batch)")

    overhead = total_ms - accounted
    if overhead > 0.01:  # Only show if significant
        pct = (overhead / total_ms * 100) if total_ms > 0 else 0
        print(f"    {'Unaccounted':16s}: {overhead:7.2f} ms ({pct:5.1f}%)")


def print_results(results: dict, profile: dict = None) -> None:
    """Pretty-print benchmark results."""
    print(f"\n{'='*60}")
    print(f"PDB: {results['pdb_id']} ({results['atoms']} atoms)")
    print(f"{'='*60}")

    c = results["ciffy"]
    print(f"ciffy:       {c['mean']*1000:7.2f} ms ± {c['std']*1000:.2f} ms")

    # Print profile breakdown if available
    if profile is not None:
        print_profile_breakdown(profile, c['mean']*1000)

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


def generate_markdown_table(all_results: list[dict]) -> str:
    """Generate a markdown table from benchmark results."""
    lines = [
        "| Structure | Atoms | ciffy | BioPython | Biotite |",
        "|-----------|------:|------:|----------:|--------:|",
    ]

    for r in all_results:
        c = r["ciffy"]
        ciffy_ms = f"{c['mean']*1000:.2f} ms"

        if r["biopython"]:
            bp = r["biopython"]
            bp_speedup = bp["mean"] / c["mean"]
            biopython_str = f"{bp['mean']*1000:.0f} ms ({bp_speedup:.0f}x)"
        else:
            biopython_str = "—"

        if r["biotite"]:
            bt = r["biotite"]
            bt_speedup = bt["mean"] / c["mean"]
            biotite_str = f"{bt['mean']*1000:.0f} ms ({bt_speedup:.0f}x)"
        else:
            biotite_str = "—"

        lines.append(
            f"| {r['pdb_id']} | {r['atoms']:,} | {ciffy_ms} | {biopython_str} | {biotite_str} |"
        )

    return "\n".join(lines)


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

        results, profile = benchmark_file(pdb_id, filepath, runs=5)
        print_results(results, profile)

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
    import argparse
    import ciffy

    parser = argparse.ArgumentParser(description="ciffy performance benchmark")
    parser.add_argument("--markdown", action="store_true", help="Output markdown table")
    parser.add_argument("--ciffy-only", action="store_true",
                        help="Only benchmark ciffy (skip biopython/biotite)")
    args = parser.parse_args()

    # Check if profiling is enabled
    has_profiling = hasattr(ciffy, '_get_profile')

    all_results = []
    all_profiles = []
    for pdb_id, filepath in TEST_FILES:
        if os.path.exists(filepath):
            results, profile = benchmark_file(pdb_id, filepath, ciffy_only=args.ciffy_only)
            all_results.append(results)
            all_profiles.append(profile)

    if args.markdown:
        print(generate_markdown_table(all_results))
    else:
        print("ciffy Performance Benchmark")
        print("="*60)
        print(f"ciffy version: {ciffy.__version__}")
        if has_profiling:
            print("C profiling: ENABLED")
        else:
            print("C profiling: disabled (rebuild with CIFFY_PROFILE=1)")
        for results, profile in zip(all_results, all_profiles):
            print_results(results, profile)
        print()
