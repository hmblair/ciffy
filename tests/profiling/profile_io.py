"""
Performance profiling for ciffy CIF parser.

Compares ciffy vs BioPython and Biotite parsing performance.

Usage:
    python tests/profiling/profile_io.py
    python tests/profiling/profile_io.py --markdown
    python tests/profiling/profile_io.py --ciffy-only
    python tests/profiling/profile_io.py --dataset /path/to/cif/files
"""

import glob
import os
import sys
import warnings

import numpy as np

# Suppress deprecation warnings during benchmarking
warnings.filterwarnings("ignore", category=DeprecationWarning, module="ciffy")

# Handle both direct execution and module import
try:
    from .timing import Timer, TimingResult, DEFAULT_RUNS
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
    from timing import Timer, TimingResult, DEFAULT_RUNS

# Get test directory (parent of profiling/)
TEST_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_DIR = os.path.join(TEST_DIR, "data")

# Find all CIF files in data directory
TEST_FILES = [
    (os.path.splitext(os.path.basename(f))[0], f)
    for f in sorted(glob.glob(os.path.join(DATA_DIR, "*.cif")))
]

# Number of iterations for benchmarking
BENCHMARK_RUNS = DEFAULT_RUNS


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




def benchmark_file(pdb_id: str, filepath: str, runs: int = BENCHMARK_RUNS,
                   ciffy_only: bool = False) -> tuple:
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

    # Benchmark each (Timer.benchmark handles warmup internally)
    result = Timer.benchmark(load_ciffy, runs=runs)
    results["ciffy"] = result.to_dict()

    if has_biopython:
        result = Timer.benchmark(load_biopython, runs=runs)
        results["biopython"] = result.to_dict()
    else:
        results["biopython"] = None

    if has_biotite:
        result = Timer.benchmark(load_biotite, runs=runs)
        results["biotite"] = result.to_dict()
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
# PolymerDataset Benchmarking
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_dataset_scan(directory: str, num_workers_list: list[int],
                           runs: int = 3) -> dict:
    """
    Benchmark PolymerDataset index building with different worker counts.

    Args:
        directory: Path to directory with CIF files.
        num_workers_list: List of worker counts to test (0 = sequential).
        runs: Number of runs per configuration.

    Returns:
        Dict mapping num_workers -> (mean_time, std_time, num_items).
    """
    from ciffy.nn import PolymerDataset
    from ciffy import Scale

    results = {}

    for num_workers in num_workers_list:
        times = []
        num_items = 0

        for _ in range(runs):
            with Timer() as t:
                dataset = PolymerDataset(directory, scale=Scale.CHAIN, num_workers=num_workers)
            times.append(t.elapsed)
            num_items = len(dataset)

        results[num_workers] = {
            "mean": float(np.mean(times)),
            "std": float(np.std(times)),
            "items": num_items,
        }

    return results


def print_dataset_benchmark(results: dict, directory: str) -> None:
    """Print dataset benchmark results."""
    print(f"\n{'='*60}")
    print(f"PolymerDataset Index Scan: {directory}")
    print(f"{'='*60}")

    # Get baseline (sequential) time
    baseline = results.get(0, {}).get("mean", 1.0)
    items = results.get(0, {}).get("items", 0)

    print(f"Items indexed: {items}")
    print()
    print(f"{'Workers':<10} {'Time':<15} {'Speedup':<10}")
    print("-" * 35)

    for num_workers in sorted(results.keys()):
        r = results[num_workers]
        speedup = baseline / r["mean"] if r["mean"] > 0 else 0
        worker_str = "sequential" if num_workers == 0 else str(num_workers)
        print(f"{worker_str:<10} {r['mean']*1000:>8.1f} ms    {speedup:>5.2f}x")


# ─────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import ciffy

    parser = argparse.ArgumentParser(description="ciffy performance benchmark")
    parser.add_argument("--markdown", action="store_true", help="Output markdown table")
    parser.add_argument("--ciffy-only", action="store_true",
                        help="Only benchmark ciffy (skip biopython/biotite)")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Benchmark PolymerDataset scanning on this directory")
    parser.add_argument("--max-workers", type=int, default=8,
                        help="Maximum workers to test for dataset benchmark")
    args = parser.parse_args()

    # Check if profiling is enabled
    has_profiling = hasattr(ciffy, '_get_profile')

    # Dataset benchmark mode
    if args.dataset:
        print("PolymerDataset Scan Benchmark")
        print("=" * 60)
        print(f"ciffy version: {ciffy.__version__}")

        # Test sequential + various worker counts
        worker_counts = [0] + list(range(1, args.max_workers + 1))
        results = benchmark_dataset_scan(args.dataset, worker_counts)
        print_dataset_benchmark(results, args.dataset)
        exit(0)

    # Standard file parsing benchmark
    if not TEST_FILES:
        print("No CIF files found in tests/data/")
        print("Run some tests first to download test structures.")
        exit(1)

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
