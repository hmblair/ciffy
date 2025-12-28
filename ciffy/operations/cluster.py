"""
Sequence-based clustering using MMseqs2.

Clusters structures by sequence identity and returns cluster representatives.
Useful for reducing redundancy in datasets or creating non-redundant test sets.

Example:
    >>> from ciffy.operations.cluster import cluster
    >>> from pathlib import Path
    >>>
    >>> # Cluster all CIF files at 50% sequence identity
    >>> paths = list(Path("data/").glob("*.cif"))
    >>> result = cluster(paths, threshold=0.5)
    >>>
    >>> # Get representative structures (one per cluster)
    >>> for rep in result.representatives:
    ...     print(rep)
    >>>
    >>> # Get all structures in a specific cluster
    >>> cluster_0_members = result.get_cluster(0)
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


def _find_mmseqs() -> str | None:
    """Find the mmseqs binary."""
    candidates = [
        shutil.which("mmseqs"),
        "/Users/hmblair/mambaforge/envs/ciffy/bin/mmseqs",
        "/Users/hmblair/mambaforge/bin/mmseqs",
        "/usr/local/bin/mmseqs",
    ]
    for path in candidates:
        if path and Path(path).exists():
            return path
    return None


@dataclass
class ClusterResult:
    """
    Result of sequence-based clustering.

    Attributes:
        paths: Original input paths in order.
        labels: Cluster label for each structure (0-indexed).
        representatives: Paths to cluster representative structures.
        threshold: Sequence identity threshold used for clustering.
    """

    paths: list[Path]
    labels: np.ndarray
    representatives: list[Path]
    threshold: float = 0.5

    @property
    def n_clusters(self) -> int:
        """Number of clusters."""
        return len(self.representatives)

    @property
    def n_structures(self) -> int:
        """Total number of structures."""
        return len(self.paths)

    def get_cluster(self, label: int) -> list[Path]:
        """Get all paths in a specific cluster."""
        return [p for p, lbl in zip(self.paths, self.labels) if lbl == label]

    def cluster_sizes(self) -> dict[int, int]:
        """Get size of each cluster."""
        unique, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))

    def summary(self) -> str:
        """Return a summary string."""
        sizes = self.cluster_sizes()
        size_dist = sorted(sizes.values(), reverse=True)
        top_5 = size_dist[:5]
        return (
            f"ClusterResult(\n"
            f"  structures={self.n_structures},\n"
            f"  clusters={self.n_clusters},\n"
            f"  threshold={self.threshold},\n"
            f"  largest_clusters={top_5}\n"
            f")"
        )

    def __repr__(self) -> str:
        return self.summary()


def _extract_sequences(paths: list[Path]) -> dict[str, tuple[Path, str]]:
    """
    Extract sequences from structure files.

    Returns dict mapping sequence ID to (path, sequence).
    Uses chain index to create unique IDs when multiple chains exist.
    """
    import ciffy

    sequences = {}
    for path in paths:
        try:
            polymer = ciffy.load(str(path))
            # Get polymer chains only
            polymer = polymer.poly()
            if polymer.size() == 0:
                continue

            # Extract sequence for each chain
            for i, chain in enumerate(polymer.chains()):
                seq = chain.sequence_str()
                if seq and len(seq) >= 3:  # Skip very short sequences
                    # Use path stem + chain index as unique ID
                    seq_id = f"{path.stem}_chain{i}"
                    sequences[seq_id] = (path, seq)
        except Exception:
            # Skip files that can't be loaded
            continue

    return sequences


def _run_mmseqs_cluster(
    sequences: dict[str, tuple[Path, str]],
    mmseqs_bin: str,
    threshold: float,
    threads: int = 4,
    coverage: float = 0.8,
) -> dict[str, str]:
    """
    Run mmseqs2 clustering and return cluster assignments.

    Args:
        sequences: Dict mapping seq_id to (path, sequence).
        mmseqs_bin: Path to mmseqs binary.
        threshold: Minimum sequence identity (0-1).
        threads: Number of threads.
        coverage: Minimum alignment coverage (0-1).

    Returns:
        Dict mapping seq_id to representative seq_id.
    """
    if len(sequences) < 2:
        # Single sequence is its own cluster
        if sequences:
            seq_id = list(sequences.keys())[0]
            return {seq_id: seq_id}
        return {}

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        fasta_path = tmp / "sequences.fasta"
        result_prefix = tmp / "result"
        tmp_path = tmp / "tmp"

        # Write sequences to FASTA
        with open(fasta_path, "w") as f:
            for seq_id, (path, seq) in sequences.items():
                f.write(f">{seq_id}\n{seq}\n")

        # Run mmseqs easy-cluster
        cmd = [
            mmseqs_bin, "easy-cluster",
            str(fasta_path),
            str(result_prefix),
            str(tmp_path),
            "--min-seq-id", str(threshold),
            "-c", str(coverage),
            "--threads", str(threads),
            "-v", "0",  # Quiet
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"mmseqs easy-cluster failed:\n{result.stderr}"
            )

        # Parse cluster assignments from TSV
        # Format: representative_id\tmember_id
        cluster_tsv = tmp / "result_cluster.tsv"
        assignments = {}

        if cluster_tsv.exists():
            with open(cluster_tsv) as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        rep_id, member_id = parts[0], parts[1]
                        assignments[member_id] = rep_id

        return assignments


def _build_structure_graph(
    sequences: dict[str, tuple[Path, str]],
    assignments: dict[str, str],
) -> dict[Path, set[Path]]:
    """
    Build a graph where structures are connected if any of their chains
    cluster together.

    Args:
        sequences: Dict mapping seq_id to (path, sequence).
        assignments: Dict mapping seq_id to representative seq_id.

    Returns:
        Adjacency dict: path -> set of connected paths.
    """
    # Group sequences by their chain-level cluster representative
    rep_to_paths: dict[str, set[Path]] = {}
    for seq_id, rep_id in assignments.items():
        if seq_id in sequences:
            path = sequences[seq_id][0]
            if rep_id not in rep_to_paths:
                rep_to_paths[rep_id] = set()
            rep_to_paths[rep_id].add(path)

    # Build adjacency: structures are connected if they share a chain cluster
    all_paths = set(path for path, _ in sequences.values())
    adjacency: dict[Path, set[Path]] = {p: set() for p in all_paths}

    for paths_in_cluster in rep_to_paths.values():
        # All structures in this chain cluster are connected
        path_list = list(paths_in_cluster)
        for i, p1 in enumerate(path_list):
            for p2 in path_list[i + 1:]:
                adjacency[p1].add(p2)
                adjacency[p2].add(p1)

    return adjacency


def _find_connected_components(adjacency: dict[Path, set[Path]]) -> list[list[Path]]:
    """
    Find connected components in the structure graph using BFS.

    Args:
        adjacency: Adjacency dict from _build_structure_graph.

    Returns:
        List of components, each component is a list of paths.
    """
    visited: set[Path] = set()
    components: list[list[Path]] = []

    for start in adjacency:
        if start in visited:
            continue

        # BFS from this node
        component: list[Path] = []
        queue = [start]
        visited.add(start)

        while queue:
            node = queue.pop(0)
            component.append(node)

            for neighbor in adjacency[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        components.append(sorted(component, key=lambda p: p.name))

    return components


def _select_representative(paths: list[Path], sequences: dict[str, tuple[Path, str]]) -> Path:
    """
    Select a representative structure from a cluster.

    Chooses the structure with the longest total sequence length.
    """
    def total_seq_length(path: Path) -> int:
        total = 0
        for seq_id, (p, seq) in sequences.items():
            if p == path:
                total += len(seq)
        return total

    return max(paths, key=total_seq_length)


def cluster(
    paths: Sequence[str | Path],
    threshold: float = 0.5,
    threads: int = 4,
    coverage: float = 0.8,
    mmseqs_path: str | None = None,
) -> ClusterResult:
    """
    Cluster structures by sequence identity using MMseqs2.

    Two structures are in the same cluster if ANY chain in one has
    sequence identity >= threshold with ANY chain in the other. This
    transitive clustering ensures no homologous chains appear in
    different clusters, making it safe for ML train/test splitting.

    Args:
        paths: Paths to CIF/PDB files to cluster.
        threshold: Sequence identity threshold for clustering (default: 0.5).
            Structures with sequence identity >= threshold are grouped together.
            Common thresholds:
            - 0.3: Remote homologs
            - 0.5: Same family
            - 0.7: High similarity
            - 0.9: Near-identical sequences
        threads: Number of threads for mmseqs (default: 4).
        coverage: Minimum alignment coverage (default: 0.8).
        mmseqs_path: Path to mmseqs binary. Auto-detected if None.

    Returns:
        ClusterResult with cluster labels and representatives.

    Raises:
        RuntimeError: If mmseqs is not installed or fails.
        ValueError: If no valid paths provided.

    Example:
        >>> result = cluster(paths, threshold=0.5)
        >>> print(f"Found {result.n_clusters} clusters")
        >>> for rep in result.representatives:
        ...     print(f"Representative: {rep}")
    """
    # Convert to Path objects and validate
    path_list = [Path(p) for p in paths]
    valid_paths = [p for p in path_list if p.exists()]

    if not valid_paths:
        raise ValueError("No valid structure files provided")

    if len(valid_paths) != len(path_list):
        missing = len(path_list) - len(valid_paths)
        import warnings
        warnings.warn(f"Skipping {missing} missing files")

    # Find mmseqs
    mmseqs_bin = mmseqs_path or _find_mmseqs()
    if mmseqs_bin is None:
        raise RuntimeError(
            "mmseqs not found. Install with:\n"
            "  mamba install -c conda-forge -c bioconda mmseqs2\n"
            "Or download from: https://github.com/soedinglab/MMseqs2"
        )

    # Extract sequences from structures
    sequences = _extract_sequences(valid_paths)

    if not sequences:
        raise ValueError("No valid sequences found in structure files")

    # Get paths that have sequences
    paths_with_seqs = list(set(path for path, _ in sequences.values()))

    # Special case: single structure
    if len(paths_with_seqs) == 1:
        path = paths_with_seqs[0]
        return ClusterResult(
            paths=[path],
            labels=np.array([0]),
            representatives=[path],
            threshold=threshold,
        )

    # Run mmseqs clustering on all chains
    assignments = _run_mmseqs_cluster(
        sequences, mmseqs_bin, threshold, threads, coverage
    )

    # Build structure-level graph based on chain clustering
    adjacency = _build_structure_graph(sequences, assignments)

    # Find connected components (each component = one structure cluster)
    components = _find_connected_components(adjacency)

    # Build result
    clustered_paths: list[Path] = []
    labels_list: list[int] = []
    representatives: list[Path] = []

    for label, component in enumerate(components):
        rep = _select_representative(component, sequences)
        representatives.append(rep)

        for path in component:
            clustered_paths.append(path)
            labels_list.append(label)

    # Sort by path to maintain consistent ordering
    sorted_indices = sorted(range(len(clustered_paths)), key=lambda i: clustered_paths[i].name)
    clustered_paths = [clustered_paths[i] for i in sorted_indices]
    labels = np.array([labels_list[i] for i in sorted_indices])

    return ClusterResult(
        paths=clustered_paths,
        labels=labels,
        representatives=representatives,
        threshold=threshold,
    )


def cluster_representatives(
    paths: Sequence[str | Path],
    threshold: float = 0.5,
    threads: int = 4,
    coverage: float = 0.8,
    mmseqs_path: str | None = None,
) -> list[Path]:
    """
    Convenience function to get just the representative structures.

    Args:
        paths: Paths to CIF/PDB files to cluster.
        threshold: Sequence identity threshold for clustering (default: 0.5).
        threads: Number of threads for mmseqs (default: 4).
        coverage: Minimum alignment coverage (default: 0.8).
        mmseqs_path: Path to mmseqs binary. Auto-detected if None.

    Returns:
        List of representative structure paths (one per cluster).

    Example:
        >>> reps = cluster_representatives(paths, threshold=0.7)
        >>> print(f"Reduced {len(paths)} structures to {len(reps)} representatives")
    """
    result = cluster(
        paths,
        threshold=threshold,
        threads=threads,
        coverage=coverage,
        mmseqs_path=mmseqs_path,
    )
    return result.representatives


__all__ = [
    "cluster",
    "cluster_representatives",
    "ClusterResult",
]
