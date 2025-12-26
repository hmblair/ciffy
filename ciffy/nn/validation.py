"""
Pre-training validation utilities.

Validates training configuration before starting to fail fast with helpful
error messages. Designed for molecular biology researchers who may not be
familiar with ML infrastructure.

Example:
    >>> from ciffy.nn.validation import validate_training_config
    >>>
    >>> issues = validate_training_config(
    ...     data_dir="./rna_structures",
    ...     residues=["A", "C", "G", "U"],
    ...     output_dir="./checkpoints",
    ... )
    >>> if issues.has_errors:
    ...     print(issues.format())
    ...     sys.exit(1)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ValidationIssue:
    """A single validation issue."""

    level: str  # 'error', 'warning', 'info'
    message: str
    suggestion: str | None = None

    def format(self) -> str:
        """Format issue for display."""
        icons = {"error": "✗", "warning": "⚠", "info": "ℹ"}
        icon = icons.get(self.level, "•")
        lines = [f"{icon} {self.message}"]
        if self.suggestion:
            lines.append(f"  → {self.suggestion}")
        return "\n".join(lines)


@dataclass
class ValidationResult:
    """Collection of validation issues."""

    issues: list[ValidationIssue] = field(default_factory=list)

    # Statistics collected during validation
    n_structures: int = 0
    structure_files: list[Path] = field(default_factory=list)
    residue_counts: dict[str, int] = field(default_factory=dict)

    @property
    def has_errors(self) -> bool:
        """Check if any errors were found."""
        return any(i.level == "error" for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        """Check if any warnings were found."""
        return any(i.level == "warning" for i in self.issues)

    def add_error(self, message: str, suggestion: str | None = None) -> None:
        """Add an error issue."""
        self.issues.append(ValidationIssue("error", message, suggestion))

    def add_warning(self, message: str, suggestion: str | None = None) -> None:
        """Add a warning issue."""
        self.issues.append(ValidationIssue("warning", message, suggestion))

    def add_info(self, message: str, suggestion: str | None = None) -> None:
        """Add an info message."""
        self.issues.append(ValidationIssue("info", message, suggestion))

    def format(self, show_info: bool = True) -> str:
        """Format all issues for display."""
        lines = []
        for issue in self.issues:
            if issue.level == "info" and not show_info:
                continue
            lines.append(issue.format())
        return "\n".join(lines)

    def format_summary(self) -> str:
        """Format a summary of the validation."""
        lines = []

        # Structure summary
        if self.n_structures > 0:
            lines.append(f"Found {self.n_structures} structure files")
            if self.residue_counts:
                counts = ", ".join(
                    f"{k}: {v:,}" for k, v in sorted(self.residue_counts.items())
                )
                lines.append(f"  Residue counts: {counts}")

        # Issues summary
        n_errors = sum(1 for i in self.issues if i.level == "error")
        n_warnings = sum(1 for i in self.issues if i.level == "warning")

        if n_errors > 0:
            lines.append(f"\n✗ {n_errors} error(s) found - training cannot proceed")
        elif n_warnings > 0:
            lines.append(f"\n⚠ {n_warnings} warning(s) - training will proceed")
        else:
            lines.append("\n✓ Validation passed")

        return "\n".join(lines)


def find_cif_files(directory: Path | str) -> list[Path]:
    """Find all CIF files in a directory (recursive)."""
    directory = Path(directory)
    if not directory.exists():
        return []

    files = []
    # Check immediate directory
    files.extend(directory.glob("*.cif"))
    # Check subdirectories
    files.extend(directory.glob("**/*.cif"))

    # Remove duplicates and sort
    return sorted(set(files))


def find_similar_directories(target: Path, search_root: Path | None = None) -> list[tuple[Path, int]]:
    """Find directories that might contain CIF files.

    Returns list of (path, count) tuples sorted by count descending.
    """
    if search_root is None:
        search_root = target.parent if target.parent.exists() else Path.cwd()

    candidates = []

    # Check siblings and parent directories
    for check_dir in [search_root] + list(search_root.iterdir()):
        if not check_dir.is_dir():
            continue
        if check_dir.name.startswith("."):
            continue

        cif_count = len(list(check_dir.glob("*.cif"))) + len(
            list(check_dir.glob("**/*.cif"))
        )
        if cif_count > 0:
            candidates.append((check_dir, cif_count))

    return sorted(candidates, key=lambda x: -x[1])[:5]


def validate_data_directory(
    data_dir: str | Path | None,
    cif_patterns: list[str] | None = None,
    min_structures: int = 10,
) -> ValidationResult:
    """Validate data directory and CIF file availability.

    Args:
        data_dir: Directory containing CIF files.
        cif_patterns: Optional glob patterns for CIF files.
        min_structures: Minimum recommended structures for training.

    Returns:
        ValidationResult with issues and statistics.
    """
    result = ValidationResult()

    # Check if any data source is specified
    if not data_dir and not cif_patterns:
        result.add_error(
            "No data source specified",
            "Set 'data_dir' to a directory containing .cif files, "
            "or use 'cif_patterns' to specify file patterns",
        )
        return result

    # Collect all CIF files
    cif_files: list[Path] = []

    if data_dir:
        data_path = Path(data_dir)

        if not data_path.exists():
            result.add_error(
                f"Data directory not found: {data_dir}",
            )
            # Try to find similar directories
            similar = find_similar_directories(data_path)
            if similar:
                suggestions = [f"{p} ({n} files)" for p, n in similar[:3]]
                result.issues[-1].suggestion = "Did you mean: " + ", ".join(suggestions)
            else:
                result.issues[-1].suggestion = (
                    "Use 'ciffy download' to fetch RNA structures from RCSB PDB"
                )
            return result

        if not data_path.is_dir():
            result.add_error(
                f"Data path is not a directory: {data_dir}",
                "Provide a directory path, not a file path",
            )
            return result

        cif_files.extend(find_cif_files(data_path))

    if cif_patterns:
        from glob import glob

        for pattern in cif_patterns:
            matches = glob(pattern)
            cif_files.extend(Path(p) for p in matches if p.endswith(".cif"))

    # Remove duplicates
    cif_files = sorted(set(cif_files))
    result.structure_files = cif_files
    result.n_structures = len(cif_files)

    # Check structure count
    if result.n_structures == 0:
        result.add_error(
            f"No .cif files found in {data_dir}",
        )
        # Check for common issues
        if data_dir:
            data_path = Path(data_dir)
            pdb_files = list(data_path.glob("*.pdb")) + list(data_path.glob("**/*.pdb"))
            if pdb_files:
                result.issues[-1].suggestion = (
                    f"Found {len(pdb_files)} .pdb files. Convert to .cif format, "
                    "or download structures with 'ciffy download'"
                )
            else:
                similar = find_similar_directories(data_path)
                if similar:
                    suggestions = [f"{p} ({n} files)" for p, n in similar[:3]]
                    result.issues[-1].suggestion = (
                        "Did you mean: " + ", ".join(suggestions)
                    )
    else:
        result.add_info(f"Found {result.n_structures} structures")

    return result


def validate_output_directory(
    output_dir: str | Path,
    warn_existing: bool = True,
) -> ValidationResult:
    """Validate output directory configuration.

    Args:
        output_dir: Directory for saving checkpoints.
        warn_existing: Warn if directory contains existing checkpoints.

    Returns:
        ValidationResult with issues.
    """
    result = ValidationResult()
    output_path = Path(output_dir)

    # Check if parent is writable
    parent = output_path.parent
    if parent.exists() and not os.access(parent, os.W_OK):
        result.add_error(
            f"Cannot write to output directory: {output_dir}",
            f"Check permissions on {parent}",
        )
        return result

    # Check for existing checkpoints
    if warn_existing and output_path.exists():
        existing_checkpoints = list(output_path.glob("*.pt")) + list(
            output_path.glob("*.pth")
        )
        if existing_checkpoints:
            result.add_warning(
                f"Output directory contains {len(existing_checkpoints)} existing checkpoint(s)",
                "Checkpoints may be overwritten. Use a new directory to preserve them.",
            )

        # Check for config.json (indicates trained model)
        if (output_path / "config.json").exists():
            result.add_warning(
                "Output directory contains a previously trained model",
                "Training will overwrite the existing model.",
            )

    return result


def validate_residues(
    residue_names: list[str] | None,
    available_residues: list[str] | None = None,
) -> ValidationResult:
    """Validate residue configuration.

    Args:
        residue_names: List of residue names to train on.
        available_residues: Optional list of valid residue names.

    Returns:
        ValidationResult with issues.
    """
    result = ValidationResult()

    if not residue_names:
        result.add_error(
            "No residues specified for training",
            "Set 'residues' to a list like ['A', 'C', 'G', 'U'] for RNA",
        )
        return result

    # Check for valid residue names
    if available_residues:
        invalid = [r for r in residue_names if r not in available_residues]
        if invalid:
            result.add_error(
                f"Unknown residue(s): {', '.join(invalid)}",
                f"Valid options: {', '.join(available_residues)}",
            )

    # Common residue sets
    rna_residues = {"A", "C", "G", "U"}
    dna_residues = {"DA", "DC", "DG", "DT"}
    protein_residues = {
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
        "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    }

    residue_set = set(residue_names)

    if residue_set.issubset(rna_residues):
        result.add_info(f"Training on {len(residue_names)} RNA residue type(s)")
    elif residue_set.issubset(dna_residues):
        result.add_info(f"Training on {len(residue_names)} DNA residue type(s)")
    elif residue_set.issubset(protein_residues):
        result.add_info(f"Training on {len(residue_names)} protein residue type(s)")
    else:
        result.add_info(f"Training on {len(residue_names)} residue type(s)")

    return result


def validate_device(device: str) -> ValidationResult:
    """Validate device configuration.

    Args:
        device: Device string ('auto', 'cuda', 'cpu', 'mps', etc.)

    Returns:
        ValidationResult with issues.
    """
    result = ValidationResult()

    try:
        import torch
    except ImportError:
        result.add_error(
            "PyTorch is not installed",
            "Install with: pip install torch",
        )
        return result

    if device == "auto":
        if torch.cuda.is_available():
            result.add_info(f"Using CUDA (GPU: {torch.cuda.get_device_name(0)})")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            result.add_info("Using MPS (Apple Silicon)")
        else:
            result.add_info("Using CPU (no GPU available)")
    elif device.startswith("cuda"):
        if not torch.cuda.is_available():
            result.add_error(
                f"CUDA device requested but CUDA is not available",
                "Use 'device: auto' or 'device: cpu'",
            )
        else:
            # Check specific device index
            if ":" in device:
                idx = int(device.split(":")[1])
                if idx >= torch.cuda.device_count():
                    result.add_error(
                        f"CUDA device {idx} not found (only {torch.cuda.device_count()} available)",
                        f"Use 'cuda:0' through 'cuda:{torch.cuda.device_count()-1}'",
                    )
    elif device == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            result.add_error(
                "MPS device requested but not available",
                "MPS requires macOS 12.3+ with Apple Silicon",
            )

    return result


def validate_training_config(
    data_dir: str | Path | None = None,
    cif_patterns: list[str] | None = None,
    residues: list[str] | None = None,
    output_dir: str | Path | None = None,
    device: str = "auto",
    min_structures: int = 10,
    quiet: bool = False,
) -> ValidationResult:
    """Validate complete training configuration.

    Runs all validation checks and returns combined results.

    Args:
        data_dir: Directory containing CIF files.
        cif_patterns: Optional glob patterns for CIF files.
        residues: List of residue names to train on.
        output_dir: Directory for saving checkpoints.
        device: Device string.
        min_structures: Minimum recommended structures.
        quiet: If True, skip info messages.

    Returns:
        ValidationResult with all issues and statistics.
    """
    result = ValidationResult()

    # Validate data
    data_result = validate_data_directory(data_dir, cif_patterns, min_structures)
    result.issues.extend(data_result.issues)
    result.n_structures = data_result.n_structures
    result.structure_files = data_result.structure_files

    # Validate residues
    if residues:
        residue_result = validate_residues(residues)
        result.issues.extend(residue_result.issues)

    # Validate output
    if output_dir:
        output_result = validate_output_directory(output_dir)
        result.issues.extend(output_result.issues)

    # Validate device
    device_result = validate_device(device)
    result.issues.extend(device_result.issues)

    return result


def print_validation_result(
    result: ValidationResult,
    show_info: bool = True,
    show_summary: bool = True,
) -> None:
    """Print validation result to console.

    Args:
        result: Validation result to print.
        show_info: Show info-level messages.
        show_summary: Show summary at end.
    """
    # Print issues
    formatted = result.format(show_info=show_info)
    if formatted:
        print(formatted)

    # Print summary
    if show_summary:
        print(result.format_summary())


__all__ = [
    "ValidationIssue",
    "ValidationResult",
    "validate_data_directory",
    "validate_output_directory",
    "validate_residues",
    "validate_device",
    "validate_training_config",
    "print_validation_result",
]
