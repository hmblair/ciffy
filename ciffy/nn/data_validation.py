"""
Data validation and compatibility checking for polymer datasets.

Provides utilities to validate dataset/model compatibility before training
and generate diagnostic reports when filtering removes too many samples.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ciffy import Polymer
    from ciffy.nn.flow.polymer import PolymerFlowModel
    from .dataset import PolymerDataset

from ciffy import Scale

logger = logging.getLogger(__name__)


@dataclass
class StructureExample:
    """Example of a problematic structure for debugging."""

    pdb_id: str
    file_path: str
    reason: str
    details: str

    def format(self) -> str:
        return f"  - {self.pdb_id} ({self.file_path}): {self.details}"


@dataclass
class DataCompatibilityReport:
    """Report on dataset/model compatibility with diagnostics."""

    total_samples: int = 0
    valid_samples: int = 0

    # Skip counts by reason
    skip_counts: dict[str, int] = field(default_factory=dict)

    # Up to N examples per skip reason
    examples: dict[str, list[StructureExample]] = field(default_factory=dict)

    # Flow model info (if applicable)
    flow_model_residues: set[int] | None = None
    unsupported_residues: set[int] | None = None

    @property
    def is_compatible(self) -> bool:
        """True if at least some samples are valid."""
        return self.valid_samples > 0

    @property
    def valid_fraction(self) -> float:
        """Fraction of samples that passed validation."""
        if self.total_samples == 0:
            return 0.0
        return self.valid_samples / self.total_samples

    def add_skip(
        self,
        reason: str,
        example: StructureExample | None = None,
        max_examples: int = 3,
    ) -> None:
        """Record a skipped sample with optional example."""
        self.skip_counts[reason] = self.skip_counts.get(reason, 0) + 1

        if example is not None:
            if reason not in self.examples:
                self.examples[reason] = []
            if len(self.examples[reason]) < max_examples:
                self.examples[reason].append(example)

    def format_summary(self) -> str:
        """Format concise summary with example problematic structures."""
        lines = [
            "Dataset compatibility report:",
            f"  Valid samples: {self.valid_samples}/{self.total_samples} "
            f"({self.valid_fraction * 100:.1f}%)",
        ]

        if self.skip_counts:
            lines.append("\nSkip breakdown:")
            for reason, count in sorted(
                self.skip_counts.items(), key=lambda x: -x[1]
            ):
                lines.append(f"  {reason}: {count}")

                # Add examples (up to max_examples)
                if reason in self.examples:
                    for ex in self.examples[reason]:
                        lines.append(ex.format())

        if self.unsupported_residues:
            from ciffy.biochemistry import Residue

            names = []
            for r in sorted(self.unsupported_residues):
                try:
                    names.append(Residue.from_index(r).name)
                except (ValueError, KeyError):
                    names.append(f"<unknown:{r}>")
            lines.append(f"\nUnsupported residue types in data: {', '.join(names)}")

            if self.flow_model_residues:
                supported = []
                for r in sorted(self.flow_model_residues):
                    try:
                        supported.append(Residue.from_index(r).name)
                    except (ValueError, KeyError):
                        supported.append(f"<{r}>")
                lines.append(f"Flow model supports: {', '.join(supported)}")

        return "\n".join(lines)


def validate_flow_model_compatibility(
    flow_model: "PolymerFlowModel",
    polymer_dataset: "PolymerDataset",
    sample_count: int = 100,
    min_residues: int = 1,
    max_residues: int = 10000,
) -> DataCompatibilityReport:
    """
    Quick compatibility check between flow model and dataset.

    Samples a subset of the dataset to estimate compatibility without
    processing the entire dataset. Use this for fast fail-fast validation.

    Args:
        flow_model: The PolymerFlowModel to check against.
        polymer_dataset: The source PolymerDataset.
        sample_count: Number of samples to check (default 100).
        min_residues: Minimum residues to accept.
        max_residues: Maximum residues to accept.

    Returns:
        DataCompatibilityReport with estimated compatibility.
    """
    report = DataCompatibilityReport()
    report.flow_model_residues = flow_model.supported_residues.copy()

    # Sample indices
    n_total = len(polymer_dataset)
    if n_total == 0:
        return report

    indices = list(range(n_total))
    if n_total > sample_count:
        random.shuffle(indices)
        indices = indices[:sample_count]

    report.total_samples = len(indices)
    seen_unsupported: set[int] = set()

    for idx in indices:
        try:
            polymer = polymer_dataset[idx]
        except Exception as e:
            report.add_skip(
                "load_error",
                StructureExample(
                    pdb_id="<unknown>",
                    file_path=str(polymer_dataset._index[idx][0]),
                    reason="load_error",
                    details=str(e)[:100],
                ),
            )
            continue

        if polymer is None:
            report.add_skip("none_empty")
            continue

        # Get file path for error reporting
        file_path = str(polymer_dataset._index[idx][0])
        pdb_id = polymer.pdb_id

        # Apply poly() filter (exclude HETATM)
        polymer = polymer.poly()

        # Check residue count
        n_res = polymer.size(Scale.RESIDUE)
        if n_res < min_residues:
            report.add_skip(
                "too_few_residues",
                StructureExample(
                    pdb_id=pdb_id,
                    file_path=file_path,
                    reason="too_few_residues",
                    details=f"{n_res} residues (min={min_residues})",
                ),
            )
            continue

        if n_res > max_residues:
            report.add_skip(
                "too_many_residues",
                StructureExample(
                    pdb_id=pdb_id,
                    file_path=file_path,
                    reason="too_many_residues",
                    details=f"{n_res} residues (max={max_residues})",
                ),
            )
            continue

        # Check for unknown residues (index < 0)
        seq = polymer.sequence
        if hasattr(seq, "numpy"):
            seq = seq.numpy()

        if any(r < 0 for r in seq):
            report.add_skip(
                "unknown_residues",
                StructureExample(
                    pdb_id=pdb_id,
                    file_path=file_path,
                    reason="unknown_residues",
                    details="Contains residues with unknown type (index < 0)",
                ),
            )
            continue

        # Check for unsupported residue types
        seq_set = set(int(r) for r in seq)
        unsupported_in_sample = seq_set - report.flow_model_residues
        if unsupported_in_sample:
            seen_unsupported.update(unsupported_in_sample)
            report.add_skip(
                "unsupported_residue_type",
                StructureExample(
                    pdb_id=pdb_id,
                    file_path=file_path,
                    reason="unsupported_residue_type",
                    details="Contains residue types not in flow model",
                ),
            )
            continue

        # Check atom count match
        try:
            expected_atoms = sum(flow_model.atom_counts[int(t)] for t in seq)
        except KeyError as e:
            # Residue type not in flow model (shouldn't happen after above check)
            report.add_skip("missing_residue_model")
            continue

        actual_atoms = polymer.size()
        if actual_atoms != expected_atoms:
            report.add_skip(
                "atom_count_mismatch",
                StructureExample(
                    pdb_id=pdb_id,
                    file_path=file_path,
                    reason="atom_count_mismatch",
                    details=f"Expected {expected_atoms} atoms, got {actual_atoms}",
                ),
            )
            continue

        # Valid sample
        report.valid_samples += 1

    report.unsupported_residues = seen_unsupported if seen_unsupported else None

    return report
