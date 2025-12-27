"""
Filtered polymer dataset with diagnostics.

Provides a reusable wrapper around PolymerDataset that applies configurable
filtering with detailed skip reason tracking and example collection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from ciffy import Polymer
    from .dataset import PolymerDataset

try:
    import torch
    from torch.utils.data import Dataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Dataset = object  # type: ignore

from ciffy import Scale

from .data_validation import DataCompatibilityReport, StructureExample

logger = logging.getLogger(__name__)


@dataclass
class FilterConfig:
    """Configuration for polymer dataset filtering."""

    min_residues: int | None = None
    max_residues: int | None = None
    poly_only: bool = True
    reject_unknown_residues: bool = True
    flow_model: Any = None  # PolymerFlowModel for atom count validation
    collect_examples: int = 3


class FilteredPolymerDataset(Dataset):
    """
    Polymer dataset with configurable filtering and diagnostics.

    Wraps PolymerDataset and applies additional filters with detailed
    skip reason tracking and example collection.

    Example:
        >>> from ciffy.nn import PolymerDataset
        >>> from ciffy.nn.filtered_dataset import FilteredPolymerDataset, FilterConfig
        >>>
        >>> base_dataset = PolymerDataset("./structures/", scale=Scale.CHAIN)
        >>> config = FilterConfig(min_residues=10, max_residues=500)
        >>> dataset = FilteredPolymerDataset(base_dataset, config)
        >>>
        >>> # Access the compatibility report
        >>> print(dataset.report.format_summary())
        >>> print(f"Valid: {len(dataset)} samples")

    Attributes:
        polymer_dataset: The underlying PolymerDataset.
        config: The FilterConfig used for filtering.
        valid_indices: List of indices that passed all filters.
        report: DataCompatibilityReport with skip statistics and examples.
    """

    def __init__(
        self,
        polymer_dataset: "PolymerDataset",
        config: FilterConfig,
    ):
        """
        Initialize filtered dataset.

        Args:
            polymer_dataset: Base PolymerDataset to wrap.
            config: FilterConfig specifying filtering criteria.

        Raises:
            ImportError: If PyTorch is not available.
            ValueError: If no samples pass the filters.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for FilteredPolymerDataset")

        self.polymer_dataset = polymer_dataset
        self.config = config

        # Build filtered index
        self.valid_indices, self.report = self._build_filtered_index()

        if len(self.valid_indices) == 0:
            raise ValueError(
                f"No valid samples after filtering!\n\n"
                f"{self.report.format_summary()}\n\n"
                f"Consider:\n"
                f"  1. Check flow model compatibility with your data\n"
                f"  2. Adjust min_residues/max_residues bounds\n"
                f"  3. Use structures with complete atom sets"
            )

        # Log summary
        logger.info(
            f"FilteredPolymerDataset: {len(self.valid_indices)}/{len(polymer_dataset)} "
            f"samples valid ({self.report.valid_fraction * 100:.1f}%)"
        )

        if self.report.skip_counts:
            skip_summary = ", ".join(
                f"{k}: {v}" for k, v in sorted(self.report.skip_counts.items())
            )
            logger.info(f"  Skip breakdown: {skip_summary}")

    def _build_filtered_index(
        self,
    ) -> tuple[list[int], DataCompatibilityReport]:
        """Build list of valid indices with diagnostics."""
        report = DataCompatibilityReport()
        valid_indices: list[int] = []

        flow_model = self.config.flow_model
        if flow_model is not None:
            report.flow_model_residues = flow_model._supported_types_set.copy()

        report.total_samples = len(self.polymer_dataset)

        for idx in range(len(self.polymer_dataset)):
            # Load sample
            try:
                polymer = self.polymer_dataset[idx]
            except Exception as e:
                report.add_skip(
                    "load_error",
                    StructureExample(
                        pdb_id="<unknown>",
                        file_path=str(self.polymer_dataset._index[idx][0]),
                        reason="load_error",
                        details=str(e)[:100],
                    ),
                    max_examples=self.config.collect_examples,
                )
                continue

            if polymer is None:
                report.add_skip(
                    "none_empty", max_examples=self.config.collect_examples
                )
                continue

            file_path = str(self.polymer_dataset._index[idx][0])
            pdb_id = polymer.pdb_id

            # Apply poly() if configured (exclude HETATM)
            if self.config.poly_only:
                polymer = polymer.poly()

            # Check residue count bounds
            n_res = polymer.size(Scale.RESIDUE)

            if self.config.min_residues and n_res < self.config.min_residues:
                report.add_skip(
                    "too_few_residues",
                    StructureExample(
                        pdb_id=pdb_id,
                        file_path=file_path,
                        reason="too_few_residues",
                        details=f"{n_res} residues (min={self.config.min_residues})",
                    ),
                    max_examples=self.config.collect_examples,
                )
                continue

            if self.config.max_residues and n_res > self.config.max_residues:
                report.add_skip(
                    "too_many_residues",
                    StructureExample(
                        pdb_id=pdb_id,
                        file_path=file_path,
                        reason="too_many_residues",
                        details=f"{n_res} residues (max={self.config.max_residues})",
                    ),
                    max_examples=self.config.collect_examples,
                )
                continue

            # Check for unknown residues
            seq = polymer.sequence
            if hasattr(seq, "numpy"):
                seq = seq.numpy()

            if self.config.reject_unknown_residues and any(r < 0 for r in seq):
                report.add_skip(
                    "unknown_residues",
                    StructureExample(
                        pdb_id=pdb_id,
                        file_path=file_path,
                        reason="unknown_residues",
                        details="Contains residues with unknown type",
                    ),
                    max_examples=self.config.collect_examples,
                )
                continue

            # Check flow model compatibility
            if flow_model is not None:
                # Check residue types are supported
                seq_set = set(int(r) for r in seq)
                unsupported = seq_set - report.flow_model_residues
                if unsupported:
                    if report.unsupported_residues is None:
                        report.unsupported_residues = set()
                    report.unsupported_residues.update(unsupported)

                    report.add_skip(
                        "unsupported_residue_type",
                        StructureExample(
                            pdb_id=pdb_id,
                            file_path=file_path,
                            reason="unsupported_residue_type",
                            details="Contains residue types not in flow model",
                        ),
                        max_examples=self.config.collect_examples,
                    )
                    continue

                # Check atom counts match
                try:
                    expected_atoms = sum(
                        flow_model._atom_counts[int(t)] for t in seq
                    )
                    actual_atoms = polymer.size()

                    if actual_atoms != expected_atoms:
                        report.add_skip(
                            "atom_count_mismatch",
                            StructureExample(
                                pdb_id=pdb_id,
                                file_path=file_path,
                                reason="atom_count_mismatch",
                                details=f"Expected {expected_atoms}, got {actual_atoms} atoms",
                            ),
                            max_examples=self.config.collect_examples,
                        )
                        continue
                except KeyError:
                    report.add_skip("missing_residue_model")
                    continue

            # Valid sample
            valid_indices.append(idx)

        report.valid_samples = len(valid_indices)
        return valid_indices, report

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Optional["Polymer"]:
        """
        Get pre-validated sample.

        Args:
            idx: Index into valid samples (not the underlying dataset).

        Returns:
            Polymer object with filtering applied (e.g., poly() if configured).
        """
        polymer_idx = self.valid_indices[idx]
        polymer = self.polymer_dataset[polymer_idx]

        if polymer is None:
            return None

        if self.config.poly_only:
            polymer = polymer.poly()

        return polymer
