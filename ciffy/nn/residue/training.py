"""Training utilities for ResidueVAE."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

import ciffy
from ciffy import Scale
from ciffy.biochemistry.linking import GLYCOSIDIC_FRAME, O3P_FRAME, P_FRAME

if TYPE_CHECKING:
    from ciffy.nn import PolymerDataset


@dataclass
class ResidueVAEConfig:
    """Configuration for ResidueVAE training.

    Attributes:
        latent_dim: Dimension of latent space per residue.
        d_model: Hidden dimension for transformer layers.
        n_heads: Number of attention heads.
        encoder_layers: Number of encoder transformer layers.
        decoder_layers: Number of decoder MLP layers.
        lr: Initial learning rate.
        n_epochs: Number of training epochs.
        warmup_epochs: Number of warmup epochs.
        batch_size: Number of structures per batch.
        coord_weight: Weight for coordinate reconstruction loss.
        transform_weight: Weight for transform reconstruction loss.
        kl_weight: Weight for KL divergence loss.
        free_bits: Free bits for KL loss.
        max_transform_magnitude: Filter structures with transforms larger than this.
    """

    latent_dim: int = 16
    d_model: int = 64
    n_heads: int = 4
    encoder_layers: int = 1
    decoder_layers: int = 2
    lr: float = 1e-2
    n_epochs: int = 200
    warmup_epochs: int = 10
    batch_size: int = 64
    coord_weight: float = 1.0
    transform_weight: float = 1.0
    kl_weight: float = 0.01
    free_bits: float = 0.25
    max_transform_magnitude: float = 10.0


def precompute_targets(
    dataset: "PolymerDataset",
    n_structures: int | None = None,
    max_transform_magnitude: float = 10.0,
    verbose: bool = True,
) -> list[dict]:
    """Precompute aligned coordinates and inter-residue transforms.

    Filters out structures with:
    - Internal gaps (unresolved residues between resolved ones)
    - Too few residues (< 2)
    - Invalid sequences
    - Transform magnitudes exceeding threshold

    Args:
        dataset: PolymerDataset to load structures from.
        n_structures: Maximum number of structures to process.
            If None, processes all structures.
        max_transform_magnitude: Skip structures with any transform
            component exceeding this value.
        verbose: Whether to print progress.

    Returns:
        List of dicts with keys:
        - 'polymer': Aligned polymer (coordinates in GLYCOSIDIC frame)
        - 'target_coords': Target coordinates tensor
        - 'transforms': Inter-residue SE(3) transforms (N, 6)
    """
    if n_structures is None:
        n_structures = len(dataset)

    cached = []
    n_filtered = 0
    n_gaps = 0
    n_errors = 0

    for i in range(min(n_structures, len(dataset))):
        if verbose and i % 50 == 0:
            print(f"  Processing structure {i}...", flush=True)

        try:
            raw_polymer = dataset[i]
            if raw_polymer is None:
                n_errors += 1
                continue

            raw_polymer = raw_polymer.torch()

            # Skip chains with internal gaps
            if raw_polymer.has_internal_gaps():
                n_gaps += 1
                continue

            polymer = raw_polymer.strip()
            if polymer.size(Scale.RESIDUE) < 2:
                continue
            if (polymer.sequence < 0).any():
                continue

            # Compute transforms FIRST (before alignment destroys frame info)
            # Use O3P_FRAME → P_FRAME to preserve phosphodiester bond geometry
            try:
                transforms = polymer.local_transforms(O3P_FRAME, P_FRAME)
            except Exception:
                n_errors += 1
                continue

            if transforms.abs().max() > max_transform_magnitude:
                n_filtered += 1
                continue

            # THEN align coordinates to GLYCOSIDIC_FRAME for encoding
            try:
                aligned, _ = polymer.align(frame=GLYCOSIDIC_FRAME)
            except Exception:
                n_errors += 1
                continue

            cached.append({
                'polymer': aligned,
                'target_coords': aligned.coordinates,
                'transforms': transforms,
            })

            if verbose and len(cached) % 100 == 0:
                print(f"  Cached {len(cached)} structures...", flush=True)

        except Exception:
            n_errors += 1
            continue

    if verbose:
        print(f"  Cached {len(cached)} structures", flush=True)
        if n_gaps > 0:
            print(f"  Skipped {n_gaps} structures with internal gaps", flush=True)
        if n_filtered > 0:
            print(f"  Filtered {n_filtered} structures with large transforms", flush=True)
        if n_errors > 0:
            print(f"  {n_errors} structures had errors", flush=True)

    return cached


def create_batches(
    cached_data: list[dict],
    batch_size: int = 64,
    device: str | torch.device = "cpu",
) -> list[dict]:
    """Create batched data using ciffy.join.

    Args:
        cached_data: List of dicts from precompute_targets.
        batch_size: Number of structures per batch.
        device: Device to move batches to.

    Returns:
        List of batch dicts with keys:
        - 'polymer': Joined polymer on device
        - 'target_coords': Target coordinates on device
        - 'transforms': Concatenated transforms on device
    """
    batches = []
    for i in range(0, len(cached_data), batch_size):
        batch_items = cached_data[i:i + batch_size]
        polymers = [item['polymer'] for item in batch_items]
        transforms = [item['transforms'] for item in batch_items]

        joined = ciffy.join(*polymers).to(device)
        joined_transforms = torch.cat(transforms, dim=0).to(device)
        joined_coords = joined.coordinates

        batches.append({
            'polymer': joined,
            'target_coords': joined_coords,
            'transforms': joined_transforms,
        })

    return batches
