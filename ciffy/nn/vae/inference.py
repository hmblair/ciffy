"""
VAE-specific inference utilities for structure generation and analysis.

Provides VAE-focused functions for reconstruction, interpolation, and sampling
from a trained PolymerVAE model.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

try:
    import torch
except ImportError:
    torch = None

if TYPE_CHECKING:
    from .vae import PolymerVAE
    from ...polymer import Polymer


def reconstruct_polymer(
    vae: PolymerVAE,
    polymer: Polymer,
    sample_latent: bool = False,
) -> Polymer:
    """
    Reconstruct a polymer through the VAE encoder-decoder.

    Encodes the input polymer to the latent space and decodes back to structure space.

    Args:
        vae: Trained VAE model.
        polymer: Input polymer to reconstruct.
        sample_latent: If True, sample z from latent distribution (stochastic).
            If False, use latent mean (deterministic reconstruction).

    Returns:
        Reconstructed polymer with same topology as input.

    Example:
        >>> from ciffy.nn import load_vae
        >>> import ciffy
        >>>
        >>> vae = load_vae("checkpoint_best.pt", device="cuda")
        >>> polymer = ciffy.load("structure.cif", backend="torch").to("cuda")
        >>> recon = reconstruct_polymer(vae, polymer, sample_latent=False)
        >>> recon.numpy().write("reconstructed.cif")
    """
    if torch is None:
        raise ImportError("PyTorch is required. Install with: pip install torch")

    vae.eval()
    with torch.no_grad():
        return vae.reconstruct(polymer, sample_latent=sample_latent)


def interpolate_structures(
    vae: PolymerVAE,
    polymer1: Polymer,
    polymer2: Polymer,
    n_steps: int = 10,
    output_dir: Optional[str | Path] = None,
) -> list[Polymer]:
    """
    Interpolate between two polymer conformations in VAE latent space.

    Creates a smooth path through latent space connecting two structures,
    then decodes interpolated latent vectors to generate intermediate structures.

    Args:
        vae: Trained VAE model.
        polymer1: Starting conformation.
        polymer2: Ending conformation.
        n_steps: Number of interpolation steps including both endpoints.
        output_dir: If provided, save interpolated structures as .cif files.

    Returns:
        List of n_steps interpolated Polymers.

    Raises:
        ValueError: If n_steps < 2.

    Example:
        >>> from ciffy.nn import load_vae
        >>> import ciffy
        >>>
        >>> vae = load_vae("checkpoint_best.pt")
        >>> p1 = ciffy.load("struct1.cif", backend="torch")
        >>> p2 = ciffy.load("struct2.cif", backend="torch")
        >>> frames = interpolate_structures(
        ...     vae, p1, p2,
        ...     n_steps=20,
        ...     output_dir="./interpolation",
        ... )
        >>> print(f"Generated {len(frames)} interpolation frames")
    """
    if n_steps < 2:
        raise ValueError(f"n_steps must be >= 2, got {n_steps}")

    if torch is None:
        raise ImportError("PyTorch is required. Install with: pip install torch")

    vae.eval()
    with torch.no_grad():
        results = vae.interpolate(polymer1, polymer2, n_steps=n_steps)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, polymer in enumerate(results):
            filename = f"interp_{i:06d}.cif"
            filepath = output_dir / filename

            try:
                # Convert to numpy if needed
                if hasattr(polymer, "numpy"):
                    polymer = polymer.numpy()
                polymer.write(str(filepath))
            except Exception as e:
                print(f"Warning: Failed to save {filepath}: {e}")

    return results


def sample_from_sequence(
    vae: PolymerVAE,
    sequence: str | list[str],
    n_samples: int = 1,
    temperature: float = 1.0,
    output_dir: Optional[str | Path] = None,
    prefix: str = "gen_",
    backend: str = "torch",
) -> list[Polymer]:
    """
    Sample conformations for sequences from a trained VAE.

    Convenience wrapper around the generic generate_samples function
    specifically configured for PolymerVAE.

    Args:
        vae: Trained VAE model.
        sequence: Single sequence or list of sequences.
        n_samples: Number of samples per sequence.
        temperature: Sampling temperature (higher = more diverse).
        output_dir: If provided, save samples as .cif files.
        prefix: Prefix for output filenames.
        backend: Backend for template creation ("torch" or "numpy").

    Returns:
        List of sampled Polymers.

    Example:
        >>> from ciffy.nn import load_vae
        >>> from ciffy.nn.vae.inference import sample_from_sequence
        >>>
        >>> vae = load_vae("checkpoint_best.pt", device="cuda")
        >>> samples = sample_from_sequence(
        ...     vae,
        ...     sequence=["MGKLF", "acgu"],
        ...     n_samples=10,
        ...     temperature=1.0,
        ...     output_dir="./samples",
        ... )
    """
    from ..inference import generate_samples

    return generate_samples(
        vae,
        sequence=sequence,
        n_samples=n_samples,
        temperature=temperature,
        output_dir=output_dir,
        prefix=prefix,
        backend=backend,
    )


__all__ = [
    "reconstruct_polymer",
    "interpolate_structures",
    "sample_from_sequence",
]
