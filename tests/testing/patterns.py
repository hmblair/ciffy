"""Reusable test patterns for common ciffy test scenarios.

These patterns encapsulate common testing workflows so tests can focus
on what they're testing rather than boilerplate.
"""

from typing import Optional, Callable
import numpy as np

from .tolerances import get_tolerances, ToleranceProfile


def assert_roundtrip_preserves_structure(
    polymer,
    threshold: Optional[float] = None,
    tolerance_key: str = "roundtrip_small",
    message: str = "",
) -> float:
    """Execute internal coordinate roundtrip and verify structure preservation.

    Args:
        polymer: Polymer to test
        threshold: Maximum allowed RMSD (overrides tolerance_key if provided)
        tolerance_key: Key from ToleranceProfile to use if threshold not provided
        message: Extra message for assertion failure

    Returns:
        The computed RMSD value

    Raises:
        AssertionError: If RMSD exceeds tolerance

    Example:
        def test_rna_roundtrip(self):
            polymer = from_sequence("acgu")
            assert_roundtrip_preserves_structure(polymer)
    """
    from ciffy import kabsch_align

    tol = get_tolerances()

    if threshold is None:
        threshold = getattr(tol, tolerance_key)

    # Clone/copy original coordinates
    if hasattr(polymer.coordinates, "clone"):
        orig_coords = polymer.coordinates.clone()
    else:
        orig_coords = polymer.coordinates.copy()

    # Trigger reconstruction by reading and setting dihedrals
    if hasattr(polymer.dihedrals, "clone"):
        dihedrals = polymer.dihedrals.clone()
    else:
        dihedrals = polymer.dihedrals.copy()
    polymer.dihedrals = dihedrals

    # Align reconstructed to original (move to CPU for alignment if needed)
    new_coords = polymer.coordinates
    if hasattr(new_coords, 'cpu'):
        aligned, _, _ = kabsch_align(new_coords.cpu(), orig_coords.cpu())
        aligned_np = np.asarray(aligned)
        orig_np = np.asarray(orig_coords.cpu())
    else:
        aligned, _, _ = kabsch_align(new_coords, orig_coords)
        aligned_np = np.asarray(aligned)
        orig_np = np.asarray(orig_coords)

    # Compute RMSD
    rmsd = float(np.sqrt(((aligned_np - orig_np) ** 2).sum(axis=1).mean()))

    assert rmsd < threshold, (
        f"Roundtrip RMSD {rmsd:.6f} exceeds threshold {threshold}. {message}"
    )
    return rmsd


def assert_gradient_flows(
    polymer,
    loss_fn: Optional[Callable] = None,
) -> None:
    """Verify gradients flow through internal coordinate reconstruction.

    Args:
        polymer: Polymer (must be torch backend)
        loss_fn: Optional loss function taking coordinates, defaults to coords.pow(2).mean()

    Raises:
        AssertionError: If polymer is not torch backend or gradients don't flow

    Example:
        def test_grad_flow(self):
            polymer = from_sequence("acgu", backend="torch")
            assert_gradient_flows(polymer)
    """
    import torch

    if polymer.backend != "torch":
        raise AssertionError("assert_gradient_flows requires torch backend")

    # Enable gradients on dihedrals
    dihedrals = polymer.dihedrals.clone()
    dihedrals.requires_grad_(True)
    polymer.dihedrals = dihedrals

    # Access coordinates (triggers reconstruction)
    coords = polymer.coordinates

    # Compute loss
    if loss_fn is None:
        loss = coords.pow(2).mean()
    else:
        loss = loss_fn(coords)

    # Backward should work
    loss.backward()

    # Gradients should exist and be non-zero
    assert dihedrals.grad is not None, "No gradients computed"
    assert not torch.all(dihedrals.grad == 0), "All gradients are zero"


def assert_cif_roundtrip(
    polymer,
    tmp_path,
    check_coordinates: bool = True,
    check_sequence: bool = True,
) -> "Polymer":
    """Test CIF write/read roundtrip.

    Args:
        polymer: Polymer to write and reload
        tmp_path: pytest tmp_path fixture
        check_coordinates: Whether to verify coordinates match
        check_sequence: Whether to verify sequence exists

    Returns:
        The reloaded polymer

    Example:
        def test_cif_roundtrip(self, tmp_path):
            polymer = from_sequence("acgu")
            reloaded = assert_cif_roundtrip(polymer, tmp_path)
    """
    from ciffy import load

    tol = get_tolerances()
    output_path = tmp_path / "test.cif"

    # Get polymer count before writing
    polymer_count = getattr(polymer, "polymer_count", polymer.size())

    # Write
    polymer.write(str(output_path))

    # Reload
    reloaded = load(str(output_path), backend=polymer.backend)

    # Verify
    if check_coordinates and polymer_count > 0:
        orig_coords = np.asarray(polymer.coordinates[:polymer_count])
        reload_coords = np.asarray(reloaded.coordinates)
        assert np.allclose(orig_coords, reload_coords, atol=tol.coord_roundtrip), (
            "Coordinates not preserved in CIF roundtrip"
        )

    if check_sequence:
        assert len(reloaded.sequence) > 0, "Reloaded polymer has no sequence"

    return reloaded
