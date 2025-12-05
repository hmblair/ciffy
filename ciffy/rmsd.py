import torch
from .main import Polymer, Scale


def _coordinate_covariance(
    polymer1: Polymer,
    polymer2: Polymer,
    scale: Scale = Scale.MOLECULE,
) -> torch.Tensor:
    """
    Get the covariance matrices between the coordintes of the two polymers.
    """

    outer_prod = torch.multiply(
        polymer1.coordinates[:, None, :],
        polymer2.coordinates[:, :, None],
    )

    return polymer1.reduce(outer_prod, scale)


def _kabsch_distance(
    polymer1: Polymer,
    polymer2: Polymer,
    scale: Scale = Scale.MOLECULE,
) -> torch.Tensor:
    """
    Get the aligned distance between the individual molecules in the polymers
    using the kabsch algorithm. The two polymers should have the same molecule
    indices. An optional weight can be provided to bias the alignment.
    """

    # Center and get the coordinate covariance matrices

    polymer1_c, _ = polymer1.center(scale)
    polymer2_c, _ = polymer2.center(scale)
    cov = _coordinate_covariance(polymer1_c, polymer2_c, scale)

    sigma = torch.linalg.svdvals(cov)
    det = torch.linalg.det(cov)

    sigma = sigma.clone()
    sigma[det < 0, -1] = - sigma[det < 0, -1]
    sigma = sigma.mean(-1)

    # Get the variances of the point clouds

    var1 = polymer1_c.moment(2, scale).mean(-1)
    var2 = polymer2_c.moment(2, scale).mean(-1)

    # Compute the kabsch distance

    return (var1 + var2 - 2 * sigma)
