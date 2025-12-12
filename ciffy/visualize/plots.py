"""
Matplotlib-based visualization plots.

Provides functions for plotting profiles and contact maps of molecular
structures using matplotlib.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Union, Any

import numpy as np

from ..types import Scale

if TYPE_CHECKING:
    from ..polymer import Polymer


# Consistent styling constants
LABEL_SIZE = 14
TITLE_SIZE = 15
TICK_SIZE = 13


def _require_matplotlib():
    """Check matplotlib availability and apply styling."""
    try:
        import matplotlib.pyplot as plt
        plt.rcParams['font.family'] = 'Helvetica'
        return plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install with: pip install matplotlib"
        )


def plot_profile(
    polymer: "Polymer",
    values: np.ndarray,
    *,
    scale: Scale = Scale.RESIDUE,
    ax: Any = None,
    color: Union[str, None] = None,
    fill: bool = True,
    xlabel: Union[str, None] = None,
    ylabel: Union[str, None] = None,
    title: Union[str, None] = None,
    **kwargs,
) -> Any:
    """
    Plot values as a profile along the sequence.

    Creates a line plot showing how values vary along the polymer sequence.

    Args:
        polymer: Polymer structure.
        values: Array of values to plot. Shape must match scale.
        scale: Scale of the values (RESIDUE or ATOM).
        ax: Matplotlib axes to plot on. Creates new figure if None.
        color: Line color. Uses RdPu colormap if None.
        fill: Whether to fill under the curve.
        xlabel: X-axis label. Defaults to scale name.
        ylabel: Y-axis label. Defaults to "Value".
        title: Plot title.
        **kwargs: Additional arguments passed to plt.plot().

    Returns:
        Matplotlib axes object.

    Example:
        >>> import ciffy
        >>> import numpy as np
        >>> polymer = ciffy.load("structure.cif")
        >>> values = np.random.rand(polymer.size(ciffy.RESIDUE))
        >>> ax = ciffy.visualize.plot_profile(polymer, values)
    """
    plt = _require_matplotlib()

    values = np.asarray(values)
    expected_size = polymer.size(scale)
    if values.shape[0] != expected_size:
        raise ValueError(
            f"Values length ({values.shape[0]}) must match polymer "
            f"size at {scale.name} scale ({expected_size})"
        )

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    x = np.arange(len(values))

    # Use RdPu colormap for consistent styling
    if color is None:
        line_color = plt.cm.RdPu(0.8)
        fill_color = plt.cm.RdPu(0.3)
    else:
        line_color = color
        fill_color = color

    ax.plot(x, values, color=line_color, linewidth=1, **kwargs)

    if fill:
        ax.fill_between(x, values, alpha=0.5, color=fill_color)

    ax.grid(axis="y", alpha=0.5)
    ax.set_xlim(0, len(values))

    if xlabel is None:
        xlabel = scale.name.capitalize()
    ax.set_xlabel(xlabel, fontsize=LABEL_SIZE)

    if ylabel is None:
        ylabel = "Value"
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)

    if title is not None:
        ax.set_title(title, fontsize=TITLE_SIZE)

    ax.tick_params(axis='both', labelsize=TICK_SIZE)

    return ax


def contact_map(
    polymer: "Polymer",
    *,
    scale: Scale = Scale.RESIDUE,
    power: float = 2.0,
    ax: Any = None,
    cmap: str = "RdPu",
    vmin: Union[float, None] = None,
    vmax: Union[float, None] = None,
    colorbar: bool = True,
    title: Union[str, None] = None,
    **kwargs,
) -> Any:
    """
    Plot a contact map showing pairwise proximity.

    Creates a heatmap of 1/r^power values, where r is the distance between
    residue (or atom) centroids. Higher values indicate closer proximity.

    Args:
        polymer: Polymer structure.
        scale: Scale for distance computation (RESIDUE or ATOM).
        power: Exponent for distance transformation (default 2.0).
            Common values: 1 (inverse), 2 (inverse square), 6 (LJ-like).
        ax: Matplotlib axes to plot on. Creates new figure if None.
        cmap: Colormap name (default: RdPu).
        vmin: Minimum value for color scaling.
        vmax: Maximum value for color scaling.
        colorbar: Whether to add a colorbar.
        title: Plot title.
        **kwargs: Additional arguments passed to plt.imshow().

    Returns:
        Matplotlib axes object.

    Example:
        >>> import ciffy
        >>> polymer = ciffy.load("structure.cif")
        >>> ax = ciffy.visualize.contact_map(polymer, scale=ciffy.RESIDUE)
    """
    plt = _require_matplotlib()

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    # Compute pairwise distances
    distances = polymer.pairwise_distances(scale)

    # Convert to numpy if torch tensor
    if hasattr(distances, 'numpy'):
        distances = distances.cpu().numpy()
    else:
        distances = np.asarray(distances)

    # Compute 1/r^power, handling zeros on diagonal
    with np.errstate(divide='ignore'):
        contact = 1.0 / (distances ** power)

    # Set diagonal to 0 (self-contacts are infinite)
    np.fill_diagonal(contact, 0)

    # Handle any remaining infinities
    contact = np.nan_to_num(contact, nan=0.0, posinf=0.0, neginf=0.0)

    # Plot with crisp pixels (no interpolation)
    im = ax.imshow(contact, cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation='none', **kwargs)

    # Styled colorbar with LaTeX label
    if colorbar:
        cbar = plt.colorbar(im, ax=ax)
        if power == int(power):
            cbar.set_label(f"$1/r^{int(power)}$", fontsize=LABEL_SIZE)
        else:
            cbar.set_label(f"$1/r^{{{power}}}$", fontsize=LABEL_SIZE)
        cbar.ax.tick_params(labelsize=TICK_SIZE)

    # Labels with consistent styling
    scale_name = scale.name.capitalize()
    ax.set_xlabel(scale_name, fontsize=LABEL_SIZE)
    ax.set_ylabel(scale_name, fontsize=LABEL_SIZE)

    if title is None:
        title = "Contact Map"
    ax.set_title(title, fontsize=TITLE_SIZE)

    ax.tick_params(axis='both', labelsize=TICK_SIZE)

    return ax
