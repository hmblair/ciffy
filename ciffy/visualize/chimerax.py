"""
ChimeraX integration for molecular visualization.

Provides functions to launch ChimeraX with custom coloring based on
per-residue or per-atom values.
"""

from __future__ import annotations
import atexit
import os
import subprocess
import tempfile
from typing import TYPE_CHECKING, Union

import numpy as np

from ..types import Scale
from .defattr import to_defattr

if TYPE_CHECKING:
    from ..polymer import Polymer

# Track temp files for cleanup on exit
_temp_files: list[str] = []


def _cleanup_temp_files() -> None:
    """Remove any remaining temp files created by visualize()."""
    for filepath in _temp_files:
        try:
            os.unlink(filepath)
        except OSError:
            pass  # File may already be deleted
    _temp_files.clear()


# Register cleanup to run when Python exits
atexit.register(_cleanup_temp_files)


def find_chimerax() -> str:
    """
    Find the ChimeraX executable.

    Returns:
        Path to ChimeraX executable.

    Raises:
        FileNotFoundError: If ChimeraX is not found.
    """
    # Common locations
    candidates = [
        "ChimeraX",  # In PATH
        "chimerax",
        "/Applications/ChimeraX.app/Contents/MacOS/ChimeraX",  # macOS
        "/usr/bin/chimerax",
        "/usr/local/bin/chimerax",
    ]

    for path in candidates:
        try:
            result = subprocess.run(
                [path, "--version"],
                capture_output=True,
                timeout=5,
            )
            if result.returncode == 0:
                return path
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue

    raise FileNotFoundError(
        "ChimeraX not found. Install ChimeraX or specify the path with "
        "chimerax_path parameter."
    )


def _build_visualization_script(
    cif_path: str,
    defattr_path: str,
    color: str,
    max_value: float,
    chain: Union[str, None],
    style: str = "sphere",
) -> str:
    """
    Build ChimeraX command script for visualization.

    Args:
        cif_path: Path to CIF file.
        defattr_path: Path to defattr file.
        color: Color for high values.
        max_value: Maximum value for color scaling.
        chain: Chain to display (None for all).
        style: Display style ("sphere", "stick", "cartoon").

    Returns:
        ChimeraX command string.
    """
    commands = [
        f"open {cif_path}",
        "close #1.2-999",  # Close extra models if mmCIF has multiple
    ]

    if chain is not None:
        commands.append(f"del ~/{chain}")

    commands.extend([
        "hide pseudobonds",
        "color grey",
        "graphics quality 5",
        "renumber start 1 relative false",
        f"open {defattr_path}",
        f"color byattribute value palette white:{color} range 0,{max_value}",
        "hide cartoons",
        "nucleotides atoms",
    ])

    if style == "sphere":
        commands.append("style sphere")
    elif style == "stick":
        commands.append("style stick")
    elif style == "cartoon":
        commands.extend(["show cartoons", "hide atoms"])

    commands.extend([
        "lighting soft",
        "lighting ambientIntensity 1.3",
    ])

    return "; ".join(commands)


def visualize(
    polymer: "Polymer",
    values: np.ndarray,
    *,
    scale: Scale = Scale.RESIDUE,
    backend: str = "chimerax",
    color: str = "indianred",
    cif_path: Union[str, None] = None,
    chain: Union[str, int, None] = None,
    style: str = "sphere",
    chimerax_path: Union[str, None] = None,
    attr_name: str = "value",
) -> None:
    """
    Visualize per-residue or per-atom values on a 3D structure.

    Opens ChimeraX (or other backend) with the structure colored by the
    provided values.

    Args:
        polymer: Polymer structure to visualize.
        values: Array of values to visualize. Shape must match scale:
            - Scale.RESIDUE: (num_residues,)
            - Scale.ATOM: (num_atoms,)
        scale: Scale of the values (RESIDUE or ATOM).
        backend: Visualization backend. Currently only "chimerax" supported.
        color: Color for high values. Low values are white.
        cif_path: Path to CIF file. If None, writes a temporary file.
        chain: Chain to display (name string or index). None for all.
        style: Display style ("sphere", "stick", "cartoon").
        chimerax_path: Path to ChimeraX executable. Auto-detected if None.
        attr_name: Attribute name for the defattr file.

    Raises:
        ValueError: If backend is not supported.
        FileNotFoundError: If ChimeraX is not found.

    Example:
        >>> import ciffy
        >>> import numpy as np
        >>> polymer = ciffy.load("structure.cif")
        >>> values = np.random.rand(polymer.size(ciffy.RESIDUE))
        >>> ciffy.visualize.visualize(polymer, values)
    """
    if backend != "chimerax":
        raise ValueError(f"Backend '{backend}' not supported. Use 'chimerax'.")

    values = np.asarray(values)

    # Get chain name if index provided
    chain_name = None
    if chain is not None:
        if isinstance(chain, int):
            chain_name = polymer.names[chain]
        else:
            chain_name = chain

    # Write CIF if not provided
    if cif_path is None:
        cif_fd, cif_path = tempfile.mkstemp(suffix=".cif")
        os.close(cif_fd)
        polymer.write(cif_path)
        _temp_files.append(cif_path)

    # Write defattr file
    defattr_fd, defattr_path = tempfile.mkstemp(suffix=".defattr")
    os.close(defattr_fd)
    to_defattr(polymer, values, defattr_path, scale=scale, attr_name=attr_name, chain=chain)
    _temp_files.append(defattr_path)

    # Find ChimeraX
    if chimerax_path is None:
        chimerax_path = find_chimerax()

    # Build command script
    max_value = float(np.nanmax(values))
    if max_value == 0:
        max_value = 1.0

    script = _build_visualization_script(
        cif_path=cif_path,
        defattr_path=defattr_path,
        color=color,
        max_value=max_value,
        chain=chain_name,
        style=style,
    )

    # Launch ChimeraX
    cmd = [chimerax_path, "--cmd", script]
    subprocess.run(cmd)

    # Note: Temp files are cleaned up via atexit handler when Python exits.
    # We don't delete immediately as ChimeraX may still be reading them.
