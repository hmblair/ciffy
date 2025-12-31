"""
Backwards-compatible re-exports from ciffy.nn.polymer.

This module re-exports PolymerModel (formerly PolymerFlowModel) and
ResidueGenerativeCore from their new location in ciffy.nn.polymer.

The classes have been moved because they are not flow-specific - they work
with any residue-level generative model implementing ResidueGenerativeCore
(including both ResidueFlowModel and ResidueVAE).

Preferred imports:
    >>> from ciffy.nn import PolymerModel, ResidueGenerativeCore

Legacy imports (still work):
    >>> from ciffy.nn.flow import PolymerFlowModel  # Deprecated name
    >>> from ciffy.nn.flow.polymer import PolymerFlowModel  # Deprecated location
"""

import warnings
from ciffy.nn.polymer import (
    PolymerModel,
    PolymerFlowModel,
    ResidueGenerativeCore,
    SequenceArray,
)


def __getattr__(name: str):
    """Emit deprecation warning when accessing PolymerFlowModel."""
    if name == "PolymerFlowModel":
        warnings.warn(
            "PolymerFlowModel is deprecated, use PolymerModel from ciffy.nn instead. "
            "The class works with any ResidueGenerativeCore (Flow, VAE, etc.).",
            DeprecationWarning,
            stacklevel=2,
        )
        return PolymerFlowModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "PolymerModel",
    "PolymerFlowModel",  # Deprecated alias
    "ResidueGenerativeCore",
    "SequenceArray",
]
