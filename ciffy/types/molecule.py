"""
Molecule type enumeration.

Re-exports Molecule from biochemistry for backwards compatibility.
The canonical definition is auto-generated in biochemistry/_generated_molecule.py.
"""

from ..biochemistry._generated_molecule import Molecule, molecule_type

__all__ = ["Molecule", "molecule_type"]
