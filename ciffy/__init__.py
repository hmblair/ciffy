__version__ = "0.3.0"

from .main import Polymer, Scale, Molecule, Reduction, load
from .rmsd import _kabsch_distance as rmsd

RESIDUE = Scale.RESIDUE
CHAIN = Scale.CHAIN
MOLECULE = Scale.MOLECULE

PROTEIN = Molecule.PROTEIN
RNA = Molecule.RNA
DNA = Molecule.DNA
