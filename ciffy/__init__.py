__version__ = "0.2.0"

from .main import Polymer, Scale, Molecule, Reduction, load

RESIDUE = Scale.RESIDUE
CHAIN = Scale.CHAIN
MOLECULE = Scale.MOLECULE

PROTEIN = Molecule.PROTEIN
RNA = Molecule.RNA
DNA = Molecule.DNA
