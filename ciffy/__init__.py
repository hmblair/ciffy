__version__ = "0.1.1"

from .main import Polymer, load, Scale, Molecule, Reduction

RESIDUE = Scale.RESIDUE
CHAIN = Scale.CHAIN
MOLECULE = Scale.MOLECULE

PROTEIN = Molecule.PROTEIN
RNA = Molecule.RNA
DNA = Molecule.DNA
