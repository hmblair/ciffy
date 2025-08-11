import ciffy
from tqdm import trange
import numpy as np
from time import time
from Bio.PDB.MMCIFParser import FastMMCIFParser

import os
dn = os.path.dirname(os.path.realpath(__file__))


def _bio_get_coords(iden: str, file: str) -> np.ndarray:

    parser = FastMMCIFParser()
    stru = parser.get_structure(iden, file)
    coords = []

    for model in stru:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    coords.append(atom.get_vector()._ar)

    return np.stack(coords, axis=0)


# ID = "3SKW"
ID = "9GCM"
file = dn + "/" + ID + ".cif"

runs = 100

start = time()
for _ in trange(runs):
    poly = ciffy.load(file)
    breakpoint()
c_elapsed = time() - start

print(poly)
print(f"Average time (ciffy): {c_elapsed / runs}")

start = time()
for _ in trange(runs):
    bp_coords = _bio_get_coords(ID, file)
bp_elapsed = time() - start

print(f"Average time (BioPython): {bp_elapsed / runs}")

print(f"Speedup factor: {bp_elapsed / c_elapsed}")
