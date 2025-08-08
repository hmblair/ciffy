import ciffy
from tqdm import trange
from time import time
from Bio.PDB.MMCIFParser import MMCIFParser

file = "9GCM.cif"
ID = "9GCM"
runs = 100
parser = MMCIFParser()

start = time()
for _ in trange(runs):
    poly = ciffy.load(file)
c_elapsed = time() - start

print(f"Average time (ciffy): {c_elapsed / runs}")

start = time()
for _ in trange(runs):
    parser.get_structure(ID, file)
bp_elapsed = time() - start

print(f"Average time (BioPython): {bp_elapsed / runs}")

print(f"Speedup factor: {bp_elapsed / c_elapsed}")
