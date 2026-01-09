#!/usr/bin/env python3
"""
Example: Extending a polymer chain with new residues.

This example shows how to:
1. Create a polymer from sequence
2. Extend it by appending residues
3. Save the extended structure

By default, residues are placed in a straight line along the Z-axis with
identical orientations. For realistic backbone geometry, provide an SE(3)
transform from a flow model.
"""

import ciffy
from ciffy import Residue, Scale
from ciffy import template


def main():
    # =========================================================================
    # Create a starting polymer
    # =========================================================================

    # Start with a short RNA sequence
    polymer = template("ac", id="extended_rna")
    print(f"Starting polymer:")
    print(f"  Sequence: {polymer.sequence_str()}")
    print(f"  Residues: {polymer.size(Scale.RESIDUE)}")
    print(f"  Atoms: {polymer.size()}")
    print()

    # =========================================================================
    # Extend the chain
    # =========================================================================

    # Add a guanine residue - just pass the residue type
    polymer = polymer.append(Residue.G)
    print(f"After adding G:")
    print(f"  Sequence: {polymer.sequence_str()}")
    print(f"  Residues: {polymer.size(Scale.RESIDUE)}")
    print()

    # Add a uracil residue
    polymer = polymer.append(Residue.U)
    print(f"After adding U:")
    print(f"  Sequence: {polymer.sequence_str()}")
    print(f"  Residues: {polymer.size(Scale.RESIDUE)}")
    print()

    # =========================================================================
    # Save the extended structure
    # =========================================================================

    output_path = "/tmp/extended_rna.cif"
    polymer.write(output_path)
    print(f"Saved extended polymer to: {output_path}")
    print()

    # =========================================================================
    # Verify by reloading
    # =========================================================================

    reloaded = ciffy.load(output_path)
    print(f"Reloaded polymer:")
    print(f"  Sequence: {reloaded.sequence_str()}")
    print(f"  Residues: {reloaded.size(Scale.RESIDUE)}")
    print(reloaded)


if __name__ == "__main__":
    main()
