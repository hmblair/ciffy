#!/usr/bin/env python3
"""
Example: Generating template polymers from sequences.

This example shows how to create Polymer objects from sequence strings,
which is useful for generative modeling workflows where you need the
correct atom types and structure but will generate coordinates separately.
"""

import numpy as np
from ciffy import from_sequence, randomize_backbone, Scale, write_cif


def main():
    # =========================================================================
    # Basic Usage
    # =========================================================================

    # RNA sequence (lowercase)
    rna = from_sequence("acgu")
    print(f"RNA 'acgu':")
    print(f"  Residues: {rna.size(Scale.RESIDUE)}")
    print(f"  Atoms: {rna.size()}")
    print(f"  Sequence indices: {list(rna.sequence)}")  # [0, 1, 2, 3] = A, C, G, U
    print()

    # Protein sequence (uppercase)
    protein = from_sequence("MGKLF")
    print(f"Protein 'MGKLF':")
    print(f"  Residues: {protein.size(Scale.RESIDUE)}")
    print(f"  Atoms: {protein.size()}")
    print(f"  Sequence indices: {list(protein.sequence)}")  # M=15, G=10, K=13, L=14, F=9
    print()

    # =========================================================================
    # Inspecting the Template
    # =========================================================================

    # Coordinates use ideal CCD (Chemical Component Dictionary) values
    print(f"Coordinates are ideal CCD values (not zeros)")
    print(f"  Coordinate range: [{rna.coordinates.min():.2f}, {rna.coordinates.max():.2f}]")

    # Atoms have correct type indices
    print(f"First 10 atom indices: {list(rna.atoms[:10])}")

    # Elements are atomic numbers (C=6, N=7, O=8, P=15, H=1)
    unique_elements = set(rna.elements.tolist())
    print(f"Unique elements: {unique_elements}")
    print()

    # =========================================================================
    # Random Backbone Conformations (Ramachandran Sampling)
    # =========================================================================

    print("Generating proteins with random backbone conformations:")

    # Option 1: Use sample_dihedrals parameter
    random_protein = from_sequence("MGKLF", sample_dihedrals=True, seed=42)
    print(f"  Random protein (seed=42): generated with realistic phi/psi angles")

    # Option 2: Use randomize_backbone on existing polymer
    ideal_protein = from_sequence("MGKLF")
    randomized = randomize_backbone(ideal_protein, seed=123)
    print(f"  Randomized backbone (seed=123): applied to existing polymer")

    # Reproducibility: same seed = same structure
    protein1 = from_sequence("MGKLF", sample_dihedrals=True, seed=42)
    protein2 = from_sequence("MGKLF", sample_dihedrals=True, seed=42)
    print(f"  Reproducible: {np.allclose(protein1.coordinates, protein2.coordinates)}")
    print()

    # =========================================================================
    # Saving Generated Structures
    # =========================================================================

    print("Saving generated structures:")

    # Generate a random protein conformation
    sequence = "ACDEFGHIKLMNPQRSTVWY"  # All 20 amino acids
    polymer = from_sequence(sequence, sample_dihedrals=True, seed=42, id="random_protein")

    # Save to CIF file
    output_path = "/tmp/random_protein.cif"
    write_cif(polymer, output_path)
    print(f"  Saved random protein to: {output_path}")

    # Alternative: use the write() method
    polymer.write("/tmp/random_protein_v2.cif")
    print(f"  Also saved via polymer.write() method")
    print()

    # =========================================================================
    # Generative Modeling Workflow
    # =========================================================================

    print("Generative modeling workflow:")

    # 1. Create template from target sequence
    target_seq = "augcaugcaugc"  # 12-mer RNA
    template = from_sequence(target_seq, id="generated_rna")
    n_atoms = template.size()
    print(f"  1. Created template for '{target_seq}' ({n_atoms} atoms)")

    # 2. Generate coordinates with your model (simulated here)
    # In practice: generated_coords = model(template.atoms, template.sequence)
    generated_coords = np.random.randn(n_atoms, 3).astype(np.float32) * 5.0
    print(f"  2. Generated coordinates shape: {generated_coords.shape}")

    # 3. Attach coordinates to template
    template.coordinates = generated_coords
    print(f"  3. Attached coordinates to template")

    # 4. Inspect the generated polymer
    print(f"  4. Generated polymer:")
    print(template)

    # 5. Save as CIF file
    output_path = "/tmp/generated_rna.cif"
    template.write(output_path)
    print(f"  5. Saved to {output_path}")
    print()

    # =========================================================================
    # Backend Options
    # =========================================================================

    # NumPy backend (default)
    np_polymer = from_sequence("acgu", backend="numpy")
    print(f"NumPy backend: {np_polymer.backend}")

    # PyTorch backend (for integration with deep learning)
    torch_polymer = from_sequence("acgu", backend="torch")
    print(f"PyTorch backend: {torch_polymer.backend}")

    # Move to GPU if available
    import torch
    if torch.cuda.is_available():
        gpu_polymer = torch_polymer.to("cuda")
        print(f"GPU polymer device: {gpu_polymer.coordinates.device}")
    print()

    # =========================================================================
    # All 20 Amino Acids with Random Conformation
    # =========================================================================

    all_aa = "ACDEFGHIKLMNPQRSTVWY"
    full_protein = from_sequence(all_aa, sample_dihedrals=True, seed=0)
    print(f"All 20 amino acids with random backbone:")
    print(f"  Sequence: {all_aa}")
    print(f"  Residues: {full_protein.size(Scale.RESIDUE)}")
    print(f"  Total atoms: {full_protein.size()}")

    # Save it
    full_protein.write("/tmp/all_amino_acids.cif")
    print(f"  Saved to: /tmp/all_amino_acids.cif")


if __name__ == "__main__":
    main()
