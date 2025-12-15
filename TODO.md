HIGH Priority

RNA CHI Angle Optimization Without Ring Deformation

Goal: Enable modification of RNA glycosidic (CHI) dihedral angles without deforming nucleobase rings.

Currently, optimizing CHI_PURINE or CHI_PYRIMIDINE dihedrals causes ring atoms to move inconsistently because they reference a mix of sugar atoms (C1') and other ring atoms in the Z-matrix. When CHI changes, atoms referencing the sugar move differently than atoms referencing other ring atoms, breaking planarity.

Potential approaches:
- Compute Jacobian of ring dihedrals w.r.t. Z-matrix dihedrals and apply compensating updates
- Ensure all ring atoms reference only other ring atoms (except the glycosidic bond attachment)
- Use a hybrid representation: Z-matrix for backbone, rigid body for bases

Files likely affected:
- ciffy/src/codegen/residue.py - Canonical reference definitions for base atoms
- ciffy/src/internal/graph.c - Z-matrix construction
- ciffy/internal/coordinates.py - Coordinate manager dihedral methods

See test: tests/test_internal.py::TestRingPreservation::test_ring_torsion_during_backbone_optimization

MEDIUM Priority
Improved Polymer Template Construction

Goal: Enhance from_sequence method with ideal dihedral angles.

Files likely affected:

    ciffy/template.py - Template construction logic
    ciffy/biochemistry/_generated_residues.py - Ideal coordinates data
    ciffy/biochemistry/constants.py - Dihedral angle constants
    tests/test_template.py - Template validation tests

CUDA Polymer Conversions

Goal: GPU-native conversion algorithms to avoid CPU-GPU memory transfers.

Files likely affected:

    ciffy/backend/torch_ops.py - CUDA operation implementations
    ciffy/src/internal/ - New CUDA source files
    ciffy/internal/nerf.py - GPU-aware NERF algorithm
    tests/test_device.py - GPU testing
