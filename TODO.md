HIGH Priority

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
