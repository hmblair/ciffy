HIGH Priority
Cartesian-to-Internal Backward Pass

Goal: Implement backward pass for C conversion functions to enable gradient flow.

Files likely affected:

    ciffy/src/internal/geometry.c - C conversion implementations
    ciffy/src/internal/batch.c - Batch processing functions
    ciffy/backend/torch_ops.py - PyTorch autograd integration
    tests/test_internal.py - Gradient testing

Full NN + Internal Test

Goal: End-to-end test: sequence → embedding → dihedral prediction → structure → RMSD → gradient flow.

Files likely affected:

    tests/test_nn.py - Neural network integration tests
    ciffy/nn/embedding.py - Embedding dimension validation
    ciffy/template.py - Template from sequence construction
    ciffy/operations/metrics.py - RMSD computation

MEDIUM Priority
Improved Polymer Template Construction

Goal: Enhance from_sequence method with ideal dihedral angles and validate internal representation compatibility.

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
