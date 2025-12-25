# Known Issues

## ResidueFlowModel: Fixed Atom Set Requirement

- **Severity**: LOW (by design)
- **Location**: `ciffy/nn/flow/residue/`

The PCA + Flow architecture requires a fixed set of atoms per residue type. Missing atoms at inference time will cause failures.

**Root Cause**: PCA projection matrix V has shape `(k, n_atoms×3)` - a fixed dimensionality. The model cannot handle variable-length inputs.

**Current Mitigation**: The `min_coverage` parameter (default 0.9) filters training data to only include atoms present in ≥90% of structures, ensuring the trained model uses commonly-available atoms.

**Potential Fix**: Replace PCA with a set-based encoder (e.g., PointNet-style shared MLP + pooling). The normalizing flow layers would remain unchanged - only the coordinate-to-latent projection needs modification.

