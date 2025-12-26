# Known Issues

## ResidueFlowModel: Fixed Atom Set Requirement

- **Severity**: LOW (by design)
- **Location**: `ciffy/nn/flow/residue/`

The PCA + Flow architecture requires a fixed set of atoms per residue type. Missing atoms at inference time will cause failures.

**Root Cause**: PCA projection matrix V has shape `(k, n_atoms×3)` - a fixed dimensionality. The model cannot handle variable-length inputs.

**Current Mitigation**: The `min_coverage` parameter (default 0.9) filters training data to only include atoms present in ≥90% of structures, ensuring the trained model uses commonly-available atoms.

**Potential Fix**: Replace PCA with a set-based encoder (e.g., PointNet-style shared MLP + pooling). The normalizing flow layers would remain unchanged - only the coordinate-to-latent projection needs modification.

## Multi-Model Structures: Only Model 1 Supported

- **Severity**: LOW (limitation)
- **Location**: `ciffy/io/loader.py`, `ciffy/src/cif/registry.c`

For multi-model structures (e.g., NMR ensembles), only model 1 is loaded. Attempting to load other models raises `NotImplementedError`.

**Root Cause**: Loading arbitrary models requires filtering atoms by model number throughout the parsing pipeline. The current implementation scans to find where model 1 ends and truncates there.

**Current Behavior**: The `model` parameter in `load()` defaults to 1. Passing any other value raises:
```python
NotImplementedError: Only model 1 is currently supported, got model=N.
```

**Potential Fix**: Add a `is_wrong_model` mask (similar to the existing `is_excluded` chain filter) that marks atoms not belonging to the target model, then filter during batch parsing and counting.
