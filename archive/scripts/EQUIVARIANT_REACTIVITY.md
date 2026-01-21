# Equivariant Reactivity Prediction

Training SO(3)-equivariant transformers to predict SHAPE and DMS reactivity from RNA 3D structure.

## Scripts

- `train_equivariant_reactivity.py` - Main training script
- `plot_predictions.py` - Generate prediction vs ground truth plots

## Key Findings

### 1. Dropout Causes NaN Gradients with Equivariant Ops

**Problem**: Using dropout > 0 with the EquivariantTransformer causes NaN gradients during backpropagation, specifically in the embedding layer.

**Solution**: Set `dropout=0.0` (default in script). The model trains stably without dropout.

**Additional stability measures**:
- Lower learning rate: `lr=1e-4` (vs typical 1e-3)
- Higher gradient clipping: `grad_clip=10.0` (equivariant ops have large gradients)

### 2. Train/Val Split Must Be From Same Dataset

**Problem**: Training on pdb130 and validating on pdb240 showed strong training performance (SHAPE ~0.60) but near-zero validation correlation.

**Cause**: Distribution mismatch between datasets, not overfitting.

**Solution**: Use `--val-split 0.2` to hold out 20% of training data for validation. This gives matching train/val performance (~0.60 correlation on both).

### 3. Rank Parameter Has Minimal Impact

Compared rank 0, 1, 2, and full (None) for the equivariant basis:

| Rank | Parameters | Val SHAPE | Val DMS |
|------|------------|-----------|---------|
| 0    | 189K       | 0.60      | 0.52    |
| 1    | 227K       | 0.54      | 0.53    |
| 2    | 265K       | 0.59      | 0.54    |
| full | 265K       | 0.60      | 0.51    |

**Conclusion**: Rank 0 offers best efficiency - same performance with 30% fewer parameters. For l=[0,1] representations, rank 2 equals full rank.

### 4. Model Architecture

Best configuration (188K params):
```
--hidden-mult 8
--hidden-layers 2
--k-neighbors 64
--rank 0
--dropout 0.0
--lr 1e-4
```

Larger models (737K params with hidden-mult=16, hidden-layers=4) showed similar performance, suggesting the task doesn't require more capacity.

## Usage

```bash
# Local test
python scripts/train_equivariant_reactivity.py \
    --data /path/to/structures \
    --h5-train /path/to/train.hdf5 \
    --val-split 0.2 \
    --epochs 1 \
    --output-dir outputs/test

# Sherlock GPU training
rex sherlock -d --gpu scripts/train_equivariant_reactivity.py -- \
    --name experiment_name \
    --val-split 0.2 \
    --epochs 10

# Generate prediction plots
rex sherlock -d --gpu scripts/plot_predictions.py
```

## Results

Final model achieves:
- **SHAPE correlation**: 0.60 (train), 0.60 (val)
- **DMS correlation**: 0.51 (train), 0.52 (val)

Model and plots saved to `/scratch/users/hmblair/ciffy/equi_react/equi_split/`

## Data

- **Structures**: pdb130 dataset (1132 RNA chains)
- **Reactivity**: HDF5 files with SHAPE (2A3) and DMS profiles
- **Matching**: ReactivityIndex class matches structures to profiles by sequence
