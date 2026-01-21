# Configuration Examples

This directory contains example configuration files for training and inference.

## Structure

- **examples/configs/** - Example/_template configuration files (version controlled)
- **configs/** - Your personal configuration files (git-ignored, not shared)

## Setup

1. **Copy examples to personal configs:**

```bash
# Create your personal configs directory if it doesn't exist
mkdir -p configs

# Copy example config as a _template
cp examples/configs/inference_example.yaml configs/inference_local.yaml
```

2. **Edit for your setup:**

```bash
# Edit with your paths and settings
vi configs/inference_local.yaml
```

3. **Run inference:**

```bash
# Use your personal config
ciffy inference configs/inference_local.yaml

# Or multiple personal configs
ciffy inference configs/*.yaml
```

## Example Configs

### `inference_example.yaml`

Basic example with inline sequences for quick testing:
- 3 example sequences (protein, RNA, all amino acids)
- 10 samples per sequence
- Fixed random seed for reproducibility

**Use this as a _template** - copy and customize for your needs.

### `inference_from_fasta.yaml`

Example using FASTA file input for larger-scale generation:
- Loads sequences from `sequences.fasta` file
- 5 samples per sequence
- Higher temperature for more diversity

**Use this _template** when you have many sequences to process.

## HPC Setup

For running on HPC systems, create an HPC-specific config:

```bash
cp examples/configs/inference_example.yaml configs/inference_hpc.yaml
```

Edit for HPC paths:

```yaml
model:
  checkpoint_path: /mnt/hpc/checkpoints/vae_best.pt  # HPC storage
  device: cuda

input:
  sequence_file: /mnt/shared/sequences.fasta

output:
  output_dir: /scratch/username/inference_output  # Use scratch for speed
```

Then submit job:

```bash
ciffy inference configs/inference_hpc.yaml --device cuda
```

## Training Configs

For VAE training, see the main `configs/` directory examples:

- Copy example VAE config
- Customize data paths
- Run: `ciffy experiment configs/vae_custom.yaml`

## Tips

- **Keep examples pristine** - Don't edit files in `examples/configs/`
- **Customize locally** - All your personal configs go in `configs/` directory
- **Version your code** - The example configs are version-controlled
- **Never commit secrets** - Personal configs with paths, usernames, etc. stay private

## Personal vs. Examples

| Location | What | Tracked | Share |
|----------|------|---------|-------|
| `examples/configs/` | Templates showing structure | ✓ Yes | ✓ Yes |
| `configs/` | Your personal settings | ✗ No | ✗ No |

This way:
- GitHub always has clean examples
- Your personal paths/settings stay private
- Easy to switch between environments (laptop, HPC, cloud)
