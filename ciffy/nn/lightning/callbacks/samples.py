"""Callback for generating and saving sample structures during training.

Useful for monitoring generative model quality.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from lightning import Callback, LightningModule, Trainer

if TYPE_CHECKING:
    from ciffy import Polymer


class SampleGenerationCallback(Callback):
    """Generate and save sample structures during validation.

    This callback:
    - Generates samples from the model at specified intervals
    - Saves ground truth and generated structures as CIF files
    - Computes RMSD metrics between samples and ground truth

    Requires the model to implement a `sample()` method that follows
    the PolymerGenerativeModel protocol.

    Args:
        sample_dir: Directory to save generated samples.
        n_samples: Number of samples to generate per template.
        num_steps: Number of diffusion steps for sampling.
        every_n_epochs: Generate samples every N epochs.
        templates: Optional list of template Polymers. If None, uses
            validation data from the datamodule.
        max_templates: Maximum templates to sample from.

    Example:
        >>> callback = SampleGenerationCallback(
        ...     sample_dir="./samples",
        ...     n_samples=5,
        ...     every_n_epochs=10,
        ... )
        >>> trainer = Trainer(callbacks=[callback])
    """

    def __init__(
        self,
        sample_dir: str | Path,
        n_samples: int = 5,
        num_steps: int = 50,
        every_n_epochs: int = 10,
        templates: list["Polymer"] | None = None,
        max_templates: int = 5,
    ) -> None:
        super().__init__()
        self.sample_dir = Path(sample_dir)
        self.n_samples = n_samples
        self.num_steps = num_steps
        self.every_n_epochs = every_n_epochs
        self.templates = templates
        self.max_templates = max_templates

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Generate samples at the end of validation epochs."""
        current_epoch = trainer.current_epoch

        # Check if we should generate samples this epoch
        if (current_epoch + 1) % self.every_n_epochs != 0:
            return

        # Skip if model doesn't have sample method
        model = getattr(pl_module, "model", pl_module)
        if not hasattr(model, "sample"):
            return

        # Get templates
        templates = self._get_templates(trainer, pl_module)
        if not templates:
            return

        # Create output directory
        epoch_dir = self.sample_dir / f"epoch_{current_epoch + 1:04d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)

        # Generate samples
        pl_module.eval()
        rmsd_values = []

        for i, template in enumerate(templates[: self.max_templates]):
            seq_dir = epoch_dir / f"template_{i}"
            seq_dir.mkdir(exist_ok=True)

            try:
                # Save ground truth
                template.write(str(seq_dir / "ground_truth.cif"))

                # Generate samples
                with torch.no_grad():
                    samples = model.sample(
                        template,
                        n_samples=self.n_samples,
                        num_steps=self.num_steps,
                    )

                # Save samples and compute RMSD
                for j, sample in enumerate(samples):
                    sample.write(str(seq_dir / f"sample_{j}.cif"))

                    # Compute RMSD if coordinates match
                    if sample.size() == template.size():
                        from ciffy import rmsd

                        rmsd_val = rmsd(
                            sample.coordinates.reshape(1, -1, 3),
                            template.coordinates.reshape(1, -1, 3),
                        )[0]
                        rmsd_values.append(float(rmsd_val))

            except Exception as e:
                # Log error but continue
                print(f"Sample generation failed for template {i}: {e}")
                continue

        # Log average RMSD
        if rmsd_values:
            avg_rmsd = sum(rmsd_values) / len(rmsd_values)
            pl_module.log("val/sample_rmsd", avg_rmsd)

    def _get_templates(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> list["Polymer"]:
        """Get template polymers for sampling."""
        if self.templates is not None:
            return self.templates

        # Try to get from datamodule
        if not hasattr(trainer, "datamodule") or trainer.datamodule is None:
            return []

        dm = trainer.datamodule

        # For latent diffusion, get polymers from filtered dataset
        if hasattr(dm, "val_dataset"):
            templates = []
            val_data = dm.val_dataset

            # Handle Subset wrapper
            if hasattr(val_data, "dataset"):
                dataset = val_data.dataset
                indices = val_data.indices if hasattr(val_data, "indices") else range(len(val_data))
            else:
                dataset = val_data
                indices = range(len(val_data))

            for idx in list(indices)[: self.max_templates]:
                try:
                    # Get underlying polymer
                    if hasattr(dataset, "filtered_dataset"):
                        polymer = dataset.filtered_dataset[idx]
                    elif hasattr(dataset, "__getitem__"):
                        item = dataset[idx]
                        if hasattr(item, "coordinates"):
                            polymer = item
                        else:
                            continue
                    else:
                        continue

                    templates.append(polymer)
                except Exception:
                    continue

            return templates

        return []


__all__ = ["SampleGenerationCallback"]
