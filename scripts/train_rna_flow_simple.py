"""Train RNA PolymerFlowModel with hardcoded paths."""

from pathlib import Path

from ciffy.biochemistry import Residue
from ciffy.nn.flow import PolymerFlowModel
from ciffy.nn.flow.residue import ResidueFlowModel
from ciffy.nn.flow.residue.model import ResidueFlowConfig

# Config
DATA_DIR = "/home/hmblair/data/rna"
OUTPUT_DIR = "/home/hmblair/models/rna_flow"
N_EPOCHS = 200
DEVICE = "cuda"

# Create output dir
output_path = Path(OUTPUT_DIR)
output_path.mkdir(parents=True, exist_ok=True)

# Get CIF files
data_path = Path(DATA_DIR)
cif_files = sorted(data_path.glob("*.cif"))
print(f"Found {len(cif_files)} CIF files")

# Config
config = ResidueFlowConfig(
    latent_dim=12,
    n_layers=6,
    use_rotation=True,
    noise_std=0.05,
)
print(f"Config: {config}")

# Train models for each residue
rna_residues = [Residue.A, Residue.U, Residue.G, Residue.C]
residue_models = {}

for residue in rna_residues:
    print(f"\n{'='*60}")
    print(f"Training model for {residue.name}")
    print(f"{'='*60}")

    try:
        model, info = ResidueFlowModel.from_structures(
            cif_files,
            residue,
            config=config,
            n_epochs=N_EPOCHS,
            device=DEVICE,
            verbose=True,
        )

        # Save individual model
        model_path = output_path / residue.name
        model.save(model_path)
        print(f"Saved to {model_path}")

        residue_models[residue] = model

        print(f"  Samples: {info.get('n_samples', 'N/A')}")
        print(f"  Test RMSD: {info.get('test_rmsd', 0):.4f} Å")
        print(f"  Test NLL: {info.get('test_nll', 0):.2f}")

    except Exception as e:
        print(f"Failed: {e}")
        import traceback
        traceback.print_exc()

if not residue_models:
    raise ValueError("No models trained")

# Create PolymerFlowModel
print(f"\n{'='*60}")
print("Creating PolymerFlowModel")
print(f"{'='*60}")

polymer_model = PolymerFlowModel(residue_models)
polymer_model.save(output_path)
print(f"Saved PolymerFlowModel to {output_path}")
print(f"Supported residues: {[r.name for r in polymer_model.supported_types]}")
print("Done!")
