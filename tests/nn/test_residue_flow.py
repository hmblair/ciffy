"""Tests for ciffy.nn.residue_flow module."""

import tempfile
import numpy as np
import pytest
import torch

from ciffy.nn.residue_flow import PCAFlow, train_pca_flow
from ciffy.nn.residue_flow.data import compute_pca


@pytest.fixture
def sample_coords():
    """Generate synthetic coordinate data for testing."""
    np.random.seed(42)
    n_samples = 100
    n_atoms = 10

    # Create coordinates with some structure
    base = np.random.randn(n_atoms, 3)
    noise = 0.1 * np.random.randn(n_samples, n_atoms, 3)
    coords = base + noise

    return coords.astype(np.float32)


class TestPCAFlow:
    """Tests for PCAFlow model."""

    def test_init(self, sample_coords):
        """Test model initialization."""
        V, mean, _, _ = compute_pca(sample_coords, n_components=6)
        V_t = torch.from_numpy(V).float()
        mean_t = torch.from_numpy(mean).float()

        flow = PCAFlow(V_t, mean_t, n_layers=4, hidden_dim=32, bound=3.0)

        assert flow.k == 6
        assert flow.d == 30  # 10 atoms * 3
        assert flow.bound == 3.0

    def test_encode_decode_roundtrip(self, sample_coords):
        """Test that encode-decode is approximately invertible."""
        flow, _ = train_pca_flow(
            sample_coords, latent_dim=6, n_layers=4, hidden_dim=32,
            n_epochs=10, verbose=False
        )
        flow.eval()

        X = torch.from_numpy(sample_coords[:10]).float()
        with torch.no_grad():
            z = flow.encode(X)
            X_recon = flow.decode(z)

        # RMSD should be small (bounded by PCA truncation)
        rmsd = torch.sqrt(((X - X_recon) ** 2).mean()).item()
        assert rmsd < 0.5  # Reasonable threshold for 6D PCA

    def test_bound_prevents_extrapolation(self, sample_coords):
        """Test that bound parameter limits latent values."""
        flow, _ = train_pca_flow(
            sample_coords, latent_dim=6, n_layers=4, hidden_dim=32,
            bound=2.0, n_epochs=10, verbose=False
        )
        flow.eval()

        # Extreme latent values
        z_extreme = torch.tensor([[10.0, -10.0, 5.0, -5.0, 8.0, -8.0]])

        with torch.no_grad():
            coords = flow.decode(z_extreme)

        # Should produce valid output (not NaN/Inf)
        assert torch.isfinite(coords).all()

    def test_sample(self, sample_coords):
        """Test sampling from the model."""
        flow, _ = train_pca_flow(
            sample_coords, latent_dim=6, n_layers=4, hidden_dim=32,
            n_epochs=10, verbose=False
        )
        flow.eval()

        samples = flow.sample(50)

        assert samples.shape == (50, 10, 3)
        assert torch.isfinite(samples).all()

    def test_log_prob(self, sample_coords):
        """Test log probability computation."""
        flow, _ = train_pca_flow(
            sample_coords, latent_dim=6, n_layers=4, hidden_dim=32,
            n_epochs=10, verbose=False
        )
        flow.eval()

        X = torch.from_numpy(sample_coords[:10]).float()
        log_prob = flow.log_prob(X)

        assert log_prob.shape == (10,)
        assert torch.isfinite(log_prob).all()

    def test_gradient_flow(self, sample_coords):
        """Test that gradients flow through decode."""
        flow, _ = train_pca_flow(
            sample_coords, latent_dim=6, n_layers=4, hidden_dim=32,
            n_epochs=10, verbose=False
        )
        flow.eval()

        z = torch.randn(1, 6, requires_grad=True)
        coords = flow.decode(z)
        loss = coords.sum()
        loss.backward()

        assert z.grad is not None
        assert z.grad.shape == (1, 6)

    def test_device_transfer(self, sample_coords):
        """Test model works on different devices."""
        flow, _ = train_pca_flow(
            sample_coords, latent_dim=6, n_layers=4, hidden_dim=32,
            n_epochs=10, verbose=False
        )

        # Test on CPU
        z_cpu = torch.randn(5, 6)
        with torch.no_grad():
            coords_cpu = flow.decode(z_cpu)
        assert coords_cpu.device.type == "cpu"

        # Test on MPS if available
        if torch.backends.mps.is_available():
            flow_mps = flow.to("mps")
            z_mps = torch.randn(5, 6, device="mps")
            with torch.no_grad():
                coords_mps = flow_mps.decode(z_mps)
            assert coords_mps.device.type == "mps"

    def test_save_load_state_dict(self, sample_coords):
        """Test model serialization via state_dict."""
        flow, _ = train_pca_flow(
            sample_coords, latent_dim=6, n_layers=4, hidden_dim=32,
            bound=2.5, n_epochs=10, verbose=False
        )

        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            torch.save(flow.state_dict(), f.name)
            state = torch.load(f.name)

        # Check key components are saved
        assert "V" in state
        assert "mean" in state
        assert "layers.0.log_scale" in state  # ActNorm

    def test_safetensors_roundtrip(self, sample_coords):
        """Test save/load with safetensors format."""
        from ciffy.nn.residue_flow import ResidueFlowModel
        from ciffy.biochemistry import Residue

        flow, info = train_pca_flow(
            sample_coords, latent_dim=6, n_layers=4, hidden_dim=32,
            bound=2.5, n_epochs=10, verbose=False
        )

        # Create a mock model with valid Residue.A atom indices
        atom_indices = list(Residue.A.index()[:10])  # First 10 atoms of adenosine
        model = ResidueFlowModel(
            flow=flow,
            residue=Residue.A,
            atom_indices=atom_indices,
            pca_rmsd=info["pca_rmsd"],
            var_explained=info["var_explained"],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = tmpdir + "/test_model"
            model.save(save_path)

            # Check files exist
            import os
            assert os.path.exists(save_path + "/tensors.safetensors")
            assert os.path.exists(save_path + "/config.json")

            # Load and verify
            loaded = ResidueFlowModel.load(save_path)
            assert loaded.residue.name == "A"
            assert len(loaded.atoms) == len(atom_indices)
            assert loaded.latent_dim == 6

            # Verify outputs match
            z_test = torch.randn(3, 6)
            with torch.no_grad():
                orig_out = model.flow.decode(z_test)
                loaded_out = loaded.flow.decode(z_test)
            assert torch.allclose(orig_out, loaded_out, atol=1e-6)

    def test_jit_compilation(self, sample_coords):
        """Test JIT compilation of decoder."""
        from ciffy.nn.residue_flow import ResidueFlowModel
        from ciffy.biochemistry import Residue

        flow, info = train_pca_flow(
            sample_coords, latent_dim=6, n_layers=4, hidden_dim=32,
            n_epochs=10, verbose=False
        )

        # Create model without JIT
        model_no_jit = ResidueFlowModel(
            flow=flow,
            residue=Residue.A,
            atom_indices=list(Residue.A.index()[:10]),
            pca_rmsd=info["pca_rmsd"],
            var_explained=info["var_explained"],
            jit=False,
        )

        # Create model with JIT
        model_jit = ResidueFlowModel(
            flow=flow,
            residue=Residue.A,
            atom_indices=list(Residue.A.index()[:10]),
            pca_rmsd=info["pca_rmsd"],
            var_explained=info["var_explained"],
            jit=True,
        )

        assert not model_no_jit.is_jit
        assert model_jit.is_jit

        # Verify outputs match
        z = torch.randn(5, 6)
        with torch.no_grad():
            out_no_jit = model_no_jit.decode(z)
            out_jit = model_jit.decode(z)
        assert torch.allclose(out_no_jit, out_jit, atol=1e-6)


class TestTrainPCAFlow:
    """Tests for train_pca_flow function."""

    def test_basic_training(self, sample_coords):
        """Test basic training workflow."""
        flow, info = train_pca_flow(
            sample_coords, latent_dim=6, n_epochs=20, verbose=False
        )

        assert isinstance(flow, PCAFlow)
        assert "pca_rmsd" in info
        assert "var_explained" in info
        assert "losses" in info
        assert len(info["losses"]) == 20

    def test_pca_rmsd_matches_flow_rmsd(self, sample_coords):
        """Test that flow RMSD equals PCA RMSD (flow is invertible)."""
        flow, info = train_pca_flow(
            sample_coords, latent_dim=6, n_epochs=50, verbose=False
        )

        # Flow should be exactly invertible, so RMSD = PCA RMSD
        assert abs(info["flow_rmsd"] - info["pca_rmsd"]) < 0.01

    def test_more_dims_lower_rmsd(self, sample_coords):
        """Test that more latent dims gives lower RMSD."""
        _, info_6d = train_pca_flow(
            sample_coords, latent_dim=6, n_epochs=10, verbose=False
        )
        _, info_12d = train_pca_flow(
            sample_coords, latent_dim=12, n_epochs=10, verbose=False
        )

        assert info_12d["pca_rmsd"] < info_6d["pca_rmsd"]
