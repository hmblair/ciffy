"""Tests for ciffy.nn.flow.residue module."""

import tempfile
import numpy as np
import pytest
import torch

from ciffy.nn.flow.residue import PCAFlow, train_pca_flow
from ciffy.nn.flow.residue.data import compute_pca


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
            X_recon = flow.decode(z)  # Returns (N, d) flat

        # Flatten X for comparison (PCAFlow now returns flat output)
        X_flat = X.reshape(X.shape[0], -1)

        # RMSD should be small (bounded by PCA truncation)
        rmsd = torch.sqrt(((X_flat - X_recon) ** 2).mean()).item()
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

        # PCAFlow returns flat output (N, d)
        assert samples.shape == (50, 30)  # 10 atoms * 3 = 30
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
        from ciffy.nn.flow.residue import ResidueFlowModel
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
            n_atoms=len(atom_indices),
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
        from ciffy.nn.flow.residue import ResidueFlowModel
        from ciffy.nn.flow.residue.data import compute_pca
        from ciffy.biochemistry import Residue

        n_samples, n_atoms = sample_coords.shape[:2]

        # Create extended representation (coords + 6D transform)
        coords_flat = sample_coords.reshape(n_samples, -1)
        transforms = np.random.randn(n_samples, 6).astype(np.float32) * 0.1
        extended = np.concatenate([coords_flat, transforms], axis=1)

        # Compute PCA and create flow
        V, mean, _, var_explained = compute_pca(extended, n_components=6)
        V_t = torch.from_numpy(V).float()
        mean_t = torch.from_numpy(mean).float()
        flow = PCAFlow(V_t, mean_t, n_layers=4, hidden_dim=32, bound=3.0)

        pca_rmsd = float(np.sqrt(((extended - (extended - mean) @ V.T @ V - mean) ** 2).mean()))

        # Create model without JIT
        model_no_jit = ResidueFlowModel(
            flow=flow,
            residue=Residue.A,
            atom_indices=list(Residue.A.index()[:n_atoms]),
            n_atoms=n_atoms,
            pca_rmsd=pca_rmsd,
            var_explained=float(var_explained[-1]),
            jit=False,
        )

        # Create model with JIT
        model_jit = ResidueFlowModel(
            flow=flow,
            residue=Residue.A,
            atom_indices=list(Residue.A.index()[:n_atoms]),
            n_atoms=n_atoms,
            pca_rmsd=pca_rmsd,
            var_explained=float(var_explained[-1]),
            jit=True,
        )

        assert not model_no_jit.is_jit
        assert model_jit.is_jit

        # Verify outputs match (decode returns coords, transforms tuple)
        z = torch.randn(5, 6)
        with torch.no_grad():
            coords_no_jit, trans_no_jit = model_no_jit.decode(z)
            coords_jit, trans_jit = model_jit.decode(z)
        assert torch.allclose(coords_no_jit, coords_jit, atol=1e-5)
        assert torch.allclose(trans_no_jit, trans_jit, atol=1e-5)


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


class TestResidueFlowModel:
    """Tests for ResidueFlowModel (with link transforms)."""

    @pytest.fixture
    def sample_extended_data(self):
        """Generate synthetic extended representation data."""
        np.random.seed(42)
        n_samples = 100
        n_atoms = 22  # Typical adenosine

        # Create coordinates with some structure
        base_coords = np.random.randn(n_atoms, 3)
        noise = 0.1 * np.random.randn(n_samples, n_atoms, 3)
        coords = base_coords + noise

        # Create transforms with realistic values
        # axis-angle: small rotations (~0.5 rad)
        # translation: typical O3'-P distance (~1.6 Å)
        axis_angles = 0.5 * np.random.randn(n_samples, 3)
        translations = np.array([0.0, 0.0, 1.6]) + 0.1 * np.random.randn(n_samples, 3)
        transforms = np.concatenate([axis_angles, translations], axis=1)

        return coords.astype(np.float32), transforms.astype(np.float32)

    def test_extended_pca_flow(self, sample_extended_data):
        """Test PCAFlow with extended representation."""
        coords, transforms = sample_extended_data
        n_samples = len(coords)

        # Create extended representation
        coords_flat = coords.reshape(n_samples, -1)
        extended = np.concatenate([coords_flat, transforms], axis=1)

        # Compute PCA
        V, mean, _, var_explained = compute_pca(extended, n_components=8)
        V_t = torch.from_numpy(V).float()
        mean_t = torch.from_numpy(mean).float()

        flow = PCAFlow(V_t, mean_t, n_layers=4, hidden_dim=32, bound=3.0)

        # Extended dim = n_atoms*3 + 6
        assert flow.d == 22 * 3 + 6

        # Test encode-decode
        X = torch.from_numpy(extended).float()
        with torch.no_grad():
            z = flow.encode(X)
            recon = flow.decode(z)

        # Flatten decoded output for comparison
        recon_flat = recon.reshape(n_samples, -1)
        rmsd = torch.sqrt(((X - recon_flat) ** 2).mean()).item()
        assert rmsd < 0.5

    def test_decode_split(self, sample_extended_data):
        """Test that decode correctly splits coords and transforms."""
        from ciffy.nn.flow.residue.model import ResidueFlowModel
        from ciffy.biochemistry import Residue

        coords, transforms = sample_extended_data
        n_samples, n_atoms = coords.shape[:2]

        # Create extended representation
        coords_flat = coords.reshape(n_samples, -1)
        extended = np.concatenate([coords_flat, transforms], axis=1)

        # Create and train flow
        V, mean, _, _ = compute_pca(extended, n_components=8)
        V_t = torch.from_numpy(V).float()
        mean_t = torch.from_numpy(mean).float()
        flow = PCAFlow(V_t, mean_t, n_layers=4, hidden_dim=32)

        # Create model (use first 22 atoms of adenosine as placeholder)
        model = ResidueFlowModel(
            flow=flow,
            residue=Residue.A,
            atom_indices=list(Residue.A.index()[:n_atoms]),
            n_atoms=n_atoms,
            pca_rmsd=0.1,
            var_explained=0.95,
        )

        # Test decode
        z = torch.randn(5, 8)
        with torch.no_grad():
            decoded_coords, decoded_transforms = model.decode(z)

        assert decoded_coords.shape == (5, n_atoms, 3)
        assert decoded_transforms.shape == (5, 6)

    def test_position_next_residue(self):
        """Test positioning next residue using transform."""
        from ciffy.nn.flow.residue.data import (
            position_next_residue,
            compute_link_frames,
            compute_relative_transform,
        )
        from ciffy.biochemistry import Residue

        np.random.seed(42)

        # Create synthetic residue coordinates
        # We need atoms: C4', C3', O3' for res1 and P, O5', OP1 for res2
        atoms = list(Residue.A.index()[:22])  # Get atom indices
        n_atoms = len(atoms)

        # Random coordinates (just for testing transform logic)
        coords1 = np.random.randn(n_atoms, 3).astype(np.float32)
        coords2 = np.random.randn(n_atoms, 3).astype(np.float32)

        # Compute transform from coords1 to coords2
        o1, R1, o2, R2 = compute_link_frames(coords1, coords2, atoms, Residue.A)
        transform = compute_relative_transform(o1, R1, o2, R2)

        # Position coords2 using the transform (should recover original positions)
        coords2_positioned = position_next_residue(
            coords1, coords2, transform, atoms, Residue.A
        )

        # The P atom should be at the target position
        atom_to_col = {a: i for i, a in enumerate(atoms)}
        p_idx = atom_to_col[Residue.A.P.value]

        # P position from original coords2 after positioning
        p_positioned = coords2_positioned[p_idx]
        # Target P position from transform
        o1_new, R1_new, _, _ = compute_link_frames(coords1, coords2, atoms, Residue.A)
        from ciffy.nn.flow.residue.data import apply_relative_transform
        target_p, _ = apply_relative_transform(o1_new, R1_new, transform)

        np.testing.assert_allclose(p_positioned, target_p, atol=1e-5)

    def test_link_transform_stats(self, sample_extended_data):
        """Test that transform statistics are reasonable."""
        _, transforms = sample_extended_data

        # axis-angle magnitude (rotation angle)
        rot_angles = np.linalg.norm(transforms[:, :3], axis=1)
        mean_rot = np.rad2deg(rot_angles.mean())

        # translation magnitude
        trans_dist = np.linalg.norm(transforms[:, 3:], axis=1)
        mean_trans = trans_dist.mean()

        # Should be reasonable values
        assert 0 < mean_rot < 180  # Degrees
        assert 0.5 < mean_trans < 5.0  # Angstroms (typical O3'-P is ~1.6)

    def test_save_load_model(self, sample_extended_data):
        """Test save/load for ResidueFlowModel."""
        from ciffy.nn.flow.residue.model import ResidueFlowModel
        from ciffy.biochemistry import Residue

        coords, transforms = sample_extended_data
        n_samples, n_atoms = coords.shape[:2]

        # Create extended representation
        coords_flat = coords.reshape(n_samples, -1)
        extended = np.concatenate([coords_flat, transforms], axis=1)

        # Create and train flow
        V, mean, _, var_explained = compute_pca(extended, n_components=8)
        V_t = torch.from_numpy(V).float()
        mean_t = torch.from_numpy(mean).float()
        flow = PCAFlow(V_t, mean_t, n_layers=4, hidden_dim=32)

        # Create model
        model = ResidueFlowModel(
            flow=flow,
            residue=Residue.A,
            atom_indices=list(Residue.A.index()[:n_atoms]),
            n_atoms=n_atoms,
            pca_rmsd=0.1,
            var_explained=float(var_explained[-1]),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = tmpdir + "/residue_model"
            model.save(save_path)

            # Load and verify
            loaded = ResidueFlowModel.load(save_path)

            assert loaded.residue.name == "A"
            assert loaded.n_atoms == n_atoms
            assert loaded.latent_dim == 8

            # Verify outputs match
            z_test = torch.randn(3, 8)
            with torch.no_grad():
                orig_coords, orig_trans = model.decode(z_test)
                loaded_coords, loaded_trans = loaded.decode(z_test)

            assert torch.allclose(orig_coords, loaded_coords, atol=1e-6)
            assert torch.allclose(orig_trans, loaded_trans, atol=1e-6)
