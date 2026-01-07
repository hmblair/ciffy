"""Tests for ciffy.nn.residue module (ResidueEncoder, ResidueDecoder, ResidueVAE)."""

import pytest
import numpy as np

import ciffy
from ciffy import Scale

from tests.utils import (
    get_test_cif,
    TORCH_AVAILABLE,
)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestPackByResidue:
    """Tests for internal pack_by_residue utility."""

    def test_pack_unpack_roundtrip(self):
        """Test that pack/unpack are inverse operations."""
        import torch
        from ciffy.nn.residue.packing import pack_by_residue, unpack_by_residue

        # Create test data
        torch.manual_seed(42)
        n_atoms = 100
        d_features = 64
        features = torch.randn(n_atoms, d_features)

        # Create counts (5 residues with varying sizes)
        counts = torch.tensor([20, 25, 15, 30, 10])

        # Pack
        packed, mask = pack_by_residue(features, counts)

        # Check packed shape
        n_residues = 5
        max_atoms = counts.max().item()
        assert packed.shape == (n_residues, max_atoms, d_features)
        assert mask.shape == (n_residues, max_atoms)

        # Unpack
        unpacked = unpack_by_residue(packed, mask)

        # Should match original
        assert unpacked.shape == features.shape
        assert torch.allclose(unpacked, features, atol=1e-6)

    def test_mask_correctness(self):
        """Test that mask correctly indicates valid positions."""
        import torch
        from ciffy.nn.residue.packing import pack_by_residue

        n_atoms = 50
        d_features = 32
        features = torch.randn(n_atoms, d_features)

        # Unequal residue sizes
        counts = torch.tensor([10, 5, 20, 15])

        packed, mask = pack_by_residue(features, counts)

        # Check mask sums match counts
        max_atoms = counts.max().item()
        for i, count in enumerate(counts):
            assert mask[i].sum().item() == count.item()
            # First `count` positions should be True
            assert mask[i, :count].all()
            # Remaining should be False
            if count < max_atoms:
                assert not mask[i, count:].any()


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestResidueEncoder:
    """Tests for ResidueEncoder class."""

    def test_output_dim_property(self):
        """Test that output_dim returns latent_dim."""
        from ciffy.nn.residue import ResidueEncoder

        encoder = ResidueEncoder(latent_dim=32, d_model=64)
        assert encoder.output_dim == 32

        encoder2 = ResidueEncoder(latent_dim=64, d_model=128)
        assert encoder2.output_dim == 64

    def test_forward_shape(self):
        """Test forward pass output shapes."""
        import torch
        from ciffy.nn.residue import ResidueEncoder

        encoder = ResidueEncoder(latent_dim=32, d_model=64, n_layers=1)
        encoder.eval()

        # Load a small test structure
        p = ciffy.load(get_test_cif("3SKW")).torch()
        chain = p.chain(0)

        with torch.no_grad():
            z = encoder(chain)

        n_residues = chain.size(Scale.RESIDUE)
        assert z.shape == (n_residues, 32)

    def test_forward_with_distribution(self):
        """Test forward pass returning distribution parameters."""
        import torch
        from ciffy.nn.residue import ResidueEncoder

        encoder = ResidueEncoder(latent_dim=32, d_model=64, n_layers=1)
        encoder.eval()

        p = ciffy.load(get_test_cif("3SKW")).torch()
        chain = p.chain(0)

        with torch.no_grad():
            z, mu, logvar = encoder(chain, return_distribution=True)

        n_residues = chain.size(Scale.RESIDUE)
        assert z.shape == (n_residues, 32)
        assert mu.shape == (n_residues, 32)
        assert logvar.shape == (n_residues, 32)

    def test_forward_with_transforms(self):
        """Test forward pass with pre-computed transforms."""
        import torch
        from ciffy.nn.residue import ResidueEncoder

        encoder = ResidueEncoder(latent_dim=32, d_model=64, n_layers=1)
        encoder.eval()

        p = ciffy.load(get_test_cif("3SKW")).torch()
        chain = p.chain(0)

        # Pre-compute transforms using encoder's internal method
        transforms = encoder._compute_transforms(chain)

        with torch.no_grad():
            z1 = encoder(chain, transforms=transforms)
            z2 = encoder(chain)  # Should compute transforms internally

        # Should be identical since same transforms
        assert z1.shape == z2.shape
        assert torch.allclose(z1, z2, atol=1e-5)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestResidueDecoder:
    """Tests for ResidueDecoder class."""

    def test_forward_shape(self):
        """Test forward pass output shapes."""
        import torch
        from ciffy.nn.residue import ResidueDecoder

        decoder = ResidueDecoder(latent_dim=32, d_model=64, n_layers=1)
        decoder.eval()

        p = ciffy.load(get_test_cif("3SKW")).torch()
        chain = p.chain(0)

        n_residues = chain.size(Scale.RESIDUE)
        z = torch.randn(n_residues, 32)

        with torch.no_grad():
            coords, transforms = decoder(z, chain)

        # Coords should have shape (n_atoms, 3)
        assert coords.shape == (chain.size(), 3)
        # Transforms should have shape (n_residues, 6)
        assert transforms.shape == (n_residues, 6)

    def test_output_coords_reasonable(self):
        """Test that output coordinates are in reasonable range."""
        import torch
        from ciffy.nn.residue import ResidueDecoder

        decoder = ResidueDecoder(latent_dim=32, d_model=64, n_layers=1)
        decoder.eval()

        p = ciffy.load(get_test_cif("3SKW")).torch()
        chain = p.chain(0)

        n_residues = chain.size(Scale.RESIDUE)
        z = torch.randn(n_residues, 32)

        with torch.no_grad():
            coords, _ = decoder(z, chain)

        # Coords should be finite
        assert torch.isfinite(coords).all()


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestResidueVAE:
    """Tests for ResidueVAE class."""

    def test_encode_decode_shapes(self):
        """Test encode/decode round-trip shapes."""
        import torch
        from ciffy.nn.residue import ResidueVAE

        vae = ResidueVAE(latent_dim=32, d_model=64, encoder_layers=1, decoder_layers=1)
        vae.eval()

        p = ciffy.load(get_test_cif("3SKW")).torch()
        chain = p.chain(0)

        with torch.no_grad():
            # Encode
            z = vae.encode(chain)
            n_residues = chain.size(Scale.RESIDUE)
            assert z.shape == (n_residues, 32)

            # Decode
            coords, transforms = vae.decode(z, chain)
            assert coords.shape == (chain.size(), 3)
            assert transforms.shape == (n_residues, 6)

    def test_forward_pass(self):
        """Test forward pass returns all outputs."""
        import torch
        from ciffy.nn.residue import ResidueVAE

        vae = ResidueVAE(latent_dim=32, d_model=64, encoder_layers=1, decoder_layers=1)
        vae.eval()

        p = ciffy.load(get_test_cif("3SKW")).torch()
        chain = p.chain(0)

        with torch.no_grad():
            coords, transforms, mu, logvar = vae(chain)

        n_residues = chain.size(Scale.RESIDUE)
        assert coords.shape == (chain.size(), 3)
        assert transforms.shape == (n_residues, 6)
        assert mu.shape == (n_residues, 32)
        assert logvar.shape == (n_residues, 32)

    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        import torch
        from ciffy.nn.residue import ResidueVAE

        vae = ResidueVAE(latent_dim=16, d_model=32, encoder_layers=1, decoder_layers=1)

        p = ciffy.load(get_test_cif("3SKW")).torch()
        chain = p.chain(0)

        # Forward pass
        coords, transforms, mu, logvar = vae(chain)

        # Compute simple loss
        loss = coords.mean() + transforms.mean() + mu.mean() + logvar.mean()

        # Backward pass
        loss.backward()

        # Check gradients exist
        for name, param in vae.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_sample(self):
        """Test sampling from prior."""
        from ciffy.nn.residue import ResidueVAE

        vae = ResidueVAE(latent_dim=16, d_model=32, encoder_layers=1, decoder_layers=1)

        # Sample a short sequence
        sampled = vae.sample("acguacgu", temperature=1.0)

        # Check output is a valid Polymer
        assert isinstance(sampled, ciffy.Polymer)
        assert sampled.size(Scale.RESIDUE) == 8
        assert sampled.sequence_str() == "acguacgu"

        # Check coordinates are finite
        assert sampled.coordinates.isfinite().all()
