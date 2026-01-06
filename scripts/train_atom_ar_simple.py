#!/usr/bin/env python3
"""
Simple AtomAR training working directly with Polymer objects.

Uses PolymerEmbedding for embeddings and extend_new() for generation.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import ciffy
from ciffy.nn import PolymerDataset, PolymerEmbedding
from ciffy.biochemistry import Residue, Scale, Molecule
from ciffy.backend import to_numpy
from ciffy import rmsd


class SimpleAtomAR(nn.Module):
    """
    Simple all-atom AR model working directly with Polymer objects.

    - Encodes context polymer with a transformer
    - Decodes one residue at a time
    """

    def __init__(self, d_model=256, n_layers=4, n_heads=8):
        super().__init__()
        self.d_model = d_model

        # Use PolymerEmbedding for atom/residue/element embeddings
        embed_dim = d_model // 4
        self.embedding = PolymerEmbedding(
            scale=Scale.ATOM,
            atom_dim=embed_dim,
            residue_dim=embed_dim,
            element_dim=embed_dim,
        )

        # Coordinate encoding
        self.coord_proj = nn.Linear(3, embed_dim)

        # Input projection (embedding output + coord encoding)
        self.input_proj = nn.Linear(self.embedding.output_dim + embed_dim, d_model)

        # Context encoder (transformer)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.context_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Learnable query for pooling
        self.pool_query = nn.Parameter(torch.randn(1, d_model))

        # Residue decoder - takes context + pooled target embedding
        self.residue_proj = nn.Linear(self.embedding.output_dim, d_model)
        self.decoder = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )

        # Per-atom output (takes decoder hidden + per-atom embedding)
        self.atom_decoder = nn.Sequential(
            nn.Linear(d_model + self.embedding.output_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, 3),
        )

    def encode_context(self, context_polymer):
        """
        Encode context polymer into a single vector.

        Args:
            context_polymer: Polymer object (torch backend) with previous atoms

        Returns:
            (d_model,) context vector
        """
        if context_polymer.size() == 0:
            device = next(self.parameters()).device
            return torch.zeros(self.d_model, device=device)

        # Get embeddings from polymer
        emb = self.embedding(context_polymer)  # (N, embed_dim*3)
        coords = context_polymer.coordinates.float()  # Ensure float32
        coord_emb = self.coord_proj(coords)  # (N, embed_dim)

        # Combine and project
        x = torch.cat([emb, coord_emb], dim=-1)  # (N, embed_dim*4)
        x = self.input_proj(x)  # (N, d_model)

        # Add batch dim for transformer
        x = x.unsqueeze(0)  # (1, N, d_model)
        x = self.context_encoder(x)
        x = x.squeeze(0)  # (N, d_model)

        # Pool with attention
        query = self.pool_query  # (1, d_model)
        scores = torch.matmul(query, x.T) / (self.d_model ** 0.5)  # (1, N)
        weights = F.softmax(scores, dim=-1)  # (1, N)
        context = torch.matmul(weights, x).squeeze(0)  # (d_model,)

        return context

    def decode_residue(self, context, target_residue):
        """
        Decode atom coordinates for a residue.

        Args:
            context: (d_model,) context from previous atoms
            target_residue: Polymer object containing the single target residue

        Returns:
            (M, 3) predicted coordinates
        """
        # Get embeddings using PolymerEmbedding (includes atom, residue, element)
        target_emb = self.embedding(target_residue)  # (M, embed_dim*3)

        # Pool residue embedding for the decoder input
        res_emb_pooled = target_emb.mean(dim=0)  # (embed_dim*3,)
        res_emb = self.residue_proj(res_emb_pooled)  # (d_model,)

        # Combine context and residue
        combined = torch.cat([context, res_emb])  # (d_model * 2,)
        h = self.decoder(combined)  # (d_model,)

        # Decode each atom
        h_expanded = h.unsqueeze(0).expand(target_residue.size(), -1)  # (M, d_model)
        atom_input = torch.cat([h_expanded, target_emb], dim=-1)  # (M, d_model + embed_dim*3)
        coords = self.atom_decoder(atom_input)  # (M, 3)

        return coords

    def forward_chain(self, polymer):
        """
        Forward pass on a single chain with teacher forcing.

        Args:
            polymer: Polymer object (torch backend)

        Returns:
            total_loss, n_atoms
        """
        n_res = polymer.size(Scale.RESIDUE)
        total_loss = 0.0
        n_atoms = 0

        for i in range(n_res):
            # Get context polymer (all residues before this one)
            if i == 0:
                context = torch.zeros(self.d_model, device=polymer.coordinates.device)
            else:
                context_poly = polymer.residue(list(range(i)))
                context = self.encode_context(context_poly)

            # Get target residue as Polymer
            target_residue = polymer.residue([i])
            target_coords = target_residue.coordinates.float()

            # Decode
            pred_coords = self.decode_residue(context, target_residue)

            # Loss: Kabsch-aligned RMSD (rotationally invariant)
            res_rmsd = rmsd(pred_coords, target_coords, eps=1e-6)
            total_loss += res_rmsd ** 2 * target_residue.size()
            n_atoms += target_residue.size()

        return total_loss, n_atoms

    @torch.no_grad()
    def generate(self, sequence_str, device='cpu'):
        """
        Generate a chain autoregressively using extend_new().

        Args:
            sequence_str: Sequence string like "ACGU"
            device: Device to use

        Returns:
            Generated Polymer object
        """
        char_to_res = {'A': Residue.A, 'C': Residue.C, 'G': Residue.G, 'U': Residue.U}

        # Start with empty polymer
        polymer = ciffy.Polymer()
        context = torch.zeros(self.d_model, device=device)

        for i, char in enumerate(sequence_str):
            residue = char_to_res[char]

            # Create a template single-residue polymer for embedding
            template = ciffy.Polymer().append(residue, residue.ideal)
            template = template.torch().to(device)

            # Decode coordinates
            pred_coords = self.decode_residue(context, template)
            coords_np = pred_coords.cpu().numpy()

            # Extend output polymer with predicted coordinates
            if i == 0:
                polymer = polymer.append(residue, coords_np)
            else:
                from ciffy.geometry import LocalCoordinates
                transform = np.zeros(6, dtype=np.float32)
                polymer = polymer.append(residue, LocalCoordinates(coords_np, transform))

            # Update context for next iteration
            polymer_torch = polymer.torch().to(device)
            context = self.encode_context(polymer_torch)

        return polymer


def train(data_dir, output_dir, num_epochs=100, lr=1e-4, max_chains=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load dataset
    dataset = PolymerDataset(
        data_dir,
        scale=Scale.CHAIN,
        molecule_types=Molecule.RNA,
        min_residues=6,
        max_residues=200,  # Limit to avoid OOM (transformer is O(n²))
        backend='torch',
        limit=max_chains,
    )
    print(f"Chains: {len(dataset)}")

    # Create model
    model = SimpleAtomAR(d_model=256, n_layers=4, n_heads=8).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Training loop
    best_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_atoms = 0
        n_chains = 0

        indices = torch.randperm(len(dataset)).tolist()
        for idx in indices:
            poly = dataset[idx]
            if poly is None:
                continue

            try:
                # Prepare polymer (strip before poly to avoid hierarchy bug)
                poly = poly.strip().poly()
                if poly.size() == 0 or poly.size(Scale.RESIDUE) < 4:
                    continue

                poly, _ = poly.center()
                poly = poly.to(device)

                optimizer.zero_grad()
                loss, n_atoms = model.forward_chain(poly)

                if n_atoms > 0:
                    loss = loss / n_atoms
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                    total_loss += loss.item() * n_atoms
                    total_atoms += n_atoms
                    n_chains += 1
            except Exception as e:
                print(f"  Skip chain {idx}: {e}")
                if 'cuda' in str(device):
                    torch.cuda.empty_cache()
                continue

        scheduler.step()

        avg_loss = total_loss / max(1, total_atoms)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), output_dir / 'model.pt')

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}: loss={avg_loss:.4f} ({n_chains} chains)")

    print(f"Best loss: {best_loss:.4f}")
    return model


def sample_100mer(model, output_dir, device='cpu'):
    """Sample a 100-mer structure."""
    output_dir = Path(output_dir)

    # Random sequence
    np.random.seed(42)
    seq_chars = [np.random.choice(['A', 'C', 'G', 'U']) for _ in range(100)]
    seq_str = ''.join(seq_chars)
    print(f"Sequence: {seq_str[:20]}...{seq_str[-20:]}")

    # Generate
    model.eval()
    model = model.to(device)
    polymer = model.generate(seq_str, device=device)

    print(f"Generated: {polymer.size()} atoms, {polymer.size(Scale.RESIDUE)} residues")

    # Save
    polymer = polymer._clone(pdb_id='AR_100mer')
    output_path = output_dir / 'sampled_100mer.cif'
    polymer.write(str(output_path))
    print(f"Saved: {output_path}")

    return polymer


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='/home/hmblair/data/rna')
    parser.add_argument('--output-dir', default='outputs/atom_ar_simple')
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max-chains', type=int, default=None)
    args = parser.parse_args()

    model = train(args.data_dir, args.output_dir, args.num_epochs, args.lr, args.max_chains)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sample_100mer(model, args.output_dir, device)


if __name__ == '__main__':
    main()
