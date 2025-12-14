#!/usr/bin/env python3
"""
Example: Energy minimization via dihedral optimization.

This example demonstrates:
1. Loading a polymer structure
2. Perturbing its dihedral angles
3. Minimizing RMSD to the original via gradient descent
4. Saving intermediate states as CIF files
5. Creating a movie with ChimeraX

Usage:
    python examples/energy_minimization.py
"""

import argparse
import copy
import os
import shutil
import subprocess

import torch
import torch.nn as nn

import ciffy


def find_chimerax():
    """Find ChimeraX executable."""
    candidates = [
        "ChimeraX",
        "chimerax",
        "/Applications/ChimeraX.app/Contents/MacOS/ChimeraX",
        "/usr/bin/chimerax",
        "/usr/local/bin/chimerax",
    ]
    for path in candidates:
        if shutil.which(path):
            return path
        if os.path.isfile(path):
            return path
    return None


def main():
    parser = argparse.ArgumentParser(description="Energy minimization example")
    parser.add_argument("--no-render", action="store_true",
                        help="Skip automatic movie rendering")
    parser.add_argument("--output", default="/tmp/minimization",
                        help="Output directory (default: /tmp/minimization)")
    args = parser.parse_args()

    # Output directory for trajectory
    output_dir = args.output
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    print("=" * 60)
    print("Energy Minimization via Dihedral Optimization")
    print("=" * 60)

    # =========================================================================
    # 1. Load target structure
    # =========================================================================
    print("\n1. Loading structure...")
    target = ciffy.load("tests/data/1ZEW.cif")

    # Get first chain (DNA)
    for chain in target.chains():
        target_chain = chain.torch()
        break

    print(f"   Target: {target_chain.size()} atoms")

    # =========================================================================
    # 2. Create perturbed template
    # =========================================================================
    print("\n2. Creating perturbed structure...")
    template = copy.deepcopy(target_chain)

    # Perturb dihedral angles
    original_dihedrals = template.dihedrals.clone()
    torch.manual_seed(42)  # Reproducibility
    perturbation = torch.randn_like(original_dihedrals) * 1.5
    perturbed_dihedrals = original_dihedrals + perturbation
    template.dihedrals = perturbed_dihedrals

    initial_rmsd = ciffy.rmsd(template, target_chain).item()
    print(f"   Initial RMSD after perturbation: {initial_rmsd:.2f} Å")

    # Collect trajectory coordinates
    trajectory_coords = [template.coordinates.detach().cpu().numpy().copy()]

    # =========================================================================
    # 3. Set up optimizer
    # =========================================================================
    print("\n3. Setting up optimizer...")

    class DihedralModel(nn.Module):
        def __init__(self, init_dihedrals):
            super().__init__()
            self.dihedrals = nn.Parameter(init_dihedrals.clone())

    model = DihedralModel(perturbed_dihedrals)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.006)

    # =========================================================================
    # 4. Optimization loop with trajectory saving
    # =========================================================================
    print("\n4. Running optimization...")
    print(f"   {'Step':>5} {'RMSD (Å)':>10} {'Grad Norm':>12}")
    print("   " + "-" * 30)

    n_steps = 500
    save_every = 5
    trajectory = []

    for step in range(n_steps + 1):
        optimizer.zero_grad()

        # Apply current dihedrals to template
        template.dihedrals = model.dihedrals

        # Compute RMSD loss
        loss = ciffy.rmsd(template, target_chain)

        if step < n_steps:
            loss.backward()
            grad_norm = model.dihedrals.grad.norm().item()
            optimizer.step()
        else:
            grad_norm = 0.0

        rmsd_val = loss.item()
        trajectory.append(rmsd_val)

        # Print progress and save frame
        if step % save_every == 0 or step == n_steps:
            print(f"   {step:5d} {rmsd_val:10.4f} {grad_norm:12.6f}")
            trajectory_coords.append(template.coordinates.detach().cpu().numpy().copy())

    final_rmsd = trajectory[-1]
    print("\n   " + "-" * 30)
    print(f"   Initial RMSD: {initial_rmsd:.4f} Å")
    print(f"   Final RMSD:   {final_rmsd:.4f} Å")
    print(f"   Improvement:  {100 * (initial_rmsd - final_rmsd) / initial_rmsd:.1f}%")

    # =========================================================================
    # 5. Generate trajectory and ChimeraX movie script
    # =========================================================================
    print("\n5. Generating trajectory and ChimeraX movie script...")

    # Save target for comparison
    target_path = os.path.join(output_dir, "target.cif")
    target_chain.numpy().write(target_path)

    # Write multi-model PDB file for trajectory (ChimeraX coordsets)
    n_frames = len(trajectory_coords)
    trajectory_path = os.path.join(output_dir, "trajectory.pdb")

    # Get atom info from template
    template_np = template.numpy()
    atom_names = list(template_np.atom_names())
    elements = template_np.elements

    with open(trajectory_path, "w") as f:
        for frame_idx, coords in enumerate(trajectory_coords):
            f.write(f"MODEL     {frame_idx + 1:4d}\n")
            for i, (x, y, z) in enumerate(coords):
                atom_name = atom_names[i] if i < len(atom_names) else "X"
                # Map element number to symbol
                elem_symbols = ["X", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
                                "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca"]
                elem = elem_symbols[min(elements[i], len(elem_symbols) - 1)]
                # PDB ATOM format
                f.write(f"ATOM  {i+1:5d} {atom_name:4s} UNK A   1    "
                        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {elem:>2s}\n")
            f.write("ENDMDL\n")
        f.write("END\n")

    print(f"   Trajectory: {trajectory_path} ({n_frames} frames)")

    # Create ChimeraX script
    script_path = os.path.join(output_dir, "movie.cxc")
    movie_path = os.path.join(output_dir, "minimization.mp4")

    with open(script_path, "w") as f:
        f.write("# ChimeraX movie script for energy minimization trajectory\n")
        f.write("# Generated by ciffy examples/energy_minimization.py\n\n")

        # Open trajectory as coordsets
        f.write(f"open {trajectory_path} coordsets true\n\n")

        # Style
        f.write("hide pseudobonds\n")
        f.write("color #1 cornflowerblue\n")
        f.write("style sphere\n")
        f.write("graphics quality 5\n")
        f.write("lighting soft\n")
        f.write("view\n\n")

        # Open target as reference (transparent)
        f.write(f"open {target_path}\n")
        f.write("color #2 indianred\n")
        f.write("style #2 sphere\n")
        f.write("transparency #2 70\n\n")

        # Start movie recording
        f.write("movie record\n\n")

        # Play through coordinate sets
        f.write(f"coordset #1 1,{n_frames} pauseFrames 1\n")
        f.write(f"wait {n_frames + 10}\n\n")

        # Final pause
        f.write("wait 30\n\n")

        # Save movie and exit
        f.write(f"movie encode {movie_path} format h264 quality high\n")
        f.write("exit\n")

    print(f"   Script: {script_path}")

    # =========================================================================
    # 6. Render movie with ChimeraX
    # =========================================================================
    movie_path = os.path.join(output_dir, "minimization.mp4")

    if not args.no_render:
        print("\n6. Rendering movie with ChimeraX...")
        chimerax = find_chimerax()

        if chimerax is None:
            print("   ChimeraX not found. Skipping movie rendering.")
            print(f"   To render manually, run in ChimeraX:")
            print(f"   open {script_path}")
        else:
            print(f"   Using: {chimerax}")
            print("   This may take a minute...")

            # Try offscreen first (Linux), fall back to GUI mode (macOS)
            result = subprocess.run(
                [chimerax, "--offscreen", "--script", script_path],
                capture_output=True,
                text=True,
            )

            # Check if offscreen failed due to OSMesa
            if "Offscreen rendering is not available" in (result.stdout + result.stderr):
                print("   Offscreen not available, using GUI mode...")
                result = subprocess.run(
                    [chimerax, "--script", script_path],
                    capture_output=True,
                    text=True,
                )

            if os.path.exists(movie_path):
                print(f"   Movie saved to: {movie_path}")
            else:
                print("   Movie rendering may have failed.")
                print(f"   To render manually, run in ChimeraX:")
                print(f"   open {script_path}")
    else:
        print(f"\n6. Skipping movie rendering (--no-render)")
        print(f"   To render manually, run in ChimeraX:")
        print(f"   open {script_path}")

    # =========================================================================
    # 7. Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("Output files:")
    print(f"  Trajectory: {trajectory_path} ({n_frames} frames)")
    print(f"  Target:     {target_path}")
    print(f"  Script:     {script_path}")
    if os.path.exists(movie_path):
        print(f"  Movie:      {movie_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
