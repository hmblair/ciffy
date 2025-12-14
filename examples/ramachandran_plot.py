#!/usr/bin/env python3
"""
Example: Ramachandran plot analysis for proteins.

This example shows how to extract backbone dihedral angles (phi/psi)
from protein structures and create Ramachandran plots for analyzing
secondary structure and backbone geometry.

Requires: matplotlib (pip install matplotlib)
"""

import numpy as np
import ciffy
from ciffy import DihedralType


def basic_ramachandran(protein):
    """Create a basic Ramachandran plot."""
    import matplotlib.pyplot as plt

    # Get phi and psi angles
    phi = np.degrees(protein.dihedral(DihedralType.PHI))
    psi = np.degrees(protein.dihedral(DihedralType.PSI))

    # Remove NaN values (terminal residues don't have all angles)
    valid = ~(np.isnan(phi) | np.isnan(psi))
    phi = phi[valid]
    psi = psi[valid]

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(phi, psi, alpha=0.5, s=20, c='steelblue')

    # Formatting
    ax.set_xlabel("φ (degrees)", fontsize=12)
    ax.set_ylabel("ψ (degrees)", fontsize=12)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_yticks([-180, -90, 0, 90, 180])
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.set_aspect('equal')
    ax.set_title(f"Ramachandran Plot - {protein.id()}")
    ax.grid(True, alpha=0.3)

    return fig, ax


def ramachandran_with_regions(protein):
    """Ramachandran plot with secondary structure regions highlighted."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    phi = np.degrees(protein.dihedral(DihedralType.PHI))
    psi = np.degrees(protein.dihedral(DihedralType.PSI))

    valid = ~(np.isnan(phi) | np.isnan(psi))
    phi = phi[valid]
    psi = psi[valid]

    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw approximate allowed regions
    # Alpha helix region
    alpha_ellipse = Ellipse((-60, -45), 40, 40, alpha=0.2, color='red', label='α-helix')
    ax.add_patch(alpha_ellipse)

    # Beta sheet region
    beta_ellipse = Ellipse((-120, 130), 50, 40, alpha=0.2, color='blue', label='β-sheet')
    ax.add_patch(beta_ellipse)

    # Left-handed helix (rare, mainly glycine)
    left_ellipse = Ellipse((60, 45), 40, 40, alpha=0.2, color='green', label='Left-handed')
    ax.add_patch(left_ellipse)

    # Plot data points
    ax.scatter(phi, psi, alpha=0.6, s=15, c='black', zorder=5)

    # Formatting
    ax.set_xlabel("φ (degrees)", fontsize=12)
    ax.set_ylabel("ψ (degrees)", fontsize=12)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_yticks([-180, -90, 0, 90, 180])
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.set_aspect('equal')
    ax.set_title(f"Ramachandran Plot - {protein.id()}")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    return fig, ax


def secondary_structure_analysis(protein):
    """Analyze secondary structure content from phi/psi angles."""
    phi = np.degrees(protein.dihedral(DihedralType.PHI))
    psi = np.degrees(protein.dihedral(DihedralType.PSI))

    valid = ~(np.isnan(phi) | np.isnan(psi))
    phi_valid = phi[valid]
    psi_valid = psi[valid]
    n_residues = len(phi_valid)

    # Define regions (approximate)
    # Alpha helix: phi ~ -60°, psi ~ -45°
    alpha_helix = (
        (phi_valid > -80) & (phi_valid < -40) &
        (psi_valid > -65) & (psi_valid < -20)
    )

    # Beta sheet: phi ~ -120°, psi ~ 120°
    beta_sheet = (
        (phi_valid > -150) & (phi_valid < -90) &
        (psi_valid > 90) & (psi_valid < 150)
    )

    # Left-handed helix (mainly glycine)
    left_helix = (
        (phi_valid > 40) & (phi_valid < 80) &
        (psi_valid > 20) & (psi_valid < 70)
    )

    # PPII helix: phi ~ -75°, psi ~ 145°
    ppii = (
        (phi_valid > -90) & (phi_valid < -55) &
        (psi_valid > 120) & (psi_valid < 170)
    )

    print(f"Secondary Structure Analysis for {protein.id()}")
    print(f"{'='*50}")
    print(f"Total residues analyzed: {n_residues}")
    print()
    print(f"  α-helix:        {np.sum(alpha_helix):4d} ({100*np.sum(alpha_helix)/n_residues:5.1f}%)")
    print(f"  β-sheet:        {np.sum(beta_sheet):4d} ({100*np.sum(beta_sheet)/n_residues:5.1f}%)")
    print(f"  Left-handed:    {np.sum(left_helix):4d} ({100*np.sum(left_helix)/n_residues:5.1f}%)")
    print(f"  PPII helix:     {np.sum(ppii):4d} ({100*np.sum(ppii)/n_residues:5.1f}%)")

    other = n_residues - np.sum(alpha_helix) - np.sum(beta_sheet) - np.sum(left_helix) - np.sum(ppii)
    print(f"  Other/loop:     {other:4d} ({100*other/n_residues:5.1f}%)")

    return {
        'alpha': np.sum(alpha_helix),
        'beta': np.sum(beta_sheet),
        'left': np.sum(left_helix),
        'ppii': np.sum(ppii),
        'other': other,
        'total': n_residues,
    }


def glycine_proline_analysis(protein):
    """Analyze phi/psi distributions for glycine and proline residues."""
    from ciffy.biochemistry import Residue

    phi = np.degrees(protein.dihedral(DihedralType.PHI))
    sequence = protein.sequence

    print(f"Glycine and Proline Analysis")
    print(f"{'='*50}")

    # Identify glycine and proline positions
    gly_mask = np.array([Residue(s) == Residue.GLY for s in sequence])
    pro_mask = np.array([Residue(s) == Residue.PRO for s in sequence])

    print(f"Glycine residues: {np.sum(gly_mask)}")
    print(f"Proline residues: {np.sum(pro_mask)}")

    # Check if phi array matches sequence length
    if len(phi) != len(sequence):
        print(f"\nNote: φ array length ({len(phi)}) differs from sequence ({len(sequence)})")
        print("Skipping detailed Gly/Pro analysis")
        return gly_mask, pro_mask

    print()

    # Glycine can access positive phi (left-handed region)
    gly_phi = phi[gly_mask]
    gly_phi_valid = gly_phi[~np.isnan(gly_phi)]
    if len(gly_phi_valid) > 0:
        gly_positive_phi = np.sum(gly_phi_valid > 0)
        print(f"Glycine with positive φ: {gly_positive_phi} ({100*gly_positive_phi/len(gly_phi_valid):.1f}%)")

    # Proline has restricted phi (around -60°)
    pro_phi = phi[pro_mask]
    pro_phi_valid = pro_phi[~np.isnan(pro_phi)]
    if len(pro_phi_valid) > 0:
        print(f"Proline φ range: {pro_phi_valid.min():.0f}° to {pro_phi_valid.max():.0f}°")
        print(f"Proline φ mean: {pro_phi_valid.mean():.0f}° (expected ~-60°)")

    return gly_mask, pro_mask


def ramachandran_by_residue_type(protein):
    """Create Ramachandran plot colored by residue type."""
    import matplotlib.pyplot as plt
    from ciffy.biochemistry import Residue

    phi = np.degrees(protein.dihedral(DihedralType.PHI))
    psi = np.degrees(protein.dihedral(DihedralType.PSI))
    sequence = protein.sequence

    valid = ~(np.isnan(phi) | np.isnan(psi))

    # Classify residues
    gly_mask = np.array([Residue(s) == Residue.GLY for s in sequence]) & valid
    pro_mask = np.array([Residue(s) == Residue.PRO for s in sequence]) & valid
    other_mask = valid & ~gly_mask & ~pro_mask

    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot each type with different colors
    ax.scatter(phi[other_mask], psi[other_mask], alpha=0.5, s=15, c='steelblue', label='General')
    ax.scatter(phi[gly_mask], psi[gly_mask], alpha=0.7, s=25, c='red', label='Glycine', marker='^')
    ax.scatter(phi[pro_mask], psi[pro_mask], alpha=0.7, s=25, c='green', label='Proline', marker='s')

    # Formatting
    ax.set_xlabel("φ (degrees)", fontsize=12)
    ax.set_ylabel("ψ (degrees)", fontsize=12)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_yticks([-180, -90, 0, 90, 180])
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.set_aspect('equal')
    ax.set_title(f"Ramachandran Plot by Residue Type - {protein.id()}")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    return fig, ax


def omega_analysis(protein):
    """Analyze omega (peptide bond) angles for cis/trans classification."""
    omega = np.degrees(protein.dihedral(DihedralType.OMEGA))
    valid_omega = omega[~np.isnan(omega)]

    print(f"Peptide Bond Analysis")
    print(f"{'='*50}")
    print(f"Total peptide bonds: {len(valid_omega)}")

    if len(valid_omega) == 0:
        print("  No omega angles available")
        return

    # Trans: omega ~ 180° (or -180°)
    trans = np.abs(np.abs(valid_omega) - 180) < 30

    # Cis: omega ~ 0°
    cis = np.abs(valid_omega) < 30

    print(f"  Trans (ω ~ 180°): {np.sum(trans)} ({100*np.sum(trans)/len(valid_omega):.1f}%)")
    print(f"  Cis (ω ~ 0°):     {np.sum(cis)} ({100*np.sum(cis)/len(valid_omega):.1f}%)")

    if np.any(cis):
        cis_positions = np.where(np.abs(omega) < 30)[0]
        print(f"\nCis peptide bonds at positions: {list(cis_positions + 1)}")  # 1-indexed


def main():
    # Load a protein structure
    # You can replace this with your own CIF file path
    import sys
    import os

    if len(sys.argv) > 1:
        cif_path = sys.argv[1]
    else:
        # Default: use test data file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cif_path = os.path.join(script_dir, "..", "tests", "data", "9GCM.cif")
        if not os.path.exists(cif_path):
            # Try relative to current directory
            cif_path = "tests/data/9GCM.cif"

    print(f"Loading: {cif_path}")
    print("(You can also run: python ramachandran_plot.py your_structure.cif)")
    print()

    try:
        polymer = ciffy.load(cif_path)
    except Exception as e:
        print(f"Error loading structure: {e}")
        print("\nPlease provide a valid CIF file.")
        return

    protein = polymer.by_type(ciffy.PROTEIN)

    if protein.size() == 0:
        print("No protein chains found!")
        return

    print(protein)
    print()

    # =========================================================================
    # Secondary Structure Analysis
    # =========================================================================
    secondary_structure_analysis(protein)
    print()

    # =========================================================================
    # Omega (Peptide Bond) Analysis
    # =========================================================================
    omega_analysis(protein)
    print()

    # =========================================================================
    # Glycine/Proline Analysis
    # =========================================================================
    glycine_proline_analysis(protein)
    print()

    # =========================================================================
    # Create Plots (if matplotlib available)
    # =========================================================================
    try:
        import matplotlib.pyplot as plt

        # Basic Ramachandran plot
        fig1, ax1 = basic_ramachandran(protein)
        fig1.savefig("/tmp/ramachandran_basic.png", dpi=150, bbox_inches='tight')
        print("\nSaved: /tmp/ramachandran_basic.png")

        # Ramachandran with regions
        fig2, ax2 = ramachandran_with_regions(protein)
        fig2.savefig("/tmp/ramachandran_regions.png", dpi=150, bbox_inches='tight')
        print("Saved: /tmp/ramachandran_regions.png")

        # Ramachandran by residue type
        fig3, ax3 = ramachandran_by_residue_type(protein)
        fig3.savefig("/tmp/ramachandran_types.png", dpi=150, bbox_inches='tight')
        print("Saved: /tmp/ramachandran_types.png")

        plt.close('all')

    except ImportError:
        print("\nMatplotlib not available - skipping plot generation")
        print("Install with: pip install matplotlib")


if __name__ == "__main__":
    main()
