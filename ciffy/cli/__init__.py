"""
Command-line interface for ciffy.

Usage:
    ciffy <file.cif>              # Load and print polymer summary
    ciffy <file1> <file2> ...     # Load and print multiple files
    ciffy <file.cif> --atoms      # Also show atom counts per residue
    ciffy <file.cif> --desc       # Show entity descriptions per chain
    ciffy map <file.cif>          # Display contact map
    ciffy split <file.cif>        # Split into per-chain files
"""

from .__main__ import main

__all__ = ["main"]
