/**
 * @file graph.h
 * @brief Bond graph construction for Z-matrix generation.
 */

#ifndef _CIFFY_GRAPH_H
#define _CIFFY_GRAPH_H

#include <stdint.h>
#include <stddef.h>

/**
 * Build bond graph edge list from polymer data.
 *
 * Combines intra-residue bonds (from precomputed patterns) and
 * inter-residue linking bonds into a single edge array.
 *
 * @param atoms         (N,) int32 atom values
 * @param sequence      (R,) int32 residue type indices
 * @param res_sizes     (R,) int32 number of atoms per residue
 * @param chain_lengths (C,) int32 number of residues per chain
 * @param n_atoms       Total number of atoms N
 * @param n_residues    Total number of residues R
 * @param n_chains      Number of chains C
 * @param out_edges     Output: (E, 2) int64 edge array (caller allocates)
 * @param max_edges     Maximum number of edges that can be stored
 * @return              Number of edges written, or -1 on error
 */
int64_t build_bond_graph_c(
    const int32_t *atoms,
    const int32_t *sequence,
    const int32_t *res_sizes,
    const int32_t *chain_lengths,
    int64_t n_atoms,
    int64_t n_residues,
    int64_t n_chains,
    int64_t *out_edges,
    int64_t max_edges
);

/**
 * Estimate maximum number of edges for allocation.
 *
 * Conservative upper bound: 2 * (sum of intra-residue bonds + inter-residue bonds).
 * Factor of 2 accounts for symmetric edges.
 *
 * @param sequence    (R,) int32 residue type indices
 * @param n_residues  Total number of residues
 * @return            Upper bound on number of symmetric edges
 */
int64_t estimate_max_edges(
    const int32_t *sequence,
    int64_t n_residues
);


/**
 * Build Z-matrix from edge list for a single chain (with pre-built CSR).
 *
 * When dihedral-aware mode is enabled (atoms, sequence, residue_starts,
 * chain_res_starts all non-NULL), the function will attempt to use
 * dihedral-specific references for atoms that own named dihedrals.
 *
 * @param offsets          (n_atoms+1,) CSR offsets
 * @param neighbors        (n_edges,) CSR neighbor indices
 * @param n_atoms          Total number of atoms
 * @param chain_start      First atom index for this chain
 * @param chain_size       Number of atoms in this chain
 * @param root             Root atom index for BFS
 * @param atoms            (n_atoms,) int32 atom types (NULL to disable dihedral-aware)
 * @param sequence         (n_residues,) int32 residue types
 * @param residue_starts   (n_residues+1,) int64 cumsum of residue sizes
 * @param n_residues       Total number of residues
 * @param chain_res_starts (n_chains,) int64 residue start indices per chain
 * @param n_chains         Number of chains
 * @param out_zmatrix      Output: (chain_size, 4) int64 Z-matrix entries
 * @param out_dihedral_types Output: (chain_size,) int8 dihedral type or -1 (can be NULL)
 * @param out_levels       Output: (chain_size,) int32 BFS level per entry (can be NULL)
 * @return                 Number of entries written, or -1 on error
 */
int64_t build_zmatrix_from_csr(
    const int64_t *offsets,
    const int64_t *neighbors,
    int64_t n_atoms,
    int64_t chain_start,
    int64_t chain_size,
    int64_t root,
    const int32_t *atoms,
    const int32_t *sequence,
    const int64_t *residue_starts,
    int64_t n_residues,
    const int64_t *chain_res_starts,
    int64_t n_chains,
    int64_t *out_zmatrix,
    int8_t *out_dihedral_types,
    int32_t *out_levels
);

/**
 * Convert edge list to CSR format.
 *
 * @param edges       (n_edges, 2) int64 edge list
 * @param n_edges     Number of edges
 * @param n_atoms     Total number of atoms
 * @param out_offsets Output: (n_atoms+1,) int64 CSR offsets (caller allocates)
 * @param out_neighbors Output: (n_edges,) int64 neighbor indices (caller allocates)
 * @return            0 on success, -1 on error
 */
int edges_to_csr(
    const int64_t *edges,
    int64_t n_edges,
    int64_t n_atoms,
    int64_t *out_offsets,
    int64_t *out_neighbors
);


/**
 * Build Z-matrix for all chains in parallel using OpenMP.
 *
 * When dihedral-aware mode is enabled (atoms, sequence, residue_starts,
 * chain_res_starts all non-NULL), uses dihedral-specific references
 * for atoms that own named dihedrals.
 *
 * @param offsets          (n_atoms+1,) CSR offsets
 * @param neighbors        (n_edges,) CSR neighbor indices
 * @param n_atoms          Total number of atoms
 * @param chain_starts     (n_chains,) int64 first atom index per chain
 * @param chain_sizes      (n_chains,) int64 number of atoms per chain
 * @param roots            (n_chains,) int64 root atom index per chain
 * @param n_chains         Number of chains
 * @param atoms            (n_atoms,) int32 atom types (NULL to disable)
 * @param sequence         (n_residues,) int32 residue types
 * @param residue_starts   (n_residues+1,) int64 cumsum of residue sizes
 * @param n_residues       Total number of residues
 * @param chain_res_starts (n_chains,) int64 residue start indices per chain
 * @param out_zmatrix      Output: (total_atoms, 4) int64 Z-matrix
 * @param out_dihedral_types Output: (total_atoms,) int8 dihedral types (can be NULL)
 * @param out_levels       Output: (total_atoms,) int32 BFS levels (can be NULL)
 * @param out_counts       Output: (n_chains,) int64 entries per chain
 * @return                 Total entries written, or -1 on error
 */
int64_t build_zmatrix_parallel(
    const int64_t *offsets,
    const int64_t *neighbors,
    int64_t n_atoms,
    const int64_t *chain_starts,
    const int64_t *chain_sizes,
    const int64_t *roots,
    int64_t n_chains,
    const int32_t *atoms,
    const int32_t *sequence,
    const int64_t *residue_starts,
    int64_t n_residues,
    const int64_t *chain_res_starts,
    int64_t *out_zmatrix,
    int8_t *out_dihedral_types,
    int32_t *out_levels,
    int64_t *out_counts
);


/**
 * Find connected components in a CSR-format graph.
 *
 * Returns atom indices grouped by component, with offsets for each component.
 * Includes isolated atoms (no bonds) as single-atom components.
 *
 * @param offsets        (n_atoms+1,) CSR offsets
 * @param neighbors      (n_edges,) CSR neighbor indices
 * @param n_atoms        Total number of atoms
 * @param out_atoms      Output: (n_atoms,) atom indices grouped by component
 * @param out_offsets    Output: (n_atoms+1,) offsets into out_atoms per component
 *                       Component i's atoms are out_atoms[out_offsets[i]:out_offsets[i+1]]
 * @return               Number of components found, or -1 on error
 */
int64_t find_connected_components_c(
    const int64_t *offsets,
    const int64_t *neighbors,
    int64_t n_atoms,
    int64_t *out_atoms,
    int64_t *out_offsets
);


/**
 * Build canonical Z-matrix in a single pass.
 *
 * Processes atoms in natural order (0, 1, 2, ..., n_atoms-1). For each atom,
 * uses pre-defined canonical references from ATOM_CANONICAL_REFS if available,
 * otherwise falls back to bond-graph-based reference selection.
 *
 * Key features:
 * - zmatrix[i] corresponds directly to atoms[i] (no reordering)
 * - Captures named dihedrals (phi, psi, alpha, beta, etc.) via ATOM_DIHEDRAL_TYPE
 * - Single pass with no Python post-processing needed
 *
 * @param atoms          (n_atoms,) int32 atom type values
 * @param sequence       (n_residues,) int32 residue type indices
 * @param res_sizes      (n_residues,) int32 atoms per residue
 * @param chain_lengths  (n_chains,) int32 residues per chain
 * @param n_atoms        Total number of atoms
 * @param n_residues     Total number of residues
 * @param n_chains       Number of chains
 * @param bond_offsets   (n_atoms+1,) int64 CSR offsets for bond graph
 * @param bond_neighbors (n_edges,) int64 CSR neighbor indices
 * @param out_zmatrix    Output: (n_atoms, 4) int64 [atom_idx, dist_ref, ang_ref, dih_ref]
 * @param out_dihedral_types Output: (n_atoms,) int8 dihedral type index or -1
 * @return               Number of atoms processed, or -1 on error
 */
int64_t build_canonical_zmatrix_c(
    const int32_t *atoms,
    const int32_t *sequence,
    const int32_t *res_sizes,
    const int32_t *chain_lengths,
    int64_t n_atoms,
    int64_t n_residues,
    int64_t n_chains,
    const int64_t *bond_offsets,
    const int64_t *bond_neighbors,
    int64_t *out_zmatrix,
    int8_t *out_dihedral_types
);

#endif /* _CIFFY_GRAPH_H */
