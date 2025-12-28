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

#endif /* _CIFFY_GRAPH_H */
