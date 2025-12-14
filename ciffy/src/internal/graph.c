/**
 * @file graph.c
 * @brief Bond graph construction for Z-matrix generation.
 *
 * Builds molecular bond graph by:
 * 1. Adding intra-residue bonds from precomputed patterns
 * 2. Adding inter-residue linking bonds (peptide/phosphodiester)
 * 3. Filtering bonds where atoms are missing (incomplete residues)
 */

#include "graph.h"
#include "bond_patterns.h"
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* Maximum atom value we can handle for the lookup table */
#define MAX_ATOM_VALUE 4096

/**
 * Build value_to_local mapping for a single residue.
 *
 * Creates a table mapping atom values -> local indices within the residue.
 * Uses -1 to indicate atoms not present.
 */
static void build_value_to_local(
    const int32_t *atoms,
    int32_t res_start,
    int32_t res_size,
    int32_t *value_to_local  /* Pre-zeroed array of size MAX_ATOM_VALUE */
) {
    /* Initialize to -1 (not present) */
    memset(value_to_local, -1, MAX_ATOM_VALUE * sizeof(int32_t));

    for (int32_t local_idx = 0; local_idx < res_size; local_idx++) {
        int32_t atom_value = atoms[res_start + local_idx];
        if (atom_value > 0 && atom_value < MAX_ATOM_VALUE) {
            value_to_local[atom_value] = local_idx;
        }
    }
}

int64_t estimate_max_edges(
    const int32_t *sequence,
    int64_t n_residues
) {
    int64_t total = 0;

    for (int64_t i = 0; i < n_residues; i++) {
        int32_t res_type = sequence[i];
        if (res_type >= 0 && res_type < NUM_RESIDUE_TYPES) {
            /* Intra-residue bonds */
            total += RESIDUE_BOND_COUNTS[res_type];
        }
    }

    /* Inter-residue bonds (at most n_residues - 1 per chain, conservative: n_residues) */
    total += n_residues;

    /* Factor of 2 for symmetric edges */
    return total * 2;
}

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
) {
    (void)n_atoms;  /* Unused but kept for consistency with Python interface */

    /* Working buffer for value -> local index mapping */
    int32_t *value_to_local = (int32_t *)malloc(MAX_ATOM_VALUE * sizeof(int32_t));
    if (value_to_local == NULL) {
        return -1;
    }

    int64_t edge_count = 0;

    /* Track atom and residue offsets as we process chains */
    int32_t atom_offset = 0;
    int32_t res_offset = 0;

    for (int64_t chain_idx = 0; chain_idx < n_chains; chain_idx++) {
        int32_t chain_len = chain_lengths[chain_idx];

        if (chain_len == 0) {
            continue;
        }

        /* Process residues in this chain */
        int32_t chain_atom_start = atom_offset;
        int32_t chain_res_start = res_offset;

        for (int32_t res_idx = 0; res_idx < chain_len; res_idx++) {
            int32_t global_res_idx = chain_res_start + res_idx;
            int32_t res_type = sequence[global_res_idx];
            int32_t res_size = res_sizes[global_res_idx];
            int32_t res_atom_start = atom_offset;

            /* Build value -> local mapping for this residue */
            build_value_to_local(atoms, res_atom_start, res_size, value_to_local);

            /* Add intra-residue bonds */
            if (res_type >= 0 && res_type < NUM_RESIDUE_TYPES) {
                const int32_t *bonds = RESIDUE_BONDS[res_type];
                int bond_count = RESIDUE_BOND_COUNTS[res_type];

                if (bonds != NULL) {
                    for (int b = 0; b < bond_count; b++) {
                        int32_t atom_val1 = bonds[b * 2];
                        int32_t atom_val2 = bonds[b * 2 + 1];

                        /* Look up local indices */
                        int32_t local1 = (atom_val1 < MAX_ATOM_VALUE) ?
                            value_to_local[atom_val1] : -1;
                        int32_t local2 = (atom_val2 < MAX_ATOM_VALUE) ?
                            value_to_local[atom_val2] : -1;

                        /* Only add if both atoms present */
                        if (local1 >= 0 && local2 >= 0) {
                            int64_t global1 = res_atom_start + local1;
                            int64_t global2 = res_atom_start + local2;

                            if (edge_count + 2 <= max_edges) {
                                /* Add both directions (symmetric) */
                                out_edges[edge_count * 2] = global1;
                                out_edges[edge_count * 2 + 1] = global2;
                                edge_count++;
                                out_edges[edge_count * 2] = global2;
                                out_edges[edge_count * 2 + 1] = global1;
                                edge_count++;
                            }
                        }
                    }
                }
            }

            atom_offset += res_size;
        }

        /* Add inter-residue bonds within this chain */
        atom_offset = chain_atom_start;  /* Reset for inter-residue processing */

        for (int32_t res_idx = 0; res_idx < chain_len - 1; res_idx++) {
            int32_t curr_res = chain_res_start + res_idx;
            int32_t next_res = chain_res_start + res_idx + 1;

            int32_t curr_type = sequence[curr_res];
            int32_t next_type = sequence[next_res];

            int32_t curr_size = res_sizes[curr_res];
            int32_t next_size = res_sizes[next_res];

            int32_t curr_atom_start = atom_offset;
            int32_t next_atom_start = atom_offset + curr_size;

            /* Get linking atom values from current residue */
            int32_t prev_atom_val = 0;  /* Atom on curr that links to next */
            int32_t next_atom_val = 0;  /* Atom on next that links from curr */

            if (curr_type >= 0 && curr_type < NUM_RESIDUE_TYPES) {
                prev_atom_val = RESIDUE_LINKING_PREV[curr_type];
            }
            if (next_type >= 0 && next_type < NUM_RESIDUE_TYPES) {
                next_atom_val = RESIDUE_LINKING_NEXT[next_type];
            }

            if (prev_atom_val > 0 && next_atom_val > 0) {
                /* Build value -> local mappings for both residues */
                build_value_to_local(atoms, curr_atom_start, curr_size, value_to_local);
                int32_t local_prev = (prev_atom_val < MAX_ATOM_VALUE) ?
                    value_to_local[prev_atom_val] : -1;

                build_value_to_local(atoms, next_atom_start, next_size, value_to_local);
                int32_t local_next = (next_atom_val < MAX_ATOM_VALUE) ?
                    value_to_local[next_atom_val] : -1;

                if (local_prev >= 0 && local_next >= 0) {
                    int64_t global_prev = curr_atom_start + local_prev;
                    int64_t global_next = next_atom_start + local_next;

                    if (edge_count + 2 <= max_edges) {
                        /* Add both directions (symmetric) */
                        out_edges[edge_count * 2] = global_prev;
                        out_edges[edge_count * 2 + 1] = global_next;
                        edge_count++;
                        out_edges[edge_count * 2] = global_next;
                        out_edges[edge_count * 2 + 1] = global_prev;
                        edge_count++;
                    }
                }
            }

            atom_offset += curr_size;
        }

        /* Account for last residue in chain */
        if (chain_len > 0) {
            atom_offset += res_sizes[chain_res_start + chain_len - 1];
        }

        res_offset += chain_len;
    }

    free(value_to_local);
    return edge_count;
}


/* ========================================================================== */
/* Z-MATRIX CONSTRUCTION */
/* ========================================================================== */

/**
 * Compare function for qsort on edges by source node.
 */
static int compare_edges_by_source(const void *a, const void *b) {
    const int64_t *ea = (const int64_t *)a;
    const int64_t *eb = (const int64_t *)b;
    if (ea[0] < eb[0]) return -1;
    if (ea[0] > eb[0]) return 1;
    /* Secondary sort by destination for determinism */
    if (ea[1] < eb[1]) return -1;
    if (ea[1] > eb[1]) return 1;
    return 0;
}

/**
 * Find a placed atom that is a child of target (has target as parent).
 * Returns -1 if not found.
 *
 * @param order       Array of global atom indices in BFS order
 * @param order_len   Number of atoms placed so far
 * @param parent      Array mapping local index -> global parent index
 * @param chain_start First atom index of the chain (for local index computation)
 * @param target      Global index of target atom
 * @param exclude     Global index of atom to exclude
 */
static int64_t find_child_of(
    const int64_t *order,
    int64_t order_len,
    const int64_t *parent,
    int64_t chain_start,
    int64_t target,
    int64_t exclude
) {
    for (int64_t i = 0; i < order_len; i++) {
        int64_t atom = order[i];  /* Global index */
        if (atom == exclude || atom == target) continue;
        int64_t atom_local = atom - chain_start;
        if (parent[atom_local] == target) return atom;
    }
    /* Fallback: any placed atom not excluded */
    for (int64_t i = order_len - 1; i >= 0; i--) {
        int64_t atom = order[i];
        if (atom != exclude && atom != target) return atom;
    }
    return -1;
}

/**
 * Find a placed neighbor (sibling or any placed atom).
 *
 * @param order       Array of global atom indices in BFS order
 * @param order_len   Number of atoms placed so far
 * @param parent      Array mapping local index -> global parent index
 * @param chain_start First atom index of the chain (for local index computation)
 * @param target      Global index of target atom
 * @param exclude1-3  Global indices of atoms to exclude
 */
static int64_t find_placed_neighbor(
    const int64_t *order,
    int64_t order_len,
    const int64_t *parent,
    int64_t chain_start,
    int64_t target,
    int64_t exclude1,
    int64_t exclude2,
    int64_t exclude3
) {
    int64_t target_local = target - chain_start;
    int64_t target_parent = parent[target_local];

    /* First, try to find a sibling */
    for (int64_t i = 0; i < order_len; i++) {
        int64_t atom = order[i];  /* Global index */
        if (atom == exclude1 || atom == exclude2 || atom == exclude3 || atom == target) continue;
        int64_t atom_local = atom - chain_start;
        if (parent[atom_local] == target_parent) return atom;
    }
    /* Fallback: any placed atom not excluded */
    for (int64_t i = order_len - 1; i >= 0; i--) {
        int64_t atom = order[i];
        if (atom != exclude1 && atom != exclude2 && atom != exclude3 && atom != target) {
            return atom;
        }
    }
    return -1;
}

int edges_to_csr(
    const int64_t *edges,
    int64_t n_edges,
    int64_t n_atoms,
    int64_t *out_offsets,
    int64_t *out_neighbors
) {
    /* Initialize offsets to zero */
    memset(out_offsets, 0, (size_t)(n_atoms + 1) * sizeof(int64_t));

    if (n_edges == 0) {
        return 0;
    }

    /* Pass 1: Count edges per source node */
    for (int64_t i = 0; i < n_edges; i++) {
        int64_t src = edges[i * 2];
        if (src >= 0 && src < n_atoms) {
            out_offsets[src + 1]++;
        }
    }

    /* Cumulative sum to get final offsets */
    for (int64_t i = 1; i <= n_atoms; i++) {
        out_offsets[i] += out_offsets[i - 1];
    }

    /* Allocate temporary write positions (copy of offsets) */
    int64_t *write_pos = (int64_t *)malloc((size_t)n_atoms * sizeof(int64_t));
    if (write_pos == NULL) {
        return -1;
    }
    memcpy(write_pos, out_offsets, (size_t)n_atoms * sizeof(int64_t));

    /* Pass 2: Scatter edges to final positions (counting sort) */
    for (int64_t i = 0; i < n_edges; i++) {
        int64_t src = edges[i * 2];
        int64_t dst = edges[i * 2 + 1];
        if (src >= 0 && src < n_atoms) {
            out_neighbors[write_pos[src]++] = dst;
        }
    }

    free(write_pos);
    return 0;
}


int64_t build_zmatrix_from_csr(
    const int64_t *offsets,
    const int64_t *neighbors,
    int64_t n_atoms,
    int64_t chain_start,
    int64_t chain_size,
    int64_t root,
    int64_t *out_zmatrix
) {
    if (chain_size == 0) return 0;

    /* Validate root is in bounds */
    if (root < 0 || root >= n_atoms) {
        return -1;
    }

    /* Check if root has any neighbors */
    if (offsets[root + 1] == offsets[root]) {
        /* No bonds from root: single atom with no references */
        out_zmatrix[0] = root;
        out_zmatrix[1] = -1;
        out_zmatrix[2] = -1;
        out_zmatrix[3] = -1;
        return 1;
    }

    /*
     * Allocate working arrays sized to chain_size, not n_atoms.
     * BFS only visits atoms in [chain_start, chain_end), so we use
     * local indices (atom - chain_start) into these arrays.
     */
    int64_t *parent = (int64_t *)malloc((size_t)chain_size * sizeof(int64_t));
    int64_t *grandparent = (int64_t *)malloc((size_t)chain_size * sizeof(int64_t));
    int8_t *visited = (int8_t *)calloc((size_t)chain_size, sizeof(int8_t));
    int64_t *order = (int64_t *)malloc((size_t)chain_size * sizeof(int64_t));
    int64_t *queue = (int64_t *)malloc((size_t)chain_size * sizeof(int64_t));

    if (!parent || !grandparent || !visited || !order || !queue) {
        free(parent);
        free(grandparent);
        free(visited);
        free(order);
        free(queue);
        return -1;
    }

    /* Initialize parent and grandparent to -1 (only chain_size elements) */
    for (int64_t i = 0; i < chain_size; i++) {
        parent[i] = -1;
        grandparent[i] = -1;
    }

    /* ------------------------------------------------------------------ */
    /* Step 1: BFS spanning tree from root                                */
    /* ------------------------------------------------------------------ */

    int64_t chain_end = chain_start + chain_size;
    int64_t order_len = 0;
    int64_t queue_head = 0, queue_tail = 0;

    /* Enqueue root (use local index for visited) */
    int64_t root_local = root - chain_start;
    queue[queue_tail++] = root;
    visited[root_local] = 1;

    while (queue_head < queue_tail) {
        int64_t current = queue[queue_head++];
        order[order_len++] = current;

        /* Get neighbors from CSR */
        int64_t start = offsets[current];
        int64_t end = offsets[current + 1];

        for (int64_t i = start; i < end; i++) {
            int64_t neighbor = neighbors[i];

            /* Only process atoms in this chain */
            if (neighbor < chain_start || neighbor >= chain_end) continue;

            int64_t neighbor_local = neighbor - chain_start;
            if (visited[neighbor_local]) continue;

            visited[neighbor_local] = 1;
            parent[neighbor_local] = current;  /* Store global index as parent */
            queue[queue_tail++] = neighbor;
        }
    }

    /* ------------------------------------------------------------------ */
    /* Step 2: Build Z-matrix entries                                     */
    /* ------------------------------------------------------------------ */

    for (int64_t i = 0; i < order_len; i++) {
        int64_t atom = order[i];  /* Global index */
        int64_t atom_local = atom - chain_start;
        int64_t p = parent[atom_local];  /* Parent is stored as global index */
        int64_t *entry = &out_zmatrix[i * 4];

        if (i == 0) {
            /* First atom: no references */
            entry[0] = atom;
            entry[1] = -1;
            entry[2] = -1;
            entry[3] = -1;
        }
        else if (i == 1) {
            /* Second atom: distance to parent only */
            entry[0] = atom;
            entry[1] = p;
            entry[2] = -1;
            entry[3] = -1;
        }
        else if (i == 2) {
            /* Third atom: distance and angle */
            int64_t p_local = p - chain_start;
            int64_t gp = parent[p_local];  /* Grandparent (global) */
            if (gp == -1) {
                gp = find_child_of(order, i, parent, chain_start, p, atom);
            }
            grandparent[atom_local] = gp;

            entry[0] = atom;
            entry[1] = p;
            entry[2] = gp;
            entry[3] = -1;
        }
        else {
            /* Full Z-matrix entry */
            int64_t p_local = p - chain_start;
            int64_t gp = parent[p_local];
            if (gp == -1) {
                gp = find_child_of(order, i, parent, chain_start, p, atom);
            }

            /* Find great-grandparent for dihedral */
            int64_t gp_local = (gp >= chain_start) ? gp - chain_start : -1;
            int64_t ggp = (gp_local >= 0) ? grandparent[p_local] : -1;
            if (ggp == atom || ggp == p || ggp == gp || ggp == -1) {
                ggp = (gp_local >= 0) ? parent[gp_local] : -1;
            }
            if (ggp == atom || ggp == p || ggp == gp || ggp == -1) {
                ggp = find_placed_neighbor(order, i, parent, chain_start, gp, atom, p, gp);
            }

            grandparent[atom_local] = gp;

            entry[0] = atom;
            entry[1] = p;
            entry[2] = gp;
            entry[3] = ggp;
        }
    }

    /* Cleanup */
    free(parent);
    free(grandparent);
    free(visited);
    free(order);
    free(queue);

    return order_len;
}


int64_t build_zmatrix_parallel(
    const int64_t *offsets,
    const int64_t *neighbors,
    int64_t n_atoms,
    const int64_t *chain_starts,
    const int64_t *chain_sizes,
    const int64_t *roots,
    int64_t n_chains,
    int64_t *out_zmatrix,
    int64_t *out_counts
) {
    if (n_chains == 0) return 0;

    /* Compute output offsets for each chain (where each chain's Z-matrix starts) */
    int64_t *output_offsets = (int64_t *)malloc((size_t)(n_chains + 1) * sizeof(int64_t));
    if (output_offsets == NULL) return -1;

    output_offsets[0] = 0;
    for (int64_t i = 0; i < n_chains; i++) {
        output_offsets[i + 1] = output_offsets[i] + chain_sizes[i];
    }

    int error_flag = 0;

    /* Process chains in parallel */
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
#endif
    for (int64_t c = 0; c < n_chains; c++) {
        if (error_flag) continue;  /* Skip if error occurred */

        int64_t chain_start = chain_starts[c];
        int64_t chain_size = chain_sizes[c];
        int64_t root = roots[c];

        if (chain_size == 0) {
            out_counts[c] = 0;
            continue;
        }

        /* Output location for this chain's Z-matrix */
        int64_t *chain_output = &out_zmatrix[output_offsets[c] * 4];

        /* Build Z-matrix for this chain */
        int64_t count = build_zmatrix_from_csr(
            offsets, neighbors, n_atoms,
            chain_start, chain_size, root,
            chain_output
        );

        if (count < 0) {
#ifdef _OPENMP
            #pragma omp atomic write
#endif
            error_flag = 1;
            out_counts[c] = 0;
        } else {
            out_counts[c] = count;
        }
    }

    free(output_offsets);

    if (error_flag) return -1;

    /* Compute total entries */
    int64_t total = 0;
    for (int64_t c = 0; c < n_chains; c++) {
        total += out_counts[c];
    }

    return total;
}
