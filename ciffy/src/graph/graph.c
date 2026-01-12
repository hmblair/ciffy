/**
 * @file graph.c
 * @brief Bond graph construction for molecular structures.
 *
 * Builds molecular bond graph by:
 * 1. Adding intra-residue bonds from precomputed patterns
 * 2. Adding inter-residue linking bonds (peptide/phosphodiester)
 * 3. Filtering bonds where atoms are missing (incomplete residues)
 */

#include "graph.h"
#include "../internal/bond_patterns.h"
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


int edges_to_csr(
    const int64_t *edges,
    int64_t n_edges,
    int64_t n_atoms,
    int64_t *out_offsets,
    int64_t *out_neighbors
) {
    /* Validate input sizes */
    if (n_atoms < 0 || n_edges < 0) {
        return -1;
    }

    /* Handle edge case of zero atoms */
    if (n_atoms == 0) {
        out_offsets[0] = 0;
        return 0;
    }

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


int64_t find_connected_components_c(
    const int64_t *offsets,
    const int64_t *neighbors,
    int64_t n_atoms,
    int64_t *out_atoms,
    int64_t *out_offsets
) {
    if (n_atoms == 0) {
        out_offsets[0] = 0;
        return 0;
    }

    /* Allocate visited array and queue */
    int8_t *visited = (int8_t *)calloc((size_t)n_atoms, sizeof(int8_t));
    int64_t *queue = (int64_t *)malloc((size_t)n_atoms * sizeof(int64_t));

    if (!visited || !queue) {
        free(visited);
        free(queue);
        return -1;
    }

    int64_t n_components = 0;
    int64_t atoms_written = 0;
    out_offsets[0] = 0;

    for (int64_t start = 0; start < n_atoms; start++) {
        if (visited[start]) continue;

        /* Check if atom has any neighbors */
        int64_t n_neighbors = offsets[start + 1] - offsets[start];

        if (n_neighbors == 0) {
            /* Isolated atom - add as single-atom component */
            visited[start] = 1;
            out_atoms[atoms_written++] = start;
            n_components++;
            out_offsets[n_components] = atoms_written;
            continue;
        }

        /* BFS to find component */
        int64_t queue_head = 0, queue_tail = 0;

        queue[queue_tail++] = start;
        visited[start] = 1;

        while (queue_head < queue_tail) {
            int64_t node = queue[queue_head++];

            /* Get neighbors from CSR */
            int64_t edge_start = offsets[node];
            int64_t edge_end = offsets[node + 1];

            for (int64_t i = edge_start; i < edge_end; i++) {
                int64_t neighbor = neighbors[i];
                if (!visited[neighbor]) {
                    visited[neighbor] = 1;
                    queue[queue_tail++] = neighbor;
                }
            }
        }

        /* Copy BFS result (queue contents) to output */
        int64_t component_size = queue_tail;
        for (int64_t i = 0; i < component_size; i++) {
            out_atoms[atoms_written++] = queue[i];
        }

        n_components++;
        out_offsets[n_components] = atoms_written;
    }

    free(visited);
    free(queue);

    return n_components;
}
