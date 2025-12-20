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
#include "canonical_refs.h"
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


/* ========================================================================== */
/* DIHEDRAL-AWARE REFERENCE RESOLUTION                                        */
/* ========================================================================== */

/**
 * Binary search to find which residue an atom belongs to.
 *
 * @param atom_idx      Global atom index
 * @param residue_starts Cumulative residue sizes (n_residues + 1)
 * @param n_residues    Number of residues
 * @return Residue index, or -1 if not found
 */
static int64_t find_residue_for_atom(
    int64_t atom_idx,
    const int64_t *residue_starts,
    int64_t n_residues
) {
    /* Guard against empty residue array */
    if (n_residues <= 0) {
        return -1;
    }
    int64_t lo = 0, hi = n_residues;
    while (lo < hi) {
        int64_t mid = lo + (hi - lo) / 2;
        if (residue_starts[mid + 1] <= atom_idx) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if (lo < n_residues && residue_starts[lo] <= atom_idx && atom_idx < residue_starts[lo + 1]) {
        return lo;
    }
    return -1;
}

/**
 * Check if crossing from res_idx to target_res crosses a chain boundary.
 *
 * @param res_idx            Current residue index
 * @param target_res         Target residue index
 * @param chain_res_starts   Residue indices where each chain starts
 * @param n_chains           Number of chains
 * @return 1 if crosses boundary, 0 otherwise
 */
static int crosses_chain_boundary(
    int64_t res_idx,
    int64_t target_res,
    const int64_t *chain_res_starts,
    int64_t n_chains
) {
    if (res_idx == target_res) return 0;

    int64_t min_res = (res_idx < target_res) ? res_idx : target_res;
    int64_t max_res = (res_idx > target_res) ? res_idx : target_res;

    /* Check if any chain boundary falls in (min_res, max_res] */
    for (int64_t c = 1; c < n_chains; c++) {
        int64_t boundary = chain_res_starts[c];
        if (boundary > min_res && boundary <= max_res) {
            return 1;
        }
    }
    return 0;
}

/**
 * Find atom of specified type within a residue.
 *
 * @param target_res     Residue index
 * @param expected_type  Expected atom type (global index)
 * @param atoms          Atom types array
 * @param residue_starts Cumulative residue sizes
 * @return Global atom index, or -1 if not found
 */
static int64_t find_atom_of_type_in_residue(
    int64_t target_res,
    int16_t expected_type,
    const int32_t *atoms,
    const int64_t *residue_starts
) {
    int64_t start = residue_starts[target_res];
    int64_t end = residue_starts[target_res + 1];

    for (int64_t i = start; i < end; i++) {
        if (atoms[i] == expected_type) {
            return i;
        }
    }
    return -1;
}

/**
 * Try to resolve dihedral-specific references for an atom.
 *
 * If the atom owns a named dihedral, looks up the reference pattern and
 * tries to resolve each reference atom. Returns success if all three
 * references can be resolved and are already placed.
 *
 * @param atom_idx          Global atom index being placed
 * @param atoms             Atom types array (global)
 * @param sequence          Residue types array
 * @param residue_starts    Cumulative residue sizes
 * @param n_residues        Number of residues
 * @param chain_res_starts  Residue indices where chains start
 * @param n_chains          Number of chains
 * @param placed            Boolean array: is atom placed? (indexed by global atom idx)
 * @param out_dist          Output: distance reference atom
 * @param out_ang           Output: angle reference atom
 * @param out_dih           Output: dihedral reference atom
 * @param out_dtype         Output: dihedral type index
 * @return 1 if successful, 0 if cannot resolve
 */
static int resolve_dihedral_refs(
    int64_t atom_idx,
    const int32_t *atoms,
    const int32_t *sequence,
    const int64_t *residue_starts,
    int64_t n_residues,
    const int64_t *chain_res_starts,
    int64_t n_chains,
    const int8_t *placed,
    int64_t *out_dist,
    int64_t *out_ang,
    int64_t *out_dih,
    int8_t *out_dtype
) {
    int32_t atom_type = atoms[atom_idx];

    /* Check bounds */
    if (atom_type < 0 || atom_type >= NUM_ATOM_TYPES) {
        return 0;
    }

    /* Check if this atom owns a named dihedral */
    int8_t dihedral_type = ATOM_DIHEDRAL_TYPE[atom_type];
    if (dihedral_type < 0) {
        return 0;  /* Not a dihedral owner */
    }

    /* Get reference pattern: [dih_off, dih_idx, ang_off, ang_idx, dist_off, dist_idx] */
    const int8_t *refs = ATOM_DIHEDRAL_REFS[atom_type];

    /* Find which residue this atom belongs to */
    int64_t res_idx = find_residue_for_atom(atom_idx, residue_starts, n_residues);
    if (res_idx < 0) {
        return 0;
    }

    /* Resolve each reference: dih (i=0), ang (i=1), dist (i=2) */
    int64_t resolved[3];  /* [dih_ref, ang_ref, dist_ref] */

    for (int i = 0; i < 3; i++) {
        int8_t offset = refs[i * 2];
        int8_t local_idx = refs[i * 2 + 1];

        int64_t target_res = res_idx + offset;

        /* Check residue bounds */
        if (target_res < 0 || target_res >= n_residues) {
            return 0;
        }

        /* Check chain boundary */
        if (offset != 0 && crosses_chain_boundary(res_idx, target_res, chain_res_starts, n_chains)) {
            return 0;
        }

        /* Look up expected atom type from canonical ordering */
        int32_t target_res_type = sequence[target_res];
        if (target_res_type < 0 || target_res_type >= NUM_RESIDUE_TYPES) {
            return 0;
        }
        if (local_idx >= RESIDUE_ATOM_COUNTS[target_res_type]) {
            return 0;
        }
        int16_t expected_atom_type = RESIDUE_CANONICAL_ATOMS[target_res_type][local_idx];

        /* Find atom of this type in target residue */
        int64_t ref_atom = find_atom_of_type_in_residue(
            target_res, expected_atom_type, atoms, residue_starts);
        if (ref_atom < 0) {
            return 0;
        }

        /* Check that reference is already placed */
        if (!placed[ref_atom]) {
            return 0;
        }

        resolved[i] = ref_atom;
    }

    /* Success - all references resolved */
    *out_dih = resolved[0];
    *out_ang = resolved[1];
    *out_dist = resolved[2];
    *out_dtype = dihedral_type;

    return 1;
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


int64_t build_zmatrix_from_csr(
    const int64_t *offsets,
    const int64_t *neighbors,
    int64_t n_atoms,
    int64_t chain_start,
    int64_t chain_size,
    int64_t root,
    /* New params for dihedral-aware selection (can all be NULL to disable) */
    const int32_t *atoms,            /* Atom types array */
    const int32_t *sequence,         /* Residue types array */
    const int64_t *residue_starts,   /* Cumsum of residue sizes */
    int64_t n_residues,
    const int64_t *chain_res_starts, /* Residue indices where chains start */
    int64_t n_chains,
    /* Outputs */
    int64_t *out_zmatrix,
    int8_t *out_dihedral_types,      /* Dihedral type per entry (can be NULL) */
    int32_t *out_levels              /* BFS level per entry (can be NULL) */
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
        if (out_dihedral_types) out_dihedral_types[0] = -1;
        if (out_levels) out_levels[0] = 0;
        return 1;
    }

    /* Check if dihedral-aware mode is enabled */
    int dihedral_aware = (atoms != NULL && sequence != NULL &&
                          residue_starts != NULL && chain_res_starts != NULL);

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
    int32_t *level = (int32_t *)calloc((size_t)chain_size, sizeof(int32_t));  /* BFS level */

    /* Check base allocations FIRST before any conditional allocations */
    if (!parent || !grandparent || !visited || !order || !queue || !level) {
        free(parent);
        free(grandparent);
        free(visited);
        free(order);
        free(queue);
        free(level);
        return -1;
    }

    /* Placed array for dihedral resolution (indexed by GLOBAL atom index) */
    int8_t *placed = NULL;
    if (dihedral_aware) {
        placed = (int8_t *)calloc((size_t)n_atoms, sizeof(int8_t));
        if (!placed) {
            free(parent);
            free(grandparent);
            free(visited);
            free(order);
            free(queue);
            free(level);
            return -1;
        }
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

            int64_t current_local = current - chain_start;
            visited[neighbor_local] = 1;
            parent[neighbor_local] = current;  /* Store global index as parent */
            level[neighbor_local] = level[current_local] + 1;  /* Child is one level deeper */
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
        int8_t dihedral_type = -1;  /* Default: no named dihedral */

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
            /* Full Z-matrix entry - try dihedral-aware refs first */
            int used_dihedral_refs = 0;

            if (dihedral_aware) {
                int64_t dih_dist, dih_ang, dih_dih;
                int8_t dtype;

                if (resolve_dihedral_refs(
                        atom, atoms, sequence, residue_starts, n_residues,
                        chain_res_starts, n_chains, placed,
                        &dih_dist, &dih_ang, &dih_dih, &dtype)) {
                    /* Successfully resolved dihedral-specific references */
                    entry[0] = atom;
                    entry[1] = dih_dist;
                    entry[2] = dih_ang;
                    entry[3] = dih_dih;
                    dihedral_type = dtype;
                    used_dihedral_refs = 1;

                    /* Still track grandparent for BFS consistency */
                    grandparent[atom_local] = dih_ang;
                }
            }

            if (!used_dihedral_refs) {
                /* Fall back to default BFS references */
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

        /* Mark this atom as placed (for dihedral resolution of later atoms) */
        if (placed) {
            placed[atom] = 1;
        }

        /* Store dihedral type if output array provided */
        if (out_dihedral_types) {
            out_dihedral_types[i] = dihedral_type;
        }

        /* Store BFS level if output array provided */
        if (out_levels) {
            out_levels[i] = level[atom_local];
        }
    }

    /* Cleanup */
    free(parent);
    free(grandparent);
    free(visited);
    free(order);
    free(queue);
    free(level);
    free(placed);

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
    /* New params for dihedral-aware selection (can all be NULL to disable) */
    const int32_t *atoms,            /* Atom types array */
    const int32_t *sequence,         /* Residue types array */
    const int64_t *residue_starts,   /* Cumsum of residue sizes */
    int64_t n_residues,
    const int64_t *chain_res_starts, /* Residue indices where chains start */
    /* Outputs */
    int64_t *out_zmatrix,
    int8_t *out_dihedral_types,      /* Dihedral type per entry (can be NULL) */
    int32_t *out_levels,             /* BFS level per entry (can be NULL) */
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

        /* Output location for this chain's Z-matrix, dihedral types, and levels */
        int64_t *chain_output = &out_zmatrix[output_offsets[c] * 4];
        int8_t *chain_dtypes = out_dihedral_types ?
            &out_dihedral_types[output_offsets[c]] : NULL;
        int32_t *chain_levels = out_levels ?
            &out_levels[output_offsets[c]] : NULL;

        /* Build Z-matrix for this chain */
        int64_t count = build_zmatrix_from_csr(
            offsets, neighbors, n_atoms,
            chain_start, chain_size, root,
            atoms, sequence, residue_starts, n_residues,
            chain_res_starts, n_chains,
            chain_output, chain_dtypes, chain_levels
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


/* ========================================================================== */
/* CANONICAL Z-MATRIX CONSTRUCTION                                            */
/* ========================================================================== */

/**
 * Find atom by type within a residue.
 *
 * @param atoms        (n_atoms,) int32 atom type values
 * @param res_start    First atom index of the residue
 * @param res_end      One past last atom index of the residue
 * @param atom_type    Atom type to find
 * @return             Global atom index, or -1 if not found
 */
static int64_t find_atom_by_type(
    const int32_t *atoms,
    int64_t res_start,
    int64_t res_end,
    int32_t atom_type
) {
    for (int64_t i = res_start; i < res_end; i++) {
        if (atoms[i] == atom_type) return i;
    }
    return -1;
}

/**
 * Find first bonded neighbor with lower index than current atom.
 *
 * @param atom_idx       Current atom's global index
 * @param bond_offsets   CSR offsets for bond graph
 * @param bond_neighbors CSR neighbor indices
 * @param excludes       Array of atoms to exclude (can be NULL)
 * @param n_excludes     Number of excludes
 * @return               Global atom index of neighbor, or -1 if not found
 */
static int64_t find_first_bonded_lower(
    int64_t atom_idx,
    const int64_t *bond_offsets,
    const int64_t *bond_neighbors,
    const int64_t *excludes,
    int n_excludes
) {
    int64_t start = bond_offsets[atom_idx];
    int64_t end = bond_offsets[atom_idx + 1];

    for (int64_t i = start; i < end; i++) {
        int64_t neighbor = bond_neighbors[i];
        if (neighbor >= atom_idx) continue;  /* Must be earlier */

        /* Check excludes */
        int excluded = 0;
        for (int e = 0; e < n_excludes; e++) {
            if (neighbor == excludes[e]) {
                excluded = 1;
                break;
            }
        }
        if (!excluded) return neighbor;
    }
    return -1;
}

/**
 * Check if two residue indices are in different chains.
 *
 * @param res1            First residue index
 * @param res2            Second residue index
 * @param chain_starts    Array of residue indices where each chain starts
 * @param n_chains        Number of chains
 * @return                1 if different chains, 0 if same chain
 */
static int residues_in_different_chains(
    int64_t res1,
    int64_t res2,
    const int64_t *chain_starts,
    int64_t n_chains
) {
    /* Find which chain each residue belongs to */
    int64_t chain1 = -1, chain2 = -1;

    for (int64_t c = 0; c < n_chains; c++) {
        int64_t start = chain_starts[c];
        int64_t end = (c < n_chains - 1) ? chain_starts[c + 1] : INT64_MAX;

        if (res1 >= start && res1 < end) chain1 = c;
        if (res2 >= start && res2 < end) chain2 = c;
    }

    return chain1 != chain2;
}

/**
 * Resolve a canonical reference.
 *
 * @param ref_value       Value from ATOM_CANONICAL_REFS (atom type or backbone ID)
 * @param ref_offset      Residue offset (-1, 0, or +1)
 * @param res_idx         Current residue index
 * @param atoms           (n_atoms,) int32 atom type values
 * @param sequence        (n_residues,) int32 residue type indices
 * @param residue_starts  (n_residues+1,) int64 cumulative atom starts
 * @param n_residues      Total number of residues
 * @param chain_starts    (n_chains,) int64 residue start index per chain
 * @param n_chains        Number of chains
 * @return                Global atom index, or -1 if cannot resolve
 */
static int64_t resolve_canonical_ref(
    int16_t ref_value,
    int8_t ref_offset,
    int64_t res_idx,
    const int32_t *atoms,
    const int32_t *sequence,
    const int64_t *residue_starts,
    int64_t n_residues,
    const int64_t *chain_starts,
    int64_t n_chains
) {
    if (ref_value < 0) return -1;  /* No reference */

    int64_t target_res = res_idx + ref_offset;
    if (target_res < 0 || target_res >= n_residues) return -1;

    /* Check chain boundary for inter-residue refs */
    if (ref_offset != 0) {
        if (residues_in_different_chains(res_idx, target_res, chain_starts, n_chains)) {
            return -1;
        }
    }

    /* Resolve atom type */
    int32_t target_atom_type;
    if (ref_offset == 0) {
        /* Intra-residue: ref_value IS the atom type */
        target_atom_type = ref_value;
    } else {
        /* Inter-residue: ref_value is backbone name ID */
        int32_t target_res_type = sequence[target_res];
        if (target_res_type < 0 || target_res_type >= NUM_RESIDUE_TYPES) {
            return -1;
        }
        target_atom_type = RESIDUE_BACKBONE_ATOMS[target_res_type][ref_value];
        if (target_atom_type < 0) return -1;
    }

    /* Find atom in target residue */
    int64_t target_start = residue_starts[target_res];
    int64_t target_end = residue_starts[target_res + 1];

    return find_atom_by_type(atoms, target_start, target_end, target_atom_type);
}


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
) {
    if (n_atoms == 0) return 0;

    /* Allocate working arrays */
    int64_t *residue_starts = (int64_t *)malloc((size_t)(n_residues + 1) * sizeof(int64_t));
    int64_t *chain_res_starts = (int64_t *)malloc((size_t)(n_chains + 1) * sizeof(int64_t));

    if (!residue_starts || !chain_res_starts) {
        free(residue_starts);
        free(chain_res_starts);
        return -1;
    }

    /* Compute residue start indices (cumulative sum of res_sizes) */
    residue_starts[0] = 0;
    for (int64_t r = 0; r < n_residues; r++) {
        int64_t prev = residue_starts[r];
        int64_t size = res_sizes[r];
        /* Check for overflow */
        if (size > 0 && prev > INT64_MAX - size) {
            free(residue_starts);
            free(chain_res_starts);
            return -1;
        }
        residue_starts[r + 1] = prev + size;
    }

    /* Compute chain start residue indices (cumulative sum of chain_lengths) */
    chain_res_starts[0] = 0;
    for (int64_t c = 0; c < n_chains; c++) {
        int64_t prev = chain_res_starts[c];
        int64_t len = chain_lengths[c];
        /* Check for overflow */
        if (len > 0 && prev > INT64_MAX - len) {
            free(residue_starts);
            free(chain_res_starts);
            return -1;
        }
        chain_res_starts[c + 1] = prev + len;
    }

    /* Process atoms in natural order */
    int64_t current_res = 0;

    for (int64_t atom_idx = 0; atom_idx < n_atoms; atom_idx++) {
        /* Find which residue this atom belongs to */
        while (current_res < n_residues - 1 &&
               atom_idx >= residue_starts[current_res + 1]) {
            current_res++;
        }

        int32_t atom_type = atoms[atom_idx];

        /* Initialize output */
        int64_t *entry = &out_zmatrix[atom_idx * 4];
        entry[0] = atom_idx;
        entry[1] = -1;
        entry[2] = -1;
        entry[3] = -1;
        out_dihedral_types[atom_idx] = -1;

        /* Try canonical refs if available */
        if (atom_type > 0 && atom_type < NUM_ATOM_TYPES &&
            ATOM_HAS_CANONICAL_REFS[atom_type]) {

            const int16_t *refs = ATOM_CANONICAL_REFS[atom_type];
            int16_t dist_ref = refs[0];
            int16_t ang_ref = refs[1];
            int16_t dih_ref = refs[2];
            int8_t dist_off = (int8_t)refs[3];
            int8_t ang_off = (int8_t)refs[4];
            int8_t dih_off = (int8_t)refs[5];

            /* Resolve references */
            int64_t dist_atom = resolve_canonical_ref(
                dist_ref, dist_off, current_res,
                atoms, sequence, residue_starts, n_residues,
                chain_res_starts, n_chains
            );
            int64_t ang_atom = resolve_canonical_ref(
                ang_ref, ang_off, current_res,
                atoms, sequence, residue_starts, n_residues,
                chain_res_starts, n_chains
            );
            int64_t dih_atom = resolve_canonical_ref(
                dih_ref, dih_off, current_res,
                atoms, sequence, residue_starts, n_residues,
                chain_res_starts, n_chains
            );

            /* IMPORTANT: Refs must point to earlier atoms (lower indices). */
            /* This is a Z-matrix invariant. If a canonical ref points to a */
            /* higher-indexed atom, we must invalidate it so fallback is used. */
            if (dist_atom >= atom_idx) dist_atom = -1;
            if (ang_atom >= atom_idx) ang_atom = -1;
            if (dih_atom >= atom_idx) dih_atom = -1;

            /* Store resolved refs */
            entry[1] = dist_atom;
            entry[2] = ang_atom;
            entry[3] = dih_atom;

            /* Assign dihedral type only if all refs are valid */
            if (dist_atom >= 0 && ang_atom >= 0 && dih_atom >= 0) {
                out_dihedral_types[atom_idx] = ATOM_DIHEDRAL_TYPE[atom_type];
            }
        }

        /* Fall back to bond graph if canonical refs are missing or failed */
        if (bond_offsets != NULL && bond_neighbors != NULL) {
            /* Fill in missing distance ref */
            if (entry[1] < 0) {
                entry[1] = find_first_bonded_lower(
                    atom_idx, bond_offsets, bond_neighbors, NULL, 0
                );
            }

            /* Fill in missing angle ref (only if we have a valid dist_ref) */
            if (entry[1] >= 0 && entry[2] < 0) {
                int64_t excludes1[1] = {atom_idx};
                entry[2] = find_first_bonded_lower(
                    entry[1], bond_offsets, bond_neighbors, excludes1, 1
                );
            }

            /* Fill in missing dihedral ref (only if we have valid dist and angle refs) */
            if (entry[1] >= 0 && entry[2] >= 0 && entry[3] < 0) {
                int64_t excludes2[2] = {atom_idx, entry[1]};
                entry[3] = find_first_bonded_lower(
                    entry[2], bond_offsets, bond_neighbors, excludes2, 2
                );
            }
        }
    }

    free(residue_starts);
    free(chain_res_starts);

    return n_atoms;
}
