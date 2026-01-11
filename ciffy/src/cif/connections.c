/**
 * @file connections.c
 * @brief Parse _struct_conn block for hydrogen bonds, covalent bonds, etc.
 */

#include "connections.h"
#include "io.h"
#include "chain_lookup.h"
#include "../log.h"

#include <string.h>


/* ============================================================================
 * CONNECTION TYPE PARSING
 * ============================================================================ */

/**
 * @brief Convert conn_type_id string to ConnType enum.
 */
static ConnType _parse_conn_type(const char *str, size_t len) {
    if (len == 6 && strncmp(str, "hydrog", 6) == 0) return CONN_TYPE_HYDROG;
    if (len == 6 && strncmp(str, "covale", 6) == 0) return CONN_TYPE_COVALE;
    if (len == 6 && strncmp(str, "metalc", 6) == 0) return CONN_TYPE_METALC;
    if (len == 6 && strncmp(str, "disulf", 6) == 0) return CONN_TYPE_DISULF;
    return CONN_TYPE_UNKNOWN;
}


/* ============================================================================
 * ATOM LOOKUP - BINARY SEARCH APPROACH
 *
 * Instead of building a hash of all atoms (O(n) insertions), we use binary
 * search on the sorted atom block. CIF atom blocks are ordered by:
 *   1. Chain (label_asym_id)
 *   2. Residue (label_seq_id) within chain
 *   3. Atom name (label_atom_id) within residue
 *
 * Lookup cost: O(log n) binary search + O(residue_size) linear scan
 * Build cost: O(n_chains) to compute chain offsets
 *
 * For typical structures with ~10k connections referencing ~10k atoms,
 * this is much faster than hashing 100k+ atoms.
 * ============================================================================ */

/**
 * @brief Atom lookup context for binary search approach.
 *
 * Uses ChainLookup from chain_lookup.h for O(1) chain name lookups.
 */
typedef struct {
    mmBlock *block;          /**< Atom site block */
    int *chain_offsets;      /**< Start row for each chain [n_chains + 1] */
    int n_chains;            /**< Number of chains */
    char **chain_names;      /**< Chain name strings */
    ChainLookup chain_lookup; /**< Hash table for O(1) chain lookup */
    int asym_idx;            /**< Attribute index for label_asym_id */
    int seq_idx;             /**< Attribute index for label_seq_id */
    int atom_idx;            /**< Attribute index for label_atom_id */
} AtomLookup;

/* Note: _fast_get_field from io.h is used for field extraction instead of
 * a local duplicate implementation. */

/**
 * @brief Compare two strings with known lengths.
 */
static inline int _strncmp_len(const char *a, size_t a_len,
                                const char *b, size_t b_len) {
    size_t min_len = a_len < b_len ? a_len : b_len;
    int cmp = strncmp(a, b, min_len);
    if (cmp != 0) return cmp;
    if (a_len < b_len) return -1;
    if (a_len > b_len) return 1;
    return 0;
}

/**
 * @brief Parse seq_id as integer for faster comparison.
 */
static inline int _parse_seq_id(const char *str, size_t len) {
    int val = 0;
    for (size_t i = 0; i < len && str[i] >= '0' && str[i] <= '9'; i++) {
        val = val * 10 + (str[i] - '0');
    }
    return val;
}

/**
 * @brief Look up atom by (chain, seq, atom) using binary search.
 *
 * @return Atom row index, or -1 if not found
 */
static int32_t _lookup_atom_bsearch(const AtomLookup *ctx,
                                     const char *chain, size_t chain_len,
                                     const char *seq, size_t seq_len,
                                     const char *atom, size_t atom_len) {
    /* Find chain index using hash lookup */
    int chain_idx = chain_lookup_find(&ctx->chain_lookup, chain, chain_len);
    if (chain_idx < 0) return -1;

    int start = ctx->chain_offsets[chain_idx];
    int end = ctx->chain_offsets[chain_idx + 1];
    if (start >= end) return -1;

    /* Parse target seq_id */
    int target_seq = _parse_seq_id(seq, seq_len);

    /* Binary search for seq_id within chain */
    mmBlock *block = ctx->block;
    char **lines = block->lines;
    const int *offsets = block->offsets;

    int lo = start, hi = end - 1;
    int first_match = -1;

    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;

        size_t mid_seq_len;
        const char *mid_seq = _fast_get_field(lines[mid], offsets, ctx->seq_idx, &mid_seq_len);
        _strip_outer_quotes(&mid_seq, &mid_seq_len);

        int mid_seq_val = _parse_seq_id(mid_seq, mid_seq_len);

        if (mid_seq_val < target_seq) {
            lo = mid + 1;
        } else if (mid_seq_val > target_seq) {
            hi = mid - 1;
        } else {
            /* Found matching seq_id - find first occurrence */
            first_match = mid;
            hi = mid - 1;
        }
    }

    if (first_match < 0) return -1;

    /* Linear search within residue for matching atom name */
    for (int row = first_match; row < end; row++) {
        size_t row_seq_len, row_atom_len;

        const char *row_seq = _fast_get_field(lines[row], offsets, ctx->seq_idx, &row_seq_len);
        _strip_outer_quotes(&row_seq, &row_seq_len);

        int row_seq_val = _parse_seq_id(row_seq, row_seq_len);
        if (row_seq_val != target_seq) break;  /* Past this residue */

        const char *row_atom = _fast_get_field(lines[row], offsets, ctx->atom_idx, &row_atom_len);
        _strip_outer_quotes(&row_atom, &row_atom_len);

        if (row_atom_len == atom_len && strncmp(row_atom, atom, atom_len) == 0) {
            return row;
        }
    }

    return -1;
}

/**
 * @brief Build atom lookup context.
 *
 * Computes chain offsets from atoms_per_chain for O(1) chain range lookup.
 */
AtomLookup *_build_atom_lookup_ctx(mmBlock *block, mmCIF *cif, CifErrorContext *ctx) {
    AtomLookup *lookup = (AtomLookup *)malloc(sizeof(AtomLookup));
    if (!lookup) {
        CIF_SET_ERROR(ctx, CIF_ERR_ALLOC, "Failed to allocate AtomLookup");
        return NULL;
    }

    lookup->block = block;
    lookup->n_chains = cif->chains;
    lookup->chain_names = cif->names;

    /* Compute chain offsets */
    lookup->chain_offsets = (int *)malloc((size_t)(cif->chains + 1) * sizeof(int));
    if (!lookup->chain_offsets) {
        free(lookup);
        CIF_SET_ERROR(ctx, CIF_ERR_ALLOC, "Failed to allocate chain_offsets");
        return NULL;
    }

    lookup->chain_offsets[0] = 0;
    for (int i = 0; i < cif->chains; i++) {
        lookup->chain_offsets[i + 1] = lookup->chain_offsets[i] + cif->atoms_per_chain[i];
    }

    /* Build chain name hash table using shared ChainLookup */
    chain_lookup_init(&lookup->chain_lookup);
    chain_lookup_build(&lookup->chain_lookup, cif->names, cif->chains);

    /* Get attribute indices */
    lookup->asym_idx = _get_attr_index(block, "label_asym_id", ctx);
    lookup->seq_idx = _get_attr_index(block, "label_seq_id", ctx);
    lookup->atom_idx = _get_attr_index(block, "label_atom_id", ctx);

    if (lookup->asym_idx < 0 || lookup->seq_idx < 0 || lookup->atom_idx < 0) {
        /* ChainLookup is embedded, no need to free it */
        free(lookup->chain_offsets);
        free(lookup);
        CIF_SET_ERROR(ctx, CIF_ERR_ATTR,
                      "Missing required attributes for atom lookup");
        return NULL;
    }

    LOG_DEBUG("Built atom lookup context: %d chains, %d total atoms",
              cif->chains, cif->atoms);

    return lookup;
}

/**
 * @brief Free atom lookup context.
 */
void _free_atom_lookup_ctx(AtomLookup *lookup) {
    if (lookup) {
        /* ChainLookup is embedded (not a pointer), no need to free it */
        free(lookup->chain_offsets);
        free(lookup);
    }
}

/* Keep hash-based lookup for backward compatibility and variable-width blocks */
AtomHash _build_atom_lookup(mmBlock *block, int n_atoms, CifErrorContext *ctx) {
    AtomHash hash = atom_hash_create((size_t)n_atoms);
    if (!hash.entries) {
        CIF_SET_ERROR(ctx, CIF_ERR_ALLOC, "Failed to allocate atom lookup hash");
        return hash;
    }

    if (block->lines == NULL) {
        CifError err = _precompute_lines(block, ctx);
        if (err != CIF_OK) {
            atom_hash_free(&hash);
            return (AtomHash){NULL, 0, 0};
        }
    }

    int asym_idx = _get_attr_index(block, "label_asym_id", ctx);
    int seq_idx = _get_attr_index(block, "label_seq_id", ctx);
    int atom_idx = _get_attr_index(block, "label_atom_id", ctx);

    if (asym_idx < 0 || seq_idx < 0 || atom_idx < 0) {
        CIF_SET_ERROR(ctx, CIF_ERR_ATTR,
                      "Missing required attributes for atom lookup");
        atom_hash_free(&hash);
        return (AtomHash){NULL, 0, 0};
    }

    if (!block->variable_width && block->offsets != NULL) {
        char **lines = block->lines;
        const int *offsets = block->offsets;

        for (int row = 0; row < n_atoms; row++) {
            char *line_start = lines[row];
            size_t asym_len, seq_len, atom_len;

            const char *asym = _fast_get_field(line_start, offsets, asym_idx, &asym_len);
            const char *seq = _fast_get_field(line_start, offsets, seq_idx, &seq_len);
            const char *atom = _fast_get_field(line_start, offsets, atom_idx, &atom_len);

            _strip_outer_quotes(&asym, &asym_len);
            _strip_outer_quotes(&seq, &seq_len);
            _strip_outer_quotes(&atom, &atom_len);

            atom_hash_insert(&hash, asym, asym_len, seq, seq_len, atom, atom_len, row);
        }
    } else {
        for (int row = 0; row < n_atoms; row++) {
            size_t asym_len, seq_len, atom_len;
            char *asym = _get_field_ptr(block, row, asym_idx, &asym_len);
            char *seq = _get_field_ptr(block, row, seq_idx, &seq_len);
            char *atom = _get_field_ptr(block, row, atom_idx, &atom_len);

            if (!asym || !seq || !atom) continue;

            _strip_outer_quotes((const char **)&asym, &asym_len);
            _strip_outer_quotes((const char **)&seq, &seq_len);
            _strip_outer_quotes((const char **)&atom, &atom_len);

            atom_hash_insert(&hash, asym, asym_len, seq, seq_len, atom, atom_len, row);
        }
    }

    LOG_DEBUG("Built atom lookup hash: %zu entries, capacity %zu",
              hash.count, hash.capacity);

    return hash;
}


/* ============================================================================
 * CONNECTION PARSING
 * ============================================================================ */

CifError _parse_connections(mmCIF *cif, mmBlock *conn_block,
                            const AtomHash *atom_hash, CifErrorContext *ctx) {
    if (!conn_block || conn_block->size == 0) {
        /* No connections to parse */
        cif->connections = NULL;
        cif->conn_types = NULL;
        cif->n_connections = 0;
        return CIF_OK;
    }

    int n_rows = conn_block->size;

    /* Precompute lines if needed */
    CifError err = _precompute_lines(conn_block, ctx);
    if (err != CIF_OK) return err;

    /* Get attribute indices */
    int type_idx = _get_attr_index(conn_block, "conn_type_id", ctx);
    int p1_asym_idx = _get_attr_index(conn_block, "ptnr1_label_asym_id", ctx);
    int p1_seq_idx = _get_attr_index(conn_block, "ptnr1_label_seq_id", ctx);
    int p1_atom_idx = _get_attr_index(conn_block, "ptnr1_label_atom_id", ctx);
    int p2_asym_idx = _get_attr_index(conn_block, "ptnr2_label_asym_id", ctx);
    int p2_seq_idx = _get_attr_index(conn_block, "ptnr2_label_seq_id", ctx);
    int p2_atom_idx = _get_attr_index(conn_block, "ptnr2_label_atom_id", ctx);

    if (type_idx < 0 || p1_asym_idx < 0 || p1_seq_idx < 0 || p1_atom_idx < 0 ||
        p2_asym_idx < 0 || p2_seq_idx < 0 || p2_atom_idx < 0) {
        LOG_DEBUG("Missing _struct_conn attributes, skipping connection parsing");
        _free_lines(conn_block);
        cif->connections = NULL;
        cif->conn_types = NULL;
        cif->n_connections = 0;
        return CIF_OK;
    }

    /* Allocate output arrays (may over-allocate if some connections can't be resolved) */
    int *connections = (int *)malloc((size_t)n_rows * 2 * sizeof(int));
    int *conn_types = (int *)malloc((size_t)n_rows * sizeof(int));

    if (!connections || !conn_types) {
        free(connections);
        free(conn_types);
        _free_lines(conn_block);
        CIF_SET_ERROR(ctx, CIF_ERR_ALLOC, "Failed to allocate connection arrays");
        return CIF_ERR_ALLOC;
    }

    int n_valid = 0;

    /* Parse each connection row */
    for (int row = 0; row < n_rows; row++) {
        size_t len;

        /* Get connection type */
        char *type_str = _get_field_ptr(conn_block, row, type_idx, &len);
        if (!type_str) continue;
        _strip_outer_quotes((const char **)&type_str, &len);
        ConnType conn_type = _parse_conn_type(type_str, len);

        /* Get partner 1 identifiers */
        size_t p1_asym_len, p1_seq_len, p1_atom_len;
        char *p1_asym = _get_field_ptr(conn_block, row, p1_asym_idx, &p1_asym_len);
        char *p1_seq = _get_field_ptr(conn_block, row, p1_seq_idx, &p1_seq_len);
        char *p1_atom = _get_field_ptr(conn_block, row, p1_atom_idx, &p1_atom_len);

        if (!p1_asym || !p1_seq || !p1_atom) continue;
        _strip_outer_quotes((const char **)&p1_asym, &p1_asym_len);
        _strip_outer_quotes((const char **)&p1_seq, &p1_seq_len);
        _strip_outer_quotes((const char **)&p1_atom, &p1_atom_len);

        /* Get partner 2 identifiers */
        size_t p2_asym_len, p2_seq_len, p2_atom_len;
        char *p2_asym = _get_field_ptr(conn_block, row, p2_asym_idx, &p2_asym_len);
        char *p2_seq = _get_field_ptr(conn_block, row, p2_seq_idx, &p2_seq_len);
        char *p2_atom = _get_field_ptr(conn_block, row, p2_atom_idx, &p2_atom_len);

        if (!p2_asym || !p2_seq || !p2_atom) continue;
        _strip_outer_quotes((const char **)&p2_asym, &p2_asym_len);
        _strip_outer_quotes((const char **)&p2_seq, &p2_seq_len);
        _strip_outer_quotes((const char **)&p2_atom, &p2_atom_len);

        /* Look up atom indices */
        int32_t atom1 = atom_hash_get(atom_hash,
                                       p1_asym, p1_asym_len,
                                       p1_seq, p1_seq_len,
                                       p1_atom, p1_atom_len);
        int32_t atom2 = atom_hash_get(atom_hash,
                                       p2_asym, p2_asym_len,
                                       p2_seq, p2_seq_len,
                                       p2_atom, p2_atom_len);

        /* Skip if either atom wasn't found */
        if (atom1 < 0 || atom2 < 0) {
            continue;
        }

        /* Store valid connection */
        connections[n_valid * 2] = atom1;
        connections[n_valid * 2 + 1] = atom2;
        conn_types[n_valid] = (int)conn_type;
        n_valid++;
    }

    _free_lines(conn_block);

    /* Shrink arrays to actual size */
    if (n_valid == 0) {
        free(connections);
        free(conn_types);
        cif->connections = NULL;
        cif->conn_types = NULL;
    } else if (n_valid < n_rows) {
        /* Realloc to shrink (optional, but saves memory) */
        int *new_conn = (int *)realloc(connections, (size_t)n_valid * 2 * sizeof(int));
        int *new_types = (int *)realloc(conn_types, (size_t)n_valid * sizeof(int));
        cif->connections = new_conn ? new_conn : connections;
        cif->conn_types = new_types ? new_types : conn_types;
    } else {
        cif->connections = connections;
        cif->conn_types = conn_types;
    }

    cif->n_connections = n_valid;

    LOG_DEBUG("Parsed %d connections (%d rows, %d resolved)",
              n_valid, n_rows, n_valid);

    return CIF_OK;
}


/**
 * @brief Parse connections using binary search (faster for large structures).
 *
 * Uses O(log n) binary search per lookup instead of building O(n) hash table.
 * For structures with many more atoms than connections, this is much faster.
 */
CifError _parse_connections_bsearch(mmCIF *cif, mmBlock *atom_block,
                                     mmBlock *conn_block, CifErrorContext *ctx) {
    if (!conn_block || conn_block->size == 0) {
        cif->connections = NULL;
        cif->conn_types = NULL;
        cif->n_connections = 0;
        return CIF_OK;
    }

    /* Build lookup context */
    AtomLookup *lookup = _build_atom_lookup_ctx(atom_block, cif, ctx);
    if (!lookup) return ctx->code;

    int n_rows = conn_block->size;

    /* Precompute lines for conn_block */
    CifError err = _precompute_lines(conn_block, ctx);
    if (err != CIF_OK) {
        _free_atom_lookup_ctx(lookup);
        return err;
    }

    /* Get attribute indices for conn_block */
    int type_idx = _get_attr_index(conn_block, "conn_type_id", ctx);
    int p1_asym_idx = _get_attr_index(conn_block, "ptnr1_label_asym_id", ctx);
    int p1_seq_idx = _get_attr_index(conn_block, "ptnr1_label_seq_id", ctx);
    int p1_atom_idx = _get_attr_index(conn_block, "ptnr1_label_atom_id", ctx);
    int p2_asym_idx = _get_attr_index(conn_block, "ptnr2_label_asym_id", ctx);
    int p2_seq_idx = _get_attr_index(conn_block, "ptnr2_label_seq_id", ctx);
    int p2_atom_idx = _get_attr_index(conn_block, "ptnr2_label_atom_id", ctx);

    if (type_idx < 0 || p1_asym_idx < 0 || p1_seq_idx < 0 || p1_atom_idx < 0 ||
        p2_asym_idx < 0 || p2_seq_idx < 0 || p2_atom_idx < 0) {
        LOG_DEBUG("Missing _struct_conn attributes, skipping connection parsing");
        _free_lines(conn_block);
        _free_atom_lookup_ctx(lookup);
        cif->connections = NULL;
        cif->conn_types = NULL;
        cif->n_connections = 0;
        return CIF_OK;
    }

    /* Allocate output arrays */
    int *connections = (int *)malloc((size_t)n_rows * 2 * sizeof(int));
    int *conn_types = (int *)malloc((size_t)n_rows * sizeof(int));

    if (!connections || !conn_types) {
        free(connections);
        free(conn_types);
        _free_lines(conn_block);
        _free_atom_lookup_ctx(lookup);
        CIF_SET_ERROR(ctx, CIF_ERR_ALLOC, "Failed to allocate connection arrays");
        return CIF_ERR_ALLOC;
    }

    int n_valid = 0;

    /* Parse each connection row */
    for (int row = 0; row < n_rows; row++) {
        size_t len;

        /* Get connection type */
        char *type_str = _get_field_ptr(conn_block, row, type_idx, &len);
        if (!type_str) continue;
        _strip_outer_quotes((const char **)&type_str, &len);
        ConnType conn_type = _parse_conn_type(type_str, len);

        /* Get partner 1 identifiers */
        size_t p1_asym_len, p1_seq_len, p1_atom_len;
        char *p1_asym = _get_field_ptr(conn_block, row, p1_asym_idx, &p1_asym_len);
        char *p1_seq = _get_field_ptr(conn_block, row, p1_seq_idx, &p1_seq_len);
        char *p1_atom = _get_field_ptr(conn_block, row, p1_atom_idx, &p1_atom_len);

        if (!p1_asym || !p1_seq || !p1_atom) continue;
        _strip_outer_quotes((const char **)&p1_asym, &p1_asym_len);
        _strip_outer_quotes((const char **)&p1_seq, &p1_seq_len);
        _strip_outer_quotes((const char **)&p1_atom, &p1_atom_len);

        /* Get partner 2 identifiers */
        size_t p2_asym_len, p2_seq_len, p2_atom_len;
        char *p2_asym = _get_field_ptr(conn_block, row, p2_asym_idx, &p2_asym_len);
        char *p2_seq = _get_field_ptr(conn_block, row, p2_seq_idx, &p2_seq_len);
        char *p2_atom = _get_field_ptr(conn_block, row, p2_atom_idx, &p2_atom_len);

        if (!p2_asym || !p2_seq || !p2_atom) continue;
        _strip_outer_quotes((const char **)&p2_asym, &p2_asym_len);
        _strip_outer_quotes((const char **)&p2_seq, &p2_seq_len);
        _strip_outer_quotes((const char **)&p2_atom, &p2_atom_len);

        /* Look up atom indices using binary search */
        int32_t atom1 = _lookup_atom_bsearch(lookup,
                                              p1_asym, p1_asym_len,
                                              p1_seq, p1_seq_len,
                                              p1_atom, p1_atom_len);
        int32_t atom2 = _lookup_atom_bsearch(lookup,
                                              p2_asym, p2_asym_len,
                                              p2_seq, p2_seq_len,
                                              p2_atom, p2_atom_len);

        /* Skip if either atom wasn't found */
        if (atom1 < 0 || atom2 < 0) {
            continue;
        }

        /* Store valid connection */
        connections[n_valid * 2] = atom1;
        connections[n_valid * 2 + 1] = atom2;
        conn_types[n_valid] = (int)conn_type;
        n_valid++;
    }

    _free_lines(conn_block);
    _free_atom_lookup_ctx(lookup);

    /* Shrink arrays to actual size */
    if (n_valid == 0) {
        free(connections);
        free(conn_types);
        cif->connections = NULL;
        cif->conn_types = NULL;
    } else if (n_valid < n_rows) {
        int *new_conn = (int *)realloc(connections, (size_t)n_valid * 2 * sizeof(int));
        int *new_types = (int *)realloc(conn_types, (size_t)n_valid * sizeof(int));
        cif->connections = new_conn ? new_conn : connections;
        cif->conn_types = new_types ? new_types : conn_types;
    } else {
        cif->connections = connections;
        cif->conn_types = conn_types;
    }

    cif->n_connections = n_valid;

    LOG_DEBUG("Parsed %d connections using binary search (%d rows, %d resolved)",
              n_valid, n_rows, n_valid);

    return CIF_OK;
}
