/**
 * @file connections.c
 * @brief Parse _struct_conn block for hydrogen bonds, covalent bonds, etc.
 */

#include "connections.h"
#include "io.h"
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
 * ATOM LOOKUP HASH
 * ============================================================================ */

/**
 * @brief Extract field pointer and length using direct pointer arithmetic.
 *
 * Faster than _get_field_ptr for fixed-width blocks in tight loops.
 */
static inline void _extract_field(char *line_start, const int *offsets, int idx,
                                   char **out_ptr, size_t *out_len) {
    char *p = line_start + offsets[idx];
    while (*p == ' ') p++;  /* Skip leading whitespace */

    char *end = p;
    while (*end != ' ' && *end != '\n' && *end != '\0') end++;

    *out_ptr = p;
    *out_len = (size_t)(end - p);
}

AtomHash _build_atom_lookup(mmBlock *block, int n_atoms, CifErrorContext *ctx) {
    AtomHash hash = atom_hash_create((size_t)n_atoms);
    if (!hash.entries) {
        CIF_SET_ERROR(ctx, CIF_ERR_ALLOC, "Failed to allocate atom lookup hash");
        return hash;
    }

    /* Ensure lines are precomputed for field access */
    if (block->lines == NULL) {
        CifError err = _precompute_lines(block, ctx);
        if (err != CIF_OK) {
            atom_hash_free(&hash);
            return (AtomHash){NULL, 0, 0};
        }
    }

    /* Get attribute indices */
    int asym_idx = _get_attr_index(block, "label_asym_id", ctx);
    int seq_idx = _get_attr_index(block, "label_seq_id", ctx);
    int atom_idx = _get_attr_index(block, "label_atom_id", ctx);

    if (asym_idx < 0 || seq_idx < 0 || atom_idx < 0) {
        CIF_SET_ERROR(ctx, CIF_ERR_ATTR,
                      "Missing required attributes for atom lookup (label_asym_id, label_seq_id, label_atom_id)");
        atom_hash_free(&hash);
        return (AtomHash){NULL, 0, 0};
    }

    /* Use fast path for fixed-width blocks (most common case) */
    if (!block->variable_width && block->offsets != NULL) {
        char **lines = block->lines;
        const int *offsets = block->offsets;

        for (int row = 0; row < n_atoms; row++) {
            char *line_start = lines[row];
            char *asym, *seq, *atom;
            size_t asym_len, seq_len, atom_len;

            /* Direct pointer arithmetic - matches BATCH_* macros */
            _extract_field(line_start, offsets, asym_idx, &asym, &asym_len);
            _extract_field(line_start, offsets, seq_idx, &seq, &seq_len);
            _extract_field(line_start, offsets, atom_idx, &atom, &atom_len);

            /* Strip quotes if present */
            _strip_outer_quotes((const char **)&asym, &asym_len);
            _strip_outer_quotes((const char **)&seq, &seq_len);
            _strip_outer_quotes((const char **)&atom, &atom_len);

            /* Insert into hash */
            atom_hash_insert(&hash, asym, asym_len, seq, seq_len, atom, atom_len, row);
        }
    } else {
        /* Fallback for variable-width blocks */
        for (int row = 0; row < n_atoms; row++) {
            size_t asym_len, seq_len, atom_len;

            char *asym = _get_field_ptr(block, row, asym_idx, &asym_len);
            char *seq = _get_field_ptr(block, row, seq_idx, &seq_len);
            char *atom = _get_field_ptr(block, row, atom_idx, &atom_len);

            if (!asym || !seq || !atom) continue;

            /* Strip quotes if present */
            _strip_outer_quotes((const char **)&asym, &asym_len);
            _strip_outer_quotes((const char **)&seq, &seq_len);
            _strip_outer_quotes((const char **)&atom, &atom_len);

            /* Insert into hash */
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
