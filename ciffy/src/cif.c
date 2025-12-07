/**
 * @file cif.c
 * @brief mmCIF-specific parsing logic.
 *
 * Implements parsing functions for extracting molecular structure
 * data from mmCIF blocks.
 */

#include "cif.h"

/* Include hash tables here to avoid duplicate symbols */
#include "hash/atom.c"
#include "hash/residue.c"
#include "hash/element.c"


/* Number of coordinate dimensions (x, y, z) */
static const size_t COORDS = 3;

/* mmCIF attribute names for coordinates */
static const char *ATTR_X = "Cartn_x";
static const char *ATTR_Y = "Cartn_y";
static const char *ATTR_Z = "Cartn_z";

/* mmCIF attribute names for structure data */
static const char *ATTR_MODEL         = "pdbx_PDB_model_num";
static const char *ATTR_CHAIN_ID      = "id";
static const char *ATTR_RES_PER_CHAIN = "asym_id";
static const char *ATTR_STRAND_ID     = "pdb_strand_id";
static const char *ATTR_RESIDUE_NAME  = "mon_id";
static const char *ATTR_ELEMENT       = "type_symbol";
static const char *ATTR_ATOM_NAME     = "label_atom_id";
static const char *ATTR_SEQ_ID        = "label_seq_id";
static const char *ATTR_LABEL_ASYM    = "label_asym_id";
static const char *ATTR_COMP_ID       = "label_comp_id";

/* Maximum length for combined token strings */
#define MAX_TOKEN_LENGTH 512


char *_get_id(char *buffer, CifErrorContext *ctx) {

    const char *prefix = "data_";

    if (_neq(buffer, prefix)) {
        CIF_SET_ERROR(ctx, CIF_ERR_PARSE,
            "Invalid mmCIF file: missing 'data_' prefix");
        return NULL;
    }

    buffer += 5;  /* Skip "data_" */

    char *start = buffer;
    while (*buffer != '\n' && *buffer != '\0') { buffer++; }

    size_t length = (size_t)(buffer - start);
    return _strdup_n(start, length, ctx);
}


/**
 * @brief Parse an integer array from a block attribute.
 */
static int *_parse_int(mmBlock *block, const char *attr, CifErrorContext *ctx) {

    int index = _get_attr_index(block, attr);
    if (index == BAD_IX) {
        CIF_SET_ERROR(ctx, CIF_ERR_ATTR,
            "Missing attribute '%s' in block '%s'", attr, block->category);
        return NULL;
    }

    int *array = calloc((size_t)block->size, sizeof(int));
    if (array == NULL) {
        CIF_SET_ERROR(ctx, CIF_ERR_ALLOC,
            "Failed to allocate int array of size %d", block->size);
        return NULL;
    }

    for (int line = 0; line < block->size; line++) {
        char *token = _get_attr_by_line(block, line, index, ctx);
        if (token == NULL) {
            free(array);
            return NULL;
        }
        array[line] = _str_to_int(token);
        free(token);
    }

    return array;
}


/**
 * @brief Parse a string array from a block attribute.
 */
static char **_parse_str(mmBlock *block, const char *attr, CifErrorContext *ctx) {

    int index = _get_attr_index(block, attr);
    if (index == BAD_IX) {
        CIF_SET_ERROR(ctx, CIF_ERR_ATTR,
            "Missing attribute '%s' in block '%s'", attr, block->category);
        return NULL;
    }

    char **array = calloc((size_t)block->size, sizeof(char *));
    if (array == NULL) {
        CIF_SET_ERROR(ctx, CIF_ERR_ALLOC,
            "Failed to allocate string array of size %d", block->size);
        return NULL;
    }

    for (int line = 0; line < block->size; line++) {
        array[line] = _get_attr_by_line(block, line, index, ctx);
        if (array[line] == NULL) {
            /* Clean up already allocated strings */
            for (int i = 0; i < line; i++) { free(array[i]); }
            free(array);
            return NULL;
        }
    }

    return array;
}


/**
 * @brief Extract unique values from a block attribute.
 *
 * Returns an array of unique string values in order of first appearance.
 * If *size > 0 on entry, uses that as the expected array size.
 * Otherwise, determines size dynamically and sets *size on output.
 */
static char **_get_unique(mmBlock *block, const char *attr, int *size, CifErrorContext *ctx) {

    int index = _get_attr_index(block, attr);
    if (index == BAD_IX) {
        CIF_SET_ERROR(ctx, CIF_ERR_ATTR,
            "Missing attribute '%s' in block '%s'", attr, block->category);
        return NULL;
    }

    size_t alloc_size = (size_t)(*size > 0 ? *size : block->size);
    char **str = calloc(alloc_size, sizeof(char *));
    if (str == NULL) {
        CIF_SET_ERROR(ctx, CIF_ERR_ALLOC,
            "Failed to allocate unique array of size %zu", alloc_size);
        return NULL;
    }

    char *prev = NULL;
    int ix = 0;

    for (int line = 0; line < block->size; line++) {

        char *token = _get_attr_by_line(block, line, index, ctx);
        if (token == NULL) {
            for (int i = 0; i <= ix; i++) { free(str[i]); }
            free(str);
            return NULL;
        }

        if (prev == NULL) {
            prev = token;
            str[ix] = token;
        } else if (_neq(prev, token)) {
            prev = token;
            ix++;
            if ((size_t)ix >= alloc_size) {
                /* Shouldn't happen if size is correct, but be safe */
                free(token);
                break;
            }
            str[ix] = token;
        } else {
            free(token);  /* Duplicate, free it */
        }
    }

    if (*size > 0) {
        return str;
    } else {
        *size = ix + 1;
        char **resized = realloc(str, (size_t)(*size) * sizeof(char *));
        return resized != NULL ? resized : str;
    }
}


/**
 * @brief Parse coordinate data (x, y, z) from atom block.
 */
static float *_parse_coords(mmBlock *block, CifErrorContext *ctx) {

    int x_index = _get_attr_index(block, ATTR_X);
    if (x_index == BAD_IX) {
        CIF_SET_ERROR(ctx, CIF_ERR_ATTR, "Missing coordinate attribute '%s'", ATTR_X);
        return NULL;
    }

    int y_index = _get_attr_index(block, ATTR_Y);
    if (y_index == BAD_IX) {
        CIF_SET_ERROR(ctx, CIF_ERR_ATTR, "Missing coordinate attribute '%s'", ATTR_Y);
        return NULL;
    }

    int z_index = _get_attr_index(block, ATTR_Z);
    if (z_index == BAD_IX) {
        CIF_SET_ERROR(ctx, CIF_ERR_ATTR, "Missing coordinate attribute '%s'", ATTR_Z);
        return NULL;
    }

    int indices[3] = { x_index, y_index, z_index };

    float *array = calloc(COORDS * (size_t)block->size, sizeof(float));
    if (array == NULL) {
        CIF_SET_ERROR(ctx, CIF_ERR_ALLOC,
            "Failed to allocate coordinate array for %d atoms", block->size);
        return NULL;
    }

    for (int line = 0; line < block->size; line++) {
        for (size_t ix = 0; ix < COORDS; ix++) {
            char *token = _get_attr_by_line(block, line, indices[ix], ctx);
            if (token == NULL) {
                free(array);
                return NULL;
            }

            /* Parse float with error checking */
            char *endptr;
            float val = strtof(token, &endptr);
            if (endptr == token) {
                CIF_SET_ERROR(ctx, CIF_ERR_PARSE,
                    "Invalid coordinate value '%s' at line %d", token, line);
                free(token);
                free(array);
                return NULL;
            }

            array[COORDS * line + ix] = val;
            free(token);
        }
    }

    return array;
}


/**
 * @brief Parse attribute values via hash table lookup.
 */
static int *_parse_via_lookup(mmBlock *block, HashTable func, const char *attr, CifErrorContext *ctx) {

    int index = _get_attr_index(block, attr);
    if (index == BAD_IX) {
        CIF_SET_ERROR(ctx, CIF_ERR_ATTR,
            "Missing attribute '%s' in block '%s'", attr, block->category);
        return NULL;
    }

    int *array = calloc((size_t)block->size, sizeof(int));
    if (array == NULL) {
        CIF_SET_ERROR(ctx, CIF_ERR_ALLOC,
            "Failed to allocate lookup array of size %d", block->size);
        return NULL;
    }

    for (int line = 0; line < block->size; line++) {
        char *token = _get_attr_by_line(block, line, index, ctx);
        if (token == NULL) {
            free(array);
            return NULL;
        }
        array[line] = _lookup(func, token);
        free(token);
    }

    return array;
}


/**
 * @brief Parse combined attribute values (e.g., residue_atom) via hash lookup.
 */
static int *_parse_via_lookup_double(
    mmBlock *block,
    HashTable func,
    const char *attr1,
    const char *attr2,
    CifErrorContext *ctx
) {

    int index1 = _get_attr_index(block, attr1);
    if (index1 == BAD_IX) {
        CIF_SET_ERROR(ctx, CIF_ERR_ATTR,
            "Missing attribute '%s' in block '%s'", attr1, block->category);
        return NULL;
    }

    int index2 = _get_attr_index(block, attr2);
    if (index2 == BAD_IX) {
        CIF_SET_ERROR(ctx, CIF_ERR_ATTR,
            "Missing attribute '%s' in block '%s'", attr2, block->category);
        return NULL;
    }

    int *array = calloc((size_t)block->size, sizeof(int));
    if (array == NULL) {
        CIF_SET_ERROR(ctx, CIF_ERR_ALLOC,
            "Failed to allocate lookup array of size %d", block->size);
        return NULL;
    }

    for (int line = 0; line < block->size; line++) {

        char *token1 = _get_attr_by_line(block, line, index1, ctx);
        if (token1 == NULL) {
            free(array);
            return NULL;
        }

        char *token2 = _get_attr_by_line(block, line, index2, ctx);
        if (token2 == NULL) {
            free(token1);
            free(array);
            return NULL;
        }

        /* Combine tokens with bounds checking */
        char result[MAX_TOKEN_LENGTH];
        int written = snprintf(result, sizeof(result), "%s_%s", token1, token2);

        free(token1);
        free(token2);

        if (written < 0 || (size_t)written >= sizeof(result)) {
            CIF_SET_ERROR(ctx, CIF_ERR_OVERFLOW,
                "Combined token too long at line %d (limit %zu)",
                line, sizeof(result) - 1);
            free(array);
            return NULL;
        }

        array[line] = _lookup(func, result);
    }

    return array;
}


/**
 * @brief Count unique values in a block attribute.
 */
static int _unique(mmBlock *block, const char *attr, CifErrorContext *ctx) {

    int index = _get_attr_index(block, attr);
    if (index == BAD_IX) {
        CIF_SET_ERROR(ctx, CIF_ERR_ATTR,
            "Missing attribute '%s' in block '%s'", attr, block->category);
        return -1;
    }

    int count = 0;
    char *prev = NULL;

    for (int line = 0; line < block->size; line++) {

        char *token = _get_attr_by_line(block, line, index, ctx);
        if (token == NULL) {
            if (prev != NULL) free(prev);
            return -1;
        }

        if (prev == NULL || _neq(prev, token)) {
            if (prev != NULL) free(prev);
            prev = token;
            count++;
        } else {
            free(token);
        }
    }

    if (prev != NULL) free(prev);
    return count;
}


/**
 * @brief Parse size counts relative to attribute changes.
 *
 * Counts items grouped by unique attribute values.
 */
static int *_parse_sizes_relative(mmBlock *block, const char *attr, int *size, CifErrorContext *ctx) {

    int index = _get_attr_index(block, attr);
    if (index == BAD_IX) {
        CIF_SET_ERROR(ctx, CIF_ERR_ATTR,
            "Missing attribute '%s' in block '%s'", attr, block->category);
        return NULL;
    }

    size_t alloc_size = (size_t)(*size > 0 ? *size : block->size);
    int *sizes = calloc(alloc_size, sizeof(int));
    if (sizes == NULL) {
        CIF_SET_ERROR(ctx, CIF_ERR_ALLOC,
            "Failed to allocate sizes array of size %zu", alloc_size);
        return NULL;
    }

    char *prev = NULL;
    int ix = 0;

    for (int line = 0; line < block->size; line++) {

        char *token = _get_attr_by_line(block, line, index, ctx);
        if (token == NULL) {
            if (prev != NULL) free(prev);
            free(sizes);
            return NULL;
        }

        if (prev == NULL) {
            prev = token;
        } else if (_neq(prev, token)) {
            free(prev);
            prev = token;
            ix++;
        } else {
            free(token);
        }

        if ((size_t)ix < alloc_size) {
            sizes[ix]++;
        }
    }

    if (prev != NULL) free(prev);

    if (*size > 0) {
        return sizes;
    } else {
        *size = ix + 1;
        int *resized = realloc(sizes, (size_t)(*size) * sizeof(int));
        return resized != NULL ? resized : sizes;
    }
}


/**
 * @brief Parse per-residue atom counts with chain awareness.
 *
 * @param block The atom_site block
 * @param attr Sequence ID attribute name
 * @param size Total number of residues
 * @param nonpoly Output count of non-polymeric atoms
 * @param is_nonpoly Output array marking non-polymer atoms (1) vs polymer (0)
 * @param lengths Array of residue counts per chain
 * @param ctx Error context
 * @return Array of atom counts per residue, or NULL on error
 */
static int *_parse_residue_sizes(
    mmBlock *block,
    const char *attr,
    int size,
    int *nonpoly,
    int *is_nonpoly,
    int *lengths,
    CifErrorContext *ctx
) {

    int index = _get_attr_index(block, attr);
    if (index == BAD_IX) {
        CIF_SET_ERROR(ctx, CIF_ERR_ATTR,
            "Missing attribute '%s' in block '%s'", attr, block->category);
        return NULL;
    }

    int cindex = _get_attr_index(block, ATTR_LABEL_ASYM);
    if (cindex == BAD_IX) {
        CIF_SET_ERROR(ctx, CIF_ERR_ATTR,
            "Missing attribute '%s' in block '%s'", ATTR_LABEL_ASYM, block->category);
        return NULL;
    }

    int *sizes = calloc((size_t)size, sizeof(int));
    if (sizes == NULL) {
        CIF_SET_ERROR(ctx, CIF_ERR_ALLOC,
            "Failed to allocate residue sizes array of size %d", size);
        return NULL;
    }

    int offset = 0;
    char *pchain = NULL;
    int *len_ptr = lengths;

    for (int line = 0; line < block->size; line++) {

        char *ctoken = _get_attr_by_line(block, line, cindex, ctx);
        if (ctoken == NULL) {
            if (pchain != NULL) free(pchain);
            free(sizes);
            return NULL;
        }

        if (pchain == NULL) {
            pchain = ctoken;
        } else if (_neq(pchain, ctoken)) {
            free(pchain);
            pchain = ctoken;
            offset += *len_ptr;
            len_ptr++;
        } else {
            free(ctoken);
        }

        char *token = _get_attr_by_line(block, line, index, ctx);
        if (token == NULL) {
            if (pchain != NULL) free(pchain);
            free(sizes);
            return NULL;
        }

        int num = _str_to_int(token) - 1;
        free(token);

        if (num < 0) {
            (*nonpoly)++;
            is_nonpoly[line] = 1;
            continue;
        }

        is_nonpoly[line] = 0;
        int idx = offset + num;
        if (idx >= 0 && idx < size) {
            sizes[idx]++;
        }
    }

    if (pchain != NULL) free(pchain);
    return sizes;
}


/**
 * @brief Reorder arrays so polymer atoms come first, non-polymer last.
 *
 * Given a boolean mask indicating non-polymer atoms, reorder all per-atom
 * arrays (coordinates, types, elements) so that:
 * - Polymer atoms occupy indices [0, polymer_count)
 * - Non-polymer atoms occupy indices [polymer_count, atom_count)
 *
 * @param cif The mmCIF struct with arrays to reorder
 * @param is_nonpoly Boolean mask (1 = non-polymer, 0 = polymer) [atoms]
 * @param ctx Error context
 * @return CIF_OK on success, error code on failure
 */
static CifError _reorder_atoms(mmCIF *cif, int *is_nonpoly, CifErrorContext *ctx) {

    int atoms = cif->atoms;
    int polymer_count = atoms - cif->nonpoly;

    /* If no reordering needed, just set polymer count */
    if (cif->nonpoly == 0) {
        cif->polymer = polymer_count;
        return CIF_OK;
    }

    /* Allocate temporary arrays for reordered data */
    float *new_coords = calloc(COORDS * (size_t)atoms, sizeof(float));
    int *new_types = calloc((size_t)atoms, sizeof(int));
    int *new_elements = calloc((size_t)atoms, sizeof(int));

    if (new_coords == NULL || new_types == NULL || new_elements == NULL) {
        free(new_coords);
        free(new_types);
        free(new_elements);
        CIF_SET_ERROR(ctx, CIF_ERR_ALLOC, "Failed to allocate reorder buffers");
        return CIF_ERR_ALLOC;
    }

    /* Two-pass copy: polymer atoms first [0, polymer), then non-polymer [polymer, atoms) */
    int poly_ix = 0;
    int nonpoly_ix = polymer_count;

    for (int i = 0; i < atoms; i++) {
        int dest = is_nonpoly[i] ? nonpoly_ix++ : poly_ix++;

        /* Copy coordinates (3 floats per atom) */
        memcpy(&new_coords[COORDS * dest], &cif->coordinates[COORDS * i], COORDS * sizeof(float));

        /* Copy types and elements */
        new_types[dest] = cif->types[i];
        new_elements[dest] = cif->elements[i];
    }

    /* Replace old arrays with reordered ones */
    free(cif->coordinates);
    free(cif->types);
    free(cif->elements);

    cif->coordinates = new_coords;
    cif->types = new_types;
    cif->elements = new_elements;
    cif->polymer = polymer_count;

    return CIF_OK;
}


CifError _fill_cif(mmCIF *cif, mmBlockList *blocks, CifErrorContext *ctx) {

    /* Validate required blocks */
    if (blocks->atom.category == NULL) {
        CIF_SET_ERROR(ctx, CIF_ERR_BLOCK, "Missing required _atom_site block");
        return CIF_ERR_BLOCK;
    }
    if (blocks->poly.category == NULL) {
        CIF_SET_ERROR(ctx, CIF_ERR_BLOCK, "Missing required _pdbx_poly_seq_scheme block");
        return CIF_ERR_BLOCK;
    }
    if (blocks->chain.category == NULL) {
        CIF_SET_ERROR(ctx, CIF_ERR_BLOCK, "Missing required _struct_asym block");
        return CIF_ERR_BLOCK;
    }

    /* Count the number of models, chains, residues, atoms */
    int model_count = _unique(&blocks->atom, ATTR_MODEL, ctx);
    if (model_count < 0) return ctx->code;
    cif->models = model_count;

    cif->chains = blocks->chain.size;
    cif->residues = blocks->poly.size;

    /* Adjust atom count for multi-model structures */
    blocks->atom.size /= cif->models;
    cif->atoms = blocks->atom.size;

    /* Count residues per chain */
    cif->res_per_chain = _parse_sizes_relative(&blocks->poly, ATTR_RES_PER_CHAIN, &cif->chains, ctx);
    if (cif->res_per_chain == NULL) return ctx->code;

    /* Get chain names */
    cif->names = _get_unique(&blocks->chain, ATTR_CHAIN_ID, &cif->chains, ctx);
    if (cif->names == NULL) return ctx->code;

    /* Get residue sequence */
    cif->sequence = _parse_via_lookup(&blocks->poly, _lookup_residue, ATTR_RESIDUE_NAME, ctx);
    if (cif->sequence == NULL) return ctx->code;

    /* Get strand IDs */
    cif->strands = _get_unique(&blocks->poly, ATTR_STRAND_ID, &cif->chains, ctx);
    if (cif->strands == NULL) return ctx->code;

    /* Parse atom types (residue_name + atom_name) */
    cif->types = _parse_via_lookup_double(&blocks->atom, _lookup_atom, ATTR_COMP_ID, ATTR_ATOM_NAME, ctx);
    if (cif->types == NULL) return ctx->code;

    /* Parse element types */
    cif->elements = _parse_via_lookup(&blocks->atom, _lookup_element, ATTR_ELEMENT, ctx);
    if (cif->elements == NULL) return ctx->code;

    /* Parse coordinates */
    cif->coordinates = _parse_coords(&blocks->atom, ctx);
    if (cif->coordinates == NULL) return ctx->code;

    /* Allocate temporary non-polymer mask array (local, freed after reordering) */
    int *is_nonpoly = calloc((size_t)cif->atoms, sizeof(int));
    if (is_nonpoly == NULL) {
        CIF_SET_ERROR(ctx, CIF_ERR_ALLOC,
            "Failed to allocate is_nonpoly array of size %d", cif->atoms);
        return CIF_ERR_ALLOC;
    }

    /* Compute atoms per residue and populate is_nonpoly mask */
    cif->nonpoly = 0;
    cif->atoms_per_res = _parse_residue_sizes(&blocks->atom, ATTR_SEQ_ID, cif->residues, &cif->nonpoly, is_nonpoly, cif->res_per_chain, ctx);
    if (cif->atoms_per_res == NULL) {
        free(is_nonpoly);
        return ctx->code;
    }

    /* Reorder atoms: polymer first [0, polymer), non-polymer last [polymer, atoms) */
    CifError reorder_err = _reorder_atoms(cif, is_nonpoly, ctx);
    free(is_nonpoly);  /* No longer needed after reordering */
    if (reorder_err != CIF_OK) return reorder_err;

    /* Compute atoms per chain */
    cif->atoms_per_chain = _parse_sizes_relative(&blocks->atom, ATTR_LABEL_ASYM, &cif->chains, ctx);
    if (cif->atoms_per_chain == NULL) return ctx->code;

    return CIF_OK;
}
