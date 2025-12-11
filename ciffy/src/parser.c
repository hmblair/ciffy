/**
 * @file parser.c
 * @brief mmCIF parsing pipeline.
 *
 * Parses mmCIF files to extract molecular structure data including
 * coordinates, atom types, residue sequences, and chain organization.
 *
 * Pipeline:
 *   1. Block Validation  - Verify required mmCIF blocks exist
 *   2. Metadata Parsing  - Extract chain/residue counts and names
 *   3. Atom Parsing      - Single-pass coordinate/type extraction
 *   4. Atom Reordering   - Separate polymer/non-polymer atoms
 */

#include "parser.h"
#include "registry.h"
#include "log.h"

#include <math.h>    /* for isnan */
#include <unistd.h>  /* for isatty */

/* Hash tables for type lookups */
#include "hash/atom.c"
#include "hash/residue.c"
#include "hash/element.c"
#include "hash/molecule.c"


/* ============================================================================
 * CONSTANTS
 * mmCIF attribute names used throughout parsing.
 * ============================================================================ */

static const size_t COORDS = 3;

/* Coordinate attributes */
static const char *ATTR_X = "Cartn_x";
static const char *ATTR_Y = "Cartn_y";
static const char *ATTR_Z = "Cartn_z";

/* Atom-level attributes (used by batch parser) */
static const char *ATTR_ELEMENT       = "type_symbol";
static const char *ATTR_ATOM_NAME     = "label_atom_id";
static const char *ATTR_SEQ_ID        = "label_seq_id";
static const char *ATTR_LABEL_ASYM    = "label_asym_id";
static const char *ATTR_COMP_ID       = "label_comp_id";
static const char *ATTR_GROUP_PDB     = "group_PDB";

/* Note: Metadata attributes (MODEL, CHAIN_ID, RES_PER_CHAIN, STRAND_ID, RESIDUE_NAME)
 * are now defined in registry.c and used via _execute_plan() */


/* ============================================================================
 * TYPES
 * Internal structures for parsing state.
 * ============================================================================ */

/**
 * Pre-computed attribute indices for single-pass atom parsing.
 *
 * Storing indices avoids repeated string lookups in the hot loop.
 * Extend this struct to add new per-atom features (b-factor, occupancy, etc).
 */
typedef struct {
    int x, y, z;              /* Coordinate column indices */
    int element;              /* Element symbol (C, N, O, etc.) */
    int comp_id;              /* Residue/component name */
    int atom_name;            /* Atom name within residue */
    int seq_id;               /* Sequence ID for polymer detection */
    int label_asym;           /* Chain label */
} AtomIndices;


/* ============================================================================
 * FILE HEADER
 * Extract PDB identifier from mmCIF header.
 * ============================================================================ */

/**
 * Parse the PDB identifier from "data_XXXX" header line.
 */
char *_get_id(char *buffer, CifErrorContext *ctx) {
    const char *prefix = "data_";

    if (_neq(buffer, prefix)) {
        CIF_SET_ERROR(ctx, CIF_ERR_PARSE,
            "Invalid mmCIF file: missing 'data_' prefix");
        return NULL;
    }

    buffer += 5;
    char *start = buffer;
    while (*buffer != '\n' && *buffer != '\0') buffer++;

    return _strdup_n(start, (size_t)(buffer - start), ctx);
}


/* ============================================================================
 * ATTRIBUTE UTILITIES
 * Generic helpers for extracting values from mmCIF blocks.
 * ============================================================================ */

/**
 * Extract unique string values from an attribute (first occurrence only).
 *
 * Used for chain names, strand IDs, etc. where values repeat across rows
 * but we only want distinct values in order of appearance.
 *
 * Uses pointer-based comparison to avoid allocating for duplicate values.
 */
char **_get_unique(mmBlock *block, const char *attr, int *size,
                   CifErrorContext *ctx) {
    LOG_DEBUG("Extracting unique '%s' from block '%s' (size=%d)",
              attr, block->category ? block->category : "unknown", block->size);

    int index = _get_attr_index(block, attr, ctx);
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

    /* Track previous value as pointer+length to avoid allocation for comparison */
    char *prev_ptr = NULL;
    size_t prev_len = 0;
    int ix = 0;

    for (int line = 0; line < block->size; line++) {
        size_t cur_len;
        char *cur_ptr = _get_field_ptr(block, line, index, &cur_len);
        if (cur_ptr == NULL) {
            CIF_SET_ERROR(ctx, CIF_ERR_PARSE,
                "Failed to get field in block '%s' at line %d/%d, attr %d/%d (lines=%s)",
                block->category ? block->category : "unknown",
                line, block->size, index, block->attributes,
                block->lines ? "ok" : "NULL");
            for (int i = 0; i < ix; i++) free(str[i]);
            free(str);
            return NULL;
        }

        /* Check if this is a new unique value */
        bool is_new = (prev_ptr == NULL) ||
                      !_field_eq_field(prev_ptr, prev_len, cur_ptr, cur_len);

        if (is_new) {
            if (ix > 0 && (size_t)ix >= alloc_size) {
                LOG_WARNING("Unique value count %d exceeds allocation %zu, truncating",
                            ix + 1, alloc_size);
                break;
            }
            /* Only allocate when we find a new unique value */
            str[ix] = _strdup_n(cur_ptr, cur_len, ctx);
            if (str[ix] == NULL) {
                for (int i = 0; i < ix; i++) free(str[i]);
                free(str);
                return NULL;
            }
            prev_ptr = cur_ptr;
            prev_len = cur_len;
            ix++;
        }
        /* No allocation needed for duplicates */
    }

    if (*size <= 0) {
        int new_size = ix;
        char **resized = realloc(str, (size_t)new_size * sizeof(char *));
        if (resized != NULL) {
            str = resized;
        } else {
            LOG_WARNING("realloc shrink failed for unique array, using oversized buffer");
        }
        *size = new_size;
    }

    LOG_DEBUG("Found %d unique values for '%s'", ix, attr);
    return str;
}

/**
 * Count unique consecutive values in an attribute.
 *
 * Uses pointer-based comparison - no allocations needed.
 */
int _count_unique(mmBlock *block, const char *attr, CifErrorContext *ctx) {
    int index = _get_attr_index(block, attr, ctx);
    if (index == BAD_IX) {
        CIF_SET_ERROR(ctx, CIF_ERR_ATTR,
            "Missing attribute '%s' in block '%s'", attr, block->category);
        return -1;
    }

    int count = 0;
    char *prev_ptr = NULL;
    size_t prev_len = 0;

    for (int line = 0; line < block->size; line++) {
        size_t cur_len;
        char *cur_ptr = _get_field_ptr(block, line, index, &cur_len);
        if (cur_ptr == NULL) {
            CIF_SET_ERROR(ctx, CIF_ERR_PARSE,
                "Failed to get '%s' in block '%s' at line %d/%d (lines=%s)",
                attr, block->category ? block->category : "unknown",
                line, block->size, block->lines ? "ok" : "NULL");
            return -1;
        }

        if (prev_ptr == NULL || !_field_eq_field(prev_ptr, prev_len, cur_ptr, cur_len)) {
            prev_ptr = cur_ptr;
            prev_len = cur_len;
            count++;
        }
    }

    return count;
}

/**
 * Parse residue types via hash table lookup.
 *
 * Used for sequence parsing where residue names map to type indices.
 * Uses inline lookup to avoid allocations in the loop.
 */
int *_parse_via_lookup(mmBlock *block, HashTable func, const char *attr,
                       CifErrorContext *ctx) {
    int index = _get_attr_index(block, attr, ctx);
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
        int value;
        LookupResult res = _lookup_inline_safe(block, line, index, func, &value);

        if (res == LOOKUP_ERROR) {
            CIF_SET_ERROR(ctx, CIF_ERR_PARSE,
                "Failed to access '%s' in block '%s' at line %d/%d",
                attr, block->category ? block->category : "unknown",
                line, block->size);
            free(array);
            return NULL;
        }
        /* LOOKUP_NOT_FOUND is OK - store -1 for unknown values */
        array[line] = (res == LOOKUP_OK) ? value : -1;
    }

    return array;
}


/* ============================================================================
 * SIZE COUNTING
 * Count items per group (residues per chain, atoms per residue, etc).
 * ============================================================================ */

/**
 * Count items grouped by attribute value changes.
 *
 * Returns array where sizes[i] = count of rows with i-th unique value.
 * Used for residues-per-chain and atoms-per-chain counting.
 *
 * Uses pointer-based comparison - no allocations in the loop.
 */
int *_count_sizes_by_group(mmBlock *block, const char *attr, int *size,
                           CifErrorContext *ctx) {
    int index = _get_attr_index(block, attr, ctx);
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

    char *prev_ptr = NULL;
    size_t prev_len = 0;
    int ix = 0;

    for (int line = 0; line < block->size; line++) {
        size_t cur_len;
        char *cur_ptr = _get_field_ptr(block, line, index, &cur_len);
        if (cur_ptr == NULL) {
            CIF_SET_ERROR(ctx, CIF_ERR_PARSE,
                "Failed to get '%s' in block '%s' at line %d/%d (lines=%s)",
                attr, block->category ? block->category : "unknown",
                line, block->size, block->lines ? "ok" : "NULL");
            free(sizes);
            return NULL;
        }

        if (prev_ptr == NULL) {
            prev_ptr = cur_ptr;
            prev_len = cur_len;
        } else if (!_field_eq_field(prev_ptr, prev_len, cur_ptr, cur_len)) {
            prev_ptr = cur_ptr;
            prev_len = cur_len;
            ix++;
        }

        if ((size_t)ix < alloc_size) {
            sizes[ix]++;
        } else {
            LOG_WARNING("Size index %d exceeds allocation %zu", ix, alloc_size);
        }
    }

    if (*size <= 0) {
        int new_size = ix + 1;
        int *resized = realloc(sizes, (size_t)new_size * sizeof(int));
        if (resized != NULL) {
            sizes = resized;
        } else {
            LOG_WARNING("realloc shrink failed for sizes array, using oversized buffer");
        }
        *size = new_size;
    }
    return sizes;
}

/**
 * Count atoms per residue with chain-aware indexing.
 *
 * Handles non-polymer atoms (HETATM) by marking them in is_nonpoly mask.
 * Returns NULL-terminated sizes array indexed by global residue index.
 *
 * Uses pointer-based comparison - minimal allocations.
 */
static int *_count_atoms_per_residue(mmBlock *block, int residue_count,
                                     int *nonpoly_count, int *is_nonpoly,
                                     int *res_per_chain, CifErrorContext *ctx) {
    int seq_index = _get_attr_index(block, ATTR_SEQ_ID, ctx);
    if (seq_index == BAD_IX) {
        CIF_SET_ERROR(ctx, CIF_ERR_ATTR, "Missing attribute '%s'", ATTR_SEQ_ID);
        return NULL;
    }

    int chain_index = _get_attr_index(block, ATTR_LABEL_ASYM, ctx);
    if (chain_index == BAD_IX) {
        CIF_SET_ERROR(ctx, CIF_ERR_ATTR, "Missing attribute '%s'", ATTR_LABEL_ASYM);
        return NULL;
    }

    int group_index = _get_attr_index(block, ATTR_GROUP_PDB, ctx);
    if (group_index == BAD_IX) {
        CIF_SET_ERROR(ctx, CIF_ERR_ATTR, "Missing attribute '%s'", ATTR_GROUP_PDB);
        return NULL;
    }

    int *sizes = calloc((size_t)residue_count, sizeof(int));
    if (sizes == NULL) {
        CIF_SET_ERROR(ctx, CIF_ERR_ALLOC,
            "Failed to allocate residue sizes array of size %d", residue_count);
        return NULL;
    }

    int chain_offset = 0;
    char *prev_chain_ptr = NULL;
    size_t prev_chain_len = 0;
    int *chain_len_ptr = res_per_chain;

    for (int line = 0; line < block->size; line++) {
        /* Check if atom is HETATM (non-polymer) using pointer comparison */
        size_t group_len;
        char *group_ptr = _get_field_ptr(block, line, group_index, &group_len);
        if (group_ptr == NULL) {
            CIF_SET_ERROR(ctx, CIF_ERR_PARSE,
                "Failed to get group_PDB at line %d/%d in atom block (lines=%s)",
                line, block->size, block->lines ? "ok" : "NULL");
            free(sizes);
            return NULL;
        }

        bool is_hetatm = _field_eq(group_ptr, group_len, "HETATM");

        if (is_hetatm) {
            (*nonpoly_count)++;
            is_nonpoly[line] = 1;
            continue;
        }

        is_nonpoly[line] = 0;

        /* Track chain changes to compute residue offset */
        size_t chain_len;
        char *chain_ptr = _get_field_ptr(block, line, chain_index, &chain_len);
        if (chain_ptr == NULL) {
            CIF_SET_ERROR(ctx, CIF_ERR_PARSE,
                "Failed to get label_asym_id at line %d/%d in atom block (lines=%s)",
                line, block->size, block->lines ? "ok" : "NULL");
            free(sizes);
            return NULL;
        }

        if (prev_chain_ptr == NULL) {
            prev_chain_ptr = chain_ptr;
            prev_chain_len = chain_len;
        } else if (!_field_eq_field(prev_chain_ptr, prev_chain_len, chain_ptr, chain_len)) {
            prev_chain_ptr = chain_ptr;
            prev_chain_len = chain_len;
            chain_offset += *chain_len_ptr++;
        }

        /* Parse sequence ID inline without allocation */
        int seq_id = _parse_int_inline(block, line, seq_index) - 1;

        /* Skip atoms with invalid seq_id (shouldn't happen for ATOM records) */
        if (seq_id < 0) {
            LOG_WARNING("ATOM record with invalid seq_id at line %d", line);
            continue;
        }

        int residue_idx = chain_offset + seq_id;
        if (residue_idx >= 0 && residue_idx < residue_count) {
            sizes[residue_idx]++;
        }
    }

    return sizes;
}


/* ============================================================================
 * BATCH ATOM PARSING
 * Single-pass parallel parsing of all per-atom data.
 * Maximizes cache efficiency by processing each line once.
 * ============================================================================ */

/**
 * Initialize attribute indices for batch parsing.
 *
 * Pre-computes column indices to avoid repeated string lookups in hot loop.
 */
static CifError _init_atom_indices(mmBlock *block, AtomIndices *idx,
                                   CifErrorContext *ctx) {
    idx->x = _get_attr_index(block, ATTR_X, ctx);
    idx->y = _get_attr_index(block, ATTR_Y, ctx);
    idx->z = _get_attr_index(block, ATTR_Z, ctx);
    idx->element = _get_attr_index(block, ATTR_ELEMENT, ctx);
    idx->comp_id = _get_attr_index(block, ATTR_COMP_ID, ctx);
    idx->atom_name = _get_attr_index(block, ATTR_ATOM_NAME, ctx);
    idx->seq_id = _get_attr_index(block, ATTR_SEQ_ID, ctx);
    idx->label_asym = _get_attr_index(block, ATTR_LABEL_ASYM, ctx);

    /* Validate required indices */
    if (idx->x == BAD_IX || idx->y == BAD_IX || idx->z == BAD_IX) {
        CIF_SET_ERROR(ctx, CIF_ERR_ATTR, "Missing coordinate attributes");
        return CIF_ERR_ATTR;
    }
    if (idx->element == BAD_IX || idx->comp_id == BAD_IX || idx->atom_name == BAD_IX) {
        CIF_SET_ERROR(ctx, CIF_ERR_ATTR, "Missing atom type attributes");
        return CIF_ERR_ATTR;
    }
    if (idx->seq_id == BAD_IX || idx->label_asym == BAD_IX) {
        CIF_SET_ERROR(ctx, CIF_ERR_ATTR, "Missing chain/residue attributes");
        return CIF_ERR_ATTR;
    }

    return CIF_OK;
}

/**
 * Parse all per-atom data in a single parallel pass.
 *
 * For each atom line, parses:
 *   - Coordinates (x, y, z)
 *   - Element type via hash lookup
 *   - Atom type via combined hash lookup (residue_atom)
 *
 * Uses pre-computed line pointers for O(1) row access.
 * Unknown types return -1 (valid for non-standard residues/ligands).
 */
static CifError _parse_atoms_batch(mmBlock *block, AtomIndices *idx,
                                   float *coords, int *elements, int *types) {
    char combine_buf[MAX_INLINE_BUFFER];

    for (int line = 0; line < block->size; line++) {
        /* Parse coordinates */
        coords[COORDS * line + 0] = _parse_float_inline(block, line, idx->x);
        coords[COORDS * line + 1] = _parse_float_inline(block, line, idx->y);
        coords[COORDS * line + 2] = _parse_float_inline(block, line, idx->z);

        /* Parse types (-1 for unknown) */
        elements[line] = _lookup_inline(block, line, idx->element, _lookup_element);
        types[line] = _lookup_double_inline(block, line, idx->comp_id,
                                            idx->atom_name, _lookup_atom, combine_buf);
    }

    return CIF_OK;
}


/* ============================================================================
 * ATOM REORDERING
 * Separate polymer atoms from non-polymer (ligands, water, ions).
 * Enables simple slicing: polymer = atoms[0:polymer_count]
 * ============================================================================ */

/**
 * Reorder arrays so polymer atoms come first.
 *
 * After reordering:
 *   - atoms[0, polymer_count) = polymer atoms
 *   - atoms[polymer_count, total) = non-polymer atoms
 */
static CifError _reorder_atoms(mmCIF *cif, int *is_nonpoly, CifErrorContext *ctx) {
    int atoms = cif->atoms;
    int polymer_count = atoms - cif->nonpoly;

    LOG_DEBUG("Reordering: %d total atoms, %d polymer, %d nonpoly",
              atoms, polymer_count, cif->nonpoly);

    if (cif->nonpoly == 0) {
        cif->polymer = polymer_count;
        return CIF_OK;
    }

    /* Allocate reorder buffers */
    float *new_coords = calloc(COORDS * (size_t)atoms, sizeof(float));
    int *new_types = calloc((size_t)atoms, sizeof(int));
    int *new_elements = calloc((size_t)atoms, sizeof(int));

    if (!new_coords || !new_types || !new_elements) {
        free(new_coords);
        free(new_types);
        free(new_elements);
        CIF_SET_ERROR(ctx, CIF_ERR_ALLOC, "Failed to allocate reorder buffers");
        return CIF_ERR_ALLOC;
    }

    /* Two-pointer copy: polymer to front, non-polymer to back */
    int poly_ix = 0;
    int nonpoly_ix = polymer_count;

    for (int i = 0; i < atoms; i++) {
        int dest = is_nonpoly[i] ? nonpoly_ix++ : poly_ix++;
        memcpy(&new_coords[COORDS * dest], &cif->coordinates[COORDS * i],
               COORDS * sizeof(float));
        new_types[dest] = cif->types[i];
        new_elements[dest] = cif->elements[i];
    }

    /* Swap arrays */
    free(cif->coordinates);
    free(cif->types);
    free(cif->elements);

    cif->coordinates = new_coords;
    cif->types = new_types;
    cif->elements = new_elements;
    cif->polymer = polymer_count;

    return CIF_OK;
}


/* ============================================================================
 * MOLECULE TYPE PARSING
 * Parse chain molecule types from _entity_poly and _struct_asym blocks.
 * ============================================================================ */

/**
 * @brief Parse molecule types for each chain from _entity_poly block.
 *
 * Uses entity_id to link chains (_struct_asym) to their polymer type
 * (_entity_poly.type). Falls back to UNKNOWN if block is missing.
 *
 * @param cif Structure with chains count already set
 * @param blocks Parsed block collection
 * @param ctx Error context
 * @return CIF_OK on success
 */
static CifError _parse_molecule_types(mmCIF *cif, mmBlockList *blocks, CifErrorContext *ctx) {
    /* Allocate molecule_types array */
    cif->molecule_types = calloc((size_t)cif->chains, sizeof(int));
    if (!cif->molecule_types) {
        CIF_SET_ERROR(ctx, CIF_ERR_ALLOC, "Failed to allocate molecule_types");
        return CIF_ERR_ALLOC;
    }

    /* Default to UNKNOWN (12) for all chains */
    for (int i = 0; i < cif->chains; i++) {
        cif->molecule_types[i] = 12;  /* Molecule.UNKNOWN */
    }

    /* Check if _entity_poly block exists */
    if (blocks->entity_poly.category == NULL) {
        LOG_DEBUG("No _entity_poly block - molecule types defaulting to UNKNOWN");
        return CIF_OK;
    }

    /* Precompute lines for entity_poly block */
    CifError err = _precompute_lines(&blocks->entity_poly, ctx);
    if (err != CIF_OK) return err;

    /* Get attribute indices for entity_poly */
    int ep_entity_idx = _get_attr_index(&blocks->entity_poly, "entity_id", ctx);
    int ep_type_idx = _get_attr_index(&blocks->entity_poly, "type", ctx);

    if (ep_entity_idx < 0 || ep_type_idx < 0) {
        LOG_WARNING("_entity_poly missing entity_id or type attribute");
        _free_lines(&blocks->entity_poly);
        return CIF_OK;  /* Not fatal - just use defaults */
    }

    /* Get attribute index for struct_asym.entity_id */
    int sa_entity_idx = _get_attr_index(&blocks->chain, "entity_id", ctx);
    if (sa_entity_idx < 0) {
        LOG_WARNING("_struct_asym missing entity_id attribute");
        _free_lines(&blocks->entity_poly);
        return CIF_OK;
    }

    /* Build entity_id -> molecule_type map from _entity_poly */
    /* We'll use a simple approach: max 100 entities should be enough */
    int entity_map[100];
    for (int i = 0; i < 100; i++) entity_map[i] = 12;  /* UNKNOWN */

    for (int row = 0; row < blocks->entity_poly.size; row++) {
        /* Get entity_id using inline parser */
        int entity_id = _parse_int_inline(&blocks->entity_poly, row, ep_entity_idx);
        if (entity_id < 0 || entity_id >= 100) continue;

        /* Get type string pointer for hash lookup */
        size_t type_len;
        const char *type_ptr = _get_field_ptr(&blocks->entity_poly, row, ep_type_idx, &type_len);
        if (!type_ptr || type_len == 0) continue;

        /* Copy to local buffer, stripping outer quotes, and null-terminate */
        char type_buf[64];
        if (type_len >= sizeof(type_buf)) continue;  /* Skip if too long */

        /* Strip outer quotes if present (e.g., 'polypeptide(L)' -> polypeptide(L)) */
        const char *src = type_ptr;
        size_t src_len = type_len;
        if (src_len >= 2 &&
            ((src[0] == '\'' && src[src_len - 1] == '\'') ||
             (src[0] == '"' && src[src_len - 1] == '"'))) {
            src++;
            src_len -= 2;
        }
        memcpy(type_buf, src, src_len);
        type_buf[src_len] = '\0';

        LOG_DEBUG("Entity %d type field: '%s' (len=%zu)", entity_id, type_buf, src_len);

        /* Look up molecule type via hash table */
        struct _LOOKUP *result = _lookup_molecule(type_buf, src_len);
        int mol_type = result ? result->value : 11;  /* OTHER if not found */

        entity_map[entity_id] = mol_type;
        LOG_DEBUG("Entity %d -> molecule type %d (result=%p)", entity_id, mol_type, (void*)result);
    }

    _free_lines(&blocks->entity_poly);

    /* Map each chain to its molecule type via entity_id */
    for (int chain = 0; chain < cif->chains; chain++) {
        int entity_id = _parse_int_inline(&blocks->chain, chain, sa_entity_idx);
        if (entity_id >= 0 && entity_id < 100) {
            cif->molecule_types[chain] = entity_map[entity_id];
        }
    }

    LOG_DEBUG("Molecule types parsed for %d chains", cif->chains);
    return CIF_OK;
}


/* ============================================================================
 * PUBLIC INTERFACE
 * Main entry point for populating mmCIF structure from parsed blocks.
 * ============================================================================ */

/**
 * Populate mmCIF structure from parsed blocks.
 *
 * Pipeline:
 *   1. Validate required blocks exist
 *   2. Count models, chains, residues, atoms
 *   3. Parse chain/residue metadata
 *   4. Batch parse atom data (parallelized)
 *   5. Reorder atoms (polymer first)
 */
CifError _fill_cif(mmCIF *cif, mmBlockList *blocks, CifErrorContext *ctx) {
    LOG_DEBUG("Starting CIF structure parsing");

    /* ── Block Validation (registry-driven) ────────────────────────────────── */

    CifError val_err = _validate_blocks_registry(blocks, ctx);
    if (val_err != CIF_OK) return val_err;

    LOG_DEBUG("Block validation complete: atom=%d rows, poly=%d rows, chain=%d rows",
              blocks->atom.size, blocks->poly.size, blocks->chain.size);

    /* ── Precompute Line Pointers ────────────────────────────────────────── */
    /* Required for _get_field_ptr used in counting and metadata extraction */

    CifError err = _precompute_lines(&blocks->atom, ctx);
    if (err != CIF_OK) return err;

    err = _precompute_lines(&blocks->poly, ctx);
    if (err != CIF_OK) {
        _free_lines(&blocks->atom);
        return err;
    }

    err = _precompute_lines(&blocks->chain, ctx);
    if (err != CIF_OK) {
        _free_lines(&blocks->atom);
        _free_lines(&blocks->poly);
        return err;
    }

    LOG_DEBUG("Line pointers precomputed for all blocks");

    /* ── Parse Metadata (registry-driven) ──────────────────────────────────── */
    /* Compute field execution order and parse: chains, residues, models, atoms,
     * names, res_per_chain, strands, sequence */

    ParsePlan plan;
    err = _plan_parse(&plan, ctx);
    if (err != CIF_OK) {
        _free_lines(&blocks->atom);
        _free_lines(&blocks->poly);
        _free_lines(&blocks->chain);
        return err;
    }

    err = _execute_plan(cif, blocks, &plan, ctx);
    if (err != CIF_OK) {
        _free_lines(&blocks->atom);
        _free_lines(&blocks->poly);
        _free_lines(&blocks->chain);
        return err;
    }

    LOG_INFO("Parsing structure: %d models, %d chains, %d residues, %d atoms",
             cif->models, cif->chains, cif->residues, cif->atoms);

    LOG_DEBUG("Metadata extracted: %d chains, %d residues in sequence",
              cif->chains, cif->residues);

    /* ── Parse Molecule Types ─────────────────────────────────────────────── */
    /* Extract molecule types from _entity_poly block (if present) */

    err = _parse_molecule_types(cif, blocks, ctx);
    if (err != CIF_OK) {
        _free_lines(&blocks->atom);
        _free_lines(&blocks->poly);
        _free_lines(&blocks->chain);
        return err;
    }

    /* ── Batch Atom Parsing ───────────────────────────────────────────────── */
    /* Note: lines already precomputed at start of function */

    LOG_DEBUG("Beginning batch atom parsing (%d atoms)...", cif->atoms);

    AtomIndices idx;
    err = _init_atom_indices(&blocks->atom, &idx, ctx);
    if (err != CIF_OK) {
        _free_lines(&blocks->atom);
        _free_lines(&blocks->poly);
        _free_lines(&blocks->chain);
        return err;
    }

    cif->coordinates = calloc(COORDS * (size_t)cif->atoms, sizeof(float));
    cif->elements = calloc((size_t)cif->atoms, sizeof(int));
    cif->types = calloc((size_t)cif->atoms, sizeof(int));

    if (!cif->coordinates || !cif->elements || !cif->types) {
        free(cif->coordinates);
        free(cif->elements);
        free(cif->types);
        _free_lines(&blocks->atom);
        _free_lines(&blocks->poly);
        _free_lines(&blocks->chain);
        CIF_SET_ERROR(ctx, CIF_ERR_ALLOC, "Failed to allocate atom arrays");
        return CIF_ERR_ALLOC;
    }

    err = _parse_atoms_batch(&blocks->atom, &idx,
                             cif->coordinates, cif->elements, cif->types);

    /* Free poly and chain line pointers - no longer needed */
    _free_lines(&blocks->poly);
    _free_lines(&blocks->chain);
    if (err != CIF_OK) {
        free(cif->coordinates);
        free(cif->elements);
        free(cif->types);
        return err;
    }

    /* Validate parsed coordinates - check for NaN values */
    int nan_count = 0;
    for (int i = 0; i < cif->atoms; i++) {
        if (isnan(cif->coordinates[COORDS * i + 0]) ||
            isnan(cif->coordinates[COORDS * i + 1]) ||
            isnan(cif->coordinates[COORDS * i + 2])) {
            nan_count++;
        }
    }
    if (nan_count > 0) {
        LOG_WARNING("Found %d atoms with invalid (NaN) coordinates", nan_count);
    }

    /* ── Atom Reordering ──────────────────────────────────────────────────── */

    LOG_DEBUG("Classifying polymer vs non-polymer atoms...");

    int *is_nonpoly = calloc((size_t)cif->atoms, sizeof(int));
    if (is_nonpoly == NULL) {
        CIF_SET_ERROR(ctx, CIF_ERR_ALLOC, "Failed to allocate nonpoly mask");
        return CIF_ERR_ALLOC;
    }

    cif->nonpoly = 0;
    cif->atoms_per_res = _count_atoms_per_residue(&blocks->atom, cif->residues,
                                                  &cif->nonpoly, is_nonpoly,
                                                  cif->res_per_chain, ctx);
    if (cif->atoms_per_res == NULL) {
        free(is_nonpoly);
        return ctx->code;
    }

    LOG_DEBUG("Found %d non-polymer atoms (HETATM), reordering...", cif->nonpoly);

    err = _reorder_atoms(cif, is_nonpoly, ctx);
    free(is_nonpoly);
    if (err != CIF_OK) return err;

    cif->atoms_per_chain = _count_sizes_by_group(&blocks->atom, ATTR_LABEL_ASYM,
                                                 &cif->chains, ctx);
    if (cif->atoms_per_chain == NULL) return ctx->code;

    cif->polymer = cif->atoms - cif->nonpoly;
    LOG_INFO("Parsed %d polymer atoms, %d non-polymer atoms", cif->polymer, cif->nonpoly);
    LOG_DEBUG("CIF structure parsing complete");

    return CIF_OK;
}


/* ============================================================================
 * BLOCK PARSING API
 * Public functions for reading and managing mmCIF blocks.
 * ============================================================================ */

bool _skip_multiline_attr(char **buffer) {
    _advance_line(buffer);
    int lines = 0;
    const int MAX_MULTILINE_LINES = 10000;
    while (**buffer != ';' && **buffer != '\0' && lines < MAX_MULTILINE_LINES) {
        _advance_line(buffer);
        lines++;
    }
    if (lines >= MAX_MULTILINE_LINES) {
        LOG_ERROR("Unterminated multiline attribute (exceeded %d lines)", MAX_MULTILINE_LINES);
        return false;
    }
    if (**buffer == ';') {
        _advance_line(buffer);
    }
    return true;
}


void _next_block(char **buffer) {
    while (**buffer != '\0' && !_is_section_end(*buffer)) {
        _advance_line(buffer);
    }
    if (**buffer != '\0') {
        _advance_line(buffer);
    }
}


mmBlock _read_block(char **buffer, CifErrorContext *ctx) {

    mmBlock block = {0};

    /* Check if this is a single-entry block (no "loop_" prefix) */
    if (_eq(*buffer, "loop_")) {
        _advance_line(buffer);
    } else {
        block.single = true;
        block.size = 1;
    }

    block.head = *buffer;
    block.category = _get_category(block.head, ctx);
    if (block.category == NULL) {
        return block;  /* Error - ctx is already set */
    }

    /* Count attributes by scanning header lines */
    while (**buffer != '\0' && _eq(*buffer, block.category)) {
        block.attributes++;
        _advance_line(buffer);
        if (**buffer == ';') {
            if (!_skip_multiline_attr(buffer)) {
                LOG_ERROR("Unterminated multiline in block %s", block.category);
                CIF_SET_ERROR(ctx, CIF_ERR_PARSE, "Unterminated multiline attribute");
                free(block.category);
                block.category = NULL;
                return block;
            }
        }
    }

    /* Validate attribute count */
    if (block.attributes == 0) {
        LOG_ERROR("Block %s has no attributes", block.category);
        CIF_SET_ERROR(ctx, CIF_ERR_PARSE, "Block has no attributes");
        free(block.category);
        block.category = NULL;
        return block;
    }

    if (!block.single) {
        /* Multi-entry block: calculate offsets and line width */
        block.start = *buffer;
        block.variable_width = false;
        block.offsets = _get_offsets(block.start, block.attributes, ctx);
        if (block.offsets == NULL) {
            free(block.category);
            block.category = NULL;
            return block;  /* Error - ctx is already set */
        }
        block.width = block.offsets[block.attributes] + 1;

        /* Validate width is positive */
        if (block.width <= 0) {
            LOG_ERROR("Invalid block width %d", block.width);
            CIF_SET_ERROR(ctx, CIF_ERR_PARSE, "Invalid block line width");
            free(block.category);
            free(block.offsets);
            block.category = NULL;
            block.offsets = NULL;
            return block;
        }

        /* Count entries until section end (assuming fixed-width) */
        while (**buffer != '\0' && !_is_section_end(*buffer)) {
            /* Check if we're at a valid position (previous char should be newline) */
            if (*buffer > block.start && (*buffer)[-1] != '\n') {
                /* Variable-width detected - fall back to line scanning */
                LOG_INFO("Variable line widths in block %s, using fallback parser",
                         block.category);
                block.variable_width = true;

                CifError err = _scan_lines(&block, ctx);
                if (err != CIF_OK) {
                    free(block.category);
                    free(block.offsets);
                    block.category = NULL;
                    block.offsets = NULL;
                    return block;
                }

                /* Advance buffer to end of data section */
                *buffer = block.end;
                break;
            }

            *buffer += block.width;
            block.size++;
        }
    }

    /* Skip past section end marker */
    _next_block(buffer);

    LOG_DEBUG("Block '%s': size=%d, attrs=%d, width=%d, var_width=%d, single=%d",
              block.category, block.size, block.attributes,
              block.width, block.variable_width, block.single);

    return block;
}


void _free_block(mmBlock *block) {
    block->head = NULL;
    block->start = NULL;
    block->end = NULL;
    block->variable_width = false;

    if (block->category != NULL) {
        free(block->category);
        block->category = NULL;
    }

    if (block->offsets != NULL) {
        free(block->offsets);
        block->offsets = NULL;
    }

    if (block->lines != NULL) {
        free(block->lines);
        block->lines = NULL;
    }
}


void _store_or_free_block(mmBlock *block, mmBlockList *blocks) {
    /* Route block to correct slot using registry */
    const BlockDef *defs = _get_blocks();

    for (int i = 0; i < BLOCK_COUNT; i++) {
        if (_eq(block->category, defs[i].category)) {
            mmBlock *slot = _get_block_by_id(blocks, defs[i].id);
            if (slot != NULL) {
                *slot = *block;
                return;
            }
        }
    }

    /* Block not in registry - free it */
    _free_block(block);
}


void _free_block_list(mmBlockList *blocks) {
    _free_block(&blocks->atom);
    _free_block(&blocks->poly);
    _free_block(&blocks->nonpoly);
    _free_block(&blocks->conn);
    _free_block(&blocks->chain);
}
