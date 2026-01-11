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
#include "chain_lookup.h"
#include "../log.h"

#include <math.h>    /* for isnan */
#include <unistd.h>  /* for isatty */

/* Hash tables for type lookups */
#include "../hash/atom.c"
#include "../hash/residue.c"
#include "../hash/element.c"
#include "../hash/molecule.c"
#include "../hash/entity.c"
#include "../hash/ion.c"


/* ============================================================================
 * CONSTANTS
 * mmCIF attribute names used in atom classification and reordering.
 * Note: Batch-parsed field attributes are now defined in registry.c.
 * ============================================================================ */

static const size_t COORDS = 3;

/* Atom-level attributes (used by residue/chain counting) */
static const char *ATTR_SEQ_ID        = "label_seq_id";
static const char *ATTR_LABEL_ASYM    = "label_asym_id";


/* ============================================================================
 * CHAIN SEQ_ID MAPPING
 * Maps (chain, seq_id) from atom table to global residue index.
 * ============================================================================ */

/**
 * @brief Per-chain seq_id to residue index mapping.
 *
 * For each chain, stores the minimum seq_id from the sequence table
 * and the global residue offset. This allows correct mapping of atoms
 * to residues even when seq_id doesn't start at 1 (e.g., unresolved
 * N-terminus residues).
 *
 * Given (chain_idx, seq_id), the global residue index is:
 *   res_offset + (seq_id - min_seq_id)
 */
typedef struct {
    int min_seq_id;   /**< First seq_id in sequence table for this chain */
    int res_offset;   /**< Global residue index where this chain starts */
} ChainSeqInfo;


/* ============================================================================
 * ATOM DATA HELPERS
 * Allocate/free shared AtomData structures for polymer and HETATM atoms.
 * ============================================================================ */

void atom_data_alloc(AtomData *data, int count, bool with_bfactors) {
    data->count = count;
    data->coords = count > 0 ? malloc((size_t)count * 3 * sizeof(float)) : NULL;
    data->elements = count > 0 ? malloc((size_t)count * sizeof(int)) : NULL;
    data->bfactors = (count > 0 && with_bfactors) ? malloc((size_t)count * sizeof(float)) : NULL;
}

void atom_data_free(AtomData *data) {
    free(data->coords);
    free(data->elements);
    free(data->bfactors);
    data->coords = NULL;
    data->elements = NULL;
    data->bfactors = NULL;
    data->count = 0;
}


/* Forward declarations for partial loading support */
static int _prescan_model_filter(mmCIF *cif, mmBlock *block, int atoms,
                                 const LoadFilter *filter, CifErrorContext *ctx);
static int _prescan_chain_filter(mmCIF *cif, mmBlock *block, int atoms,
                                 const LoadFilter *filter, CifErrorContext *ctx);
static int _prescan_alt_loc_filter(mmCIF *cif, mmBlock *block, int atoms,
                                   const LoadFilter *filter, CifErrorContext *ctx);
static CifError _compact_chain_arrays(mmCIF *cif, CifErrorContext *ctx);
static int *_count_atoms_per_chain_filtered(mmCIF *cif, mmBlock *block,
                                             const ChainLookup *chain_lookup,
                                             CifErrorContext *ctx);

/**
 * @brief Recount polymer/non-polymer atoms after exclusion filtering.
 *
 * Updates cif->polymer and cif->nonpoly based on is_excluded and is_nonpoly masks.
 *
 * @param cif Structure with is_excluded and is_nonpoly masks set
 * @param original_atoms Total atom count before filtering
 */
static void _recount_polymer_nonpoly(mmCIF *cif, int original_atoms) {
    int polymer = 0;
    for (int i = 0; i < original_atoms; i++) {
        if (!cif->is_excluded[i] && !cif->is_nonpoly[i]) {
            polymer++;
        }
    }
    cif->polymer = polymer;
    cif->nonpoly = cif->atoms - polymer;
}


/* ============================================================================
 * ATOM FILTER SYSTEM
 * Extensible atom filtering with common interface.
 * ============================================================================ */

/**
 * @brief Prescan function signature for atom filters.
 *
 * Each filter marks atoms for exclusion in cif->is_excluded.
 * Filters should skip atoms already marked (is_excluded[i] != 0).
 *
 * @param cif       Structure with is_excluded array allocated
 * @param block     Atom block with precomputed lines
 * @param atoms     Total atom count (original, before filtering)
 * @param filter    Filter options
 * @param ctx       Error context
 * @return Number of atoms excluded by this filter, or -1 on error
 */
typedef int (*AtomFilterFunc)(mmCIF *cif, mmBlock *block, int atoms,
                              const LoadFilter *filter, CifErrorContext *ctx);

/**
 * @brief Filter definition with metadata.
 */
typedef struct {
    const char *name;                              /**< Filter name for logging */
    AtomFilterFunc prescan;                        /**< Prescan function */
    bool (*is_active)(const mmCIF *cif, const LoadFilter *filter);  /**< Check if filter should run */
} AtomFilter;

/* Filter activation checks */
static bool _model_filter_active(const mmCIF *cif, const LoadFilter *filter) {
    /* Model filter is active for any multi-model structure */
    return cif->models > 1 && filter != NULL && filter->model >= 1;
}

static bool _chain_filter_active(const mmCIF *cif, const LoadFilter *filter) {
    (void)filter;
    return cif->chain_mask != NULL;
}

static bool _alt_loc_filter_active(const mmCIF *cif, const LoadFilter *filter) {
    (void)cif;
    return filter != NULL && filter->alt_loc != '\0';
}

/**
 * @brief Filter registry - order matters (model filter runs first).
 */
static const AtomFilter ATOM_FILTERS[] = {
    {"model",   _prescan_model_filter,   _model_filter_active},
    {"chain",   _prescan_chain_filter,   _chain_filter_active},
    {"alt_loc", _prescan_alt_loc_filter, _alt_loc_filter_active},
    {NULL, NULL, NULL}  /* Sentinel */
};

/**
 * @brief Apply all active atom filters.
 *
 * Allocates is_excluded if any filter is active, runs each active filter's
 * prescan function, and recounts polymer/nonpoly once at the end.
 *
 * @param cif           Structure to filter
 * @param block         Atom block with precomputed lines
 * @param original_atoms Total atom count before filtering
 * @param filter        Filter options
 * @param ctx           Error context
 * @return CIF_OK on success, error code on failure
 */
static CifError _apply_atom_filters(mmCIF *cif, mmBlock *block, int original_atoms,
                                    const LoadFilter *filter, CifErrorContext *ctx) {
    /* Check if any filter is active */
    bool any_active = false;
    for (int i = 0; ATOM_FILTERS[i].name != NULL; i++) {
        if (ATOM_FILTERS[i].is_active(cif, filter)) {
            any_active = true;
            break;
        }
    }
    if (!any_active) return CIF_OK;

    /* Allocate is_excluded array once for all filters */
    cif->is_excluded = calloc((size_t)original_atoms, sizeof(int));
    if (!cif->is_excluded) {
        CIF_SET_ERROR(ctx, CIF_ERR_ALLOC, "Failed to allocate is_excluded");
        return CIF_ERR_ALLOC;
    }

    /* Run each active filter */
    int total_excluded = 0;
    for (int i = 0; ATOM_FILTERS[i].name != NULL; i++) {
        const AtomFilter *f = &ATOM_FILTERS[i];
        if (!f->is_active(cif, filter)) continue;

        int excluded = f->prescan(cif, block, original_atoms, filter, ctx);
        if (excluded < 0) {
            free(cif->is_excluded);
            cif->is_excluded = NULL;
            return ctx->code;
        }

        if (excluded > 0) {
            total_excluded += excluded;
            LOG_DEBUG("%s filter: excluded %d atoms", f->name, excluded);
        }
    }

    /* Update counts once after all filters */
    if (total_excluded > 0) {
        cif->original_atoms = original_atoms;
        cif->atoms -= total_excluded;
        cif->excluded_count = total_excluded;
        _recount_polymer_nonpoly(cif, original_atoms);
        LOG_DEBUG("After filtering: %d atoms (%d polymer, %d non-polymer)",
                  cif->atoms, cif->polymer, cif->nonpoly);
    }

    return CIF_OK;
}


/* ============================================================================
 * FILE HEADER
 * Extract PDB identifier from mmCIF header.
 * ============================================================================ */

/**
 * Parse the PDB identifier from "data_XXXX" header line.
 */
char *_get_id(ParseCursor *cursor, CifErrorContext *ctx) {
    const char *prefix = "data_";

    if (_neq(cursor->ptr, prefix)) {
        CIF_SET_ERROR(ctx, CIF_ERR_PARSE,
            "Invalid mmCIF file: missing 'data_' prefix");
        return NULL;
    }

    cursor->ptr += 5;
    char *start = cursor->ptr;
    CURSOR_SKIP_TO_EOL(cursor);
    char *id = _strdup_n(start, (size_t)(cursor->ptr - start), ctx);

    /* Advance past the header line */
    CURSOR_PASS_NEWLINE(cursor);

    return id;
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
        /* Use _lookup_inline which emits warnings for unknown values */
        array[line] = _lookup_inline(block, line, index, func);
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

        if ((size_t)ix >= alloc_size) {
            if (*size > 0) {
                /* Caller specified expected size - other arrays depend on it.
                 * Skip extras (matches original behavior). */
                continue;
            }
            /* Dynamic sizing: grow array */
            size_t old_alloc = alloc_size;
            size_t new_alloc = (size_t)(ix + 1) * 2;
            int *resized = realloc(sizes, new_alloc * sizeof(int));
            if (resized == NULL) {
                CIF_SET_ERROR(ctx, CIF_ERR_ALLOC,
                    "Failed to realloc sizes array from %zu to %zu", old_alloc, new_alloc);
                free(sizes);
                return NULL;
            }
            memset(resized + old_alloc, 0, (new_alloc - old_alloc) * sizeof(int));
            sizes = resized;
            alloc_size = new_alloc;
        }
        sizes[ix]++;
    }

    /* For dynamic sizing, shrink to actual count */
    if (*size <= 0) {
        int final_count = ix + 1;
        int *resized = realloc(sizes, (size_t)final_count * sizeof(int));
        if (resized != NULL) {
            sizes = resized;
        }
        *size = final_count;
    }
    return sizes;
}


/**
 * Build chain seq_id mapping from sequence table.
 *
 * Creates mapping from (chain, seq_id) to global residue index using
 * _pdbx_poly_seq_scheme data. This allows atoms in _atom_site to be
 * correctly assigned to residues even when seq_id doesn't start at 1.
 *
 * @param cif Structure with chains count already set
 * @param poly_block Sequence table block (BLOCK_POLY)
 * @param ctx Error context
 * @return Array of ChainSeqInfo[chains], or NULL on error. Caller must free.
 */
static ChainSeqInfo *_build_chain_seq_info(mmCIF *cif, mmBlock *poly_block,
                                           const ChainLookup *chain_lookup,
                                           CifErrorContext *ctx) {
    /* Get attribute indices for asym_id (chain) and seq_id */
    int asym_index = _get_attr_index(poly_block, "asym_id", ctx);
    if (asym_index == BAD_IX) {
        CIF_SET_ERROR(ctx, CIF_ERR_ATTR, "Missing asym_id in sequence table");
        return NULL;
    }

    int seq_index = _get_attr_index(poly_block, "seq_id", ctx);
    if (seq_index == BAD_IX) {
        CIF_SET_ERROR(ctx, CIF_ERR_ATTR, "Missing seq_id in sequence table");
        return NULL;
    }

    int chain_count = cif->original_chains > 0 ? cif->original_chains : cif->chains;
    ChainSeqInfo *info = calloc((size_t)chain_count, sizeof(ChainSeqInfo));
    if (info == NULL) {
        CIF_SET_ERROR(ctx, CIF_ERR_ALLOC, "Failed to allocate ChainSeqInfo array");
        return NULL;
    }

    /* Initialize all min_seq_id to -1 (not yet set) */
    for (int i = 0; i < chain_count; i++) {
        info[i].min_seq_id = -1;
        info[i].res_offset = 0;
    }

    /* Iterate through sequence table to find min_seq_id per chain */
    int current_chain = 0;
    int res_count = 0;
    char *prev_asym_ptr = NULL;
    size_t prev_asym_len = 0;

    for (int line = 0; line < poly_block->size; line++) {
        /* Get chain (asym_id) */
        size_t asym_len;
        char *asym_ptr = _get_field_ptr(poly_block, line, asym_index, &asym_len);
        if (asym_ptr == NULL) {
            LOG_WARNING("Failed to get asym_id at line %d in sequence table", line);
            continue;
        }

        /* Detect chain change */
        bool chain_changed = (prev_asym_ptr == NULL) ||
            !_field_eq_field(prev_asym_ptr, prev_asym_len, asym_ptr, asym_len);

        if (chain_changed) {
            /* Find chain index using hash lookup */
            int new_chain = chain_lookup_find(chain_lookup, asym_ptr, asym_len);

            if (new_chain >= 0) {
                current_chain = new_chain;
                info[current_chain].res_offset = res_count;
            }

            prev_asym_ptr = asym_ptr;
            prev_asym_len = asym_len;
        }

        /* Get seq_id and record min for current chain */
        int seq_id = _parse_int_inline(poly_block, line, seq_index);
        if (current_chain >= 0 && current_chain < chain_count) {
            if (info[current_chain].min_seq_id < 0 || seq_id < info[current_chain].min_seq_id) {
                info[current_chain].min_seq_id = seq_id;
            }
        }

        res_count++;
    }

    /* Set default min_seq_id=1 for chains that had no entries */
    for (int i = 0; i < chain_count; i++) {
        if (info[i].min_seq_id < 0) {
            info[i].min_seq_id = 1;
        }
    }

    return info;
}


/**
 * Count atoms per chain with exclusion and nonpoly support.
 *
 * Like _count_sizes_by_group but skips excluded and non-polymer atoms.
 * This ensures atoms_per_chain matches atoms_per_res (both polymer-only).
 */
static int *_count_atoms_per_chain_filtered(mmCIF *cif, mmBlock *block,
                                             const ChainLookup *chain_lookup,
                                             CifErrorContext *ctx) {
    int index = _get_attr_index(block, ATTR_LABEL_ASYM, ctx);
    if (index == BAD_IX) {
        CIF_SET_ERROR(ctx, CIF_ERR_ATTR, "Missing attribute '%s'", ATTR_LABEL_ASYM);
        return NULL;
    }

    int chain_count = cif->original_chains > 0 ? cif->original_chains : cif->chains;
    int *sizes = calloc((size_t)chain_count, sizeof(int));
    if (sizes == NULL) {
        CIF_SET_ERROR(ctx, CIF_ERR_ALLOC, "Failed to allocate chain sizes array");
        return NULL;
    }

    int current_chain = 0;
    char *prev_ptr = NULL;
    size_t prev_len = 0;

    for (int line = 0; line < block->size; line++) {
        /* Skip excluded atoms */
        if (cif->is_excluded && cif->is_excluded[line]) {
            continue;
        }

        /* Skip non-polymer atoms (HETATM) - these go to separate arrays */
        if (cif->is_nonpoly && cif->is_nonpoly[line]) {
            continue;
        }

        size_t cur_len;
        char *cur_ptr = _get_field_ptr(block, line, index, &cur_len);
        if (cur_ptr == NULL) {
            CIF_SET_ERROR(ctx, CIF_ERR_PARSE, "Failed to get chain field");
            free(sizes);
            return NULL;
        }

        /* Track chain changes */
        if (prev_ptr == NULL) {
            prev_ptr = cur_ptr;
            prev_len = cur_len;
            /* Find initial chain index using hash lookup */
            current_chain = chain_lookup_find(chain_lookup, cur_ptr, cur_len);
        } else if (!_field_eq_field(prev_ptr, prev_len, cur_ptr, cur_len)) {
            prev_ptr = cur_ptr;
            prev_len = cur_len;
            /* Find new chain index using hash lookup */
            current_chain = chain_lookup_find(chain_lookup, cur_ptr, cur_len);
        }

        if (current_chain >= 0 && current_chain < chain_count) {
            sizes[current_chain]++;
        }
    }

    return sizes;
}


/**
 * Count atoms per residue using seq_id mapping.
 *
 * Maps each atom to its correct global residue index using the ChainSeqInfo
 * mapping built from the sequence table. This correctly handles CIF files
 * where seq_id doesn't start at 1 (e.g., unresolved N-terminus residues).
 *
 * @param cif Structure with chains and names already set
 * @param block Atom table block (BLOCK_ATOM)
 * @param residue_count Total residue count from sequence table
 * @param seq_info Per-chain seq_id mapping from _build_chain_seq_info
 * @param ctx Error context
 * @return Array of atom counts per residue, or NULL on error
 */
static int *_count_atoms_per_residue(mmCIF *cif, mmBlock *block, int residue_count,
                                     ChainSeqInfo *seq_info,
                                     const ChainLookup *chain_lookup,
                                     CifErrorContext *ctx) {
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

    int *sizes = calloc((size_t)residue_count, sizeof(int));
    if (sizes == NULL) {
        CIF_SET_ERROR(ctx, CIF_ERR_ALLOC,
            "Failed to allocate residue sizes array of size %d", residue_count);
        return NULL;
    }

    /* Track current chain by detecting chain name changes */
    int chain_count = cif->original_chains > 0 ? cif->original_chains : cif->chains;
    int current_chain = 0;
    char *prev_chain_ptr = NULL;
    size_t prev_chain_len = 0;

    for (int line = 0; line < block->size; line++) {
        /* Skip excluded atoms (from chain filter) */
        if (cif->is_excluded && cif->is_excluded[line]) {
            continue;
        }

        /* Skip non-polymer atoms (already classified during pre-scan) */
        if (cif->is_nonpoly && cif->is_nonpoly[line]) {
            continue;
        }

        /* Get chain ID */
        size_t chain_len;
        char *chain_ptr = _get_field_ptr(block, line, chain_index, &chain_len);
        if (chain_ptr == NULL) {
            CIF_SET_ERROR(ctx, CIF_ERR_PARSE,
                "Failed to get label_asym_id at line %d/%d in atom block",
                line, block->size);
            free(sizes);
            return NULL;
        }

        /* Track chain changes to update current_chain index */
        if (prev_chain_ptr == NULL) {
            prev_chain_ptr = chain_ptr;
            prev_chain_len = chain_len;
            /* Find initial chain index using hash lookup */
            current_chain = chain_lookup_find(chain_lookup, chain_ptr, chain_len);
        } else if (!_field_eq_field(prev_chain_ptr, prev_chain_len, chain_ptr, chain_len)) {
            prev_chain_ptr = chain_ptr;
            prev_chain_len = chain_len;
            /* Find new chain index using hash lookup */
            current_chain = chain_lookup_find(chain_lookup, chain_ptr, chain_len);
        }

        /* Parse sequence ID and compute global residue index */
        int seq_id = _parse_int_inline(block, line, seq_index);

        if (current_chain >= 0 && current_chain < chain_count) {
            ChainSeqInfo *info = &seq_info[current_chain];
            int local_res_idx = seq_id - info->min_seq_id;
            int global_res_idx = info->res_offset + local_res_idx;

            /* Validate bounds and count atom */
            if (global_res_idx >= 0 && global_res_idx < residue_count) {
                sizes[global_res_idx]++;
            } else {
                LOG_DEBUG("Atom seq_id %d out of range for chain %d (offset=%d, min=%d)",
                          seq_id, current_chain, info->res_offset, info->min_seq_id);
            }
        }
    }

    return sizes;
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
 *   4. Batch parse atom data (parallelized) - skipped if skip_mask includes batch fields
 *   5. Reorder atoms (polymer first) - skipped if batch fields skipped
 *
 * @param skip_mask Bitmask of fields to skip (SKIP_NONE for all, SKIP_METADATA for metadata only).
 * @param filter Optional filter for partial loading (NULL = load all).
 */
CifError _fill_cif(mmCIF *cif, mmBlockList *blocks, FieldSkipMask skip_mask,
                   const LoadFilter *filter, CifErrorContext *ctx) {
    LOG_DEBUG("Starting CIF structure parsing");

    /* ── Block Validation (registry-driven) ────────────────────────────────── */

    CifError val_err = _validate_blocks_registry(blocks, ctx);
    if (val_err != CIF_OK) return val_err;

    LOG_DEBUG("Block validation complete: atom=%d rows, poly=%d rows, chain=%d rows",
              blocks->b[BLOCK_ATOM].size, blocks->b[BLOCK_POLY].size, blocks->b[BLOCK_CHAIN].size);

    /* ── Precompute Line Pointers ────────────────────────────────────────── */
    /* Required for _get_field_ptr used in counting and metadata extraction */

    CifError err = _precompute_lines(&blocks->b[BLOCK_ATOM], ctx);
    if (err != CIF_OK) return err;

    err = _precompute_lines(&blocks->b[BLOCK_POLY], ctx);
    if (err != CIF_OK) {
        _free_lines(&blocks->b[BLOCK_ATOM]);
        return err;
    }

    err = _precompute_lines(&blocks->b[BLOCK_CHAIN], ctx);
    if (err != CIF_OK) {
        _free_lines(&blocks->b[BLOCK_ATOM]);
        _free_lines(&blocks->b[BLOCK_POLY]);
        return err;
    }

    LOG_DEBUG("Line pointers precomputed for all blocks");

    /* ── Parse Metadata (registry-driven) ──────────────────────────────────── */
    /* Compute field execution order and parse: chains, residues, models, atoms,
     * names, res_per_chain, strands, sequence */

    ParsePlan plan;
    err = _plan_parse(&plan, ctx);
    if (err != CIF_OK) {
        _free_lines(&blocks->b[BLOCK_ATOM]);
        _free_lines(&blocks->b[BLOCK_POLY]);
        _free_lines(&blocks->b[BLOCK_CHAIN]);
        return err;
    }

    /* Set target model for _op_compute_atoms */
    cif->target_model = (filter && filter->model > 0) ? filter->model : 1;

    err = _execute_plan(cif, blocks, &plan, ctx);
    if (err != CIF_OK) {
        _free_lines(&blocks->b[BLOCK_ATOM]);
        _free_lines(&blocks->b[BLOCK_POLY]);
        _free_lines(&blocks->b[BLOCK_CHAIN]);
        return err;
    }

    LOG_INFO("Parsing structure: %d models, %d chains, %d residues, %d atoms",
             cif->models, cif->chains, cif->residues, cif->atoms);

    LOG_DEBUG("Metadata extracted: %d chains, %d residues in sequence",
              cif->chains, cif->residues);

    /* ── Build Chain Mask for Partial Loading ─────────────────────────────── */
    /* Build inclusion mask based on filter criteria (molecule_types, chain_names) */
    if (filter && (filter->molecule_types || filter->chain_names)) {
        cif->original_chains = cif->chains;
        int included = _build_chain_mask(cif, filter, ctx);
        if (included < 0) {
            _free_lines(&blocks->b[BLOCK_ATOM]);
            _free_lines(&blocks->b[BLOCK_POLY]);
            _free_lines(&blocks->b[BLOCK_CHAIN]);
            return ctx->code;
        }
        LOG_INFO("Chain filter: %d/%d chains included", included, cif->original_chains);
    }

    /* ── Check if we should skip batch parsing ────────────────────────────── */
    /* Skip if coordinates are skipped (the main heavy field) */
    bool skip_batch = _is_field_skipped(FIELD_COORDS, skip_mask);

    /* Build chain lookup hash table for O(1) chain name -> index lookups.
     * Use original_chains if set (before chain filtering) since functions
     * operate on uncompacted arrays. */
    int chain_count_for_lookup = cif->original_chains > 0 ? cif->original_chains : cif->chains;
    ChainLookup chain_lookup = {0};
    chain_lookup_init(&chain_lookup);
    chain_lookup_build(&chain_lookup, cif->names, chain_count_for_lookup);

    if (skip_batch) {
        LOG_DEBUG("skip_mask includes batch fields: skipping batch parsing");

        _free_lines(&blocks->b[BLOCK_POLY]);
        _free_lines(&blocks->b[BLOCK_CHAIN]);

        int original_atoms = cif->atoms;

        /* Allocate is_nonpoly (always needed) */
        cif->is_nonpoly = calloc((size_t)original_atoms, sizeof(int));
        if (!cif->is_nonpoly) {
            _free_lines(&blocks->b[BLOCK_ATOM]);
            CIF_SET_ERROR(ctx, CIF_ERR_ALLOC, "Failed to allocate is_nonpoly (metadata)");
            return CIF_ERR_ALLOC;
        }

        /* Determine filtering parameters */
        int target_model = _model_filter_active(cif, filter) ?
                           (filter->model > 0 ? filter->model : 1) : 0;
        char keep_alt = _alt_loc_filter_active(cif, filter) ? filter->alt_loc : '\0';
        bool needs_exclusion = (target_model > 0 || keep_alt != '\0');

        /* Allocate is_excluded if any filtering needed */
        if (needs_exclusion) {
            cif->is_excluded = calloc((size_t)original_atoms, sizeof(int));
            if (!cif->is_excluded) {
                free(cif->is_nonpoly);
                _free_lines(&blocks->b[BLOCK_ATOM]);
                CIF_SET_ERROR(ctx, CIF_ERR_ALLOC, "Failed to allocate is_excluded (metadata)");
                return CIF_ERR_ALLOC;
            }
        }

        /* Unified prescan: classify polymer/nonpoly and apply model/alt filters */
        PrescanResult prescan = _prescan_unified(&blocks->b[BLOCK_ATOM], original_atoms,
                                                 cif->is_nonpoly, cif->is_excluded,
                                                 target_model, keep_alt, ctx);
        if (prescan.error != CIF_OK) {
            free(cif->is_nonpoly);
            if (cif->is_excluded) free(cif->is_excluded);
            _free_lines(&blocks->b[BLOCK_ATOM]);
            return prescan.error;
        }

        cif->polymer = prescan.polymer_count;
        cif->nonpoly = original_atoms - prescan.polymer_count;

        /* Adjust atom count - polymer only, minus excluded polymer atoms */
        if (cif->is_excluded && prescan.excluded_count > 0) {
            int excluded_polymer = 0;
            for (int i = 0; i < original_atoms; i++) {
                if (cif->is_excluded[i] && !cif->is_nonpoly[i]) {
                    excluded_polymer++;
                }
            }
            cif->atoms = cif->polymer - excluded_polymer;
            cif->excluded_count = prescan.excluded_count;
        } else {
            cif->atoms = cif->polymer;
        }

        LOG_DEBUG("metadata mode: %d polymer atoms (after filters)", cif->atoms);

        /* Count atoms per chain - always use filtered version since is_nonpoly is set */
        cif->atoms_per_chain = _count_atoms_per_chain_filtered(cif, &blocks->b[BLOCK_ATOM],
                                                               &chain_lookup, ctx);

        _free_lines(&blocks->b[BLOCK_ATOM]);
        free(cif->is_nonpoly);
        cif->is_nonpoly = NULL;
        if (cif->is_excluded) {
            free(cif->is_excluded);
            cif->is_excluded = NULL;
        }
        if (cif->atoms_per_chain == NULL) return ctx->code;

        LOG_DEBUG("metadata only: computed atoms_per_chain for %d chains", cif->chains);
        return CIF_OK;
    }

    /* ── Batch Atom Parsing with Two-Pointer Placement ───────────────────── */
    /* Note: lines already precomputed at start of function */

    int original_atoms = cif->atoms;
    LOG_DEBUG("Beginning batch atom parsing (%d atoms)...", cif->atoms);

    /* Allocate is_nonpoly for two-pointer placement (sized for ORIGINAL atom count) */
    cif->is_nonpoly = calloc((size_t)original_atoms, sizeof(int));
    if (!cif->is_nonpoly) {
        CIF_SET_ERROR(ctx, CIF_ERR_ALLOC, "Failed to allocate is_nonpoly");
        return CIF_ERR_ALLOC;
    }

    /* Unified prescan for polymer classification (no model/alt filtering here,
     * that's handled by _apply_atom_filters for batch path) */
    PrescanResult prescan = _prescan_unified(&blocks->b[BLOCK_ATOM], original_atoms,
                                             cif->is_nonpoly, NULL, 0, '\0', ctx);
    if (prescan.error != CIF_OK) {
        free(cif->is_nonpoly);
        return prescan.error;
    }
    cif->polymer = prescan.polymer_count;
    cif->nonpoly = cif->atoms - prescan.polymer_count;

    LOG_DEBUG("Pre-scan: %d polymer, %d non-polymer atoms", cif->polymer, cif->nonpoly);

    /* Allocate HETATM arrays for non-polymer atoms */
    bool has_bfactors = !(skip_mask & SKIP_BFACTORS);
    atom_data_alloc(&cif->hetatm, cif->nonpoly, has_bfactors);
    if (cif->nonpoly > 0) {
        cif->hetatm_chains = calloc((size_t)cif->nonpoly, sizeof(int));
        if (!cif->hetatm_chains) {
            atom_data_free(&cif->hetatm);
            free(cif->is_nonpoly);
            CIF_SET_ERROR(ctx, CIF_ERR_ALLOC, "Failed to allocate hetatm_chains");
            return CIF_ERR_ALLOC;
        }
    }

    /* Apply atom filters (chain, alt_loc, etc.) */
    err = _apply_atom_filters(cif, &blocks->b[BLOCK_ATOM], original_atoms, filter, ctx);
    if (err != CIF_OK) {
        atom_data_free(&cif->hetatm);
        free(cif->hetatm_chains);
        free(cif->is_nonpoly);
        _free_lines(&blocks->b[BLOCK_ATOM]);
        _free_lines(&blocks->b[BLOCK_POLY]);
        _free_lines(&blocks->b[BLOCK_CHAIN]);
        return err;
    }

    /* Set atoms to polymer count only - HETATM data goes in separate arrays */
    cif->atoms = cif->polymer;

    /* Allocate arrays for fields with size_source set (coordinates, types, elements) */
    /* NOTE: cif->atoms is now the POLYMER count only */
    err = _allocate_field_arrays(cif, skip_mask, ctx);
    if (err != CIF_OK) {
        free(cif->is_nonpoly);
        if (cif->is_excluded) free(cif->is_excluded);
        _free_lines(&blocks->b[BLOCK_ATOM]);
        _free_lines(&blocks->b[BLOCK_POLY]);
        _free_lines(&blocks->b[BLOCK_CHAIN]);
        return err;
    }

    /* Compute and execute batch groups - data written directly to final positions */
    BatchGroup batch_groups[BLOCK_COUNT];
    int batch_group_count = 0;
    _compute_batch_groups(batch_groups, &batch_group_count, BLOCK_COUNT);

    for (int g = 0; g < batch_group_count; g++) {
        err = _execute_batch_group(cif, blocks, &batch_groups[g], ctx);
        if (err != CIF_OK) {
            free(cif->is_nonpoly);
            _free_lines(&blocks->b[BLOCK_ATOM]);
            _free_lines(&blocks->b[BLOCK_POLY]);
            _free_lines(&blocks->b[BLOCK_CHAIN]);
            return err;
        }
    }

    /* Build chain seq_id mapping from sequence table before freeing lines */
    ChainSeqInfo *seq_info = _build_chain_seq_info(cif, &blocks->b[BLOCK_POLY],
                                                   &chain_lookup, ctx);
    if (seq_info == NULL) {
        _free_lines(&blocks->b[BLOCK_POLY]);
        _free_lines(&blocks->b[BLOCK_CHAIN]);
        free(cif->is_nonpoly);
        return ctx->code;
    }

    /* Free poly and chain line pointers - no longer needed */
    _free_lines(&blocks->b[BLOCK_POLY]);
    _free_lines(&blocks->b[BLOCK_CHAIN]);

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
        LOG_WARNING("[%s] Found %d atoms with invalid (NaN) coordinates",
                    cif->id ? cif->id : "unknown", nan_count);
    }

    /* ── Residue/Chain Counting ────────────────────────────────────────────── */

    /* Count atoms per residue using seq_id mapping */
    cif->atoms_per_res = _count_atoms_per_residue(cif, &blocks->b[BLOCK_ATOM], cif->residues,
                                                  seq_info, &chain_lookup, ctx);
    free(seq_info);  /* No longer needed after counting */
    seq_info = NULL;

    if (cif->atoms_per_res == NULL) {
        free(cif->is_nonpoly);
        return ctx->code;
    }

    /* Count atoms per chain - always use filtered version to skip nonpoly atoms.
     * This ensures atoms_per_chain matches atoms_per_res (both polymer-only). */
    cif->atoms_per_chain = _count_atoms_per_chain_filtered(cif, &blocks->b[BLOCK_ATOM],
                                                           &chain_lookup, ctx);
    if (cif->atoms_per_chain == NULL) {
        free(cif->is_nonpoly);
        return ctx->code;
    }

    /* Free is_nonpoly - no longer needed */
    free(cif->is_nonpoly);
    cif->is_nonpoly = NULL;

    /* Free is_excluded - no longer needed */
    if (cif->is_excluded) {
        free(cif->is_excluded);
        cif->is_excluded = NULL;
    }

    /* Compact chain arrays to remove excluded chains */
    if (cif->chain_mask) {
        err = _compact_chain_arrays(cif, ctx);
        if (err != CIF_OK) {
            free(cif->chain_mask);
            return err;
        }
        free(cif->chain_mask);
        cif->chain_mask = NULL;
    }

    LOG_INFO("Parsed %d polymer atoms, %d non-polymer atoms", cif->polymer, cif->nonpoly);
    LOG_DEBUG("CIF structure parsing complete");

    return CIF_OK;
}


/* ============================================================================
 * PARTIAL LOADING SUPPORT
 * Functions for filtering chains during loading.
 * ============================================================================ */

/**
 * Build a chain inclusion mask based on filter criteria.
 *
 * Creates a boolean mask indicating which chains to include based on
 * molecule_types and/or chain_names filters.
 *
 * @param cif Structure with chains and molecule_types already parsed
 * @param filter Filter criteria
 * @param ctx Error context, populated on failure
 * @return Number of included chains, or -1 on error
 */
int _build_chain_mask(mmCIF *cif, const LoadFilter *filter, CifErrorContext *ctx) {
    if (!cif || !filter) return -1;

    /* Allocate chain mask */
    cif->chain_mask = calloc((size_t)cif->chains, sizeof(int));
    if (!cif->chain_mask) {
        CIF_SET_ERROR(ctx, CIF_ERR_ALLOC, "Failed to allocate chain_mask");
        return -1;
    }

    int included = 0;
    for (int i = 0; i < cif->chains; i++) {
        bool include = true;

        /* Check molecule type filter */
        if (include && filter->molecule_types != NULL) {
            include = false;
            for (int j = 0; filter->molecule_types[j] != -1; j++) {
                if (cif->molecule_types && cif->molecule_types[i] == filter->molecule_types[j]) {
                    include = true;
                    break;
                }
            }
        }

        /* Check chain name filter */
        if (include && filter->chain_names != NULL) {
            include = false;
            for (int j = 0; filter->chain_names[j] != NULL; j++) {
                if (cif->names && cif->names[i] &&
                    strcmp(cif->names[i], filter->chain_names[j]) == 0) {
                    include = true;
                    break;
                }
            }
        }

        cif->chain_mask[i] = include ? 1 : 0;
        if (include) included++;
    }

    LOG_DEBUG("Chain mask built: %d/%d chains included", included, cif->chains);
    return included;
}


/**
 * @brief Chain filter: mark atoms from excluded chains.
 *
 * Iterates through all atoms and marks those belonging to excluded chains.
 *
 * @param cif Structure with chain_mask, names, and is_excluded already set
 * @param block Atom block
 * @param atoms Total atom count
 * @param filter Filter options (unused, chain_mask is in cif)
 * @param ctx Error context
 * @return Number of excluded atoms, or -1 on error
 */
static int _prescan_chain_filter(mmCIF *cif, mmBlock *block, int atoms,
                                 const LoadFilter *filter, CifErrorContext *ctx) {
    (void)filter;  /* Chain mask is in cif, not filter */

    /* Get label_asym_id attribute index */
    int chain_idx = _get_attr_index(block, ATTR_LABEL_ASYM, ctx);
    if (chain_idx == BAD_IX) {
        CIF_SET_ERROR(ctx, CIF_ERR_ATTR, "Missing attribute 'label_asym_id' for chain filtering");
        return -1;
    }

    /* Build chain lookup for O(1) name->index lookups */
    ChainLookup chain_lookup = {0};
    chain_lookup_init(&chain_lookup);
    chain_lookup_build(&chain_lookup, cif->names, cif->original_chains);

    /* Track chain transitions to efficiently map atoms to chains */
    int current_chain = 0;
    char *prev_chain_ptr = NULL;
    size_t prev_chain_len = 0;
    int excluded = 0;

    for (int row = 0; row < atoms; row++) {
        if (cif->is_excluded[row]) continue;  /* Already excluded */

        size_t chain_len;
        char *chain_ptr = _get_field_ptr(block, row, chain_idx, &chain_len);
        if (!chain_ptr) {
            CIF_SET_ERROR(ctx, CIF_ERR_PARSE, "Failed to get chain field at row %d", row);
            return -1;
        }

        /* Check if chain changed - use O(1) lookup instead of O(n) loop */
        if (prev_chain_ptr == NULL ||
            !_field_eq_field(prev_chain_ptr, prev_chain_len, chain_ptr, chain_len)) {
            prev_chain_ptr = chain_ptr;
            prev_chain_len = chain_len;
            current_chain = chain_lookup_find(&chain_lookup, chain_ptr, chain_len);
        }

        /* Mark as excluded if chain not in mask */
        if (current_chain < 0 || !cif->chain_mask[current_chain]) {
            cif->is_excluded[row] = 1;
            excluded++;
        }
    }

    return excluded;
}

/**
 * @brief Model filter: mark atoms not belonging to the target model.
 *
 * @param cif Structure with is_excluded already set and target_model set
 * @param block Atom block with precomputed lines
 * @param atoms Total atom count (original, before filtering)
 * @param filter Filter options (uses filter->model)
 * @param ctx Error context
 * @return Number of excluded atoms, or -1 on error
 */
static int _prescan_model_filter(mmCIF *cif, mmBlock *block, int atoms,
                                 const LoadFilter *filter, CifErrorContext *ctx) {
    int target_model = filter->model > 0 ? filter->model : 1;

    /* Get pdbx_PDB_model_num attribute index */
    int model_idx = _get_attr_index(block, "pdbx_PDB_model_num", ctx);
    if (model_idx == BAD_IX) {
        /* No model column - single model structure, nothing to filter */
        LOG_DEBUG("No pdbx_PDB_model_num column, treating as single-model");
        return 0;
    }

    int excluded = 0;
    for (int row = 0; row < atoms; row++) {
        if (cif->is_excluded[row]) continue;  /* Already excluded */

        int model_num = _parse_int_inline(block, row, model_idx);
        if (model_num != target_model) {
            cif->is_excluded[row] = 1;
            excluded++;
        }
    }

    LOG_DEBUG("Model filter: excluded %d atoms (keeping model %d)", excluded, target_model);
    return excluded;
}

/**
 * @brief Alt loc filter: mark atoms with non-matching alternate conformations.
 *
 * @param cif Structure with is_excluded already set
 * @param block Atom block
 * @param atoms Total atom count
 * @param filter Filter options (uses filter->alt_loc)
 * @param ctx Error context
 * @return Number of excluded atoms, or -1 on error
 */
static int _prescan_alt_loc_filter(mmCIF *cif, mmBlock *block, int atoms,
                                   const LoadFilter *filter, CifErrorContext *ctx) {
    return _prescan_alt_locs(block, atoms, cif->is_excluded, filter->alt_loc, ctx);
}


/**
 * Compact chain arrays to remove excluded chains.
 *
 * After filtering, removes chains with 0 atoms from all chain-indexed arrays:
 * names, strands, molecule_types, res_per_chain, atoms_per_chain.
 * Updates cif->chains to reflect the new count.
 *
 * @param cif Structure with chain_mask and atoms_per_chain populated
 * @param ctx Error context
 * @return CIF_OK on success, error code on failure
 */
static CifError _compact_chain_arrays(mmCIF *cif, CifErrorContext *ctx) {
    if (!cif->chain_mask) return CIF_OK;  /* No filtering, nothing to compact */

    /* Count included chains */
    int included = 0;
    for (int i = 0; i < cif->original_chains; i++) {
        if (cif->chain_mask[i]) included++;
    }

    /* If all chains included, nothing to do */
    if (included == cif->original_chains) return CIF_OK;

    /* If no chains included, create empty arrays (not NULL, since Python expects arrays) */
    if (included == 0) {
        /* Free old string arrays */
        if (cif->names) {
            for (int i = 0; i < cif->original_chains; i++) {
                if (cif->names[i]) free(cif->names[i]);
            }
            free(cif->names);
        }
        if (cif->strands) {
            for (int i = 0; i < cif->original_chains; i++) {
                if (cif->strands[i]) free(cif->strands[i]);
            }
            free(cif->strands);
        }
        /* Allocate empty arrays (size 1 to avoid malloc(0) issues) */
        cif->names = calloc(1, sizeof(char *));
        cif->strands = calloc(1, sizeof(char *));
        if (cif->molecule_types) free(cif->molecule_types);
        cif->molecule_types = calloc(1, sizeof(int));
        if (cif->res_per_chain) free(cif->res_per_chain);
        cif->res_per_chain = calloc(1, sizeof(int));
        if (cif->atoms_per_chain) free(cif->atoms_per_chain);
        cif->atoms_per_chain = calloc(1, sizeof(int));
        if (cif->atoms_per_res) free(cif->atoms_per_res);
        cif->atoms_per_res = calloc(1, sizeof(int));
        cif->chains = 0;
        cif->residues = 0;
        return CIF_OK;
    }

    /* Count total residues in included chains */
    int new_residues = 0;
    for (int i = 0; i < cif->original_chains; i++) {
        if (cif->chain_mask[i] && cif->res_per_chain) {
            new_residues += cif->res_per_chain[i];
        }
    }

    /* Allocate new compacted arrays */
    char **new_names = calloc(included, sizeof(char *));
    char **new_strands = calloc(included, sizeof(char *));
    int *new_mol_types = calloc(included, sizeof(int));
    int *new_res_per_chain = calloc(included, sizeof(int));
    int *new_atoms_per_chain = calloc(included, sizeof(int));
    int *new_atoms_per_res = NULL;

    if (cif->atoms_per_res && new_residues > 0) {
        new_atoms_per_res = calloc(new_residues, sizeof(int));
    }

    if (!new_names || !new_strands || !new_mol_types ||
        !new_res_per_chain || !new_atoms_per_chain ||
        (cif->atoms_per_res && new_residues > 0 && !new_atoms_per_res)) {
        free(new_names); free(new_strands); free(new_mol_types);
        free(new_res_per_chain); free(new_atoms_per_chain);
        free(new_atoms_per_res);
        CIF_SET_ERROR(ctx, CIF_ERR_ALLOC, "Failed to allocate compacted chain arrays");
        return CIF_ERR_ALLOC;
    }

    /* Copy included chains, free excluded ones */
    int j = 0;
    int src_res_offset = 0;  /* Residue offset in source (uncompacted) arrays */
    int dst_res_offset = 0;  /* Residue offset in destination (compacted) arrays */
    for (int i = 0; i < cif->original_chains; i++) {
        int chain_res_count = cif->res_per_chain ? cif->res_per_chain[i] : 0;

        if (cif->chain_mask[i]) {
            /* Copy to new arrays */
            new_names[j] = cif->names ? cif->names[i] : NULL;
            new_strands[j] = cif->strands ? cif->strands[i] : NULL;
            new_mol_types[j] = cif->molecule_types ? cif->molecule_types[i] : 0;
            new_res_per_chain[j] = chain_res_count;
            new_atoms_per_chain[j] = cif->atoms_per_chain ? cif->atoms_per_chain[i] : 0;

            /* Copy atoms_per_res for this chain */
            if (cif->atoms_per_res && new_atoms_per_res) {
                for (int r = 0; r < chain_res_count; r++) {
                    new_atoms_per_res[dst_res_offset + r] = cif->atoms_per_res[src_res_offset + r];
                }
            }
            dst_res_offset += chain_res_count;
            j++;
        } else {
            /* Free excluded chain's strings */
            if (cif->names && cif->names[i]) free(cif->names[i]);
            if (cif->strands && cif->strands[i]) free(cif->strands[i]);
        }
        src_res_offset += chain_res_count;
    }

    /* Free old arrays (but not the strings we copied) */
    free(cif->names);
    free(cif->strands);
    free(cif->molecule_types);
    free(cif->res_per_chain);
    free(cif->atoms_per_chain);
    if (cif->atoms_per_res) free(cif->atoms_per_res);

    /* Assign new arrays */
    cif->names = new_names;
    cif->strands = new_strands;
    cif->molecule_types = new_mol_types;
    cif->res_per_chain = new_res_per_chain;
    cif->atoms_per_chain = new_atoms_per_chain;
    cif->atoms_per_res = new_atoms_per_res;
    cif->chains = included;
    cif->residues = new_residues;

    LOG_DEBUG("Compacted chains: %d -> %d", cif->original_chains, included);
    return CIF_OK;
}


/**
 * Free LoadFilter resources.
 *
 * @param filter Filter to free (fields are set to NULL)
 */
void _free_filter(LoadFilter *filter) {
    if (!filter) return;

    if (filter->molecule_types) {
        free(filter->molecule_types);
        filter->molecule_types = NULL;
    }

    if (filter->chain_names) {
        for (int i = 0; filter->chain_names[i] != NULL; i++) {
            free(filter->chain_names[i]);
        }
        free(filter->chain_names);
        filter->chain_names = NULL;
    }

    filter->mol_type_count = 0;
    filter->chain_count = 0;
    filter->model = 0;
}


/**
 * Create a LoadFilter with default values.
 *
 * @return LoadFilter with sensible defaults
 */
LoadFilter _default_filter(void) {
    return (LoadFilter){
        .molecule_types = NULL,
        .chain_names = NULL,
        .mol_type_count = 0,
        .chain_count = 0,
        .model = 1,
        .connections = false,
        .alt_loc = 'A',
    };
}


/* ============================================================================
 * BLOCK PARSING API
 * Public functions for reading and managing mmCIF blocks.
 * ============================================================================ */

bool _skip_multiline_attr(ParseCursor *cursor) {
    CURSOR_NEXT_LINE(cursor);
    int lines = 0;
    const int MAX_MULTILINE_LINES = 10000;
    while (*cursor->ptr != ';' && !CURSOR_AT_END(cursor) && lines < MAX_MULTILINE_LINES) {
        CURSOR_NEXT_LINE(cursor);
        lines++;
    }
    if (lines >= MAX_MULTILINE_LINES) {
        LOG_ERROR("Unterminated multiline attribute (exceeded %d lines)", MAX_MULTILINE_LINES);
        return false;
    }
    if (*cursor->ptr == ';') {
        CURSOR_NEXT_LINE(cursor);
    }
    return true;
}


/* ============================================================================
 * BLOCK PRUNING
 * Functions for skipping unneeded blocks during parsing.
 * ============================================================================ */

int _peek_block_id(ParseCursor *cursor) {
    char *ptr = cursor->ptr;

    /* Skip "loop_" prefix if present */
    if (_eq(ptr, "loop_")) {
        while (*ptr != '\n' && *ptr != '\0') ptr++;
        if (*ptr == '\n') ptr++;
    }

    /* Now ptr points to first attribute line, e.g. "_atom_site.id" */
    if (*ptr != '_') return -1;

    /* Compare against registry categories */
    const BlockDef *blocks = _get_blocks();
    for (int i = 0; i < BLOCK_COUNT; i++) {
        if (_eq(ptr, blocks[i].category)) {
            return i;
        }
    }

    return -1;  /* Not in registry */
}


void _skip_current_block(ParseCursor *cursor) {
    /* Skip "loop_" line if present */
    if (_eq(cursor->ptr, "loop_")) {
        CURSOR_NEXT_LINE(cursor);
    }

    /* Skip attribute header lines (lines starting with category prefix) */
    while (!CURSOR_AT_END(cursor) && cursor->ptr[0] == '_') {
        CURSOR_NEXT_LINE(cursor);
        /* Handle multiline attributes starting with ; */
        if (cursor->ptr[0] == ';') {
            _skip_multiline_attr(cursor);
        }
    }

    /* Skip data rows until section end (line starting with #) */
    while (!CURSOR_AT_END(cursor) && !_is_section_end(cursor->ptr)) {
        CURSOR_NEXT_LINE(cursor);
    }

    /* Skip the # line itself */
    if (!CURSOR_AT_END(cursor)) {
        CURSOR_NEXT_LINE(cursor);
    }
}


void _next_block(ParseCursor *cursor) {
    while (!CURSOR_AT_END(cursor) && !_is_section_end(cursor->ptr)) {
        CURSOR_NEXT_LINE(cursor);
    }
    if (!CURSOR_AT_END(cursor)) {
        CURSOR_NEXT_LINE(cursor);
    }
}


mmBlock _read_block(ParseCursor *cursor, CifErrorContext *ctx) {

    mmBlock block = {0};

    /* Check if this is a single-entry block (no "loop_" prefix) */
    if (_eq(cursor->ptr, "loop_")) {
        CURSOR_NEXT_LINE(cursor);
    } else {
        block.single = true;
        block.size = 1;
    }

    block.head = cursor->ptr;
    block.category = _get_category(block.head, ctx);
    if (block.category == NULL) {
        return block;  /* Error - ctx is already set */
    }

    /* Count attributes by scanning header lines */
    while (!CURSOR_AT_END(cursor) && _eq(cursor->ptr, block.category)) {
        block.attributes++;
        CURSOR_NEXT_LINE(cursor);
        if (*cursor->ptr == ';') {
            if (!_skip_multiline_attr(cursor)) {
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
        /* Multi-entry block: save cursor position and calculate offsets */
        block.data = *cursor;  /* Copy cursor (ptr + line) atomically */
        block.variable_width = false;
        block.offsets = _get_offsets(block.data.ptr, block.attributes, ctx);
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
        while (!CURSOR_AT_END(cursor) && !_is_section_end(cursor->ptr)) {
            /* Check if we're at a valid position (previous char should be newline) */
            if (cursor->ptr > block.data.ptr && (cursor->ptr)[-1] != '\n') {
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

                /* Update cursor to end of block data */
                cursor->ptr = block.end;
                cursor->line = block.data.line + block.size;
                break;
            }

            /* Advance by fixed width and track line */
            cursor->ptr += block.width;
            cursor->line++;
            block.size++;
        }
    }

    /* Skip past section end marker */
    _next_block(cursor);

    LOG_DEBUG("Block '%s': size=%d, attrs=%d, width=%d, var_width=%d, single=%d, data.line=%d",
              block.category, block.size, block.attributes,
              block.width, block.variable_width, block.single, block.data.line);

    return block;
}


void _free_block(mmBlock *block) {
    block->head = NULL;
    block->data.ptr = NULL;
    block->data.line = 0;
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
    for (int i = 0; i < BLOCK_COUNT; i++) {
        _free_block(&blocks->b[i]);
    }
}
