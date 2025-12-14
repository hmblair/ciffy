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
#include "../log.h"
#include "../profile.h"

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
static int *_count_atoms_per_residue(mmCIF *cif, mmBlock *block, int residue_count,
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

    int *sizes = calloc((size_t)residue_count, sizeof(int));
    if (sizes == NULL) {
        CIF_SET_ERROR(ctx, CIF_ERR_ALLOC,
            "Failed to allocate residue sizes array of size %d", residue_count);
        return NULL;
    }

    /* Note: cif->nonpoly and cif->polymer are already set by _prescan_group_pdb */

    int chain_offset = 0;
    char *prev_chain_ptr = NULL;
    size_t prev_chain_len = 0;
    int *chain_len_ptr = res_per_chain;

    for (int line = 0; line < block->size; line++) {
        /* Skip non-polymer atoms (already classified during pre-scan) */
        if (cif->is_nonpoly[line]) {
            continue;
        }

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
 *   4. Batch parse atom data (parallelized) - skipped if metadata_only
 *   5. Reorder atoms (polymer first) - skipped if metadata_only
 *
 * @param metadata_only If true, skip batch parsing and only compute counts.
 *                      Used for fast dataset indexing.
 */
CifError _fill_cif(mmCIF *cif, mmBlockList *blocks, bool metadata_only, CifErrorContext *ctx) {
    LOG_DEBUG("Starting CIF structure parsing");

    /* ── Block Validation (registry-driven) ────────────────────────────────── */

    CifError val_err = _validate_blocks_registry(blocks, ctx);
    if (val_err != CIF_OK) return val_err;

    LOG_DEBUG("Block validation complete: atom=%d rows, poly=%d rows, chain=%d rows",
              blocks->b[BLOCK_ATOM].size, blocks->b[BLOCK_POLY].size, blocks->b[BLOCK_CHAIN].size);

    /* ── Precompute Line Pointers ────────────────────────────────────────── */
    /* Required for _get_field_ptr used in counting and metadata extraction */

    PROFILE_START(line_precomp);
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
    PROFILE_END(line_precomp);

    LOG_DEBUG("Line pointers precomputed for all blocks");

    /* ── Parse Metadata (registry-driven) ──────────────────────────────────── */
    /* Compute field execution order and parse: chains, residues, models, atoms,
     * names, res_per_chain, strands, sequence */

    PROFILE_START(metadata);
    ParsePlan plan;
    err = _plan_parse(&plan, ctx);
    if (err != CIF_OK) {
        _free_lines(&blocks->b[BLOCK_ATOM]);
        _free_lines(&blocks->b[BLOCK_POLY]);
        _free_lines(&blocks->b[BLOCK_CHAIN]);
        return err;
    }

    err = _execute_plan(cif, blocks, &plan, ctx);
    if (err != CIF_OK) {
        _free_lines(&blocks->b[BLOCK_ATOM]);
        _free_lines(&blocks->b[BLOCK_POLY]);
        _free_lines(&blocks->b[BLOCK_CHAIN]);
        return err;
    }
    PROFILE_END(metadata);

    LOG_INFO("Parsing structure: %d models, %d chains, %d residues, %d atoms",
             cif->models, cif->chains, cif->residues, cif->atoms);

    LOG_DEBUG("Metadata extracted: %d chains, %d residues in sequence",
              cif->chains, cif->residues);

    /* ── metadata_only: Skip batch parsing, just compute atoms_per_chain ───── */
    if (metadata_only) {
        LOG_DEBUG("metadata_only mode: skipping batch parsing");

        _free_lines(&blocks->b[BLOCK_POLY]);
        _free_lines(&blocks->b[BLOCK_CHAIN]);

        /* Only compute atoms_per_chain for fast indexing */
        cif->atoms_per_chain = _count_sizes_by_group(&blocks->b[BLOCK_ATOM], ATTR_LABEL_ASYM,
                                                     &cif->chains, ctx);
        _free_lines(&blocks->b[BLOCK_ATOM]);
        if (cif->atoms_per_chain == NULL) return ctx->code;

        LOG_DEBUG("metadata_only: computed atoms_per_chain for %d chains", cif->chains);
        return CIF_OK;
    }

    /* ── Batch Atom Parsing with Two-Pointer Placement ───────────────────── */
    /* Note: lines already precomputed at start of function */

    LOG_DEBUG("Beginning batch atom parsing (%d atoms)...", cif->atoms);

    PROFILE_START(batch_parse);

    /* Allocate arrays for fields with size_source set (coordinates, types, elements) */
    err = _allocate_field_arrays(cif, ctx);
    if (err != CIF_OK) {
        _free_lines(&blocks->b[BLOCK_ATOM]);
        _free_lines(&blocks->b[BLOCK_POLY]);
        _free_lines(&blocks->b[BLOCK_CHAIN]);
        return err;
    }

    /* Allocate is_nonpoly for two-pointer placement */
    cif->is_nonpoly = calloc((size_t)cif->atoms, sizeof(int));
    if (!cif->is_nonpoly) {
        CIF_SET_ERROR(ctx, CIF_ERR_ALLOC, "Failed to allocate is_nonpoly");
        return CIF_ERR_ALLOC;
    }

    /* Pre-scan group_PDB to classify atoms */
    int polymer_count = _prescan_group_pdb(&blocks->b[BLOCK_ATOM], cif->atoms,
                                           cif->is_nonpoly, ctx);
    if (polymer_count < 0) {
        free(cif->is_nonpoly);
        return ctx->code;
    }
    cif->polymer = polymer_count;
    cif->nonpoly = cif->atoms - polymer_count;

    LOG_DEBUG("Pre-scan: %d polymer, %d non-polymer atoms", cif->polymer, cif->nonpoly);

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
        LOG_WARNING("Found %d atoms with invalid (NaN) coordinates", nan_count);
    }
    PROFILE_END(batch_parse);

    /* ── Residue/Chain Counting ────────────────────────────────────────────── */

    PROFILE_START(residue_count);

    /* Count atoms per residue (is_nonpoly already filled, nonpoly count already set) */
    cif->atoms_per_res = _count_atoms_per_residue(cif, &blocks->b[BLOCK_ATOM], cif->residues,
                                                  cif->res_per_chain, ctx);
    if (cif->atoms_per_res == NULL) {
        free(cif->is_nonpoly);
        return ctx->code;
    }

    /* Free is_nonpoly - no longer needed */
    free(cif->is_nonpoly);
    cif->is_nonpoly = NULL;

    cif->atoms_per_chain = _count_sizes_by_group(&blocks->b[BLOCK_ATOM], ATTR_LABEL_ASYM,
                                                 &cif->chains, ctx);
    if (cif->atoms_per_chain == NULL) return ctx->code;

    PROFILE_END(residue_count);

    LOG_INFO("Parsed %d polymer atoms, %d non-polymer atoms", cif->polymer, cif->nonpoly);
    LOG_DEBUG("CIF structure parsing complete");

    return CIF_OK;
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
