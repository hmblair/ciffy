/**
 * @file io.c
 * @brief Low-level I/O and block parsing utilities for mmCIF files.
 */

#include "io.h"
#include "../log.h"

#include <errno.h>
#include <limits.h>
#include <stdint.h>

/** Maximum file size (1GB) - prevents memory exhaustion attacks */
#define MAX_CIF_FILE_SIZE (1024L * 1024L * 1024L)


CifError _load_file(const char *name, char **buffer, CifErrorContext *ctx) {

    *buffer = NULL;

    FILE *file = fopen(name, "r");
    if (file == NULL) {
        CIF_SET_ERROR(ctx, CIF_ERR_IO, "Failed to open file: %s", name);
        return CIF_ERR_IO;
    }

    /* Get file size */
    if (fseek(file, 0, SEEK_END) != 0) {
        CIF_SET_ERROR(ctx, CIF_ERR_IO, "Failed to seek to end of file: %s", name);
        fclose(file);
        return CIF_ERR_IO;
    }

    long size = ftell(file);
    if (size < 0) {
        CIF_SET_ERROR(ctx, CIF_ERR_IO, "Failed to get file size: %s", name);
        fclose(file);
        return CIF_ERR_IO;
    }

    /* Enforce maximum file size limit */
    if (size > MAX_CIF_FILE_SIZE) {
        CIF_SET_ERROR(ctx, CIF_ERR_IO,
            "File too large: %ld bytes (max %ld): %s",
            size, MAX_CIF_FILE_SIZE, name);
        fclose(file);
        return CIF_ERR_IO;
    }

    if (fseek(file, 0, SEEK_SET) != 0) {
        CIF_SET_ERROR(ctx, CIF_ERR_IO, "Failed to seek to start of file: %s", name);
        fclose(file);
        return CIF_ERR_IO;
    }

    /* Allocate buffer */
    char *buf = malloc((size_t)size + 1);
    if (buf == NULL) {
        CIF_SET_ERROR(ctx, CIF_ERR_ALLOC,
            "Failed to allocate %ld bytes for file: %s", size + 1, name);
        fclose(file);
        return CIF_ERR_ALLOC;
    }

    /* Read file contents */
    size_t bytes_read = fread(buf, 1, (size_t)size, file);
    if (bytes_read != (size_t)size) {
        CIF_SET_ERROR(ctx, CIF_ERR_IO,
            "Failed to read file (expected %ld bytes, got %zu): %s",
            size, bytes_read, name);
        free(buf);
        fclose(file);
        return CIF_ERR_IO;
    }

    buf[size] = '\0';
    fclose(file);

    *buffer = buf;
    return CIF_OK;
}


/**
 * @brief Simple line advancement without tracking (for internal use).
 *
 * Used when scanning already-parsed block headers where line tracking
 * is not needed. For tracked line advancement, use CURSOR_NEXT_LINE macro.
 */
static inline void _advance_line_simple(char **buffer) {
    while (**buffer != '\n' && **buffer != '\0') (*buffer)++;
    if (**buffer == '\n') (*buffer)++;
}


int _get_offset(char *buffer, char delimiter, int n) {

    int offset = 0;

    /* Delimiters within single quotes are ignored.
     * Single quotes within double quotes are ignored. */
    bool squotes = false;
    bool dquotes = false;

    for (int ix = 0; ix < n; ix++) {
        while ((*buffer != delimiter && *buffer != '\n' && *buffer != '\0') || squotes) {
            if (*buffer == '\'' && !dquotes) { squotes = !squotes; }
            if (*buffer == '\"') { dquotes = !dquotes; }
            buffer++;
            offset++;
        }
        while (*buffer == delimiter) {
            buffer++;
            offset++;
        }
    }

    return offset;
}


int *_get_offsets(char *buffer, int fields, CifErrorContext *ctx) {

    int *offsets = calloc((size_t)(fields + 1), sizeof(int));
    if (offsets == NULL) {
        if (ctx != NULL) {
            CIF_SET_ERROR(ctx, CIF_ERR_ALLOC,
                "Failed to allocate offset array for %d fields", fields + 1);
        }
        return NULL;
    }

    for (int ix = 0; ix <= fields; ix++) {
        offsets[ix] = _get_offset(buffer, ' ', ix);
    }

    return offsets;
}


char *_get_field(char *buffer, CifErrorContext *ctx) {

    /* Skip leading whitespace */
    while (*buffer == ' ') { buffer++; }

    /* Read until whitespace, handling quotes.
     * Single quotes toggle quote mode (ignore spaces within).
     * Double quotes affect single quote interpretation. */
    bool squotes = false;
    bool dquotes = false;

    char *start = buffer;
    while ((*buffer != ' ' && *buffer != '\n' && *buffer != '\0') || squotes) {
        if (*buffer == '\'' && !dquotes) { squotes = !squotes; }
        if (*buffer == '\"') { dquotes = !dquotes; }
        buffer++;
    }

    size_t length = (size_t)(buffer - start);
    return _strdup_n(start, length, ctx);
}


char *_get_field_and_advance(char **buffer, CifErrorContext *ctx) {

    /* Skip leading whitespace */
    while (**buffer == ' ') { (*buffer)++; }

    /* Read until whitespace or end of line */
    char *start = *buffer;
    while (**buffer != ' ' && **buffer != '\n' && **buffer != '\0') { (*buffer)++; }

    size_t length = (size_t)(*buffer - start);
    return _strdup_n(start, length, ctx);
}


char *_get_category(char *buffer, CifErrorContext *ctx) {

    char *pos = strchr(buffer, '.');
    if (pos == NULL) {
        if (ctx != NULL) {
            CIF_SET_ERROR(ctx, CIF_ERR_PARSE,
                "Invalid attribute format (missing '.'): %.50s", buffer);
        }
        return NULL;
    }

    size_t length = (size_t)(pos - buffer + 1);

    char *result = malloc(length + 1);
    if (result == NULL) {
        if (ctx != NULL) {
            CIF_SET_ERROR(ctx, CIF_ERR_ALLOC,
                "Failed to allocate %zu bytes for category", length + 1);
        }
        return NULL;
    }

    strncpy(result, buffer, length);
    result[length - 1] = '.';
    result[length] = '\0';

    return result;
}


char *_get_attr(char *buffer, CifErrorContext *ctx) {

    char *start = strchr(buffer, '.');
    if (start == NULL) {
        if (ctx != NULL) {
            CIF_SET_ERROR(ctx, CIF_ERR_PARSE,
                "Invalid attribute format (missing '.'): %.50s", buffer);
        }
        return NULL;
    }
    start++;  /* Skip the '.' */

    char *end = strchr(start, ' ');
    if (end == NULL) {
        /* Attribute extends to end of string - find newline or end */
        end = strchr(start, '\n');
        if (end == NULL) {
            end = start + strlen(start);
        }
    }

    size_t length = (size_t)(end - start);
    return _strdup_n(start, length, ctx);
}


int _get_attr_index(mmBlock *block, const char *attr, CifErrorContext *ctx) {

    char *ptr = block->head;
    const char *name = block->category ? block->category : "unknown";

    for (int ix = 0; ix < block->attributes; ix++) {
        char *curr = _get_attr(ptr, ctx);
        if (curr != NULL) {
            bool match = _eq(curr, attr);
            free(curr);
            if (match) { return ix; }
        } else {
            /* Log allocation failure but continue searching */
            LOG_DEBUG("_get_attr_index: failed to get attr %d in block '%s'", ix, name);
        }
        _advance_line_simple(&ptr);
    }

    return BAD_IX;
}


int _str_to_int(const char *str) {

    errno = 0;
    char *endptr = NULL;

    long val = strtol(str, &endptr, 10);

    /* Check for parse errors and overflow */
    if (*endptr != '\0' || errno == ERANGE ||
        val < INT_MIN || val > INT_MAX) {
        return -1;
    }

    return (int)val;
}


/* ─────────────────────────────────────────────────────────────────────────────
 * Inline parsing functions (no allocation, cache-friendly)
 * ───────────────────────────────────────────────────────────────────────────── */

CifError _scan_lines(mmBlock *block, CifErrorContext *ctx) {
    /* Count lines first by scanning for newlines */
    int count = 0;
    char *ptr = block->data.ptr;

    while (*ptr != '\0' && !_is_section_end(ptr)) {
        count++;
        /* Advance to next line */
        while (*ptr != '\n' && *ptr != '\0') ptr++;
        if (*ptr == '\n') ptr++;
    }

    block->end = ptr;
    block->size = count;

    if (count == 0) {
        block->lines = NULL;
        return CIF_OK;
    }

    /* Allocate line pointer array */
    block->lines = malloc((size_t)count * sizeof(char *));
    if (block->lines == NULL) {
        CIF_SET_ERROR(ctx, CIF_ERR_ALLOC,
            "Failed to allocate line pointers for %d lines", count);
        return CIF_ERR_ALLOC;
    }

    /* Second pass: populate pointers */
    ptr = block->data.ptr;
    for (int i = 0; i < count; i++) {
        block->lines[i] = ptr;
        while (*ptr != '\n' && *ptr != '\0') ptr++;
        if (*ptr == '\n') ptr++;
    }

    return CIF_OK;
}


CifError _precompute_lines(mmBlock *block, CifErrorContext *ctx) {
    const char *name = block->category ? block->category : "unknown";

    LOG_DEBUG("Precomputing lines for block '%s' (size=%d, single=%d, var_width=%d)",
              name, block->size, block->single, block->variable_width);

    if (block->single || block->size <= 0) {
        LOG_DEBUG("Skipping line precomputation for '%s': single=%d, size=%d",
                  name, block->single, block->size);
        block->lines = NULL;
        return CIF_OK;
    }

    /* For variable-width blocks, lines are already populated by _scan_lines() */
    if (block->variable_width) {
        if (block->lines == NULL) {
            CIF_SET_ERROR(ctx, CIF_ERR_PARSE,
                "Variable-width block '%s' missing line pointers", name);
            return CIF_ERR_PARSE;
        }
        LOG_DEBUG("Variable-width block '%s' already has %d line pointers",
                  name, block->size);
        return CIF_OK;
    }

    /* Fixed-width: compute from width */
    block->lines = malloc((size_t)block->size * sizeof(char *));
    if (block->lines == NULL) {
        CIF_SET_ERROR(ctx, CIF_ERR_ALLOC,
            "Failed to allocate line pointers for %d lines in '%s'", block->size, name);
        return CIF_ERR_ALLOC;
    }

    for (int i = 0; i < block->size; i++) {
        block->lines[i] = block->data.ptr + i * block->width;
    }

    LOG_DEBUG("Precomputed %d line pointers for '%s' (width=%d)",
              block->size, name, block->width);
    return CIF_OK;
}


void _free_lines(mmBlock *block) {
    if (block->lines != NULL) {
        free(block->lines);
        block->lines = NULL;
    }
}


char *_get_field_ptr(mmBlock *block, int line, int index, size_t *len) {
    const char *name = block->category ? block->category : "unknown";

    /* Handle single-value blocks (no loop_, values inline with attributes) */
    if (block->single) {
        if (line != 0) {
            LOG_DEBUG("_get_field_ptr: single block '%s' only has line 0, got %d",
                      name, line);
            return NULL;
        }
        if (index < 0 || index >= block->attributes) {
            LOG_DEBUG("_get_field_ptr: attr %d out of bounds [0, %d) for single block '%s'",
                      index, block->attributes, name);
            return NULL;
        }

        /* Navigate to the attribute line */
        char *ptr = block->head;
        for (int ix = 0; ix < index; ix++) {
            while (*ptr != '\n' && *ptr != '\0') ptr++;
            if (*ptr == '\n') ptr++;
        }

        /* Skip the attribute name (first field) to get to the value */
        while (*ptr != ' ' && *ptr != '\n' && *ptr != '\0') ptr++;
        while (*ptr == ' ') ptr++;

        if (len != NULL) {
            char *end = ptr;
            bool squotes = false;
            bool dquotes = false;
            while ((*end != ' ' && *end != '\n' && *end != '\0') || squotes) {
                if (*end == '\'' && !dquotes) squotes = !squotes;
                if (*end == '"') dquotes = !dquotes;
                end++;
            }
            *len = (size_t)(end - ptr);
        }
        return ptr;
    }

    if (block->lines == NULL) {
        LOG_DEBUG("_get_field_ptr: lines=NULL for block '%s' (size=%d, single=%d)",
                  name, block->size, block->single);
        return NULL;
    }

    /* Bounds validation */
    if (line < 0 || line >= block->size) {
        LOG_DEBUG("_get_field_ptr: line %d out of bounds [0, %d) for block '%s'",
                  line, block->size, name);
        return NULL;
    }
    if (index < 0 || index >= block->attributes) {
        LOG_DEBUG("_get_field_ptr: attr %d out of bounds [0, %d) for block '%s'",
                  index, block->attributes, name);
        return NULL;
    }

    char *ptr;
    if (block->variable_width) {
        /* Variable-width: calculate offset for this specific line */
        char *line_start = block->lines[line];
        int offset = _get_offset(line_start, ' ', index);
        ptr = line_start + offset;
    } else {
        /* Fixed-width: use precomputed offsets */
        ptr = block->lines[line] + block->offsets[index];
    }

    /* Skip leading whitespace */
    while (*ptr == ' ') ptr++;

    if (len != NULL) {
        /* Calculate field length (until whitespace or newline) */
        char *end = ptr;
        bool squotes = false;
        bool dquotes = false;

        while ((*end != ' ' && *end != '\n' && *end != '\0') || squotes) {
            if (*end == '\'' && !dquotes) squotes = !squotes;
            if (*end == '"') dquotes = !dquotes;
            end++;
        }
        *len = (size_t)(end - ptr);
    }

    return ptr;
}


float _parse_float_inline(mmBlock *block, int line, int index) {

    char *ptr = _get_field_ptr(block, line, index, NULL);
    if (ptr == NULL) return 0.0f / 0.0f;  /* NaN */

    return strtof(ptr, NULL);
}


/* ============================================================================
 * FAST BATCH PARSING
 *
 * Optimized parsing functions for batch atom data extraction.
 * These functions are designed for the hot path where we parse the same
 * fields from thousands of rows. Key optimizations:
 *
 * 1. Compute line_start once per row (not per field)
 * 2. Direct pointer arithmetic with precomputed offsets
 * 3. Custom float parser avoiding locale/error handling overhead
 * 4. Inline field length calculation (no separate function call)
 * 5. memcpy for string copies instead of byte-by-byte loops
 * ============================================================================ */

/* _fast_parse_float and _fast_get_field are now defined as static inline in io.h */


/**
 * @brief Parse 3 coordinates (x,y,z) with single line_start computation.
 */
void _parse_coords_inline(mmBlock *block, int line, const int *idx, float *out) {
    if (block->lines != NULL && !block->variable_width) {
        char *line_start = block->lines[line];
        const int *offsets = block->offsets;

        char *ptr = line_start + offsets[idx[0]];
        while (*ptr == ' ') ptr++;
        out[0] = _fast_parse_float(ptr);

        ptr = line_start + offsets[idx[1]];
        while (*ptr == ' ') ptr++;
        out[1] = _fast_parse_float(ptr);

        ptr = line_start + offsets[idx[2]];
        while (*ptr == ' ') ptr++;
        out[2] = _fast_parse_float(ptr);
    } else {
        out[0] = _parse_float_inline(block, line, idx[0]);
        out[1] = _parse_float_inline(block, line, idx[1]);
        out[2] = _parse_float_inline(block, line, idx[2]);
    }
}


/**
 * @brief Fast element lookup with single line_start computation.
 *
 * @param block Block containing atom data
 * @param line Row index
 * @param index Attribute index for element symbol
 * @param func Hash lookup function (_lookup_element)
 * @return Element index or PARSE_FAIL
 */
int _lookup_element_fast(mmBlock *block, int line, int index, HashTable func) {
    if (block->lines != NULL && !block->variable_width) {
        char *line_start = block->lines[line];
        size_t len;
        char *ptr = _fast_get_field(line_start, block->offsets, index, &len);

        if (len == 0 || len >= MAX_INLINE_BUFFER) return PARSE_FAIL;

        /* Strip quotes and copy to buffer */
        _strip_outer_quotes((const char **)&ptr, &len);

        char buffer[MAX_INLINE_BUFFER];
        memcpy(buffer, ptr, len);
        buffer[len] = '\0';

        struct _LOOKUP *lookup = func(buffer, len);
        if (lookup == NULL) {
            LOG_WARNING("Unknown element '%s' at line %d", buffer, block->data.line + line);
            return PARSE_FAIL;
        }
        return lookup->value;
    }
    return _lookup_inline(block, line, index, func);
}


/**
 * @brief Fast atom type lookup (residue_atom) with single line_start computation.
 *
 * Combines two fields (comp_id, atom_id) into "COMP_ATOM" format for lookup.
 *
 * @param block Block containing atom data
 * @param line Row index
 * @param idx1 Attribute index for comp_id (residue name)
 * @param idx2 Attribute index for atom_id (atom name)
 * @param func Hash lookup function (_lookup_atom)
 * @param buffer Scratch buffer (must be MAX_INLINE_BUFFER size)
 * @return Atom type index or PARSE_FAIL
 */
int _lookup_atom_type_fast(mmBlock *block, int line, int idx1, int idx2,
                           HashTable func, char *buffer) {
    if (block->lines != NULL && !block->variable_width) {
        char *line_start = block->lines[line];
        const int *offsets = block->offsets;

        size_t len1, len2;
        char *ptr1 = _fast_get_field(line_start, offsets, idx1, &len1);
        char *ptr2 = _fast_get_field(line_start, offsets, idx2, &len2);

        if (len1 == 0 || len2 == 0) return PARSE_FAIL;
        if (len1 + 1 + len2 + 1 > MAX_INLINE_BUFFER) return PARSE_FAIL;

        /* Strip quotes */
        _strip_outer_quotes((const char **)&ptr1, &len1);
        _strip_outer_quotes((const char **)&ptr2, &len2);

        /* Build "COMP_ATOM" key with memcpy */
        memcpy(buffer, ptr1, len1);
        buffer[len1] = '_';
        memcpy(buffer + len1 + 1, ptr2, len2);
        size_t total_len = len1 + 1 + len2;
        buffer[total_len] = '\0';

        struct _LOOKUP *lookup = func(buffer, total_len);
        if (lookup == NULL) {
            LOG_WARNING("Unknown atom '%s' at line %d", buffer, block->data.line + line);
            return PARSE_FAIL;
        }
        return lookup->value;
    }
    return _lookup_double_inline(block, line, idx1, idx2, func, buffer);
}


int _parse_int_inline(mmBlock *block, int line, int index) {

    char *ptr = _get_field_ptr(block, line, index, NULL);
    if (ptr == NULL) return PARSE_FAIL;

    return (int)strtol(ptr, NULL, 10);
}


IntParseResult _parse_int_safe(mmBlock *block, int line, int index, int *result) {

    size_t len;
    char *ptr = _get_field_ptr(block, line, index, &len);
    if (ptr == NULL) return PARSE_INT_ERROR;

    /* Check for empty or missing value marker '.' */
    if (len == 0 || (len == 1 && ptr[0] == '.')) {
        return PARSE_INT_EMPTY;
    }

    /* Parse the integer */
    char *endptr = NULL;
    long val = strtol(ptr, &endptr, 10);

    /* Check if any digits were consumed */
    if (endptr == ptr) return PARSE_INT_EMPTY;

    *result = (int)val;
    return PARSE_INT_OK;
}


FloatParseResult _parse_float_safe(mmBlock *block, int line, int index, float *result) {

    size_t len;
    char *ptr = _get_field_ptr(block, line, index, &len);
    if (ptr == NULL) return PARSE_FLOAT_ERROR;

    /* Check for empty or missing value marker '.' */
    if (len == 0 || (len == 1 && ptr[0] == '.')) {
        return PARSE_FLOAT_EMPTY;
    }

    /* Parse the float */
    char *endptr = NULL;
    float val = strtof(ptr, &endptr);

    /* Check if any characters were consumed */
    if (endptr == ptr) return PARSE_FLOAT_EMPTY;

    *result = val;
    return PARSE_FLOAT_OK;
}


/**
 * @brief Copy field to buffer, stripping only outer quotes.
 *
 * CIF uses "..." or '...' to quote strings containing special characters.
 * For example, "C2'" is the string C2' (with an internal prime).
 * Uses _strip_outer_quotes() from io.h for the quote detection.
 *
 * @param ptr Source field pointer
 * @param len Length of source field
 * @param buffer Destination buffer
 * @param out_len Current position in buffer (updated)
 */
static void _copy_field_strip_outer_quotes(const char *ptr, size_t len,
                                            char *buffer, size_t *out_len) {
    _strip_outer_quotes(&ptr, &len);
    for (size_t i = 0; i < len; i++) {
        buffer[(*out_len)++] = ptr[i];
    }
}


int _lookup_inline(mmBlock *block, int line, int index, HashTable func) {

    size_t len;
    char *ptr = _get_field_ptr(block, line, index, &len);
    if (ptr == NULL || len == 0) return PARSE_FAIL;
    if (len + 1 > MAX_INLINE_BUFFER) return PARSE_FAIL;

    /* Copy field to buffer, stripping only outer quotes */
    char buffer[MAX_INLINE_BUFFER];
    size_t out_len = 0;
    _copy_field_strip_outer_quotes(ptr, len, buffer, &out_len);
    buffer[out_len] = '\0';

    struct _LOOKUP *lookup = func(buffer, out_len);
    if (lookup == NULL) {
        LOG_WARNING("Unknown residue '%s' at line %d", buffer, block->data.line + line);
        return PARSE_FAIL;
    }
    return lookup->value;
}


LookupResult _lookup_inline_safe(mmBlock *block, int line, int index,
                                  HashTable func, int *result) {

    size_t len;
    char *ptr = _get_field_ptr(block, line, index, &len);
    if (ptr == NULL) return LOOKUP_ERROR;           /* Field access failed */
    if (len == 0) return LOOKUP_NOT_FOUND;          /* Empty field */
    if (len + 1 > MAX_INLINE_BUFFER) return LOOKUP_ERROR;  /* Buffer overflow */

    /* Copy field to buffer, stripping only outer quotes */
    char buffer[MAX_INLINE_BUFFER];
    size_t out_len = 0;
    _copy_field_strip_outer_quotes(ptr, len, buffer, &out_len);
    buffer[out_len] = '\0';

    struct _LOOKUP *lookup = func(buffer, out_len);
    if (lookup == NULL) return LOOKUP_NOT_FOUND;    /* Not in hash table */

    *result = lookup->value;
    return LOOKUP_OK;
}


int _lookup_double_inline(mmBlock *block, int line, int index1, int index2,
                          HashTable func, char *buffer) {

    size_t len1, len2;
    char *ptr1 = _get_field_ptr(block, line, index1, &len1);
    char *ptr2 = _get_field_ptr(block, line, index2, &len2);

    if (ptr1 == NULL || ptr2 == NULL) return PARSE_FAIL;
    if (len1 == 0 || len2 == 0) return PARSE_FAIL;

    /* Check buffer overflow (need space for both fields + underscore + null) */
    if (len1 + 1 + len2 + 1 > MAX_INLINE_BUFFER) return PARSE_FAIL;

    /* Copy first field, stripping only outer quotes */
    size_t out_len = 0;
    _copy_field_strip_outer_quotes(ptr1, len1, buffer, &out_len);

    buffer[out_len++] = '_';

    /* Copy second field, stripping only outer quotes */
    _copy_field_strip_outer_quotes(ptr2, len2, buffer, &out_len);

    buffer[out_len] = '\0';

    struct _LOOKUP *lookup = func(buffer, out_len);
    if (lookup == NULL) {
        LOG_WARNING("Unknown atom '%s' at line %d", buffer, block->data.line + line);
        return PARSE_FAIL;
    }
    return lookup->value;
}


/* ============================================================================
 * TWO-POINTER REORDER PREPARATION
 * Pre-scan to classify atoms for direct placement during batch parsing.
 * ============================================================================ */

/**
 * @brief Pre-scan label_seq_id to build is_nonpoly mask.
 *
 * Classifies atoms as polymer or non-polymer based on residue membership:
 * - Polymer: has valid label_seq_id (>= 1)
 * - Non-polymer: missing or invalid label_seq_id
 *
 * This ensures polymer_count == sum(atoms_per_res) by definition,
 * since only atoms with valid residue sequence IDs are counted as polymer.
 * The group_PDB field (ATOM/HETATM) is ignored - residue membership is
 * the sole criterion.
 *
 * Enables two-pointer placement during batch parsing:
 * - Polymer atoms write to indices [0, polymer_count)
 * - Non-polymer atoms write to indices [polymer_count, total)
 *
 * @param block Atom block (must have lines pre-computed)
 * @param atoms Total atom count
 * @param is_nonpoly Output: non-polymer mask [atoms]
 * @param ctx Error context
 * @return Polymer count, or -1 on error
 */
int _prescan_group_pdb(mmBlock *block, int atoms, int *is_nonpoly,
                       CifErrorContext *ctx) {
    /* Get label_seq_id attribute index for residue membership check */
    int seq_idx = _get_attr_index(block, "label_seq_id", ctx);
    if (seq_idx == BAD_IX) {
        CIF_SET_ERROR(ctx, CIF_ERR_ATTR, "Missing attribute 'label_seq_id'");
        return -1;
    }

    /* Single pass: classify each atom as polymer or non-polymer
     * Polymer = has valid seq_id (>= 1), regardless of group_PDB */
    int nonpoly_count = 0;
    for (int row = 0; row < atoms; row++) {
        int seq_id;
        IntParseResult seq_result = _parse_int_safe(block, row, seq_idx, &seq_id);
        int has_valid_seq = (seq_result == PARSE_INT_OK && seq_id >= 1);

        /* Polymer requires only valid seq_id (residue membership) */
        is_nonpoly[row] = has_valid_seq ? 0 : 1;
        nonpoly_count += is_nonpoly[row];
    }

    return atoms - nonpoly_count;
}
