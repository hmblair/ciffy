/**
 * @file io.c
 * @brief Low-level I/O and block parsing utilities for mmCIF files.
 */

#include "io.h"
#include "../log.h"

#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <zlib.h>

/** Maximum decompressed file size (1GB) - prevents memory exhaustion attacks */
#define MAX_CIF_FILE_SIZE (1024L * 1024L * 1024L)

/** Initial buffer size for reading (1MB) */
#define INITIAL_BUFFER_SIZE (1024 * 1024)


CifError _load_file(const char *name, FileBuffer *fb, CifErrorContext *ctx) {
    fb->data = NULL;
    fb->size = 0;

    gzFile gz = gzopen(name, "rb");
    if (!gz) {
        CIF_SET_ERROR(ctx, CIF_ERR_IO, "Failed to open file: %s", name);
        return CIF_ERR_IO;
    }

    /* Start with 1MB buffer, grow as needed */
    size_t capacity = INITIAL_BUFFER_SIZE;
    char *buf = malloc(capacity);
    if (!buf) {
        CIF_SET_ERROR(ctx, CIF_ERR_ALLOC, "Failed to allocate buffer");
        gzclose(gz);
        return CIF_ERR_ALLOC;
    }

    size_t size = 0;
    int n;
    while ((n = gzread(gz, buf + size, (unsigned)(capacity - size - 1))) > 0) {
        size += (size_t)n;

        /* Check size limit */
        if (size > MAX_CIF_FILE_SIZE) {
            CIF_SET_ERROR(ctx, CIF_ERR_IO,
                "File too large: %zu bytes (max %ld): %s",
                size, MAX_CIF_FILE_SIZE, name);
            free(buf);
            gzclose(gz);
            return CIF_ERR_IO;
        }

        /* Grow buffer if needed */
        if (size >= capacity - 1) {
            capacity *= 2;
            char *new_buf = realloc(buf, capacity);
            if (!new_buf) {
                CIF_SET_ERROR(ctx, CIF_ERR_ALLOC,
                    "Failed to reallocate buffer to %zu bytes", capacity);
                free(buf);
                gzclose(gz);
                return CIF_ERR_ALLOC;
            }
            buf = new_buf;
        }
    }

    /* Check for read errors */
    int err;
    const char *err_msg = gzerror(gz, &err);
    if (err < 0 && err != Z_STREAM_END) {
        CIF_SET_ERROR(ctx, CIF_ERR_IO, "Failed to read file %s: %s", name, err_msg);
        free(buf);
        gzclose(gz);
        return CIF_ERR_IO;
    }

    gzclose(gz);

    /* Null-terminate */
    buf[size] = '\0';

    /* Shrink buffer to actual size */
    char *final_buf = realloc(buf, size + 1);
    if (final_buf) {
        buf = final_buf;
    }
    /* If realloc fails, keep the larger buffer - not a critical error */

    fb->data = buf;
    fb->size = size;

    LOG_DEBUG("Loaded file: %zu bytes", size);
    return CIF_OK;
}


void _free_file_buffer(FileBuffer *fb) {
    if (fb->data == NULL) return;

    free(fb->data);

    fb->data = NULL;
    fb->size = 0;
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
        while (*buffer != '\0' && ((*buffer != delimiter && *buffer != '\n') || squotes)) {
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


/**
 * @brief Calculate length of a potentially quoted field.
 *
 * Scans from ptr until field end, handling quoted strings properly.
 * Does not modify ptr. Fields end at whitespace unless inside quotes.
 *
 * @param ptr Start of field (may or may not be quoted)
 * @return Length of field including any outer quotes
 */
static inline size_t _field_length(const char *ptr) {
    const char *end = ptr;
    bool in_squotes = false;
    bool in_dquotes = false;

    while (*end != '\0' && ((*end != ' ' && *end != '\n') || in_squotes || in_dquotes)) {
        if (*end == '\'' && !in_dquotes) in_squotes = !in_squotes;
        if (*end == '"' && !in_squotes) in_dquotes = !in_dquotes;
        end++;
    }
    return (size_t)(end - ptr);
}


char *_get_field(char *buffer, CifErrorContext *ctx) {

    /* Skip leading whitespace */
    while (*buffer == ' ') { buffer++; }

    /* Calculate field length using unified helper (handles quotes) */
    size_t length = _field_length(buffer);
    return _strdup_n(buffer, length, ctx);
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


/* ============================================================================
 * QUOTE AND COLUMN DETECTION HELPERS
 *
 * These helpers consolidate common patterns for:
 * - Detecting column 0 position (for semicolon blocks)
 * - Detecting quote terminators (for quoted string fields)
 * - Skipping quoted fields
 *
 * Safety notes on pointer access:
 * - _at_column_zero: Uses short-circuit evaluation; ptr[-1] only accessed
 *   when ptr > data_start, which guarantees ptr[-1] is valid.
 * - _is_terminating_quote: Only called when *ptr is a quote character
 *   (verified by caller), and buffer is null-terminated, so ptr[1] always
 *   exists (at minimum it's the '\0' terminator).
 * ============================================================================ */

/**
 * @brief Check if position is at column 0 (start of line).
 *
 * A position is at column 0 if it's at the start of the data section
 * or immediately follows a newline character.
 *
 * SAFETY: Uses short-circuit evaluation. ptr[-1] is only accessed when
 * ptr > data_start, ensuring the access is within buffer bounds.
 *
 * @param ptr Current position in buffer
 * @param data_start Start of data section (first valid position)
 * @return true if at column 0
 */
static inline bool _at_column_zero(const char *ptr, const char *data_start) {
    return ptr == data_start || (ptr > data_start && ptr[-1] == '\n');
}

/**
 * @brief Check if current quote character terminates a field.
 *
 * In CIF, a quoted field ends when the closing quote is followed by
 * whitespace (space, tab, newline) or end of string. This allows
 * quotes to appear inside quoted strings when not at field boundaries.
 *
 * SAFETY: This function accesses ptr[1]. It should only be called when
 * *ptr is a quote character (verified by the caller's condition check).
 * Since the buffer is null-terminated and *ptr != '\0' (it's a quote),
 * there is always at least one more character at ptr[1] (the '\0' if
 * nothing else).
 *
 * @param ptr Pointer to a quote character (must not be at '\0')
 * @return true if the quote terminates the field
 */
static inline bool _is_terminating_quote(const char *ptr) {
    char next = ptr[1];
    return next == ' ' || next == '\t' || next == '\n' || next == '\0';
}

/**
 * @brief Skip past a quoted string field.
 *
 * Advances ptr from opening quote to just past the closing quote.
 * The closing quote is identified by being followed by whitespace or '\0'.
 *
 * @param ptr Pointer to opening quote (modified in place to after closing quote)
 */
static inline void _skip_quoted_string(char **ptr) {
    char quote = *(*ptr)++;  /* Save quote char and skip it */
    while (**ptr != '\0') {
        if (**ptr == quote && _is_terminating_quote(*ptr)) {
            (*ptr)++;  /* Skip closing quote */
            return;
        }
        (*ptr)++;
    }
    /* Unterminated quote - ptr is at '\0' */
}


/**
 * @brief Extract content from a semicolon multi-line value.
 *
 * Given a pointer to the opening ';', returns a pointer to the content
 * and its length. The content excludes the opening/closing ';' lines.
 *
 * @param ptr Pointer to opening ';' at column 0
 * @param data_start Start of data section (for column 0 detection)
 * @param len Output: content length
 * @return Pointer to content start
 */
static char *_extract_semicolon_content(char *ptr, const char *data_start, size_t *len) {
    /* Move past the opening ';' to the content */
    char *content_start = ptr + 1;

    /* Find end of opening line (content may start here) */
    while (*content_start == ' ' || *content_start == '\t') content_start++;
    char *line_end = content_start;
    while (*line_end != '\n' && *line_end != '\0') line_end++;

    /* Check if opening line has content or if content starts on next line */
    if (content_start == line_end) {
        /* No content on opening line, start from next line */
        if (*line_end == '\n') content_start = line_end + 1;
    }

    /* Find closing ';' at column 0 */
    char *scan = content_start;
    char *content_end = content_start;
    while (*scan != '\0') {
        if (*scan == ';' && _at_column_zero(scan, data_start)) {
            /* Found closing semicolon */
            content_end = scan - 1;  /* Exclude the newline before closing ';' */
            if (content_end < content_start) content_end = content_start;
            break;
        }
        scan++;
    }

    if (len != NULL) {
        *len = (size_t)(content_end - content_start);
    }
    return content_start;
}


/**
 * @brief Skip over a semicolon text block in CIF data.
 *
 * CIF uses lines starting with ';' to delimit multi-line text values.
 * This function advances the pointer past the entire block.
 *
 * @param ptr Pointer to ';' at start of block (modified in place)
 * @return true if block was properly terminated, false if unterminated
 */
static bool _skip_semicolon_block(char **ptr) {
    /* Skip the opening ';' line */
    while (**ptr != '\n' && **ptr != '\0') (*ptr)++;
    if (**ptr == '\n') (*ptr)++;

    /* Skip until we find a line starting with ';' (block end) */
    int lines = 0;
    const int MAX_LINES = 10000;
    while (**ptr != '\0' && **ptr != ';' && lines < MAX_LINES) {
        while (**ptr != '\n' && **ptr != '\0') (*ptr)++;
        if (**ptr == '\n') (*ptr)++;
        lines++;
    }

    if (**ptr == ';') {
        /* Skip the closing ';' line */
        while (**ptr != '\n' && **ptr != '\0') (*ptr)++;
        if (**ptr == '\n') (*ptr)++;
        return true;
    }
    return false;
}


/**
 * @brief Skip a single field value, handling quotes and semicolon blocks.
 *
 * Advances ptr past one field. Fields may be:
 * - Regular unquoted values (whitespace-delimited)
 * - Quoted strings ('...' or "...")
 * - Semicolon multi-line blocks (when ';' is at column 0)
 *
 * @param ptr Pointer to current position (modified in place)
 * @param data_start Start of data section (for column 0 detection)
 * @return true if field was found and skipped, false if at section end/EOF
 */
static bool _skip_field(char **ptr, const char *data_start) {
    /* Skip leading whitespace (not newlines) */
    while (**ptr == ' ' || **ptr == '\t') (*ptr)++;

    /* Check for newline - field might be on next line */
    if (**ptr == '\n') {
        (*ptr)++;
        /* Check if next line starts with semicolon block */
        if (**ptr == ';') {
            return _skip_semicolon_block(ptr);
        }
        /* Continue looking for field on new line */
        return _skip_field(ptr, data_start);
    }

    /* Check for semicolon block at column 0 */
    if (**ptr == ';' && _at_column_zero(*ptr, data_start)) {
        return _skip_semicolon_block(ptr);
    }

    /* End of data */
    if (**ptr == '\0' || _is_section_end(*ptr)) {
        return false;
    }

    /* Quoted string - use helper for safe quote handling */
    if (**ptr == '\'' || **ptr == '"') {
        _skip_quoted_string(ptr);
        return true;
    }

    /* Regular unquoted field */
    while (**ptr != '\0' && **ptr != ' ' && **ptr != '\t' && **ptr != '\n') {
        (*ptr)++;
    }
    return true;
}


char *_find_block_end(char *start) {
    char *p = start;

    /* Scan for '#' at start of line (section end marker) */
    while (*p != '\0') {
        if (*p == '#') {
            return p;
        }
        /* Skip to end of line */
        while (*p != '\n' && *p != '\0') p++;
        if (*p == '\n') p++;
    }
    return p;
}


int _count_lines_fast(const char *start, const char *end) {
    int count = 0;
    const char *p = start;
    size_t remaining = (size_t)(end - start);

    while (remaining > 0) {
        const char *nl = memchr(p, '\n', remaining);
        if (nl == NULL) break;
        count++;
        remaining -= (size_t)(nl - p + 1);
        p = nl + 1;
    }
    return count;
}


CifError _scan_lines(mmBlock *block, CifErrorContext *ctx) {
    const char *name = block->category ? block->category : "unknown";
    int num_attrs = block->attributes;
    char *data_start = block->data.ptr;

    /*
     * Count logical rows. Each row consists of exactly num_attrs fields.
     * Fields may span multiple physical lines when using semicolon blocks.
     */
    int count = 0;
    char *ptr = data_start;

    while (*ptr != '\0' && !_is_section_end(ptr)) {
        /* Skip leading whitespace and empty lines */
        while (*ptr == ' ' || *ptr == '\t' || *ptr == '\n') ptr++;

        if (*ptr == '\0' || _is_section_end(ptr)) break;

        /* Found start of a row - consume num_attrs fields */
        int fields = 0;
        while (fields < num_attrs) {
            if (!_skip_field(&ptr, data_start)) {
                /* Unexpected end - partial row */
                if (fields > 0) {
                    LOG_WARNING("Partial row in %s: got %d/%d fields",
                                name, fields, num_attrs);
                }
                goto done_counting;
            }
            fields++;
        }
        count++;
    }

done_counting:
    block->end = ptr;
    block->size = count;

    if (count == 0) {
        block->lines = NULL;
        return CIF_OK;
    }

    /* Allocate row pointer array */
    block->lines = malloc((size_t)count * sizeof(char *));
    if (block->lines == NULL) {
        CIF_SET_ERROR(ctx, CIF_ERR_ALLOC,
            "Failed to allocate row pointers for %d rows", count);
        return CIF_ERR_ALLOC;
    }

    /* Second pass: populate row start pointers */
    ptr = data_start;
    for (int i = 0; i < count; i++) {
        /* Skip leading whitespace and empty lines */
        while (*ptr == ' ' || *ptr == '\t' || *ptr == '\n') ptr++;

        block->lines[i] = ptr;

        /* Skip num_attrs fields to advance to next row */
        for (int f = 0; f < num_attrs; f++) {
            _skip_field(&ptr, data_start);
        }
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
            *len = _field_length(ptr);
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

    /* Fixed-width blocks are the common case (atom_site), check first */
    if (__builtin_expect(!block->variable_width, 1)) {
        /* Fixed-width: use precomputed offsets (fast path) */
        ptr = block->lines[line] + block->offsets[index];
        while (*ptr == ' ') ptr++;

        if (len != NULL) {
            /* Inline length calculation for performance */
            char *end = ptr;
            bool squotes = false;
            bool dquotes = false;
            while (*end != '\0' && ((*end != ' ' && *end != '\n') || squotes)) {
                if (*end == '\'' && !dquotes) squotes = !squotes;
                if (*end == '"') dquotes = !dquotes;
                end++;
            }
            *len = (size_t)(end - ptr);
        }
        return ptr;
    }

    /* Variable-width: navigate field-by-field (slow path, rare) */
    {
        /*
         * Variable-width: navigate field-by-field to handle multi-line values.
         * Cannot use simple offset calculation because rows may span multiple
         * physical lines when containing semicolon blocks.
         */
        const char *data_start = block->data.ptr;
        ptr = block->lines[line];

        /* Skip to the target field */
        for (int f = 0; f < index; f++) {
            /* Skip whitespace */
            while (*ptr == ' ' || *ptr == '\t') ptr++;

            /* Handle newline - next field might be on next line */
            while (*ptr == '\n') {
                ptr++;
                while (*ptr == ' ' || *ptr == '\t') ptr++;
            }

            /* Skip this field */
            if (*ptr == ';' && _at_column_zero(ptr, data_start)) {
                /* Semicolon block - skip entire block */
                while (*ptr != '\n' && *ptr != '\0') ptr++;  /* Skip opening line */
                if (*ptr == '\n') ptr++;
                while (*ptr != '\0' && !(*ptr == ';' && _at_column_zero(ptr, data_start))) {
                    while (*ptr != '\n' && *ptr != '\0') ptr++;
                    if (*ptr == '\n') ptr++;
                }
                if (*ptr == ';') {
                    while (*ptr != '\n' && *ptr != '\0') ptr++;  /* Skip closing line */
                    if (*ptr == '\n') ptr++;
                }
            } else if (*ptr == '\'' || *ptr == '"') {
                /* Quoted string - use helper for safe terminator detection */
                _skip_quoted_string(&ptr);
            } else {
                /* Regular field */
                while (*ptr != '\0' && *ptr != ' ' && *ptr != '\t' && *ptr != '\n') {
                    ptr++;
                }
            }
        }

        /* Skip whitespace before target field */
        while (*ptr == ' ' || *ptr == '\t') ptr++;
        while (*ptr == '\n') {
            ptr++;
            while (*ptr == ' ' || *ptr == '\t') ptr++;
        }

        /*
         * Handle semicolon multi-line value (variable-width only).
         * Semicolon blocks start with ';' at column 0 (start of line).
         * Fixed-width blocks (atom_site) never contain semicolon values.
         */
        if (*ptr == ';' && _at_column_zero(ptr, data_start)) {
            return _extract_semicolon_content(ptr, data_start, len);
        }

        /* Regular field in variable-width block */
        if (len != NULL) {
            *len = _field_length(ptr);
        }
        return ptr;
    }
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
            LOG_INFO("Unknown element '%s' at line %d", buffer, block->data.line + line);
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
            LOG_INFO("Unknown atom '%s' at line %d", buffer, block->data.line + line);
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
        LOG_INFO("Unknown residue '%s' at line %d", buffer, block->data.line + line);
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
        LOG_INFO("Unknown atom '%s' at line %d", buffer, block->data.line + line);
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

/**
 * @brief Pre-scan label_alt_id to filter alternate conformations.
 *
 * Marks atoms for exclusion if their alt_id doesn't match the desired
 * conformation. Atoms with '.' (no alternate) are always kept.
 *
 * @param block Atom block (must have lines pre-computed)
 * @param atoms Total atom count
 * @param is_excluded Exclusion mask to update (modified in place)
 * @param keep_alt Alt conformation to keep ('A', 'B', etc.), '\0' = keep all
 * @param ctx Error context
 * @return Number of atoms excluded, or -1 on error
 */
int _prescan_alt_locs(mmBlock *block, int atoms, int *is_excluded,
                      char keep_alt, CifErrorContext *ctx) {
    /* If no alt filtering requested, skip */
    if (keep_alt == '\0') {
        return 0;
    }

    /* Get label_alt_id attribute index */
    int alt_idx = _get_attr_index(block, "label_alt_id", ctx);
    if (alt_idx == BAD_IX) {
        /* No alt_id column - nothing to filter */
        return 0;
    }

    int excluded = 0;
    for (int row = 0; row < atoms; row++) {
        if (is_excluded[row]) continue;

        size_t alt_len;
        char *alt = _get_field_ptr(block, row, alt_idx, &alt_len);

        /* Keep '.' (no alternate) and matching alt_id */
        if (alt_len == 1 && alt[0] != '.' && alt[0] != keep_alt) {
            is_excluded[row] = 1;
            excluded++;
        }
    }

    LOG_DEBUG("Alt loc filtering: excluded %d atoms (keeping '%c')", excluded, keep_alt);
    return excluded;
}


/**
 * @brief Unified prescan: classify atoms and apply filters in a single pass.
 *
 * Combines polymer classification, model filtering, and alt_loc filtering
 * into a single loop over atoms, reducing from 3 passes to 1.
 *
 * For each atom:
 * - Polymer classification: has valid label_seq_id (>= 1)
 * - Model filtering: pdbx_PDB_model_num matches target_model (if specified)
 * - Alt loc filtering: label_alt_id matches keep_alt or is '.' (if specified)
 *
 * @param block Atom block (must have lines pre-computed)
 * @param atoms Total atom count
 * @param is_nonpoly Output: non-polymer mask [atoms] (required)
 * @param is_excluded Output: exclusion mask [atoms] (may be NULL if no filtering)
 * @param target_model Target model number (0 or -1 = no model filtering)
 * @param keep_alt Alt conformation to keep ('A', 'B', etc.), '\0' = keep all
 * @param ctx Error context
 * @return PrescanResult with polymer_count, excluded_count, and error status
 */
PrescanResult _prescan_unified(mmBlock *block, int atoms,
                               int *is_nonpoly, int *is_excluded,
                               int target_model, char keep_alt,
                               CifErrorContext *ctx) {
    PrescanResult result = {0, 0, CIF_OK};

    /* Get attribute indices (once for all atoms) */
    int seq_idx = _get_attr_index(block, "label_seq_id", ctx);
    if (seq_idx == BAD_IX) {
        CIF_SET_ERROR(ctx, CIF_ERR_ATTR, "Missing attribute 'label_seq_id'");
        result.error = CIF_ERR_ATTR;
        return result;
    }

    /* Optional attributes for filtering */
    int model_idx = BAD_IX;
    if (target_model > 0) {
        model_idx = _get_attr_index(block, "pdbx_PDB_model_num", ctx);
        /* Missing model column in single-model file is OK - no filtering needed */
    }

    int alt_idx = BAD_IX;
    if (keep_alt != '\0') {
        alt_idx = _get_attr_index(block, "label_alt_id", ctx);
        /* Missing alt column is OK - no alternate conformations */
    }

    /* Single pass over all atoms */
    int nonpoly_count = 0;
    int excluded_count = 0;

    for (int row = 0; row < atoms; row++) {
        /* 1. Polymer classification: check seq_id */
        int seq_id;
        IntParseResult seq_result = _parse_int_safe(block, row, seq_idx, &seq_id);
        int has_valid_seq = (seq_result == PARSE_INT_OK && seq_id >= 1);

        is_nonpoly[row] = has_valid_seq ? 0 : 1;
        if (is_nonpoly[row]) nonpoly_count++;

        /* Skip exclusion checks if not filtering or already excluded */
        if (is_excluded == NULL) continue;

        /* 2. Model filtering: check model_num */
        if (model_idx != BAD_IX) {
            int model_num = _parse_int_inline(block, row, model_idx);
            if (model_num != target_model) {
                is_excluded[row] = 1;
                excluded_count++;
                continue;  /* Already excluded, skip alt check */
            }
        }

        /* 3. Alt loc filtering: check alt_id */
        if (alt_idx != BAD_IX) {
            size_t alt_len;
            char *alt = _get_field_ptr(block, row, alt_idx, &alt_len);

            /* Exclude if alt_id is not '.' and doesn't match keep_alt */
            if (alt_len == 1 && alt[0] != '.' && alt[0] != keep_alt) {
                is_excluded[row] = 1;
                excluded_count++;
            }
        }
    }

    result.polymer_count = atoms - nonpoly_count;
    result.excluded_count = excluded_count;

    LOG_DEBUG("Unified prescan: %d polymer, %d excluded (model=%d, alt='%c')",
              result.polymer_count, excluded_count,
              target_model > 0 ? target_model : 0,
              keep_alt ? keep_alt : '.');

    return result;
}
