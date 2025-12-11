/**
 * @file io.c
 * @brief Low-level I/O and block parsing utilities for mmCIF files.
 */

#include "io.h"
#include "log.h"


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


void _advance_line(char **buffer) {

    while (**buffer != '\n' && **buffer != '\0') { (*buffer)++; }
    if (**buffer == '\n') { (*buffer)++; }

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


int _get_attr_index(mmBlock *block, const char *attr) {

    char *ptr = block->head;

    for (int ix = 0; ix < block->attributes; ix++) {
        char *curr = _get_attr(ptr, NULL);  /* Ignore allocation errors here */
        if (curr != NULL) {
            bool match = _eq(curr, attr);
            free(curr);
            if (match) { return ix; }
        }
        _advance_line(&ptr);
    }

    return BAD_IX;
}


char *_get_attr_by_line(mmBlock *block, int line, int index, CifErrorContext *ctx) {

    if (block->single) {

        char *ptr = block->head;
        for (int ix = 0; ix < index; ix++) {
            _advance_line(&ptr);
        }

        char *skip = _get_field_and_advance(&ptr, ctx);
        if (skip != NULL) { free(skip); }
        return _get_field_and_advance(&ptr, ctx);

    } else {

        /* Bounds check for multi-entry blocks */
        if (line < 0 || line >= block->size) {
            if (ctx != NULL) {
                CIF_SET_ERROR(ctx, CIF_ERR_BOUNDS,
                    "Line index %d out of bounds (size=%d)", line, block->size);
            }
            return NULL;
        }
        if (index < 0 || index >= block->attributes) {
            if (ctx != NULL) {
                CIF_SET_ERROR(ctx, CIF_ERR_BOUNDS,
                    "Attribute index %d out of bounds (attributes=%d)",
                    index, block->attributes);
            }
            return NULL;
        }

        char *ptr;
        if (block->variable_width) {
            /* Variable-width: use line pointers and calculate offset per-line */
            char *line_start = block->lines[line];
            int offset = _get_offset(line_start, ' ', index);
            ptr = line_start + offset;
        } else {
            /* Fixed-width: use precomputed offsets */
            ptr = block->start + line * block->width + block->offsets[index];
        }
        return _get_field(ptr, ctx);

    }
}


int _str_to_int(const char *str) {

    int base = 10;
    char *endptr = NULL;

    long val = strtol(str, &endptr, base);
    if (*endptr != '\0') { return -1; }

    return (int)val;
}


static inline char *_strip_quotes(char *str) {

    char *write_ptr = str;
    char *read_ptr = str;

    while (*read_ptr) {
        if (*read_ptr != '"') {
            *write_ptr = *read_ptr;
            write_ptr++;
        }
        read_ptr++;
    }
    *write_ptr = '\0';

    return str;
}


int _lookup(HashTable func, char *token) {

    token = _strip_quotes(token);
    struct _LOOKUP *lookup = func(token, strlen(token));

    if (lookup != NULL) {
        return lookup->value;
    }

    return -1;
}


CifError _lookup_safe(HashTable func, char *token, int *result, CifErrorContext *ctx) {

    token = _strip_quotes(token);
    struct _LOOKUP *lookup = func(token, strlen(token));

    if (lookup != NULL) {
        *result = lookup->value;
        return CIF_OK;
    }

    CIF_SET_ERROR(ctx, CIF_ERR_LOOKUP, "Unknown token: '%s'", token);
    return CIF_ERR_LOOKUP;
}


/* ─────────────────────────────────────────────────────────────────────────────
 * Inline parsing functions (no allocation, cache-friendly)
 * ───────────────────────────────────────────────────────────────────────────── */

CifError _scan_lines(mmBlock *block, CifErrorContext *ctx) {
    /* Count lines first by scanning for newlines */
    int count = 0;
    char *ptr = block->start;

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
    ptr = block->start;
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
        block->lines[i] = block->start + i * block->width;
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


int _parse_int_inline(mmBlock *block, int line, int index) {

    char *ptr = _get_field_ptr(block, line, index, NULL);
    if (ptr == NULL) return PARSE_FAIL;

    return (int)strtol(ptr, NULL, 10);
}


int _lookup_inline(mmBlock *block, int line, int index, HashTable func) {

    size_t len;
    char *ptr = _get_field_ptr(block, line, index, &len);
    if (ptr == NULL || len == 0) return PARSE_FAIL;
    if (len >= MAX_INLINE_BUFFER) return PARSE_FAIL;

    /* Copy to thread-local buffer for null-termination */
    char buffer[MAX_INLINE_BUFFER];
    memcpy(buffer, ptr, len);
    buffer[len] = '\0';

    /* Use existing lookup which handles quote stripping */
    return _lookup(func, buffer);
}


/**
 * @brief Copy field to buffer, stripping only outer quotes.
 *
 * CIF uses "..." to quote strings containing special characters.
 * For example, "C2'" is the string C2' (with an internal prime).
 * We must preserve internal ' characters while removing outer quotes.
 *
 * @param ptr Source field pointer
 * @param len Length of source field
 * @param buffer Destination buffer
 * @param out_len Current position in buffer (updated)
 */
static void _copy_field_strip_outer_quotes(const char *ptr, size_t len,
                                            char *buffer, size_t *out_len) {
    /* Check for outer double quotes: "..." */
    if (len >= 2 && ptr[0] == '"' && ptr[len - 1] == '"') {
        for (size_t i = 1; i < len - 1; i++) {
            buffer[(*out_len)++] = ptr[i];
        }
    }
    /* Check for outer single quotes: '...' */
    else if (len >= 2 && ptr[0] == '\'' && ptr[len - 1] == '\'') {
        for (size_t i = 1; i < len - 1; i++) {
            buffer[(*out_len)++] = ptr[i];
        }
    }
    /* No outer quotes - copy as-is */
    else {
        for (size_t i = 0; i < len; i++) {
            buffer[(*out_len)++] = ptr[i];
        }
    }
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
    return lookup != NULL ? lookup->value : PARSE_FAIL;
}
