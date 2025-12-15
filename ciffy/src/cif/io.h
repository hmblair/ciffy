#ifndef _CIFFY_IO_H
#define _CIFFY_IO_H

/**
 * @file io.h
 * @brief Low-level I/O and block parsing utilities for mmCIF files.
 *
 * Provides functions for loading files, parsing mmCIF blocks,
 * and extracting field values from the parsed structure.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#include "../error.h"
#include "../common.h"
#include "../lookup.h"
#include "../hash/reverse.h"

/** Sentinel value indicating attribute index not found */
#define BAD_IX -1

/** Sentinel value for failed inline parsing (aliased from reverse.h) */
#define PARSE_FAIL UNKNOWN_INDEX


/* ============================================================================
 * BLOCK REGISTRY
 * Block identifiers and category prefixes. Defined here (in io.h) because
 * both parser.h and registry.h need access to BLOCK_COUNT.
 * ============================================================================ */

/**
 * @brief Block definition list using X-macro pattern.
 *
 * This macro defines all mmCIF blocks in one place. It auto-generates:
 * - BlockId enum values
 * - BLOCKS[] array in registry.c
 *
 * Format: X(NAME, category_prefix, is_required)
 */
#define BLOCK_LIST \
    X(ATOM,           "_atom_site.",            true)  \
    X(POLY,           "_pdbx_poly_seq_scheme.", true)  \
    X(CHAIN,          "_struct_asym.",          true)  \
    X(NONPOLY,        "_pdbx_nonpoly_scheme.",  false) \
    X(CONN,           "_struct_conn.",          false) \
    X(ENTITY_POLY,    "_entity_poly.",          false) \
    X(ENTITY,         "_entity.",               false) \
    X(ENTITY_NONPOLY, "_pdbx_entity_nonpoly.",  false)

/**
 * @brief Block identifier enum.
 *
 * Each value corresponds to an mmCIF category block.
 * Auto-generated from BLOCK_LIST macro.
 */
typedef enum {
    #define X(name, category, required) BLOCK_##name,
    BLOCK_LIST
    #undef X
    BLOCK_COUNT        /**< Total number of block types */
} BlockId;


/* ============================================================================
 * PARSE CURSOR
 * Tracks buffer position and line number during parsing.
 * ============================================================================ */

/**
 * @brief Cursor for tracking position during CIF parsing.
 *
 * Bundles the buffer pointer with a line counter for accurate
 * error reporting with file line numbers. Used both for active parsing
 * and stored in blocks to enable line number calculation during extraction.
 */
typedef struct {
    char *ptr;    /**< Current position in buffer */
    int line;     /**< Current 1-indexed line number */
} ParseCursor;


/* ============================================================================
 * CURSOR MACROS
 * Centralized cursor operations that ensure line counting is always correct.
 * ALL code that advances past newlines MUST use these macros.
 * ============================================================================ */

/** @brief Skip to end of current line (stops AT the newline, doesn't pass it). */
#define CURSOR_SKIP_TO_EOL(cursor) do { \
    while (*(cursor)->ptr != '\n' && *(cursor)->ptr != '\0') (cursor)->ptr++; \
} while(0)

/** @brief Advance past a newline character, incrementing line counter. */
#define CURSOR_PASS_NEWLINE(cursor) do { \
    if (*(cursor)->ptr == '\n') { (cursor)->ptr++; (cursor)->line++; } \
} while(0)

/** @brief Advance to start of next line (skip to EOL, then pass newline). */
#define CURSOR_NEXT_LINE(cursor) do { \
    CURSOR_SKIP_TO_EOL(cursor); \
    CURSOR_PASS_NEWLINE(cursor); \
} while(0)

/** @brief Skip whitespace (spaces and tabs only, NOT newlines). */
#define CURSOR_SKIP_WHITESPACE(cursor) do { \
    while (*(cursor)->ptr == ' ' || *(cursor)->ptr == '\t') (cursor)->ptr++; \
} while(0)

/** @brief Check if cursor is at end of buffer. */
#define CURSOR_AT_END(cursor) (*(cursor)->ptr == '\0')

/** @brief Check if cursor is at a newline. */
#define CURSOR_AT_NEWLINE(cursor) (*(cursor)->ptr == '\n')

/** @brief Get absolute line number for a row within a block. */
#define CURSOR_LINE_AT(cursor, row) ((cursor)->line + (row))


/**
 * @brief Represents a parsed mmCIF block (loop or single-value).
 *
 * An mmCIF file contains multiple blocks, each with a category name
 * (e.g., "_atom_site.") and a set of attributes. Blocks can be either
 * single-entry (key-value pairs) or multi-entry (tabular data after "loop_").
 */
typedef struct {

    char *category;     /**< Block category (e.g., "_atom_site.") */
    int  attributes;    /**< Number of attributes/columns in the block */
    int  size;          /**< Number of data entries (rows) */
    int  width;         /**< Bytes per line (for fixed-width blocks, 0 for variable) */
    bool single;        /**< true if single-entry block, false if loop */
    bool variable_width;/**< true if lines have variable widths (fallback mode) */
    char *head;         /**< Pointer to start of header (attribute definitions) */
    ParseCursor data;   /**< Cursor marking start of data section (ptr + line number) */
    char *end;          /**< Pointer to end of data section (for variable-width) */
    int  *offsets;      /**< Column byte offsets (template from first line) */
    char **lines;       /**< Line start pointers (always populated for variable-width) */

} mmBlock;


/**
 * @brief Load an entire file into memory.
 *
 * Reads the complete contents of a file into a null-terminated buffer.
 * The caller is responsible for freeing the returned buffer.
 *
 * @param name Path to the file to load
 * @param buffer Output pointer to allocated buffer containing file contents
 * @param ctx Error context, populated on failure
 * @return CIF_OK on success, CIF_ERR_IO or CIF_ERR_ALLOC on failure
 */
CifError _load_file(const char *name, char **buffer, CifErrorContext *ctx);

/**
 * @brief Advance cursor to the next line.
 *
 * Convenience function wrapping CURSOR_NEXT_LINE macro.
 * Prefer using the macro directly in performance-critical code.
 *
 * @param cursor Pointer to parse cursor (modified in place)
 */
static inline void _advance_line(ParseCursor *cursor) {
    CURSOR_NEXT_LINE(cursor);
}

/**
 * @brief Calculate byte offset to the nth field.
 *
 * Counts delimiters while respecting quote escaping to find the
 * byte offset to the start of the nth field.
 *
 * @param buffer Start of the line to parse
 * @param delimiter Field delimiter character (usually space)
 * @param n Number of fields to skip (0-indexed)
 * @return Byte offset to the nth field
 */
int _get_offset(char *buffer, char delimiter, int n);

/**
 * @brief Calculate offsets for all fields in a line.
 *
 * @param buffer Start of the line to parse
 * @param fields Number of fields expected
 * @param ctx Error context, populated on failure
 * @return Array of field offsets (caller must free), or NULL on error
 */
int *_get_offsets(char *buffer, int fields, CifErrorContext *ctx);

/**
 * @brief Extract a field value from a buffer position.
 *
 * Parses a whitespace-delimited field, handling quoted strings.
 *
 * @param buffer Position to start parsing from
 * @param ctx Error context, populated on failure (may be NULL)
 * @return Allocated string containing field value, or NULL on error
 */
char *_get_field(char *buffer, CifErrorContext *ctx);

/**
 * @brief Extract a field and advance the buffer pointer.
 *
 * Like _get_field but also advances the buffer pointer past the field.
 *
 * @param buffer Pointer to buffer pointer (modified in place)
 * @param ctx Error context, populated on failure (may be NULL)
 * @return Allocated string containing field value, or NULL on error
 */
char *_get_field_and_advance(char **buffer, CifErrorContext *ctx);

/**
 * @brief Extract the category name from an attribute line.
 *
 * Parses "_category.attr" and returns "_category.".
 *
 * @param buffer Line containing the attribute
 * @param ctx Error context, populated on failure (may be NULL)
 * @return Allocated category string, or NULL on error
 */
char *_get_category(char *buffer, CifErrorContext *ctx);

/**
 * @brief Extract the attribute name from an attribute line.
 *
 * Parses "_category.attr" and returns "attr".
 *
 * @param buffer Line containing the attribute
 * @param ctx Error context, populated on failure (may be NULL)
 * @return Allocated attribute name string, or NULL on error
 */
char *_get_attr(char *buffer, CifErrorContext *ctx);

/**
 * @brief Find the index of an attribute in a block.
 *
 * Searches the block's attribute list for the given name.
 *
 * @param block The block to search
 * @param attr Attribute name to find
 * @param ctx Error context for logging (may be NULL)
 * @return Index of attribute (0-based), or BAD_IX if not found
 */
int _get_attr_index(mmBlock *block, const char *attr, CifErrorContext *ctx);

/**
 * @brief Convert a string to an integer.
 *
 * @param str String to parse
 * @return Parsed integer value, or -1 on parse error
 */
int _str_to_int(const char *str);

/** Function pointer type for gperf hash table lookup functions */
typedef struct _LOOKUP *(*HashTable)(const char *, size_t);

/**
 * @brief Result codes for lookup operations.
 *
 * Distinguishes between "value not found" (expected) and "error" (unexpected).
 * This prevents callers from conflating missing data with system failures.
 */
typedef enum {
    LOOKUP_OK = 0,           /**< Lookup succeeded, value is valid */
    LOOKUP_NOT_FOUND = 1,    /**< Value not in hash table (not an error) */
    LOOKUP_ERROR = -1        /**< Field access or buffer error */
} LookupResult;

/**
 * @brief Result codes for integer parsing operations.
 *
 * Distinguishes between parse errors and missing/empty fields.
 */
typedef enum {
    PARSE_INT_OK = 0,        /**< Parse succeeded, value is valid */
    PARSE_INT_EMPTY = 1,     /**< Field is empty or missing value (e.g., '.') */
    PARSE_INT_ERROR = -1     /**< Field access failed or parse error */
} IntParseResult;

/**
 * @brief Result codes for float parsing operations.
 *
 * Distinguishes between parse errors and missing/empty fields.
 */
typedef enum {
    PARSE_FLOAT_OK = 0,      /**< Parse succeeded, value is valid */
    PARSE_FLOAT_EMPTY = 1,   /**< Field is empty or missing value (e.g., '.') */
    PARSE_FLOAT_ERROR = -1   /**< Field access failed or parse error */
} FloatParseResult;

/* ─────────────────────────────────────────────────────────────────────────────
 * Inline parsing functions (no allocation, cache-friendly)
 * ───────────────────────────────────────────────────────────────────────────── */

/**
 * @brief Scan variable-width block for line boundaries.
 *
 * Populates block->lines by scanning for newlines. Sets block->size
 * and block->end. Used as fallback when fixed-width parsing fails.
 *
 * @param block Block to scan (start must be set)
 * @param ctx Error context, populated on failure
 * @return CIF_OK on success, CIF_ERR_ALLOC on failure
 */
CifError _scan_lines(mmBlock *block, CifErrorContext *ctx);

/**
 * @brief Pre-compute line pointers for a block.
 *
 * Populates block->lines with pointers to each line start for O(1) access.
 * Must be called before using inline parsing functions.
 *
 * @param block Block to compute line pointers for
 * @param ctx Error context, populated on failure
 * @return CIF_OK on success, CIF_ERR_ALLOC on failure
 */
CifError _precompute_lines(mmBlock *block, CifErrorContext *ctx);

/**
 * @brief Free pre-computed line pointers.
 *
 * @param block Block to free line pointers for
 */
void _free_lines(mmBlock *block);

/**
 * @brief Get pointer to field in block (no allocation).
 *
 * Returns pointer to start of field and optionally the field length.
 * The returned pointer points into the original buffer.
 *
 * @param block Block to read from
 * @param line Line index
 * @param index Attribute index
 * @param len Output for field length (may be NULL)
 * @return Pointer to field start, or NULL on error
 */
char *_get_field_ptr(mmBlock *block, int line, int index, size_t *len);

/**
 * @brief Parse float from block without allocation.
 *
 * @param block Block to read from
 * @param line Line index
 * @param index Attribute index
 * @return Parsed float value (NaN on error)
 * @deprecated Use _parse_float_safe() for better error handling
 */
float _parse_float_inline(mmBlock *block, int line, int index);

/* ============================================================================
 * FAST BATCH PARSING
 * Optimized functions for high-throughput atom data extraction.
 * ============================================================================ */

/**
 * @brief Parse 3 coordinates (x,y,z) with single line_start computation.
 */
void _parse_coords_inline(mmBlock *block, int line, const int *idx, float *out);

/**
 * @brief Pre-scan group_PDB to build is_nonpoly mask.
 *
 * This enables two-pointer placement during batch parsing:
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
                       CifErrorContext *ctx);

/**
 * @brief Fast element lookup optimized for batch processing.
 *
 * @param block Block containing atom data
 * @param line Row index
 * @param index Attribute index for element symbol
 * @param func Hash lookup function (_lookup_element)
 * @return Element index or PARSE_FAIL
 */
int _lookup_element_fast(mmBlock *block, int line, int index, HashTable func);

/**
 * @brief Fast atom type lookup (residue_atom) optimized for batch processing.
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
                           HashTable func, char *buffer);

/**
 * @brief Parse int from block without allocation.
 *
 * @param block Block to read from
 * @param line Line index
 * @param index Attribute index
 * @return Parsed int value (PARSE_FAIL on error)
 * @deprecated Use _parse_int_safe() for better error handling
 */
int _parse_int_inline(mmBlock *block, int line, int index);

/**
 * @brief Parse int from block with distinct error handling.
 *
 * Unlike _parse_int_inline(), this function distinguishes between:
 * - PARSE_INT_OK: Value parsed, result is valid
 * - PARSE_INT_EMPTY: Field is empty or contains '.' (missing value)
 * - PARSE_INT_ERROR: Field access failed
 *
 * @param block Block to read from
 * @param line Line index
 * @param index Attribute index
 * @param result Output: parsed value (valid only if PARSE_INT_OK returned)
 * @return PARSE_INT_OK, PARSE_INT_EMPTY, or PARSE_INT_ERROR
 */
IntParseResult _parse_int_safe(mmBlock *block, int line, int index, int *result);

/**
 * @brief Parse float from block with distinct error handling.
 *
 * Unlike _parse_float_inline(), this function distinguishes between:
 * - PARSE_FLOAT_OK: Value parsed, result is valid
 * - PARSE_FLOAT_EMPTY: Field is empty or contains '.' (missing value)
 * - PARSE_FLOAT_ERROR: Field access failed
 *
 * @param block Block to read from
 * @param line Line index
 * @param index Attribute index
 * @param result Output: parsed value (valid only if PARSE_FLOAT_OK returned)
 * @return PARSE_FLOAT_OK, PARSE_FLOAT_EMPTY, or PARSE_FLOAT_ERROR
 */
FloatParseResult _parse_float_safe(mmBlock *block, int line, int index, float *result);

/**
 * @brief Lookup value from block without allocation.
 *
 * Parses field in-place and performs hash lookup.
 *
 * @param block Block to read from
 * @param line Line index
 * @param index Attribute index
 * @param func Hash table lookup function
 * @return Lookup result, or PARSE_FAIL if not found
 */
int _lookup_inline(mmBlock *block, int line, int index, HashTable func);

/**
 * @brief Lookup value from block with distinct error handling.
 *
 * Unlike _lookup_inline(), this function distinguishes between:
 * - LOOKUP_OK: Value found, result is valid
 * - LOOKUP_NOT_FOUND: Field accessible but value not in hash table
 * - LOOKUP_ERROR: Field access failed (bounds, allocation, etc.)
 *
 * @param block Block to read from
 * @param line Line index
 * @param index Attribute index
 * @param func Hash table lookup function
 * @param result Output: lookup value (valid only if LOOKUP_OK returned)
 * @return LOOKUP_OK, LOOKUP_NOT_FOUND, or LOOKUP_ERROR
 */
LookupResult _lookup_inline_safe(mmBlock *block, int line, int index,
                                  HashTable func, int *result);

/**
 * @brief Lookup combined value from two fields without allocation.
 *
 * Combines two fields with underscore separator and performs hash lookup.
 *
 * @param block Block to read from
 * @param line Line index
 * @param index1 First attribute index
 * @param index2 Second attribute index
 * @param func Hash table lookup function
 * @param buffer Thread-local buffer for combining (must be MAX_INLINE_BUFFER bytes)
 * @return Lookup result, or PARSE_FAIL if not found
 */
int _lookup_double_inline(mmBlock *block, int line, int index1, int index2,
                          HashTable func, char *buffer);

/** Size of thread-local buffer for combined lookups */
#define MAX_INLINE_BUFFER 128


/* ============================================================================
 * QUOTE STRIPPING
 * CIF uses "..." or '...' to quote strings containing special characters.
 * ============================================================================ */

/**
 * @brief Strip outer quotes from a field by adjusting pointer and length.
 *
 * CIF uses double quotes ("...") or single quotes ('...') to protect
 * strings containing special characters. This function adjusts the pointer
 * and length to exclude the outer quotes if present.
 *
 * @param ptr Input/output: pointer to field start (adjusted if quoted)
 * @param len Input/output: field length (reduced by 2 if quoted)
 */
static inline void _strip_outer_quotes(const char **ptr, size_t *len) {
    if (*len >= 2 &&
        (((*ptr)[0] == '\'' && (*ptr)[*len - 1] == '\'') ||
         ((*ptr)[0] == '"' && (*ptr)[*len - 1] == '"'))) {
        (*ptr)++;
        *len -= 2;
    }
}

/**
 * @brief Get field pointer and length from line_start using precomputed offset.
 *
 * Inline helper for batch parsing - avoids full _get_field_ptr overhead.
 * Does not handle quoted fields (only for simple fields like group_PDB).
 *
 * @param line_start Pointer to start of line
 * @param offsets Pre-computed column byte offsets
 * @param index Column index
 * @param len Output: field length
 * @return Pointer to field start
 */
static inline char *_fast_get_field(char *line_start, const int *offsets,
                                     int index, size_t *len) {
    char *ptr = line_start + offsets[index];
    while (*ptr == ' ') ptr++;

    /* Calculate length inline (no quote handling needed for simple fields) */
    char *end = ptr;
    while (*end != ' ' && *end != '\n' && *end != '\0') end++;
    *len = (size_t)(end - ptr);

    return ptr;
}

/**
 * @brief Fast float parsing optimized for mmCIF coordinates.
 *
 * Optimized for: [-]digits[.digits]
 * Does NOT handle: exponents, NaN, Inf, locale decimals.
 * Uses integer accumulation to minimize floating-point operations.
 *
 * @param ptr Pointer to start of float string (whitespace must be skipped)
 * @return Parsed float value
 */
static inline float _fast_parse_float(const char *ptr) {
    int64_t integer_part = 0;
    int64_t decimal_part = 0;
    int decimal_places = 0;
    int sign = 1;

    if (*ptr == '-') {
        sign = -1;
        ptr++;
    } else if (*ptr == '+') {
        ptr++;
    }

    while (*ptr >= '0' && *ptr <= '9') {
        integer_part = integer_part * 10 + (*ptr - '0');
        ptr++;
    }

    if (*ptr == '.') {
        ptr++;
        while (*ptr >= '0' && *ptr <= '9') {
            decimal_part = decimal_part * 10 + (*ptr - '0');
            decimal_places++;
            ptr++;
        }
    }

    static const double pow10[] = {
        1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001
    };

    double result;
    if (decimal_places > 0 && decimal_places < 8) {
        result = (double)integer_part + (double)decimal_part * pow10[decimal_places];
    } else if (decimal_places >= 8) {
        double divisor = 1.0;
        for (int i = 0; i < decimal_places; i++) divisor *= 10.0;
        result = (double)integer_part + (double)decimal_part / divisor;
    } else {
        result = (double)integer_part;
    }

    return (float)(sign * result);
}


#endif /* _CIFFY_IO_H */
