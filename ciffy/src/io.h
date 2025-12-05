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

#include "error.h"
#include "common.h"
#include "hash/lookup.h"

/** Sentinel value indicating attribute index not found */
#define BAD_IX -1

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
    int  width;         /**< Bytes per line (for multi-entry blocks) */
    bool single;        /**< true if single-entry block, false if loop */
    char *head;         /**< Pointer to start of header (attribute definitions) */
    char *start;        /**< Pointer to start of data section */
    int  *offsets;      /**< Column byte offsets for multi-entry blocks */

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
 * @brief Advance buffer pointer to the next line.
 *
 * Moves the buffer pointer past the current line's newline character.
 *
 * @param buffer Pointer to buffer pointer (modified in place)
 */
void _advance_line(char **buffer);

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
 * @return Index of attribute (0-based), or BAD_IX if not found
 */
int _get_attr_index(mmBlock *block, const char *attr);

/**
 * @brief Get an attribute value by line number and attribute index.
 *
 * @param block The block to read from
 * @param line Line number (0-based, for multi-entry blocks)
 * @param index Attribute index (0-based)
 * @param ctx Error context, populated on failure (may be NULL)
 * @return Allocated string containing value, or NULL on error
 */
char *_get_attr_by_line(mmBlock *block, int line, int index, CifErrorContext *ctx);

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
 * @brief Look up a token in a hash table.
 *
 * Strips quotes from the token and performs the lookup.
 *
 * @param func Hash table lookup function
 * @param token Token to look up
 * @return Lookup result value, or -1 if not found
 */
int _lookup(HashTable func, char *token);

/**
 * @brief Look up a token with error reporting.
 *
 * Like _lookup but populates error context on failure.
 *
 * @param func Hash table lookup function
 * @param token Token to look up
 * @param result Output pointer for lookup result
 * @param ctx Error context, populated on failure
 * @return CIF_OK on success, CIF_ERR_LOOKUP if not found
 */
CifError _lookup_safe(HashTable func, char *token, int *result, CifErrorContext *ctx);

#endif /* _CIFFY_IO_H */
