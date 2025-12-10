#ifndef _CIFFY_COMMON_H
#define _CIFFY_COMMON_H

#include <string.h>
#include <stdbool.h>
#include <stdlib.h>

#include "error.h"

/**
 * @file common.h
 * @brief Common helper functions for ciffy.
 *
 * Provides string comparison utilities and safe memory allocation
 * helpers used throughout the C codebase.
 */

/**
 * @brief Compare two strings for prefix equality.
 *
 * Checks if str1 starts with str2.
 *
 * @param str1 The string to check (haystack)
 * @param str2 The prefix to look for (needle)
 * @return true if str1 starts with str2, false otherwise
 *
 * @example
 *   _eq("_atom_site.id", "_atom_site") -> true
 *   _eq("data_4V5D", "data_") -> true
 *   _eq("loop_", "data_") -> false
 */
static inline bool _eq(const char *str1, const char *str2) {
    return strncmp(str1, str2, strlen(str2)) == 0;
}

/**
 * @brief Compare two strings for prefix inequality.
 *
 * Checks if str1 does NOT start with str2.
 *
 * @param str1 The string to check (haystack)
 * @param str2 The prefix to look for (needle)
 * @return true if str1 does not start with str2, false otherwise
 */
static inline bool _neq(const char *str1, const char *str2) {
    return strncmp(str1, str2, strlen(str2)) != 0;
}

/**
 * @brief Compare a non-null-terminated field against a null-terminated string.
 *
 * @param ptr Field pointer (not null-terminated)
 * @param len Length of the field
 * @param str Null-terminated string to compare against
 * @return true if the field equals the string
 */
static inline bool _field_eq(const char *ptr, size_t len, const char *str) {
    size_t str_len = strlen(str);
    if (len != str_len) return false;
    return memcmp(ptr, str, len) == 0;
}

/**
 * @brief Compare two non-null-terminated fields for equality.
 *
 * @param ptr1 First field pointer
 * @param len1 Length of first field
 * @param ptr2 Second field pointer
 * @param len2 Length of second field
 * @return true if fields are equal
 */
static inline bool _field_eq_field(const char *ptr1, size_t len1,
                                   const char *ptr2, size_t len2) {
    if (len1 != len2) return false;
    return memcmp(ptr1, ptr2, len1) == 0;
}

/**
 * @brief Check if the current line marks the end of a CIF section.
 *
 * In mmCIF format, sections end with a line starting with '#'.
 *
 * @param line Pointer to the start of the line
 * @return true if line starts with '#', indicating section end
 */
static inline bool _is_section_end(const char *line) {
    return *line == '#';
}

/**
 * @brief Safely duplicate a substring with error handling.
 *
 * Allocates memory and copies exactly `length` characters from src,
 * adding a null terminator. Sets error context on allocation failure.
 *
 * @param src Source string pointer
 * @param length Number of characters to copy
 * @param ctx Error context for reporting failures (may be NULL)
 * @return Allocated null-terminated string, or NULL on error
 *
 * @note Caller is responsible for freeing the returned string.
 */
static inline char *_strdup_n(const char *src, size_t length, CifErrorContext *ctx) {
    char *result = malloc(length + 1);
    if (result == NULL) {
        if (ctx != NULL) {
            CIF_SET_ERROR(ctx, CIF_ERR_ALLOC,
                "Failed to allocate %zu bytes for string copy", length + 1);
        }
        return NULL;
    }
    memcpy(result, src, length);
    result[length] = '\0';
    return result;
}

/**
 * @brief Safe string duplication with error handling.
 *
 * Duplicates an entire null-terminated string.
 *
 * @param src Source string to duplicate
 * @param ctx Error context for reporting failures (may be NULL)
 * @return Allocated copy of the string, or NULL on error
 *
 * @note Caller is responsible for freeing the returned string.
 */
static inline char *_strdup_safe(const char *src, CifErrorContext *ctx) {
    return _strdup_n(src, strlen(src), ctx);
}

#endif /* _CIFFY_COMMON_H */
