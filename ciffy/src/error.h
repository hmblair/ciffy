#ifndef _CIFFY_ERROR_H
#define _CIFFY_ERROR_H

#include <stdio.h>
#include <string.h>

/**
 * @file error.h
 * @brief Error handling infrastructure for ciffy.
 *
 * Provides error codes, error context structure, and helper macros
 * for consistent error handling throughout the C codebase.
 */

/**
 * @brief Error codes for ciffy internal functions.
 *
 * Zero indicates success; negative values indicate errors.
 * These codes are mapped to Python exceptions at the API boundary.
 */
typedef enum {
    CIF_OK           =  0,  /**< Success */
    CIF_ERR_ALLOC    = -1,  /**< Memory allocation failed */
    CIF_ERR_IO       = -2,  /**< File I/O error */
    CIF_ERR_PARSE    = -3,  /**< Parsing error (malformed data) */
    CIF_ERR_ATTR     = -4,  /**< Missing or invalid attribute */
    CIF_ERR_BLOCK    = -5,  /**< Missing required CIF block */
    CIF_ERR_BOUNDS   = -6,  /**< Index out of bounds */
    CIF_ERR_OVERFLOW = -7,  /**< Buffer overflow prevented */
    CIF_ERR_LOOKUP   = -8,  /**< Hash table lookup failed (unknown token) */
} CifError;

/**
 * @brief Error context structure for detailed error messages.
 *
 * Passed through function calls and populated when errors occur.
 * Contains both the error code and a human-readable message with
 * source location information for debugging.
 */
typedef struct {
    CifError code;          /**< Error code */
    char message[512];      /**< Human-readable error message */
    const char *file;       /**< Source file where error occurred */
    int line;               /**< Line number where error occurred */
    const char *function;   /**< Function name where error occurred */
} CifErrorContext;

/**
 * @brief Initialize an error context to a clean state.
 */
#define CIF_ERROR_INIT { CIF_OK, "", NULL, 0, NULL }

/**
 * @brief Set error with source location information.
 *
 * @param ctx Pointer to CifErrorContext
 * @param err_code CifError code
 * @param fmt printf-style format string
 * @param ... Format arguments
 */
#define CIF_SET_ERROR(ctx, err_code, fmt, ...) do { \
    if ((ctx) != NULL) { \
        (ctx)->code = (err_code); \
        (ctx)->file = __FILE__; \
        (ctx)->line = __LINE__; \
        (ctx)->function = __func__; \
        snprintf((ctx)->message, sizeof((ctx)->message), fmt, ##__VA_ARGS__); \
    } \
} while(0)

/**
 * @brief Check if an error occurred and return early if so.
 *
 * @param ctx Pointer to CifErrorContext
 * @param expr Expression that returns CifError
 */
#define CIF_CHECK(ctx, expr) do { \
    CifError _err = (expr); \
    if (_err != CIF_OK) return _err; \
} while(0)

/**
 * @brief Check if an error occurred and goto cleanup label if so.
 *
 * @param ctx Pointer to CifErrorContext
 * @param expr Expression that returns CifError
 * @param label Label to jump to on error
 */
#define CIF_CHECK_GOTO(ctx, expr, label) do { \
    CifError _err = (expr); \
    if (_err != CIF_OK) goto label; \
} while(0)

/**
 * @brief Safe malloc with error handling.
 *
 * Sets error context and returns CIF_ERR_ALLOC on failure.
 *
 * @param ctx Pointer to CifErrorContext
 * @param ptr Variable to assign allocated pointer to
 * @param size Number of bytes to allocate
 */
#define CIF_ALLOC(ctx, ptr, size) do { \
    (ptr) = malloc(size); \
    if ((ptr) == NULL) { \
        CIF_SET_ERROR(ctx, CIF_ERR_ALLOC, \
            "Failed to allocate %zu bytes", (size_t)(size)); \
        return CIF_ERR_ALLOC; \
    } \
} while(0)

/**
 * @brief Safe calloc with error handling.
 *
 * Sets error context and returns CIF_ERR_ALLOC on failure.
 *
 * @param ctx Pointer to CifErrorContext
 * @param ptr Variable to assign allocated pointer to
 * @param count Number of elements to allocate
 * @param size Size of each element
 */
#define CIF_CALLOC(ctx, ptr, count, size) do { \
    (ptr) = calloc((count), (size)); \
    if ((ptr) == NULL) { \
        CIF_SET_ERROR(ctx, CIF_ERR_ALLOC, \
            "Failed to allocate %zu elements of %zu bytes", \
            (size_t)(count), (size_t)(size)); \
        return CIF_ERR_ALLOC; \
    } \
} while(0)

/**
 * @brief Safe malloc with goto on failure.
 *
 * Sets error context and jumps to label on failure.
 *
 * @param ctx Pointer to CifErrorContext
 * @param ptr Variable to assign allocated pointer to
 * @param size Number of bytes to allocate
 * @param label Label to jump to on error
 */
#define CIF_ALLOC_GOTO(ctx, ptr, size, label) do { \
    (ptr) = malloc(size); \
    if ((ptr) == NULL) { \
        CIF_SET_ERROR(ctx, CIF_ERR_ALLOC, \
            "Failed to allocate %zu bytes", (size_t)(size)); \
        goto label; \
    } \
} while(0)

/**
 * @brief Safe calloc with goto on failure.
 *
 * Sets error context and jumps to label on failure.
 *
 * @param ctx Pointer to CifErrorContext
 * @param ptr Variable to assign allocated pointer to
 * @param count Number of elements to allocate
 * @param size Size of each element
 * @param label Label to jump to on error
 */
#define CIF_CALLOC_GOTO(ctx, ptr, count, size, label) do { \
    (ptr) = calloc((count), (size)); \
    if ((ptr) == NULL) { \
        CIF_SET_ERROR(ctx, CIF_ERR_ALLOC, \
            "Failed to allocate %zu elements of %zu bytes", \
            (size_t)(count), (size_t)(size)); \
        goto label; \
    } \
} while(0)

/**
 * @brief Check a condition and set error if false.
 *
 * @param ctx Pointer to CifErrorContext
 * @param cond Condition to check
 * @param err_code CifError code to set if condition is false
 * @param fmt printf-style format string
 * @param ... Format arguments
 */
#define CIF_REQUIRE(ctx, cond, err_code, fmt, ...) do { \
    if (!(cond)) { \
        CIF_SET_ERROR(ctx, err_code, fmt, ##__VA_ARGS__); \
        return err_code; \
    } \
} while(0)

/**
 * @brief Check a condition and goto label if false.
 *
 * @param ctx Pointer to CifErrorContext
 * @param cond Condition to check
 * @param err_code CifError code to set if condition is false
 * @param label Label to jump to
 * @param fmt printf-style format string
 * @param ... Format arguments
 */
#define CIF_REQUIRE_GOTO(ctx, cond, err_code, label, fmt, ...) do { \
    if (!(cond)) { \
        CIF_SET_ERROR(ctx, err_code, fmt, ##__VA_ARGS__); \
        goto label; \
    } \
} while(0)

#endif /* _CIFFY_ERROR_H */
