#ifndef _CIFFY_LOG_H
#define _CIFFY_LOG_H

/**
 * @file log.h
 * @brief Logging infrastructure with configurable levels.
 *
 * Provides logging macros with four severity levels:
 *   - ERROR:   Critical errors that prevent operation
 *   - WARNING: Potential issues that don't stop execution
 *   - INFO:    General information about program flow
 *   - DEBUG:   Detailed debugging information
 *
 * Usage:
 *   LOG_ERROR("Failed to open file: %s", filename);
 *   LOG_WARNING("Unknown atom type: %d", atom_type);
 *   LOG_INFO("Loaded %d atoms", count);
 *   LOG_DEBUG("Processing chain %s at index %d", name, idx);
 *
 * Configuration:
 *   Set log level via environment variable CIFFY_LOG_LEVEL:
 *     export CIFFY_LOG_LEVEL=DEBUG   # Show all messages
 *     export CIFFY_LOG_LEVEL=INFO    # Show INFO, WARNING, ERROR
 *     export CIFFY_LOG_LEVEL=WARNING # Show WARNING, ERROR
 *     export CIFFY_LOG_LEVEL=ERROR   # Show only ERROR (default)
 *
 *   Or compile with -DCIFFY_LOG_LEVEL=LOG_LEVEL_DEBUG
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ============================================================================
 * LOG LEVELS
 * Higher values = more verbose
 * ============================================================================ */

#define LOG_LEVEL_NONE    0
#define LOG_LEVEL_ERROR   1
#define LOG_LEVEL_WARNING 2
#define LOG_LEVEL_INFO    3
#define LOG_LEVEL_DEBUG   4

/* Default compile-time log level (can be overridden with -DCIFFY_LOG_LEVEL=X) */
/* Set to DEBUG to allow runtime configuration of all log levels */
#ifndef CIFFY_LOG_LEVEL
#define CIFFY_LOG_LEVEL LOG_LEVEL_DEBUG
#endif


/* ============================================================================
 * RUNTIME LOG LEVEL
 * Allows changing log level at runtime via environment variable
 * ============================================================================ */

/**
 * @brief Get the current runtime log level.
 *
 * Checks CIFFY_LOG_LEVEL environment variable on first call.
 * Falls back to compile-time CIFFY_LOG_LEVEL if not set.
 *
 * @return Current log level (LOG_LEVEL_*)
 */
static inline int _ciffy_log_level(void) {
    static int level = -1;

    if (level == -1) {
        const char *env = getenv("CIFFY_LOG_LEVEL");
        if (env != NULL) {
            if (strcmp(env, "DEBUG") == 0)        level = LOG_LEVEL_DEBUG;
            else if (strcmp(env, "INFO") == 0)    level = LOG_LEVEL_INFO;
            else if (strcmp(env, "WARNING") == 0) level = LOG_LEVEL_WARNING;
            else if (strcmp(env, "ERROR") == 0)   level = LOG_LEVEL_ERROR;
            else if (strcmp(env, "NONE") == 0)    level = LOG_LEVEL_NONE;
            else                                  level = LOG_LEVEL_WARNING;  /* Default */
        } else {
            level = LOG_LEVEL_WARNING;  /* Default: only warnings and errors */
        }
    }

    return level;
}


/* ============================================================================
 * LOGGING MACROS
 * Each macro checks both compile-time and runtime levels for efficiency
 * ============================================================================ */

/* ANSI color codes for terminal output */
#define LOG_COLOR_RESET   "\033[0m"
#define LOG_COLOR_RED     "\033[31m"
#define LOG_COLOR_YELLOW  "\033[33m"
#define LOG_COLOR_BLUE    "\033[34m"
#define LOG_COLOR_GRAY    "\033[90m"

/* Check if we should use colors (only if stderr is a terminal) */
#ifndef CIFFY_LOG_NO_COLOR
#define LOG_USE_COLOR (isatty(fileno(stderr)))
#else
#define LOG_USE_COLOR 0
#endif

/**
 * @brief Internal logging function.
 *
 * @param level Log level of this message
 * @param level_str String representation ("ERROR", "WARNING", etc.)
 * @param color ANSI color code
 * @param file Source file name
 * @param line Source line number
 * @param func Function name
 * @param fmt Format string
 * @param ... Format arguments
 */
#define _LOG_IMPL(level, level_str, color, file, line, func, fmt, ...) \
    do { \
        if ((level) <= _ciffy_log_level()) { \
            if (_ciffy_log_level() >= LOG_LEVEL_DEBUG) { \
                fprintf(stderr, "%s[%-7s]%s %s:%d (%s): " fmt "\n", \
                    color, level_str, LOG_COLOR_RESET, \
                    file, line, func, ##__VA_ARGS__); \
            } else { \
                fprintf(stderr, "%s[%-7s]%s " fmt "\n", \
                    color, level_str, LOG_COLOR_RESET, ##__VA_ARGS__); \
            } \
        } \
    } while (0)


/**
 * @brief Log an error message.
 *
 * Use for critical errors that prevent normal operation.
 * These are always shown unless logging is disabled.
 */
#if CIFFY_LOG_LEVEL >= LOG_LEVEL_ERROR
#define LOG_ERROR(fmt, ...) \
    _LOG_IMPL(LOG_LEVEL_ERROR, "ERROR", LOG_COLOR_RED, \
              __FILE__, __LINE__, __func__, fmt, ##__VA_ARGS__)
#else
#define LOG_ERROR(fmt, ...) ((void)0)
#endif

/**
 * @brief Log a warning message.
 *
 * Use for potential issues that don't stop execution but may indicate problems.
 */
#if CIFFY_LOG_LEVEL >= LOG_LEVEL_WARNING
#define LOG_WARNING(fmt, ...) \
    _LOG_IMPL(LOG_LEVEL_WARNING, "WARNING", LOG_COLOR_YELLOW, \
              __FILE__, __LINE__, __func__, fmt, ##__VA_ARGS__)
#else
#define LOG_WARNING(fmt, ...) ((void)0)
#endif

/**
 * @brief Log an informational message.
 *
 * Use for general information about program flow and state.
 */
#if CIFFY_LOG_LEVEL >= LOG_LEVEL_INFO
#define LOG_INFO(fmt, ...) \
    _LOG_IMPL(LOG_LEVEL_INFO, "INFO", LOG_COLOR_BLUE, \
              __FILE__, __LINE__, __func__, fmt, ##__VA_ARGS__)
#else
#define LOG_INFO(fmt, ...) ((void)0)
#endif

/**
 * @brief Log a debug message.
 *
 * Use for detailed debugging information during development.
 * Includes file, line, and function name in output.
 */
#if CIFFY_LOG_LEVEL >= LOG_LEVEL_DEBUG
#define LOG_DEBUG(fmt, ...) \
    _LOG_IMPL(LOG_LEVEL_DEBUG, "DEBUG", LOG_COLOR_GRAY, \
              __FILE__, __LINE__, __func__, fmt, ##__VA_ARGS__)
#else
#define LOG_DEBUG(fmt, ...) ((void)0)
#endif


/* ============================================================================
 * CONDITIONAL LOGGING
 * Log only if a condition is true
 * ============================================================================ */

#define LOG_ERROR_IF(cond, fmt, ...)   do { if (cond) LOG_ERROR(fmt, ##__VA_ARGS__); } while(0)
#define LOG_WARNING_IF(cond, fmt, ...) do { if (cond) LOG_WARNING(fmt, ##__VA_ARGS__); } while(0)
#define LOG_INFO_IF(cond, fmt, ...)    do { if (cond) LOG_INFO(fmt, ##__VA_ARGS__); } while(0)
#define LOG_DEBUG_IF(cond, fmt, ...)   do { if (cond) LOG_DEBUG(fmt, ##__VA_ARGS__); } while(0)


/* ============================================================================
 * ONCE-ONLY LOGGING
 * Log a message only once (useful for repeated warnings)
 * ============================================================================ */

#define LOG_ONCE(log_macro, fmt, ...) \
    do { \
        static int _logged = 0; \
        if (!_logged) { \
            _logged = 1; \
            log_macro(fmt, ##__VA_ARGS__); \
        } \
    } while(0)

#define LOG_ERROR_ONCE(fmt, ...)   LOG_ONCE(LOG_ERROR, fmt, ##__VA_ARGS__)
#define LOG_WARNING_ONCE(fmt, ...) LOG_ONCE(LOG_WARNING, fmt, ##__VA_ARGS__)
#define LOG_INFO_ONCE(fmt, ...)    LOG_ONCE(LOG_INFO, fmt, ##__VA_ARGS__)
#define LOG_DEBUG_ONCE(fmt, ...)   LOG_ONCE(LOG_DEBUG, fmt, ##__VA_ARGS__)

#endif /* _CIFFY_LOG_H */
