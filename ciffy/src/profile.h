/**
 * @file profile.h
 * @brief Compile-time profiling infrastructure for CIF parsing.
 *
 * When CIFFY_PROFILE is defined, timing instrumentation is enabled.
 * When not defined, all macros expand to nothing (zero runtime cost).
 *
 * Usage:
 *   PROFILE_RESET();           // Reset all timers
 *   PROFILE_START(file_load);  // Start timing
 *   ... code ...
 *   PROFILE_END(file_load);    // End timing, accumulate to g_profile.file_load
 */

#ifndef _CIFFY_PROFILE_H
#define _CIFFY_PROFILE_H

#ifdef CIFFY_PROFILE

#include <time.h>
#include <string.h>

/**
 * @brief Profile data structure holding timing for each parsing phase.
 *
 * All times are in seconds (double precision).
 */
typedef struct {
    double file_load;      /**< File I/O (fopen, fread, fclose) */
    double block_parse;    /**< Block scanning and offset computation */
    double line_precomp;   /**< Line pointer array precomputation */
    double metadata;       /**< Chain/residue/atom counting and metadata */
    double batch_parse;    /**< Atom data extraction (coords, types, elements) */
    double residue_count;  /**< Per-residue and per-chain atom counting */
    double py_convert;     /**< C struct to Python/NumPy conversion */
    /* Sub-phases of batch_parse */
    double batch_coords;   /**< Float parsing for x,y,z coordinates */
    double batch_elements; /**< Element symbol hash lookup */
    double batch_types;    /**< Atom type double hash lookup */
} CifProfile;

/** Global profile instance - defined in module.c */
extern CifProfile g_profile;

/**
 * @brief Start timing a named phase.
 *
 * Creates a local timespec variable to store the start time.
 * Must be paired with PROFILE_END(name) in the same scope.
 */
#define PROFILE_START(name) \
    struct timespec _prof_##name##_start; \
    clock_gettime(CLOCK_MONOTONIC, &_prof_##name##_start)

/**
 * @brief End timing a named phase and accumulate to global profile.
 *
 * Computes elapsed time since PROFILE_START(name) and adds to g_profile.name.
 */
#define PROFILE_END(name) do { \
    struct timespec _prof_##name##_end; \
    clock_gettime(CLOCK_MONOTONIC, &_prof_##name##_end); \
    g_profile.name += (_prof_##name##_end.tv_sec - _prof_##name##_start.tv_sec) \
                    + (_prof_##name##_end.tv_nsec - _prof_##name##_start.tv_nsec) / 1e9; \
} while(0)

/**
 * @brief Reset all profile timers to zero.
 */
#define PROFILE_RESET() memset(&g_profile, 0, sizeof(g_profile))

/**
 * @brief Check if profiling is enabled (for conditional code).
 */
#define CIFFY_PROFILING_ENABLED 1

#else /* !CIFFY_PROFILE */

/* When profiling is disabled, all macros expand to nothing */
#define PROFILE_START(name)
#define PROFILE_END(name)
#define PROFILE_RESET()
#define CIFFY_PROFILING_ENABLED 0

#endif /* CIFFY_PROFILE */

#endif /* _CIFFY_PROFILE_H */
