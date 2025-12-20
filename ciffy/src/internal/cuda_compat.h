/**
 * @file cuda_compat.h
 * @brief Platform compatibility macros for CPU/CUDA code sharing.
 *
 * This header provides macros that allow the same code to compile for both
 * CPU and GPU. When compiled with nvcc (__CUDACC__ defined), functions are
 * marked as __host__ __device__. When compiled with a standard C compiler,
 * the macros expand to nothing.
 *
 * Usage:
 *   #include "cuda_compat.h"
 *
 *   CIFFY_HOST_DEVICE
 *   static inline float compute_distance(const float *a, const float *b) {
 *       // Same code runs on CPU and GPU
 *   }
 */

#ifndef CIFFY_CUDA_COMPAT_H
#define CIFFY_CUDA_COMPAT_H

/* ========================================================================= */
/* Platform detection and function decorators                                */
/* ========================================================================= */

#ifdef __CUDACC__
    /* Compiling with NVIDIA CUDA compiler */
    #define CIFFY_HAS_CUDA 1
    #define CIFFY_HOST_DEVICE __host__ __device__
    #define CIFFY_DEVICE __device__
    #define CIFFY_HOST __host__
    #define CIFFY_GLOBAL __global__

    #include <cuda_runtime.h>

    /*
     * Load a float3 from memory using __ldg() for read-only cache optimization.
     * The __ldg() intrinsic hints to the GPU that this data is read-only,
     * enabling use of the texture cache for better performance on scattered reads.
     */
    __device__ __forceinline__ float3 ciffy_load_float3_ldg(const float *p) {
        return make_float3(__ldg(p), __ldg(p + 1), __ldg(p + 2));
    }
#else
    /* Standard C/C++ compiler */
    #define CIFFY_HAS_CUDA 0
    #define CIFFY_HOST_DEVICE
    #define CIFFY_DEVICE
    #define CIFFY_HOST
    #define CIFFY_GLOBAL
#endif

/* ========================================================================= */
/* Math function wrappers                                                    */
/* ========================================================================= */

/*
 * These macros allow using the appropriate math functions for each platform.
 * CUDA device code requires specific function names, while CPU code uses
 * standard C math library functions.
 *
 * Both CPU and GPU use the same function names (sqrtf, etc.) but this
 * abstraction allows for future optimization, e.g., using CUDA fast math
 * intrinsics like __fsqrt_rn() when appropriate.
 */

#include <math.h>

#define CIFFY_SQRTF  sqrtf
#define CIFFY_COSF   cosf
#define CIFFY_SINF   sinf
#define CIFFY_ACOSF  acosf
#define CIFFY_ATAN2F atan2f
#define CIFFY_FABSF  fabsf
#define CIFFY_FMINF  fminf
#define CIFFY_FMAXF  fmaxf

/* ========================================================================= */
/* Numeric constants                                                         */
/* ========================================================================= */

/* Small epsilon for numerical stability in divisions and normalizations */
#ifndef CIFFY_EPS
#define CIFFY_EPS 1e-6f
#endif

/* Pi constant */
#ifndef CIFFY_PI
#define CIFFY_PI 3.14159265358979323846f
#endif

/* ========================================================================= */
/* Error handling                                                            */
/* ========================================================================= */

#include <stdio.h>
#include <stdlib.h>

/**
 * Report a fatal error and abort.
 *
 * Use this for unrecoverable errors that indicate bugs (not user errors).
 * For user-facing errors in Python bindings, use TORCH_CHECK instead.
 *
 * Usage:
 *   CIFFY_FATAL("component_id %d out of bounds [0, %d)", comp_id, n_components);
 */
#define CIFFY_FATAL(...) do { \
    fprintf(stderr, "CIFFY FATAL [%s:%d]: ", __FILE__, __LINE__); \
    fprintf(stderr, __VA_ARGS__); \
    fprintf(stderr, "\n"); \
    abort(); \
} while (0)

#ifdef __CUDACC__

/**
 * Check for CUDA errors after kernel launches.
 * Call this immediately after any kernel launch to detect launch failures,
 * invalid arguments, or device errors.
 */
#define CIFFY_CUDA_CHECK_KERNEL() do { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        CIFFY_FATAL("CUDA error: %s", cudaGetErrorString(err)); \
    } \
} while (0)

#endif /* __CUDACC__ */

#endif /* CIFFY_CUDA_COMPAT_H */
