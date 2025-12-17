/**
 * @file batch.cu
 * @brief CUDA kernels for batch coordinate conversions.
 *
 * This file contains GPU implementations of the batch coordinate conversion
 * operations. It uses the shared implementations from geometry_impl.h,
 * which are marked with CIFFY_HOST_DEVICE for CPU/GPU compatibility.
 *
 * Internal coordinates are stored as (N, 3) arrays in row-major order:
 *   internal[i * 3 + 0] = distance
 *   internal[i * 3 + 1] = angle
 *   internal[i * 3 + 2] = dihedral
 */

#include "cuda_compat.h"
#include "geometry_impl.h"
#include "batch.h"

#include <cuda_runtime.h>
#include <stdint.h>

/* Use the same constants as batch.h for consistency */
#ifndef INTERNAL_DIST
#define INTERNAL_DIST  0
#define INTERNAL_ANGLE 1
#define INTERNAL_DIHE  2
#define INTERNAL_COLS  3
#define INTERNAL_IDX(i, col) ((i) * INTERNAL_COLS + (col))
#endif


/* ========================================================================= */
/* CUDA Kernels                                                              */
/* ========================================================================= */

/**
 * Kernel: Convert Cartesian coordinates to internal coordinates.
 *
 * Each thread processes one Z-matrix entry independently (embarrassingly parallel).
 * Uses __ldg() via ciffy_load_float3_ldg() for read-only cache optimization on
 * scattered coordinate reads. Each coordinate is loaded once and reused.
 *
 * Output: internal array with shape (n_entries, 3) in row-major order.
 */
__global__ void kernel_cartesian_to_internal(
    const float *coords,       /* (n_atoms, 3) */
    const int64_t *indices,    /* (n_entries, 4) */
    int n_entries,
    int n_atoms,
    float *internal            /* (n_entries, 3) output: [dist, angle, dihedral] */
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_entries) return;

    int64_t atom_idx = indices[i * 4 + 0];
    int64_t dist_ref = indices[i * 4 + 1];
    int64_t angl_ref = indices[i * 4 + 2];
    int64_t dihe_ref = indices[i * 4 + 3];

    /* Bounds check */
    if (atom_idx < 0 || atom_idx >= n_atoms) {
        internal[INTERNAL_IDX(i, INTERNAL_DIST)] = 0.0f;
        internal[INTERNAL_IDX(i, INTERNAL_ANGLE)] = 0.0f;
        internal[INTERNAL_IDX(i, INTERNAL_DIHE)] = 0.0f;
        return;
    }

    /* Load all coordinates ONCE using __ldg() for read-only cache optimization.
     * This reduces memory traffic by not re-loading the same coords multiple times. */
    float3 atom_f3 = ciffy_load_float3_ldg(&coords[atom_idx * 3]);
    float atom[3] = {atom_f3.x, atom_f3.y, atom_f3.z};

    /* Check if we have valid references and load them once */
    int has_dist = (dist_ref >= 0 && dist_ref < n_atoms);
    int has_angl = has_dist && (angl_ref >= 0 && angl_ref < n_atoms);
    int has_dihe = has_angl && (dihe_ref >= 0 && dihe_ref < n_atoms);

    float ref1[3], ref2[3], ref3[3];

    if (has_dist) {
        float3 ref1_f3 = ciffy_load_float3_ldg(&coords[dist_ref * 3]);
        ref1[0] = ref1_f3.x; ref1[1] = ref1_f3.y; ref1[2] = ref1_f3.z;
    }
    if (has_angl) {
        float3 ref2_f3 = ciffy_load_float3_ldg(&coords[angl_ref * 3]);
        ref2[0] = ref2_f3.x; ref2[1] = ref2_f3.y; ref2[2] = ref2_f3.z;
    }
    if (has_dihe) {
        float3 ref3_f3 = ciffy_load_float3_ldg(&coords[dihe_ref * 3]);
        ref3[0] = ref3_f3.x; ref3[1] = ref3_f3.y; ref3[2] = ref3_f3.z;
    }

    /* Compute internal coordinates using pre-loaded data */
    internal[INTERNAL_IDX(i, INTERNAL_DIST)] = has_dist ? compute_distance_impl(atom, ref1) : 0.0f;
    internal[INTERNAL_IDX(i, INTERNAL_ANGLE)] = has_angl ? compute_angle_impl(atom, ref1, ref2) : 0.0f;
    internal[INTERNAL_IDX(i, INTERNAL_DIHE)] = has_dihe ? compute_dihedral_impl(ref3, ref2, ref1, atom) : 0.0f;
}


/**
 * Kernel: Backward pass for cartesian_to_internal.
 *
 * Each thread processes one Z-matrix entry and uses atomicAdd for gradient
 * accumulation since multiple entries may reference the same atoms.
 *
 * Input: internal (n_entries, 3) and grad_internal (n_entries, 3) arrays.
 */
__global__ void kernel_cartesian_to_internal_backward(
    const float *coords,
    const int64_t *indices,
    int n_entries,
    int n_atoms,
    const float *internal,      /* (n_entries, 3) [dist, angle, dihedral] */
    const float *grad_internal, /* (n_entries, 3) gradient w.r.t. internal */
    float *grad_coords          /* Output: atomically accumulated */
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_entries) return;

    int64_t atom_idx = indices[i * 4 + 0];
    int64_t dist_ref = indices[i * 4 + 1];
    int64_t angl_ref = indices[i * 4 + 2];
    int64_t dihe_ref = indices[i * 4 + 3];

    if (atom_idx < 0 || atom_idx >= n_atoms) return;

    const float *atom = &coords[atom_idx * 3];
    float grad_atom[3], grad_ref1[3], grad_ref2[3], grad_ref3[3];

    /* Extract values from internal arrays */
    float distance = internal[INTERNAL_IDX(i, INTERNAL_DIST)];
    float angle = internal[INTERNAL_IDX(i, INTERNAL_ANGLE)];
    float grad_distance = grad_internal[INTERNAL_IDX(i, INTERNAL_DIST)];
    float grad_angle = grad_internal[INTERNAL_IDX(i, INTERNAL_ANGLE)];
    float grad_dihedral = grad_internal[INTERNAL_IDX(i, INTERNAL_DIHE)];

    /* Distance backward */
    if (dist_ref >= 0 && dist_ref < n_atoms) {
        const float *ref1 = &coords[dist_ref * 3];
        compute_distance_backward_impl(
            atom, ref1, distance, grad_distance,
            grad_atom, grad_ref1
        );
        /* Atomic accumulation */
        atomicAdd(&grad_coords[atom_idx * 3 + 0], grad_atom[0]);
        atomicAdd(&grad_coords[atom_idx * 3 + 1], grad_atom[1]);
        atomicAdd(&grad_coords[atom_idx * 3 + 2], grad_atom[2]);
        atomicAdd(&grad_coords[dist_ref * 3 + 0], grad_ref1[0]);
        atomicAdd(&grad_coords[dist_ref * 3 + 1], grad_ref1[1]);
        atomicAdd(&grad_coords[dist_ref * 3 + 2], grad_ref1[2]);
    }

    /* Angle backward */
    if (angl_ref >= 0 && dist_ref >= 0 && angl_ref < n_atoms && dist_ref < n_atoms) {
        const float *ref1 = &coords[dist_ref * 3];
        const float *ref2 = &coords[angl_ref * 3];
        compute_angle_backward_impl(
            atom, ref1, ref2, angle, grad_angle,
            grad_atom, grad_ref1, grad_ref2
        );
        atomicAdd(&grad_coords[atom_idx * 3 + 0], grad_atom[0]);
        atomicAdd(&grad_coords[atom_idx * 3 + 1], grad_atom[1]);
        atomicAdd(&grad_coords[atom_idx * 3 + 2], grad_atom[2]);
        atomicAdd(&grad_coords[dist_ref * 3 + 0], grad_ref1[0]);
        atomicAdd(&grad_coords[dist_ref * 3 + 1], grad_ref1[1]);
        atomicAdd(&grad_coords[dist_ref * 3 + 2], grad_ref1[2]);
        atomicAdd(&grad_coords[angl_ref * 3 + 0], grad_ref2[0]);
        atomicAdd(&grad_coords[angl_ref * 3 + 1], grad_ref2[1]);
        atomicAdd(&grad_coords[angl_ref * 3 + 2], grad_ref2[2]);
    }

    /* Dihedral backward */
    if (dihe_ref >= 0 && angl_ref >= 0 && dist_ref >= 0 &&
        dihe_ref < n_atoms && angl_ref < n_atoms && dist_ref < n_atoms) {
        const float *ref1 = &coords[dist_ref * 3];
        const float *ref2 = &coords[angl_ref * 3];
        const float *ref3 = &coords[dihe_ref * 3];
        compute_dihedral_backward_impl(
            ref3, ref2, ref1, atom, grad_dihedral,
            grad_ref3, grad_ref2, grad_ref1, grad_atom
        );
        atomicAdd(&grad_coords[dihe_ref * 3 + 0], grad_ref3[0]);
        atomicAdd(&grad_coords[dihe_ref * 3 + 1], grad_ref3[1]);
        atomicAdd(&grad_coords[dihe_ref * 3 + 2], grad_ref3[2]);
        atomicAdd(&grad_coords[angl_ref * 3 + 0], grad_ref2[0]);
        atomicAdd(&grad_coords[angl_ref * 3 + 1], grad_ref2[1]);
        atomicAdd(&grad_coords[angl_ref * 3 + 2], grad_ref2[2]);
        atomicAdd(&grad_coords[dist_ref * 3 + 0], grad_ref1[0]);
        atomicAdd(&grad_coords[dist_ref * 3 + 1], grad_ref1[1]);
        atomicAdd(&grad_coords[dist_ref * 3 + 2], grad_ref1[2]);
        atomicAdd(&grad_coords[atom_idx * 3 + 0], grad_atom[0]);
        atomicAdd(&grad_coords[atom_idx * 3 + 1], grad_atom[1]);
        atomicAdd(&grad_coords[atom_idx * 3 + 2], grad_atom[2]);
    }
}


/**
 * Kernel: NERF reconstruction for a single level.
 *
 * Atoms within the same BFS level can be processed in parallel since they
 * don't depend on each other. Different levels must be processed sequentially.
 *
 * Input: internal array with shape (n_entries, 3) in row-major order.
 */
__global__ void kernel_nerf_reconstruct_level(
    float *coords,
    const int64_t *indices,
    const float *internal,     /* (n_entries, 3) [dist, angle, dihedral] */
    int level_start,
    int level_end,
    int n_atoms
) {
    int i = level_start + blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= level_end) return;

    int64_t atom_idx = indices[i * 4 + 0];
    int64_t dist_ref = indices[i * 4 + 1];
    int64_t angl_ref = indices[i * 4 + 2];
    int64_t dihe_ref = indices[i * 4 + 3];

    if (atom_idx < 0 || atom_idx >= n_atoms) return;

    float *result = &coords[atom_idx * 3];

    /* Extract internal coordinates for this entry */
    float distance = internal[INTERNAL_IDX(i, INTERNAL_DIST)];
    float angle = internal[INTERNAL_IDX(i, INTERNAL_ANGLE)];
    float dihedral = internal[INTERNAL_IDX(i, INTERNAL_DIHE)];

    if (dist_ref < 0) {
        /* First atom: place at origin */
        result[0] = 0.0f;
        result[1] = 0.0f;
        result[2] = 0.0f;

    } else if (angl_ref < 0) {
        /* Second atom: place along +X from reference */
        if (dist_ref < n_atoms) {
            const float *ref = &coords[dist_ref * 3];
            nerf_place_along_x_impl(ref, distance, result);
        }

    } else if (dihe_ref < 0) {
        /* Third atom: place in plane */
        if (dist_ref < n_atoms && angl_ref < n_atoms) {
            const float *ref1 = &coords[dist_ref * 3];
            const float *ref2 = &coords[angl_ref * 3];
            nerf_place_in_plane_impl(ref1, ref2, distance, angle, result);
        }

    } else {
        /* Full NERF placement */
        if (dihe_ref < n_atoms && angl_ref < n_atoms && dist_ref < n_atoms) {
            const float *p1 = &coords[dihe_ref * 3];
            const float *p2 = &coords[angl_ref * 3];
            const float *p3 = &coords[dist_ref * 3];
            nerf_place_atom_impl(p1, p2, p3, distance, angle, dihedral, result);
        }
    }
}


/**
 * Kernel: Backward pass for NERF reconstruction at a single level.
 *
 * Processes in reverse BFS order. Uses atomicAdd for gradient accumulation.
 *
 * Input/Output: internal and grad_internal arrays with shape (n_entries, 3).
 */
__global__ void kernel_nerf_reconstruct_backward_level(
    const float *coords,
    const int64_t *indices,
    const float *internal,      /* (n_entries, 3) [dist, angle, dihedral] */
    float *grad_coords,
    float *grad_internal,       /* (n_entries, 3) gradient output */
    int level_start,
    int level_end,
    int n_atoms
) {
    int i = level_start + blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= level_end) return;

    int64_t atom_idx = indices[i * 4 + 0];
    int64_t dist_ref = indices[i * 4 + 1];
    int64_t angl_ref = indices[i * 4 + 2];
    int64_t dihe_ref = indices[i * 4 + 3];

    if (atom_idx < 0 || atom_idx >= n_atoms) {
        grad_internal[INTERNAL_IDX(i, INTERNAL_DIST)] = 0.0f;
        grad_internal[INTERNAL_IDX(i, INTERNAL_ANGLE)] = 0.0f;
        grad_internal[INTERNAL_IDX(i, INTERNAL_DIHE)] = 0.0f;
        return;
    }

    /* Extract internal coordinates */
    float distance = internal[INTERNAL_IDX(i, INTERNAL_DIST)];
    float angle = internal[INTERNAL_IDX(i, INTERNAL_ANGLE)];
    float dihedral = internal[INTERNAL_IDX(i, INTERNAL_DIHE)];

    const float *grad_result = &grad_coords[atom_idx * 3];
    float grad_a[3], grad_b[3], grad_c[3];
    float grad_dist, grad_ang, grad_dih;

    if (dist_ref < 0) {
        /* First atom at origin: no gradients */
        grad_internal[INTERNAL_IDX(i, INTERNAL_DIST)] = 0.0f;
        grad_internal[INTERNAL_IDX(i, INTERNAL_ANGLE)] = 0.0f;
        grad_internal[INTERNAL_IDX(i, INTERNAL_DIHE)] = 0.0f;

    } else if (angl_ref < 0) {
        /* Second atom: along +X */
        grad_internal[INTERNAL_IDX(i, INTERNAL_DIST)] = grad_result[0];
        grad_internal[INTERNAL_IDX(i, INTERNAL_ANGLE)] = 0.0f;
        grad_internal[INTERNAL_IDX(i, INTERNAL_DIHE)] = 0.0f;

        if (dist_ref < n_atoms) {
            atomicAdd(&grad_coords[dist_ref * 3 + 0], grad_result[0]);
            atomicAdd(&grad_coords[dist_ref * 3 + 1], grad_result[1]);
            atomicAdd(&grad_coords[dist_ref * 3 + 2], grad_result[2]);
        }

    } else if (dihe_ref < 0) {
        /* Third atom: in plane */
        if (angl_ref < n_atoms && dist_ref < n_atoms) {
            const float *ref1 = &coords[dist_ref * 3];
            const float *ref2 = &coords[angl_ref * 3];
            float grad_ref1[3], grad_ref2[3];

            nerf_place_in_plane_backward_impl(
                ref1, ref2, distance, angle, grad_result,
                grad_ref1, grad_ref2, &grad_dist, &grad_ang
            );
            grad_internal[INTERNAL_IDX(i, INTERNAL_DIST)] = grad_dist;
            grad_internal[INTERNAL_IDX(i, INTERNAL_ANGLE)] = grad_ang;
            grad_internal[INTERNAL_IDX(i, INTERNAL_DIHE)] = 0.0f;

            atomicAdd(&grad_coords[dist_ref * 3 + 0], grad_ref1[0]);
            atomicAdd(&grad_coords[dist_ref * 3 + 1], grad_ref1[1]);
            atomicAdd(&grad_coords[dist_ref * 3 + 2], grad_ref1[2]);
            atomicAdd(&grad_coords[angl_ref * 3 + 0], grad_ref2[0]);
            atomicAdd(&grad_coords[angl_ref * 3 + 1], grad_ref2[1]);
            atomicAdd(&grad_coords[angl_ref * 3 + 2], grad_ref2[2]);
        } else {
            grad_internal[INTERNAL_IDX(i, INTERNAL_DIST)] = 0.0f;
            grad_internal[INTERNAL_IDX(i, INTERNAL_ANGLE)] = 0.0f;
            grad_internal[INTERNAL_IDX(i, INTERNAL_DIHE)] = 0.0f;
        }

    } else {
        /* Full NERF */
        if (dihe_ref < n_atoms && angl_ref < n_atoms && dist_ref < n_atoms) {
            const float *p1 = &coords[dihe_ref * 3];
            const float *p2 = &coords[angl_ref * 3];
            const float *p3 = &coords[dist_ref * 3];

            nerf_place_atom_backward_impl(
                p1, p2, p3, distance, angle, dihedral, grad_result,
                grad_a, grad_b, grad_c,
                &grad_dist, &grad_ang, &grad_dih
            );
            grad_internal[INTERNAL_IDX(i, INTERNAL_DIST)] = grad_dist;
            grad_internal[INTERNAL_IDX(i, INTERNAL_ANGLE)] = grad_ang;
            grad_internal[INTERNAL_IDX(i, INTERNAL_DIHE)] = grad_dih;

            atomicAdd(&grad_coords[dihe_ref * 3 + 0], grad_a[0]);
            atomicAdd(&grad_coords[dihe_ref * 3 + 1], grad_a[1]);
            atomicAdd(&grad_coords[dihe_ref * 3 + 2], grad_a[2]);
            atomicAdd(&grad_coords[angl_ref * 3 + 0], grad_b[0]);
            atomicAdd(&grad_coords[angl_ref * 3 + 1], grad_b[1]);
            atomicAdd(&grad_coords[angl_ref * 3 + 2], grad_b[2]);
            atomicAdd(&grad_coords[dist_ref * 3 + 0], grad_c[0]);
            atomicAdd(&grad_coords[dist_ref * 3 + 1], grad_c[1]);
            atomicAdd(&grad_coords[dist_ref * 3 + 2], grad_c[2]);
        } else {
            grad_internal[INTERNAL_IDX(i, INTERNAL_DIST)] = 0.0f;
            grad_internal[INTERNAL_IDX(i, INTERNAL_ANGLE)] = 0.0f;
            grad_internal[INTERNAL_IDX(i, INTERNAL_DIHE)] = 0.0f;
        }
    }
}


/* ========================================================================= */
/* Host-callable C functions                                                 */
/* ========================================================================= */

extern "C" {

/**
 * CUDA implementation of batch_cartesian_to_internal.
 *
 * All arrays must be device pointers.
 * Output: d_internal is (n_entries, 3) array in row-major order.
 */
void cuda_batch_cartesian_to_internal(
    const float *d_coords,
    size_t n_atoms,
    const int64_t *d_indices,
    size_t n_entries,
    float *d_internal,      /* (n_entries, 3) output */
    cudaStream_t stream
) {
    if (n_entries == 0) return;

    int threads = 256;
    int blocks = ((int)n_entries + threads - 1) / threads;

    kernel_cartesian_to_internal<<<blocks, threads, 0, stream>>>(
        d_coords, d_indices, (int)n_entries, (int)n_atoms, d_internal
    );
}


/**
 * CUDA implementation of batch_cartesian_to_internal_backward.
 *
 * d_grad_coords must be zero-initialized before calling.
 */
void cuda_batch_cartesian_to_internal_backward(
    const float *d_coords,
    size_t n_atoms,
    const int64_t *d_indices,
    size_t n_entries,
    const float *d_internal,      /* (n_entries, 3) */
    const float *d_grad_internal, /* (n_entries, 3) */
    float *d_grad_coords,
    cudaStream_t stream
) {
    if (n_entries == 0) return;

    int threads = 256;
    int blocks = ((int)n_entries + threads - 1) / threads;

    kernel_cartesian_to_internal_backward<<<blocks, threads, 0, stream>>>(
        d_coords, d_indices, (int)n_entries, (int)n_atoms,
        d_internal, d_grad_internal, d_grad_coords
    );
}


/**
 * CUDA implementation of batch_nerf_reconstruct.
 *
 * For polymers without complex branching (linear chains), all entries
 * can typically be processed in a single level. For branched structures,
 * the caller should precompute level offsets based on the BFS structure.
 *
 * Simple version: processes all entries sequentially (one level).
 */
void cuda_batch_nerf_reconstruct(
    float *d_coords,
    size_t n_atoms,
    const int64_t *d_indices,
    size_t n_entries,
    const float *d_internal,    /* (n_entries, 3) */
    cudaStream_t stream
) {
    if (n_entries == 0) return;

    /*
     * For linear polymers (most common case), we process in chunks.
     * Each chunk can be processed in parallel, but chunks must be sequential
     * to respect dependencies.
     *
     * Simple approach: process entry by entry for correctness.
     * Future optimization: precompute dependency levels for parallelism.
     */
    for (size_t i = 0; i < n_entries; i++) {
        kernel_nerf_reconstruct_level<<<1, 1, 0, stream>>>(
            d_coords, d_indices, d_internal,
            (int)i, (int)(i + 1), (int)n_atoms
        );
        /* Synchronize to ensure coordinates are written before next entry reads them */
        cudaStreamSynchronize(stream);
    }
}


/**
 * CUDA implementation of batch_nerf_reconstruct with level-based parallelism.
 *
 * level_offsets: array of n_levels+1 integers where level_offsets[i] is the
 *                starting index of level i in the Z-matrix.
 */
void cuda_batch_nerf_reconstruct_leveled(
    float *d_coords,
    size_t n_atoms,
    const int64_t *d_indices,
    size_t n_entries,
    const float *d_internal,    /* (n_entries, 3) */
    const int *level_offsets,
    int n_levels,
    cudaStream_t stream
) {
    if (n_entries == 0) return;

    int threads = 256;

    for (int level = 0; level < n_levels; level++) {
        int level_start = level_offsets[level];
        int level_end = level_offsets[level + 1];
        int level_size = level_end - level_start;

        if (level_size > 0) {
            int blocks = (level_size + threads - 1) / threads;
            kernel_nerf_reconstruct_level<<<blocks, threads, 0, stream>>>(
                d_coords, d_indices, d_internal,
                level_start, level_end, (int)n_atoms
            );
            /*
             * No explicit sync needed between levels on same stream.
             * CUDA guarantees kernel execution order and memory consistency
             * for kernels launched on the same stream.
             */
        }
    }
}


/**
 * CUDA implementation of batch_nerf_reconstruct_backward.
 *
 * Simple version: processes in reverse order, entry by entry.
 */
void cuda_batch_nerf_reconstruct_backward(
    const float *d_coords,
    size_t n_atoms,
    const int64_t *d_indices,
    size_t n_entries,
    const float *d_internal,      /* (n_entries, 3) */
    float *d_grad_coords,
    float *d_grad_internal,       /* (n_entries, 3) output */
    cudaStream_t stream
) {
    if (n_entries == 0) return;

    /* Process in reverse order */
    for (size_t i = n_entries; i > 0; i--) {
        kernel_nerf_reconstruct_backward_level<<<1, 1, 0, stream>>>(
            d_coords, d_indices, d_internal,
            d_grad_coords, d_grad_internal,
            (int)(i - 1), (int)i, (int)n_atoms
        );
        /* Synchronize to ensure gradients are accumulated before next entry */
        cudaStreamSynchronize(stream);
    }
}


/**
 * CUDA implementation of batch_nerf_reconstruct_backward with level-based parallelism.
 */
void cuda_batch_nerf_reconstruct_backward_leveled(
    const float *d_coords,
    size_t n_atoms,
    const int64_t *d_indices,
    size_t n_entries,
    const float *d_internal,      /* (n_entries, 3) */
    float *d_grad_coords,
    float *d_grad_internal,       /* (n_entries, 3) output */
    const int *level_offsets,
    int n_levels,
    cudaStream_t stream
) {
    if (n_entries == 0) return;

    int threads = 256;

    /* Process levels in reverse order */
    for (int level = n_levels - 1; level >= 0; level--) {
        int level_start = level_offsets[level];
        int level_end = level_offsets[level + 1];
        int level_size = level_end - level_start;

        if (level_size > 0) {
            int blocks = (level_size + threads - 1) / threads;
            kernel_nerf_reconstruct_backward_level<<<blocks, threads, 0, stream>>>(
                d_coords, d_indices, d_internal,
                d_grad_coords, d_grad_internal,
                level_start, level_end, (int)n_atoms
            );
            /*
             * No explicit sync needed between levels on same stream.
             * CUDA guarantees kernel execution order and memory consistency.
             */
        }
    }
}


/* ========================================================================= */
/* Anchored NERF CUDA functions                                              */
/* ========================================================================= */


/**
 * Kernel: NERF reconstruction for a single level with anchor coordinates.
 *
 * Input: internal array with shape (n_entries, 3) in row-major order.
 */
__global__ void kernel_nerf_reconstruct_level_anchored(
    float *coords,
    const int64_t *indices,
    const float *internal,         /* (n_entries, 3) [dist, angle, dihedral] */
    int level_start,
    int level_end,
    int n_atoms,
    const float *anchor_coords,    /* (n_components, 3, 3) flattened */
    const int32_t *component_ids   /* (n_entries,) */
) {
    int i = level_start + blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= level_end) return;

    int64_t atom_idx = indices[i * 4 + 0];
    int64_t dist_ref = indices[i * 4 + 1];
    int64_t angl_ref = indices[i * 4 + 2];
    int64_t dihe_ref = indices[i * 4 + 3];

    if (atom_idx < 0 || atom_idx >= n_atoms) return;

    float *result = &coords[atom_idx * 3];

    /* Extract internal coordinates for this entry */
    float distance = internal[INTERNAL_IDX(i, INTERNAL_DIST)];
    float angle = internal[INTERNAL_IDX(i, INTERNAL_ANGLE)];
    float dihedral = internal[INTERNAL_IDX(i, INTERNAL_DIHE)];

    /* Get anchors for this component */
    const float *anchor0 = NULL;
    const float *anchor1 = NULL;
    const float *anchor2 = NULL;
    if (anchor_coords != NULL && component_ids != NULL) {
        int32_t comp_id = component_ids[i];
        anchor0 = &anchor_coords[comp_id * 9 + 0];
        anchor1 = &anchor_coords[comp_id * 9 + 3];
        anchor2 = &anchor_coords[comp_id * 9 + 6];
    }

    if (dist_ref < 0) {
        /* First atom: place at anchor0 if available, else origin */
        if (anchor0 != NULL) {
            float3 a0 = ciffy_load_float3_ldg(anchor0);
            result[0] = a0.x;
            result[1] = a0.y;
            result[2] = a0.z;
        } else {
            result[0] = 0.0f;
            result[1] = 0.0f;
            result[2] = 0.0f;
        }

    } else if (angl_ref < 0) {
        /* Second atom: place along anchor direction if available */
        if (dist_ref < n_atoms) {
            const float *ref = &coords[dist_ref * 3];
            if (anchor1 != NULL) {
                nerf_place_along_direction_impl(ref, anchor1, distance, result);
            } else {
                nerf_place_along_x_impl(ref, distance, result);
            }
        }

    } else if (dihe_ref < 0) {
        /* Third atom: place in anchored plane if available */
        if (dist_ref < n_atoms && angl_ref < n_atoms) {
            const float *ref1 = &coords[dist_ref * 3];
            const float *ref2 = &coords[angl_ref * 3];
            if (anchor2 != NULL) {
                nerf_place_in_plane_anchored_impl(ref1, ref2, anchor2, distance, angle, result);
            } else {
                nerf_place_in_plane_impl(ref1, ref2, distance, angle, result);
            }
        }

    } else {
        /* Full NERF placement */
        if (dihe_ref < n_atoms && angl_ref < n_atoms && dist_ref < n_atoms) {
            const float *p1 = &coords[dihe_ref * 3];
            const float *p2 = &coords[angl_ref * 3];
            const float *p3 = &coords[dist_ref * 3];
            nerf_place_atom_impl(p1, p2, p3, distance, angle, dihedral, result);
        }
    }
}


/**
 * Kernel: Backward pass for anchored NERF reconstruction at a single level.
 *
 * Input/Output: internal and grad_internal arrays with shape (n_entries, 3).
 */
__global__ void kernel_nerf_reconstruct_backward_level_anchored(
    const float *coords,
    const int64_t *indices,
    const float *internal,      /* (n_entries, 3) [dist, angle, dihedral] */
    float *grad_coords,
    float *grad_internal,       /* (n_entries, 3) gradient output */
    int level_start,
    int level_end,
    int n_atoms,
    const float *anchor_coords,
    const int32_t *component_ids
) {
    int i = level_start + blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= level_end) return;

    int64_t atom_idx = indices[i * 4 + 0];
    int64_t dist_ref = indices[i * 4 + 1];
    int64_t angl_ref = indices[i * 4 + 2];
    int64_t dihe_ref = indices[i * 4 + 3];

    if (atom_idx < 0 || atom_idx >= n_atoms) {
        grad_internal[INTERNAL_IDX(i, INTERNAL_DIST)] = 0.0f;
        grad_internal[INTERNAL_IDX(i, INTERNAL_ANGLE)] = 0.0f;
        grad_internal[INTERNAL_IDX(i, INTERNAL_DIHE)] = 0.0f;
        return;
    }

    /* Extract internal coordinates */
    float distance = internal[INTERNAL_IDX(i, INTERNAL_DIST)];
    float angle = internal[INTERNAL_IDX(i, INTERNAL_ANGLE)];
    float dihedral = internal[INTERNAL_IDX(i, INTERNAL_DIHE)];

    const float *grad_result = &grad_coords[atom_idx * 3];
    float grad_dist, grad_ang, grad_dih;

    /* Get anchors for this component */
    const float *anchor1 = NULL;
    const float *anchor2 = NULL;
    if (anchor_coords != NULL && component_ids != NULL) {
        int32_t comp_id = component_ids[i];
        anchor1 = &anchor_coords[comp_id * 9 + 3];
        anchor2 = &anchor_coords[comp_id * 9 + 6];
    }

    if (dist_ref < 0) {
        /* First atom: no gradients */
        grad_internal[INTERNAL_IDX(i, INTERNAL_DIST)] = 0.0f;
        grad_internal[INTERNAL_IDX(i, INTERNAL_ANGLE)] = 0.0f;
        grad_internal[INTERNAL_IDX(i, INTERNAL_DIHE)] = 0.0f;

    } else if (angl_ref < 0) {
        /* Second atom */
        if (dist_ref < n_atoms) {
            if (anchor1 != NULL) {
                const float *ref = &coords[dist_ref * 3];
                float grad_ref[3];
                nerf_place_along_direction_backward_impl(
                    ref, anchor1, distance,
                    grad_result, grad_ref, &grad_dist
                );
                grad_internal[INTERNAL_IDX(i, INTERNAL_DIST)] = grad_dist;
                grad_internal[INTERNAL_IDX(i, INTERNAL_ANGLE)] = 0.0f;
                grad_internal[INTERNAL_IDX(i, INTERNAL_DIHE)] = 0.0f;
                atomicAdd(&grad_coords[dist_ref * 3 + 0], grad_ref[0]);
                atomicAdd(&grad_coords[dist_ref * 3 + 1], grad_ref[1]);
                atomicAdd(&grad_coords[dist_ref * 3 + 2], grad_ref[2]);
            } else {
                grad_internal[INTERNAL_IDX(i, INTERNAL_DIST)] = grad_result[0];
                grad_internal[INTERNAL_IDX(i, INTERNAL_ANGLE)] = 0.0f;
                grad_internal[INTERNAL_IDX(i, INTERNAL_DIHE)] = 0.0f;
                atomicAdd(&grad_coords[dist_ref * 3 + 0], grad_result[0]);
                atomicAdd(&grad_coords[dist_ref * 3 + 1], grad_result[1]);
                atomicAdd(&grad_coords[dist_ref * 3 + 2], grad_result[2]);
            }
        }

    } else if (dihe_ref < 0) {
        /* Third atom */
        if (angl_ref < n_atoms && dist_ref < n_atoms) {
            const float *ref1 = &coords[dist_ref * 3];
            const float *ref2 = &coords[angl_ref * 3];
            float grad_ref1[3], grad_ref2[3];

            if (anchor2 != NULL) {
                nerf_place_in_plane_anchored_backward_impl(
                    ref1, ref2, anchor2,
                    distance, angle, grad_result,
                    grad_ref1, grad_ref2,
                    &grad_dist, &grad_ang
                );
            } else {
                nerf_place_in_plane_backward_impl(
                    ref1, ref2,
                    distance, angle, grad_result,
                    grad_ref1, grad_ref2,
                    &grad_dist, &grad_ang
                );
            }
            grad_internal[INTERNAL_IDX(i, INTERNAL_DIST)] = grad_dist;
            grad_internal[INTERNAL_IDX(i, INTERNAL_ANGLE)] = grad_ang;
            grad_internal[INTERNAL_IDX(i, INTERNAL_DIHE)] = 0.0f;

            atomicAdd(&grad_coords[dist_ref * 3 + 0], grad_ref1[0]);
            atomicAdd(&grad_coords[dist_ref * 3 + 1], grad_ref1[1]);
            atomicAdd(&grad_coords[dist_ref * 3 + 2], grad_ref1[2]);
            atomicAdd(&grad_coords[angl_ref * 3 + 0], grad_ref2[0]);
            atomicAdd(&grad_coords[angl_ref * 3 + 1], grad_ref2[1]);
            atomicAdd(&grad_coords[angl_ref * 3 + 2], grad_ref2[2]);
        } else {
            grad_internal[INTERNAL_IDX(i, INTERNAL_DIST)] = 0.0f;
            grad_internal[INTERNAL_IDX(i, INTERNAL_ANGLE)] = 0.0f;
            grad_internal[INTERNAL_IDX(i, INTERNAL_DIHE)] = 0.0f;
        }

    } else {
        /* Full NERF */
        if (dihe_ref < n_atoms && angl_ref < n_atoms && dist_ref < n_atoms) {
            const float *p1 = &coords[dihe_ref * 3];
            const float *p2 = &coords[angl_ref * 3];
            const float *p3 = &coords[dist_ref * 3];
            float grad_a[3], grad_b[3], grad_c[3];

            nerf_place_atom_backward_impl(
                p1, p2, p3, distance, angle, dihedral, grad_result,
                grad_a, grad_b, grad_c,
                &grad_dist, &grad_ang, &grad_dih
            );
            grad_internal[INTERNAL_IDX(i, INTERNAL_DIST)] = grad_dist;
            grad_internal[INTERNAL_IDX(i, INTERNAL_ANGLE)] = grad_ang;
            grad_internal[INTERNAL_IDX(i, INTERNAL_DIHE)] = grad_dih;

            atomicAdd(&grad_coords[dihe_ref * 3 + 0], grad_a[0]);
            atomicAdd(&grad_coords[dihe_ref * 3 + 1], grad_a[1]);
            atomicAdd(&grad_coords[dihe_ref * 3 + 2], grad_a[2]);
            atomicAdd(&grad_coords[angl_ref * 3 + 0], grad_b[0]);
            atomicAdd(&grad_coords[angl_ref * 3 + 1], grad_b[1]);
            atomicAdd(&grad_coords[angl_ref * 3 + 2], grad_b[2]);
            atomicAdd(&grad_coords[dist_ref * 3 + 0], grad_c[0]);
            atomicAdd(&grad_coords[dist_ref * 3 + 1], grad_c[1]);
            atomicAdd(&grad_coords[dist_ref * 3 + 2], grad_c[2]);
        } else {
            grad_internal[INTERNAL_IDX(i, INTERNAL_DIST)] = 0.0f;
            grad_internal[INTERNAL_IDX(i, INTERNAL_ANGLE)] = 0.0f;
            grad_internal[INTERNAL_IDX(i, INTERNAL_DIHE)] = 0.0f;
        }
    }
}


/**
 * CUDA implementation of batch_nerf_reconstruct_leveled_anchored.
 */
void cuda_batch_nerf_reconstruct_leveled_anchored(
    float *d_coords,
    size_t n_atoms,
    const int64_t *d_indices,
    size_t n_entries,
    const float *d_internal,        /* (n_entries, 3) */
    const int *level_offsets,
    int n_levels,
    const float *d_anchor_coords,
    const int32_t *d_component_ids,
    cudaStream_t stream
) {
    if (n_entries == 0) return;

    int threads = 256;

    for (int level = 0; level < n_levels; level++) {
        int level_start = level_offsets[level];
        int level_end = level_offsets[level + 1];
        int level_size = level_end - level_start;

        if (level_size > 0) {
            int blocks = (level_size + threads - 1) / threads;
            kernel_nerf_reconstruct_level_anchored<<<blocks, threads, 0, stream>>>(
                d_coords, d_indices, d_internal,
                level_start, level_end, (int)n_atoms,
                d_anchor_coords, d_component_ids
            );
        }
    }
}


/**
 * CUDA implementation of batch_nerf_reconstruct_backward_leveled_anchored.
 */
void cuda_batch_nerf_reconstruct_backward_leveled_anchored(
    const float *d_coords,
    size_t n_atoms,
    const int64_t *d_indices,
    size_t n_entries,
    const float *d_internal,        /* (n_entries, 3) */
    float *d_grad_coords,
    float *d_grad_internal,         /* (n_entries, 3) output */
    const int *level_offsets,
    int n_levels,
    const float *d_anchor_coords,
    const int32_t *d_component_ids,
    cudaStream_t stream
) {
    if (n_entries == 0) return;

    int threads = 256;

    for (int level = n_levels - 1; level >= 0; level--) {
        int level_start = level_offsets[level];
        int level_end = level_offsets[level + 1];
        int level_size = level_end - level_start;

        if (level_size > 0) {
            int blocks = (level_size + threads - 1) / threads;
            kernel_nerf_reconstruct_backward_level_anchored<<<blocks, threads, 0, stream>>>(
                d_coords, d_indices, d_internal,
                d_grad_coords, d_grad_internal,
                level_start, level_end, (int)n_atoms,
                d_anchor_coords, d_component_ids
            );
        }
    }
}

} /* extern "C" */
