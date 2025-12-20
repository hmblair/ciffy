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

/* INTERNAL_* macros are defined in batch.h (included above) */


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
    CIFFY_CUDA_CHECK_KERNEL();
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
    CIFFY_CUDA_CHECK_KERNEL();
}


/* ========================================================================= */
/* Component-parallel NERF (one thread per component)                        */
/* ========================================================================= */


/**
 * Kernel: Component-parallel NERF reconstruction.
 *
 * Each thread processes one connected component sequentially.
 * Components are independent so no synchronization needed.
 */
__global__ void kernel_nerf_reconstruct_component_parallel(
    float *coords,
    const int64_t *indices,
    const float *internal,           /* (n_entries, 3) */
    const int32_t *component_offsets, /* (n_components+1,) */
    int n_components,
    int n_atoms,
    const float *anchor_coords,
    const int32_t *component_ids
) {
    int comp = blockIdx.x * blockDim.x + threadIdx.x;
    if (comp >= n_components) return;

    int start = component_offsets[comp];
    int end = component_offsets[comp + 1];

    /* Process all entries in this component sequentially */
    for (int i = start; i < end; i++) {
        int64_t atom_idx = indices[i * 4 + 0];
        int64_t dist_ref = indices[i * 4 + 1];
        int64_t angl_ref = indices[i * 4 + 2];
        int64_t dihe_ref = indices[i * 4 + 3];

        if (atom_idx < 0 || atom_idx >= n_atoms) continue;

        float *result = &coords[atom_idx * 3];

        float distance = internal[INTERNAL_IDX(i, INTERNAL_DIST)];
        float angle = internal[INTERNAL_IDX(i, INTERNAL_ANGLE)];
        float dihedral = internal[INTERNAL_IDX(i, INTERNAL_DIHE)];

        /* Get anchors for this component */
        const float *anchor0 = NULL;
        const float *anchor1 = NULL;
        const float *anchor2 = NULL;
        if (anchor_coords != NULL && component_ids != NULL) {
            int32_t comp_id = component_ids[i];
            /* Skip entry if component_id is out of bounds */
            if (comp_id < 0 || comp_id >= n_components) {
                continue;
            }
            anchor0 = &anchor_coords[comp_id * 9 + 0];
            anchor1 = &anchor_coords[comp_id * 9 + 3];
            anchor2 = &anchor_coords[comp_id * 9 + 6];
        }

        if (dist_ref < 0) {
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
            if (dist_ref < n_atoms) {
                const float *ref = &coords[dist_ref * 3];
                if (anchor1 != NULL) {
                    nerf_place_along_direction_impl(ref, anchor1, distance, result);
                } else {
                    nerf_place_along_x_impl(ref, distance, result);
                }
            }
        } else if (dihe_ref < 0) {
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
            if (dihe_ref < n_atoms && angl_ref < n_atoms && dist_ref < n_atoms) {
                const float *p1 = &coords[dihe_ref * 3];
                const float *p2 = &coords[angl_ref * 3];
                const float *p3 = &coords[dist_ref * 3];
                nerf_place_atom_impl(p1, p2, p3, distance, angle, dihedral, result);
            }
        }
    }
}


/**
 * Kernel: Component-parallel backward pass for NERF reconstruction.
 *
 * Each thread processes one component in reverse order.
 */
__global__ void kernel_nerf_reconstruct_backward_component_parallel(
    const float *coords,
    const int64_t *indices,
    const float *internal,
    float *grad_coords,
    float *grad_internal,
    const int32_t *component_offsets,
    int n_components,
    int n_atoms,
    const float *anchor_coords,
    const int32_t *component_ids
) {
    int comp = blockIdx.x * blockDim.x + threadIdx.x;
    if (comp >= n_components) return;

    int start = component_offsets[comp];
    int end = component_offsets[comp + 1];

    /* Process in reverse order for gradient flow */
    for (int i = end - 1; i >= start; i--) {
        int64_t atom_idx = indices[i * 4 + 0];
        int64_t dist_ref = indices[i * 4 + 1];
        int64_t angl_ref = indices[i * 4 + 2];
        int64_t dihe_ref = indices[i * 4 + 3];

        if (atom_idx < 0 || atom_idx >= n_atoms) {
            grad_internal[INTERNAL_IDX(i, INTERNAL_DIST)] = 0.0f;
            grad_internal[INTERNAL_IDX(i, INTERNAL_ANGLE)] = 0.0f;
            grad_internal[INTERNAL_IDX(i, INTERNAL_DIHE)] = 0.0f;
            continue;
        }

        float distance = internal[INTERNAL_IDX(i, INTERNAL_DIST)];
        float angle = internal[INTERNAL_IDX(i, INTERNAL_ANGLE)];
        float dihedral = internal[INTERNAL_IDX(i, INTERNAL_DIHE)];

        const float *grad_result = &grad_coords[atom_idx * 3];
        float grad_dist, grad_ang, grad_dih;

        const float *anchor1 = NULL;
        const float *anchor2 = NULL;
        if (anchor_coords != NULL && component_ids != NULL) {
            int32_t comp_id = component_ids[i];
            /* Skip entry if component_id is out of bounds */
            if (comp_id < 0 || comp_id >= n_components) {
                grad_internal[INTERNAL_IDX(i, INTERNAL_DIST)] = 0.0f;
                grad_internal[INTERNAL_IDX(i, INTERNAL_ANGLE)] = 0.0f;
                grad_internal[INTERNAL_IDX(i, INTERNAL_DIHE)] = 0.0f;
                continue;
            }
            anchor1 = &anchor_coords[comp_id * 9 + 3];
            anchor2 = &anchor_coords[comp_id * 9 + 6];
        }

        if (dist_ref < 0) {
            grad_internal[INTERNAL_IDX(i, INTERNAL_DIST)] = 0.0f;
            grad_internal[INTERNAL_IDX(i, INTERNAL_ANGLE)] = 0.0f;
            grad_internal[INTERNAL_IDX(i, INTERNAL_DIHE)] = 0.0f;

        } else if (angl_ref < 0) {
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
}


/**
 * CUDA implementation of batch_nerf_reconstruct_leveled_anchored.
 *
 * Uses component-parallel approach: single kernel launch with one thread
 * per component. Each thread processes its component sequentially.
 */
void cuda_batch_nerf_reconstruct_leveled_anchored(
    float *d_coords,
    size_t n_atoms,
    const int64_t *d_indices,
    size_t n_entries,
    const float *d_internal,
    const int32_t *component_offsets,
    int n_components,
    const float *d_anchor_coords,
    const int32_t *d_component_ids,
    cudaStream_t stream
) {
    if (n_entries == 0 || n_components == 0) return;

    int threads = 256;
    int blocks = (n_components + threads - 1) / threads;

    kernel_nerf_reconstruct_component_parallel<<<blocks, threads, 0, stream>>>(
        d_coords, d_indices, d_internal,
        component_offsets, n_components, (int)n_atoms,
        d_anchor_coords, d_component_ids
    );
    CIFFY_CUDA_CHECK_KERNEL();
}


/**
 * CUDA implementation of batch_nerf_reconstruct_backward_leveled_anchored.
 *
 * Uses component-parallel approach with single kernel launch.
 */
void cuda_batch_nerf_reconstruct_backward_leveled_anchored(
    const float *d_coords,
    size_t n_atoms,
    const int64_t *d_indices,
    size_t n_entries,
    const float *d_internal,
    float *d_grad_coords,
    float *d_grad_internal,
    const int32_t *component_offsets,
    int n_components,
    const float *d_anchor_coords,
    const int32_t *d_component_ids,
    cudaStream_t stream
) {
    if (n_entries == 0 || n_components == 0) return;

    int threads = 256;
    int blocks = (n_components + threads - 1) / threads;

    kernel_nerf_reconstruct_backward_component_parallel<<<blocks, threads, 0, stream>>>(
        d_coords, d_indices, d_internal,
        d_grad_coords, d_grad_internal,
        component_offsets, n_components, (int)n_atoms,
        d_anchor_coords, d_component_ids
    );
    CIFFY_CUDA_CHECK_KERNEL();
}

} /* extern "C" */
