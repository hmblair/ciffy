/**
 * @file batch.cu
 * @brief CUDA kernels for batch coordinate conversions.
 *
 * This file contains GPU implementations of the batch coordinate conversion
 * operations. It uses the shared implementations from geometry_impl.h,
 * which are marked with CIFFY_HOST_DEVICE for CPU/GPU compatibility.
 */

#include "cuda_compat.h"
#include "geometry_impl.h"
#include "batch.h"

#include <cuda_runtime.h>
#include <stdint.h>


/* ========================================================================= */
/* CUDA Kernels                                                              */
/* ========================================================================= */

/**
 * Kernel: Convert Cartesian coordinates to internal coordinates.
 *
 * Each thread processes one Z-matrix entry independently (embarrassingly parallel).
 */
__global__ void kernel_cartesian_to_internal(
    const float *coords,       /* (n_atoms, 3) */
    const int64_t *indices,    /* (n_entries, 4) */
    int n_entries,
    int n_atoms,
    float *distances,
    float *angles,
    float *dihedrals
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_entries) return;

    int64_t atom_idx = indices[i * 4 + 0];
    int64_t dist_ref = indices[i * 4 + 1];
    int64_t angl_ref = indices[i * 4 + 2];
    int64_t dihe_ref = indices[i * 4 + 3];

    /* Bounds check */
    if (atom_idx < 0 || atom_idx >= n_atoms) {
        distances[i] = 0.0f;
        angles[i] = 0.0f;
        dihedrals[i] = 0.0f;
        return;
    }

    const float *atom = &coords[atom_idx * 3];

    /* Bond length */
    if (dist_ref >= 0 && dist_ref < n_atoms) {
        const float *ref1 = &coords[dist_ref * 3];
        distances[i] = compute_distance_impl(atom, ref1);
    } else {
        distances[i] = 0.0f;
    }

    /* Bond angle */
    if (angl_ref >= 0 && dist_ref >= 0 && angl_ref < n_atoms && dist_ref < n_atoms) {
        const float *ref1 = &coords[dist_ref * 3];
        const float *ref2 = &coords[angl_ref * 3];
        angles[i] = compute_angle_impl(atom, ref1, ref2);
    } else {
        angles[i] = 0.0f;
    }

    /* Dihedral angle */
    if (dihe_ref >= 0 && angl_ref >= 0 && dist_ref >= 0 &&
        dihe_ref < n_atoms && angl_ref < n_atoms && dist_ref < n_atoms) {
        const float *ref1 = &coords[dist_ref * 3];
        const float *ref2 = &coords[angl_ref * 3];
        const float *ref3 = &coords[dihe_ref * 3];
        dihedrals[i] = compute_dihedral_impl(ref3, ref2, ref1, atom);
    } else {
        dihedrals[i] = 0.0f;
    }
}


/**
 * Kernel: Backward pass for cartesian_to_internal.
 *
 * Each thread processes one Z-matrix entry and uses atomicAdd for gradient
 * accumulation since multiple entries may reference the same atoms.
 */
__global__ void kernel_cartesian_to_internal_backward(
    const float *coords,
    const int64_t *indices,
    int n_entries,
    int n_atoms,
    const float *distances,
    const float *angles,
    const float *grad_distances,
    const float *grad_angles,
    const float *grad_dihedrals,
    float *grad_coords  /* Output: atomically accumulated */
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

    /* Distance backward */
    if (dist_ref >= 0 && dist_ref < n_atoms) {
        const float *ref1 = &coords[dist_ref * 3];
        compute_distance_backward_impl(
            atom, ref1, distances[i], grad_distances[i],
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
            atom, ref1, ref2, angles[i], grad_angles[i],
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
            ref3, ref2, ref1, atom, grad_dihedrals[i],
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
 */
__global__ void kernel_nerf_reconstruct_level(
    float *coords,
    const int64_t *indices,
    const float *distances,
    const float *angles,
    const float *dihedrals,
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

    if (dist_ref < 0) {
        /* First atom: place at origin */
        result[0] = 0.0f;
        result[1] = 0.0f;
        result[2] = 0.0f;

    } else if (angl_ref < 0) {
        /* Second atom: place along +X from reference */
        if (dist_ref < n_atoms) {
            const float *ref = &coords[dist_ref * 3];
            nerf_place_along_x_impl(ref, distances[i], result);
        }

    } else if (dihe_ref < 0) {
        /* Third atom: place in plane */
        if (dist_ref < n_atoms && angl_ref < n_atoms) {
            const float *ref1 = &coords[dist_ref * 3];
            const float *ref2 = &coords[angl_ref * 3];
            nerf_place_in_plane_impl(ref1, ref2, distances[i], angles[i], result);
        }

    } else {
        /* Full NERF placement */
        if (dihe_ref < n_atoms && angl_ref < n_atoms && dist_ref < n_atoms) {
            const float *p1 = &coords[dihe_ref * 3];
            const float *p2 = &coords[angl_ref * 3];
            const float *p3 = &coords[dist_ref * 3];
            nerf_place_atom_impl(p1, p2, p3, distances[i], angles[i], dihedrals[i], result);
        }
    }
}


/**
 * Kernel: Backward pass for NERF reconstruction at a single level.
 *
 * Processes in reverse BFS order. Uses atomicAdd for gradient accumulation.
 */
__global__ void kernel_nerf_reconstruct_backward_level(
    const float *coords,
    const int64_t *indices,
    const float *distances,
    const float *angles,
    const float *dihedrals,
    float *grad_coords,
    float *grad_distances,
    float *grad_angles,
    float *grad_dihedrals,
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
        grad_distances[i] = 0.0f;
        grad_angles[i] = 0.0f;
        grad_dihedrals[i] = 0.0f;
        return;
    }

    const float *grad_result = &grad_coords[atom_idx * 3];
    float grad_a[3], grad_b[3], grad_c[3];

    if (dist_ref < 0) {
        /* First atom at origin: no gradients */
        grad_distances[i] = 0.0f;
        grad_angles[i] = 0.0f;
        grad_dihedrals[i] = 0.0f;

    } else if (angl_ref < 0) {
        /* Second atom: along +X */
        grad_distances[i] = grad_result[0];
        grad_angles[i] = 0.0f;
        grad_dihedrals[i] = 0.0f;

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
                ref1, ref2, distances[i], angles[i], grad_result,
                grad_ref1, grad_ref2, &grad_distances[i], &grad_angles[i]
            );
            grad_dihedrals[i] = 0.0f;

            atomicAdd(&grad_coords[dist_ref * 3 + 0], grad_ref1[0]);
            atomicAdd(&grad_coords[dist_ref * 3 + 1], grad_ref1[1]);
            atomicAdd(&grad_coords[dist_ref * 3 + 2], grad_ref1[2]);
            atomicAdd(&grad_coords[angl_ref * 3 + 0], grad_ref2[0]);
            atomicAdd(&grad_coords[angl_ref * 3 + 1], grad_ref2[1]);
            atomicAdd(&grad_coords[angl_ref * 3 + 2], grad_ref2[2]);
        } else {
            grad_distances[i] = 0.0f;
            grad_angles[i] = 0.0f;
            grad_dihedrals[i] = 0.0f;
        }

    } else {
        /* Full NERF */
        if (dihe_ref < n_atoms && angl_ref < n_atoms && dist_ref < n_atoms) {
            const float *p1 = &coords[dihe_ref * 3];
            const float *p2 = &coords[angl_ref * 3];
            const float *p3 = &coords[dist_ref * 3];

            nerf_place_atom_backward_impl(
                p1, p2, p3, distances[i], angles[i], dihedrals[i], grad_result,
                grad_a, grad_b, grad_c,
                &grad_distances[i], &grad_angles[i], &grad_dihedrals[i]
            );

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
            grad_distances[i] = 0.0f;
            grad_angles[i] = 0.0f;
            grad_dihedrals[i] = 0.0f;
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
 */
void cuda_batch_cartesian_to_internal(
    const float *d_coords,
    size_t n_atoms,
    const int64_t *d_indices,
    size_t n_entries,
    float *d_distances,
    float *d_angles,
    float *d_dihedrals,
    cudaStream_t stream
) {
    if (n_entries == 0) return;

    int threads = 256;
    int blocks = ((int)n_entries + threads - 1) / threads;

    kernel_cartesian_to_internal<<<blocks, threads, 0, stream>>>(
        d_coords, d_indices, (int)n_entries, (int)n_atoms,
        d_distances, d_angles, d_dihedrals
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
    const float *d_distances,
    const float *d_angles,
    const float *d_grad_distances,
    const float *d_grad_angles,
    const float *d_grad_dihedrals,
    float *d_grad_coords,
    cudaStream_t stream
) {
    if (n_entries == 0) return;

    int threads = 256;
    int blocks = ((int)n_entries + threads - 1) / threads;

    kernel_cartesian_to_internal_backward<<<blocks, threads, 0, stream>>>(
        d_coords, d_indices, (int)n_entries, (int)n_atoms,
        d_distances, d_angles,
        d_grad_distances, d_grad_angles, d_grad_dihedrals,
        d_grad_coords
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
    const float *d_distances,
    const float *d_angles,
    const float *d_dihedrals,
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
            d_coords, d_indices, d_distances, d_angles, d_dihedrals,
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
    const float *d_distances,
    const float *d_angles,
    const float *d_dihedrals,
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
                d_coords, d_indices, d_distances, d_angles, d_dihedrals,
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
    const float *d_distances,
    const float *d_angles,
    const float *d_dihedrals,
    float *d_grad_coords,
    float *d_grad_distances,
    float *d_grad_angles,
    float *d_grad_dihedrals,
    cudaStream_t stream
) {
    if (n_entries == 0) return;

    /* Process in reverse order */
    for (size_t i = n_entries; i > 0; i--) {
        kernel_nerf_reconstruct_backward_level<<<1, 1, 0, stream>>>(
            d_coords, d_indices, d_distances, d_angles, d_dihedrals,
            d_grad_coords, d_grad_distances, d_grad_angles, d_grad_dihedrals,
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
    const float *d_distances,
    const float *d_angles,
    const float *d_dihedrals,
    float *d_grad_coords,
    float *d_grad_distances,
    float *d_grad_angles,
    float *d_grad_dihedrals,
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
                d_coords, d_indices, d_distances, d_angles, d_dihedrals,
                d_grad_coords, d_grad_distances, d_grad_angles, d_grad_dihedrals,
                level_start, level_end, (int)n_atoms
            );
            /*
             * No explicit sync needed between levels on same stream.
             * CUDA guarantees kernel execution order and memory consistency.
             */
        }
    }
}

} /* extern "C" */
