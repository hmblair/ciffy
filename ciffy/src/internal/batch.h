/**
 * @file batch.h
 * @brief Batch operations for internal coordinate conversion.
 *
 * Provides batch versions of coordinate conversion that operate on
 * arrays, suitable for calling from Python with NumPy arrays.
 */

#ifndef CIFFY_INTERNAL_BATCH_H
#define CIFFY_INTERNAL_BATCH_H

#include <stdint.h>
#include <stddef.h>

/**
 * Batch conversion from Cartesian to internal coordinates.
 *
 * Computes bond lengths, angles, and dihedrals for each Z-matrix entry.
 *
 * @param coords Input Cartesian coordinates, shape (n_atoms, 3), row-major.
 * @param n_atoms Number of atoms.
 * @param indices Z-matrix indices, shape (n_entries, 4).
 *                Each row: [atom_idx, distance_ref, angle_ref, dihedral_ref].
 *                Use -1 for missing references.
 * @param n_entries Number of Z-matrix entries.
 * @param distances Output bond lengths, size (n_entries,).
 * @param angles Output bond angles in radians, size (n_entries,).
 * @param dihedrals Output dihedral angles in radians, size (n_entries,).
 */
void batch_cartesian_to_internal(
    const float *coords, size_t n_atoms,
    const int64_t *indices, size_t n_entries,
    float *distances, float *angles, float *dihedrals
);

/**
 * Batch NERF reconstruction from internal to Cartesian coordinates.
 *
 * Reconstructs Cartesian coordinates from internal coordinates.
 * MUST be called with entries in placement order (BFS order),
 * as each atom depends on previously placed atoms.
 *
 * @param coords Output Cartesian coordinates, shape (n_atoms, 3).
 *               Pre-allocated and zero-initialized.
 * @param n_atoms Number of atoms.
 * @param indices Z-matrix indices, shape (n_entries, 4).
 * @param n_entries Number of Z-matrix entries.
 * @param distances Bond lengths, size (n_entries,).
 * @param angles Bond angles in radians, size (n_entries,).
 * @param dihedrals Dihedral angles in radians, size (n_entries,).
 */
void batch_nerf_reconstruct(
    float *coords, size_t n_atoms,
    const int64_t *indices, size_t n_entries,
    const float *distances, const float *angles, const float *dihedrals
);

/* ========================================================================= */
/* Backward (gradient) functions for automatic differentiation              */
/* ========================================================================= */

/**
 * Backward pass for batch_cartesian_to_internal.
 *
 * Computes gradients of internal coordinates with respect to Cartesian coords.
 *
 * @param coords Input Cartesian coordinates, shape (n_atoms, 3).
 * @param n_atoms Number of atoms.
 * @param indices Z-matrix indices, shape (n_entries, 4).
 * @param n_entries Number of Z-matrix entries.
 * @param distances Forward pass distances (for efficiency).
 * @param angles Forward pass angles (for efficiency).
 * @param grad_distances Upstream gradients for distances, size (n_entries,).
 * @param grad_angles Upstream gradients for angles, size (n_entries,).
 * @param grad_dihedrals Upstream gradients for dihedrals, size (n_entries,).
 * @param grad_coords Output gradients for coords, shape (n_atoms, 3).
 *                    MUST be pre-initialized (gradients are accumulated).
 */
void batch_cartesian_to_internal_backward(
    const float *coords, size_t n_atoms,
    const int64_t *indices, size_t n_entries,
    const float *distances, const float *angles,
    const float *grad_distances, const float *grad_angles, const float *grad_dihedrals,
    float *grad_coords
);

/**
 * Backward pass for batch_nerf_reconstruct.
 *
 * Computes gradients with respect to internal coordinates.
 * MUST be called with entries in REVERSE placement order.
 *
 * @param coords Forward pass reconstructed coordinates, shape (n_atoms, 3).
 * @param n_atoms Number of atoms.
 * @param indices Z-matrix indices, shape (n_entries, 4).
 * @param n_entries Number of Z-matrix entries.
 * @param distances Bond lengths, size (n_entries,).
 * @param angles Bond angles, size (n_entries,).
 * @param dihedrals Dihedral angles, size (n_entries,).
 * @param grad_coords Upstream gradients for coords, shape (n_atoms, 3).
 *                    Will be modified during backward pass.
 * @param grad_distances Output gradients for distances, size (n_entries,).
 * @param grad_angles Output gradients for angles, size (n_entries,).
 * @param grad_dihedrals Output gradients for dihedrals, size (n_entries,).
 */
void batch_nerf_reconstruct_backward(
    const float *coords, size_t n_atoms,
    const int64_t *indices, size_t n_entries,
    const float *distances, const float *angles, const float *dihedrals,
    float *grad_coords,
    float *grad_distances, float *grad_angles, float *grad_dihedrals
);

/* ========================================================================= */
/* Level-parallel NERF (OpenMP)                                              */
/* ========================================================================= */

/**
 * Level-parallel NERF reconstruction.
 *
 * Processes atoms level-by-level where all atoms at the same BFS level
 * can be placed in parallel (they only depend on lower levels).
 * Uses OpenMP for parallelism within each level.
 *
 * @param coords Output coordinates, shape (n_atoms, 3). Pre-allocated.
 * @param n_atoms Number of atoms.
 * @param indices Z-matrix indices, shape (n_entries, 4). MUST be sorted by level.
 * @param n_entries Number of Z-matrix entries.
 * @param distances Bond lengths, size (n_entries,).
 * @param angles Bond angles in radians, size (n_entries,).
 * @param dihedrals Dihedral angles in radians, size (n_entries,).
 * @param level_offsets CSR-style offsets, size (n_levels+1,).
 *                      Level i's entries span [level_offsets[i], level_offsets[i+1]).
 * @param n_levels Number of BFS levels.
 */
void batch_nerf_reconstruct_leveled(
    float *coords, size_t n_atoms,
    const int64_t *indices, size_t n_entries,
    const float *distances, const float *angles, const float *dihedrals,
    const int32_t *level_offsets, int n_levels
);

/**
 * Level-parallel backward pass for NERF reconstruction.
 *
 * Processes levels in REVERSE order. Uses atomic adds for grad_coords
 * to handle concurrent writes from multiple entries referencing the same atom.
 *
 * @param coords Forward pass coordinates, shape (n_atoms, 3).
 * @param n_atoms Number of atoms.
 * @param indices Z-matrix indices, shape (n_entries, 4). MUST be sorted by level.
 * @param n_entries Number of Z-matrix entries.
 * @param distances Bond lengths, size (n_entries,).
 * @param angles Bond angles, size (n_entries,).
 * @param dihedrals Dihedral angles, size (n_entries,).
 * @param grad_coords Upstream gradients for coords, shape (n_atoms, 3). Modified in place.
 * @param grad_distances Output gradients for distances, size (n_entries,).
 * @param grad_angles Output gradients for angles, size (n_entries,).
 * @param grad_dihedrals Output gradients for dihedrals, size (n_entries,).
 * @param level_offsets CSR-style offsets, size (n_levels+1,).
 * @param n_levels Number of BFS levels.
 */
void batch_nerf_reconstruct_backward_leveled(
    const float *coords, size_t n_atoms,
    const int64_t *indices, size_t n_entries,
    const float *distances, const float *angles, const float *dihedrals,
    float *grad_coords,
    float *grad_distances, float *grad_angles, float *grad_dihedrals,
    const int32_t *level_offsets, int n_levels
);

#endif /* CIFFY_INTERNAL_BATCH_H */
