/**
 * @file batch.h
 * @brief Batch operations for internal coordinate conversion.
 *
 * Provides batch versions of coordinate conversion that operate on
 * arrays, suitable for calling from Python with NumPy arrays.
 *
 * Internal coordinates are stored as (N, 3) arrays where each row contains
 * [distance, angle, dihedral]. Use the INTERNAL_* constants below for indexing.
 */

#ifndef CIFFY_INTERNAL_BATCH_H
#define CIFFY_INTERNAL_BATCH_H

#include <stdint.h>
#include <stddef.h>

/* Column indices for internal coordinate (N, 3) arrays */
#define INTERNAL_DIST  0   /* Bond length (Angstroms) */
#define INTERNAL_ANGLE 1   /* Bond angle (radians) */
#define INTERNAL_DIHE  2   /* Dihedral angle (radians) */
#define INTERNAL_COLS  3   /* Number of columns in internal array */

/* Helper macro for accessing internal[i, col] in row-major layout */
#define INTERNAL_IDX(i, col) ((i) * INTERNAL_COLS + (col))

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
 * @param internal Output internal coordinates, shape (n_entries, 3), row-major.
 *                 Each row: [distance, angle, dihedral].
 */
void batch_cartesian_to_internal(
    const float *coords, size_t n_atoms,
    const int64_t *indices, size_t n_entries,
    float *internal
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
 * @param internal Forward pass internal coordinates, shape (n_entries, 3).
 *                 Each row: [distance, angle, dihedral].
 * @param grad_internal Upstream gradients for internal, shape (n_entries, 3).
 * @param grad_coords Output gradients for coords, shape (n_atoms, 3).
 *                    MUST be pre-initialized (gradients are accumulated).
 */
void batch_cartesian_to_internal_backward(
    const float *coords, size_t n_atoms,
    const int64_t *indices, size_t n_entries,
    const float *internal,
    const float *grad_internal,
    float *grad_coords
);

/* ========================================================================= */
/* Anchored NERF (places atoms in reference frame defined by anchor coords)  */
/* ========================================================================= */

/**
 * Component-parallel NERF reconstruction with anchor coordinates.
 *
 * Instead of placing the first 3 atoms in a canonical frame (origin, +X, XY plane),
 * this function places them using anchor coordinates from a reference structure.
 * This eliminates the need for post-reconstruction Kabsch rotation.
 *
 * Parallelizes across connected components (each component is independent),
 * while processing entries within each component sequentially.
 *
 * @param coords Output coordinates, shape (n_atoms, 3). Pre-allocated.
 * @param n_atoms Number of atoms.
 * @param indices Z-matrix indices, shape (n_entries, 4). MUST be sorted by component.
 * @param n_entries Number of Z-matrix entries.
 * @param internal Internal coordinates, shape (n_entries, 3), row-major.
 * @param component_offsets CSR-style offsets, size (n_components+1,).
 * @param n_components Number of connected components.
 * @param anchor_coords Anchor coordinates, shape (n_components, 3, 3).
 *                      For each component: anchor_coords[comp*9..comp*9+8] are
 *                      the 3 anchor positions (each 3 floats).
 *                      NULL to use canonical frame.
 * @param component_ids Component ID for each Z-matrix entry, size (n_entries,).
 *                      NULL if anchor_coords is NULL.
 */
void batch_nerf_reconstruct_leveled_anchored(
    float *coords, size_t n_atoms,
    const int64_t *indices, size_t n_entries,
    const float *internal,
    const int32_t *component_offsets, int n_components,
    const float *anchor_coords, const int32_t *component_ids
);

/**
 * Component-parallel backward pass for anchored NERF reconstruction.
 *
 * Parallelizes across connected components, processing entries within each
 * component in reverse order to properly propagate gradients.
 *
 * Note: Gradients do NOT flow through anchor_coords (they are frozen references).
 */
void batch_nerf_reconstruct_backward_leveled_anchored(
    const float *coords, size_t n_atoms,
    const int64_t *indices, size_t n_entries,
    const float *internal,
    float *grad_coords,
    float *grad_internal,
    const int32_t *component_offsets, int n_components,
    const float *anchor_coords, const int32_t *component_ids
);

#endif /* CIFFY_INTERNAL_BATCH_H */
