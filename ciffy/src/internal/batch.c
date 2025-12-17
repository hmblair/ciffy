/**
 * @file batch.c
 * @brief Batch operations for internal coordinate conversion.
 */

#include "batch.h"
#include "geometry.h"
#include "geometry_impl.h"
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif


/* ========================================================================= */
/* Helper: single-entry NERF placement (reused by sequential and parallel)  */
/* ========================================================================= */


static inline void nerf_place_single_entry(
    float *coords, size_t n_atoms,
    const int64_t *indices,
    const float *distances, const float *angles, const float *dihedrals,
    size_t i
) {
    int64_t atom_idx = indices[i * 4 + 0];
    int64_t dist_ref = indices[i * 4 + 1];
    int64_t angl_ref = indices[i * 4 + 2];
    int64_t dihe_ref = indices[i * 4 + 3];

#ifdef CIFFY_INTERNAL_BOUNDS_CHECK
    if (atom_idx < 0 || (size_t)atom_idx >= n_atoms) {
        return;
    }
#endif
    float *result = &coords[atom_idx * 3];

    if (dist_ref < 0) {
        /* First atom: place at origin */
        result[0] = 0.0f;
        result[1] = 0.0f;
        result[2] = 0.0f;

    } else if (angl_ref < 0) {
        /* Second atom: place along +X from distance reference */
#ifdef CIFFY_INTERNAL_BOUNDS_CHECK
        if ((size_t)dist_ref >= n_atoms) {
            result[0] = result[1] = result[2] = 0.0f;
        } else
#endif
        {
            const float *ref = &coords[dist_ref * 3];
            nerf_place_along_x(ref, distances[i], result);
        }

    } else if (dihe_ref < 0) {
        /* Third atom: place in plane */
#ifdef CIFFY_INTERNAL_BOUNDS_CHECK
        if ((size_t)dist_ref >= n_atoms || (size_t)angl_ref >= n_atoms) {
            result[0] = result[1] = result[2] = 0.0f;
        } else
#endif
        {
            const float *ref1 = &coords[dist_ref * 3];
            const float *ref2 = &coords[angl_ref * 3];
            nerf_place_in_plane(ref1, ref2, distances[i], angles[i], result);
        }

    } else {
        /* Full NERF placement */
#ifdef CIFFY_INTERNAL_BOUNDS_CHECK
        if ((size_t)dihe_ref >= n_atoms ||
            (size_t)angl_ref >= n_atoms ||
            (size_t)dist_ref >= n_atoms) {
            result[0] = result[1] = result[2] = 0.0f;
        } else
#endif
        {
            const float *p1 = &coords[dihe_ref * 3];
            const float *p2 = &coords[angl_ref * 3];
            const float *p3 = &coords[dist_ref * 3];
            nerf_place_atom(p1, p2, p3, distances[i], angles[i], dihedrals[i], result);
        }
    }
}


void batch_cartesian_to_internal(
    const float *coords, size_t n_atoms,
    const int64_t *indices, size_t n_entries,
    float *distances, float *angles, float *dihedrals
) {
    /* Embarrassingly parallel: each entry reads independently from coords */
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (size_t i = 0; i < n_entries; i++) {
        int64_t atom_idx = indices[i * 4 + 0];
        int64_t dist_ref = indices[i * 4 + 1];
        int64_t angl_ref = indices[i * 4 + 2];
        int64_t dihe_ref = indices[i * 4 + 3];

#ifdef CIFFY_INTERNAL_BOUNDS_CHECK
        /* Guard against invalid indices in debug builds */
        if (atom_idx < 0 || (size_t)atom_idx >= n_atoms) {
            distances[i] = angles[i] = dihedrals[i] = 0.0f;
            continue;
        }
#endif
        const float *atom = &coords[atom_idx * 3];

        /* Bond length */
        if (dist_ref >= 0) {
#ifdef CIFFY_INTERNAL_BOUNDS_CHECK
            if ((size_t)dist_ref >= n_atoms) {
                distances[i] = 0.0f;
            } else
#endif
            {
                const float *ref1 = &coords[dist_ref * 3];
                distances[i] = compute_distance(atom, ref1);
            }
        } else {
            distances[i] = 0.0f;
        }

        /* Bond angle */
        if (angl_ref >= 0 && dist_ref >= 0) {
#ifdef CIFFY_INTERNAL_BOUNDS_CHECK
            if ((size_t)angl_ref >= n_atoms || (size_t)dist_ref >= n_atoms) {
                angles[i] = 0.0f;
            } else
#endif
            {
                const float *ref1 = &coords[dist_ref * 3];
                const float *ref2 = &coords[angl_ref * 3];
                /* Angle at dist_ref between atom and angl_ref */
                angles[i] = compute_angle(atom, ref1, ref2);
            }
        } else {
            angles[i] = 0.0f;
        }

        /* Dihedral angle */
        if (dihe_ref >= 0 && angl_ref >= 0 && dist_ref >= 0) {
#ifdef CIFFY_INTERNAL_BOUNDS_CHECK
            if ((size_t)dihe_ref >= n_atoms ||
                (size_t)angl_ref >= n_atoms ||
                (size_t)dist_ref >= n_atoms) {
                dihedrals[i] = 0.0f;
            } else
#endif
            {
                const float *ref1 = &coords[dist_ref * 3];
                const float *ref2 = &coords[angl_ref * 3];
                const float *ref3 = &coords[dihe_ref * 3];
                /* Dihedral: dihe_ref - angl_ref - dist_ref - atom */
                dihedrals[i] = compute_dihedral(ref3, ref2, ref1, atom);
            }
        } else {
            dihedrals[i] = 0.0f;
        }
    }
}


void batch_nerf_reconstruct(
    float *coords, size_t n_atoms,
    const int64_t *indices, size_t n_entries,
    const float *distances, const float *angles, const float *dihedrals
) {
    for (size_t i = 0; i < n_entries; i++) {
        int64_t atom_idx = indices[i * 4 + 0];
        int64_t dist_ref = indices[i * 4 + 1];
        int64_t angl_ref = indices[i * 4 + 2];
        int64_t dihe_ref = indices[i * 4 + 3];

#ifdef CIFFY_INTERNAL_BOUNDS_CHECK
        if (atom_idx < 0 || (size_t)atom_idx >= n_atoms) {
            continue;
        }
#endif
        float *result = &coords[atom_idx * 3];

        if (dist_ref < 0) {
            /* First atom: place at origin */
            result[0] = 0.0f;
            result[1] = 0.0f;
            result[2] = 0.0f;

        } else if (angl_ref < 0) {
            /* Second atom: place along +X from distance reference */
#ifdef CIFFY_INTERNAL_BOUNDS_CHECK
            if ((size_t)dist_ref >= n_atoms) {
                result[0] = result[1] = result[2] = 0.0f;
            } else
#endif
            {
                const float *ref = &coords[dist_ref * 3];
                nerf_place_along_x(ref, distances[i], result);
            }

        } else if (dihe_ref < 0) {
            /* Third atom: place in plane */
#ifdef CIFFY_INTERNAL_BOUNDS_CHECK
            if ((size_t)dist_ref >= n_atoms || (size_t)angl_ref >= n_atoms) {
                result[0] = result[1] = result[2] = 0.0f;
            } else
#endif
            {
                const float *ref1 = &coords[dist_ref * 3];
                const float *ref2 = &coords[angl_ref * 3];
                nerf_place_in_plane(ref1, ref2, distances[i], angles[i], result);
            }

        } else {
            /* Full NERF placement */
#ifdef CIFFY_INTERNAL_BOUNDS_CHECK
            if ((size_t)dihe_ref >= n_atoms ||
                (size_t)angl_ref >= n_atoms ||
                (size_t)dist_ref >= n_atoms) {
                result[0] = result[1] = result[2] = 0.0f;
            } else
#endif
            {
                const float *p1 = &coords[dihe_ref * 3];
                const float *p2 = &coords[angl_ref * 3];
                const float *p3 = &coords[dist_ref * 3];
                nerf_place_atom(p1, p2, p3, distances[i], angles[i], dihedrals[i], result);
            }
        }
    }
}


/* ========================================================================= */
/* Backward (gradient) functions                                             */
/* ========================================================================= */


void batch_cartesian_to_internal_backward(
    const float *coords, size_t n_atoms,
    const int64_t *indices, size_t n_entries,
    const float *distances, const float *angles,
    const float *grad_distances, const float *grad_angles, const float *grad_dihedrals,
    float *grad_coords
) {
    /* Parallel with atomic accumulation since multiple entries reference same atoms */
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (size_t i = 0; i < n_entries; i++) {
        /* Thread-local gradient storage */
        float grad_atom[3], grad_ref1[3], grad_ref2[3], grad_ref3[3];

        int64_t atom_idx = indices[i * 4 + 0];
        int64_t dist_ref = indices[i * 4 + 1];
        int64_t angl_ref = indices[i * 4 + 2];
        int64_t dihe_ref = indices[i * 4 + 3];

        if (atom_idx < 0 || (size_t)atom_idx >= n_atoms) continue;

        const float *atom = &coords[atom_idx * 3];

        /* Distance gradient */
        if (dist_ref >= 0 && (size_t)dist_ref < n_atoms) {
            const float *ref1 = &coords[dist_ref * 3];
            compute_distance_backward(
                atom, ref1,
                distances[i], grad_distances[i],
                grad_atom, grad_ref1
            );
            /* Atomic accumulate gradients for thread safety */
#ifdef _OPENMP
            #pragma omp atomic
#endif
            grad_coords[atom_idx * 3 + 0] += grad_atom[0];
#ifdef _OPENMP
            #pragma omp atomic
#endif
            grad_coords[atom_idx * 3 + 1] += grad_atom[1];
#ifdef _OPENMP
            #pragma omp atomic
#endif
            grad_coords[atom_idx * 3 + 2] += grad_atom[2];
#ifdef _OPENMP
            #pragma omp atomic
#endif
            grad_coords[dist_ref * 3 + 0] += grad_ref1[0];
#ifdef _OPENMP
            #pragma omp atomic
#endif
            grad_coords[dist_ref * 3 + 1] += grad_ref1[1];
#ifdef _OPENMP
            #pragma omp atomic
#endif
            grad_coords[dist_ref * 3 + 2] += grad_ref1[2];
        }

        /* Angle gradient */
        if (angl_ref >= 0 && dist_ref >= 0 &&
            (size_t)angl_ref < n_atoms && (size_t)dist_ref < n_atoms) {
            const float *ref1 = &coords[dist_ref * 3];
            const float *ref2 = &coords[angl_ref * 3];
            /* Angle at dist_ref between atom and angl_ref */
            compute_angle_backward(
                atom, ref1, ref2,
                angles[i], grad_angles[i],
                grad_atom, grad_ref1, grad_ref2
            );
#ifdef _OPENMP
            #pragma omp atomic
#endif
            grad_coords[atom_idx * 3 + 0] += grad_atom[0];
#ifdef _OPENMP
            #pragma omp atomic
#endif
            grad_coords[atom_idx * 3 + 1] += grad_atom[1];
#ifdef _OPENMP
            #pragma omp atomic
#endif
            grad_coords[atom_idx * 3 + 2] += grad_atom[2];
#ifdef _OPENMP
            #pragma omp atomic
#endif
            grad_coords[dist_ref * 3 + 0] += grad_ref1[0];
#ifdef _OPENMP
            #pragma omp atomic
#endif
            grad_coords[dist_ref * 3 + 1] += grad_ref1[1];
#ifdef _OPENMP
            #pragma omp atomic
#endif
            grad_coords[dist_ref * 3 + 2] += grad_ref1[2];
#ifdef _OPENMP
            #pragma omp atomic
#endif
            grad_coords[angl_ref * 3 + 0] += grad_ref2[0];
#ifdef _OPENMP
            #pragma omp atomic
#endif
            grad_coords[angl_ref * 3 + 1] += grad_ref2[1];
#ifdef _OPENMP
            #pragma omp atomic
#endif
            grad_coords[angl_ref * 3 + 2] += grad_ref2[2];
        }

        /* Dihedral gradient */
        if (dihe_ref >= 0 && angl_ref >= 0 && dist_ref >= 0 &&
            (size_t)dihe_ref < n_atoms &&
            (size_t)angl_ref < n_atoms &&
            (size_t)dist_ref < n_atoms) {
            const float *ref1 = &coords[dist_ref * 3];
            const float *ref2 = &coords[angl_ref * 3];
            const float *ref3 = &coords[dihe_ref * 3];
            /* Dihedral: dihe_ref - angl_ref - dist_ref - atom */
            compute_dihedral_backward(
                ref3, ref2, ref1, atom,
                grad_dihedrals[i],
                grad_ref3, grad_ref2, grad_ref1, grad_atom
            );
#ifdef _OPENMP
            #pragma omp atomic
#endif
            grad_coords[dihe_ref * 3 + 0] += grad_ref3[0];
#ifdef _OPENMP
            #pragma omp atomic
#endif
            grad_coords[dihe_ref * 3 + 1] += grad_ref3[1];
#ifdef _OPENMP
            #pragma omp atomic
#endif
            grad_coords[dihe_ref * 3 + 2] += grad_ref3[2];
#ifdef _OPENMP
            #pragma omp atomic
#endif
            grad_coords[angl_ref * 3 + 0] += grad_ref2[0];
#ifdef _OPENMP
            #pragma omp atomic
#endif
            grad_coords[angl_ref * 3 + 1] += grad_ref2[1];
#ifdef _OPENMP
            #pragma omp atomic
#endif
            grad_coords[angl_ref * 3 + 2] += grad_ref2[2];
#ifdef _OPENMP
            #pragma omp atomic
#endif
            grad_coords[dist_ref * 3 + 0] += grad_ref1[0];
#ifdef _OPENMP
            #pragma omp atomic
#endif
            grad_coords[dist_ref * 3 + 1] += grad_ref1[1];
#ifdef _OPENMP
            #pragma omp atomic
#endif
            grad_coords[dist_ref * 3 + 2] += grad_ref1[2];
#ifdef _OPENMP
            #pragma omp atomic
#endif
            grad_coords[atom_idx * 3 + 0] += grad_atom[0];
#ifdef _OPENMP
            #pragma omp atomic
#endif
            grad_coords[atom_idx * 3 + 1] += grad_atom[1];
#ifdef _OPENMP
            #pragma omp atomic
#endif
            grad_coords[atom_idx * 3 + 2] += grad_atom[2];
        }
    }
}


void batch_nerf_reconstruct_backward(
    const float *coords, size_t n_atoms,
    const int64_t *indices, size_t n_entries,
    const float *distances, const float *angles, const float *dihedrals,
    float *grad_coords,
    float *grad_distances, float *grad_angles, float *grad_dihedrals
) {
    /* Temporary gradient storage */
    float grad_a[3], grad_b[3], grad_c[3];

    /* Process in REVERSE order since later atoms depend on earlier ones */
    for (size_t i = n_entries; i > 0; i--) {
        size_t idx = i - 1;
        int64_t atom_idx = indices[idx * 4 + 0];
        int64_t dist_ref = indices[idx * 4 + 1];
        int64_t angl_ref = indices[idx * 4 + 2];
        int64_t dihe_ref = indices[idx * 4 + 3];

        if (atom_idx < 0 || (size_t)atom_idx >= n_atoms) {
            grad_distances[idx] = 0.0f;
            grad_angles[idx] = 0.0f;
            grad_dihedrals[idx] = 0.0f;
            continue;
        }

        const float *grad_result = &grad_coords[atom_idx * 3];

        if (dist_ref < 0) {
            /* First atom at origin: no gradients to propagate */
            grad_distances[idx] = 0.0f;
            grad_angles[idx] = 0.0f;
            grad_dihedrals[idx] = 0.0f;

        } else if (angl_ref < 0) {
            /* Second atom: placed along +X from distance reference */
            /* result = ref + (distance, 0, 0) */
            /* ∂result/∂distance = (1, 0, 0) */
            /* ∂result/∂ref = I */
            grad_distances[idx] = grad_result[0];  /* Only x component */
            grad_angles[idx] = 0.0f;
            grad_dihedrals[idx] = 0.0f;

            if ((size_t)dist_ref < n_atoms) {
                /* Propagate gradient to reference atom */
                grad_coords[dist_ref * 3 + 0] += grad_result[0];
                grad_coords[dist_ref * 3 + 1] += grad_result[1];
                grad_coords[dist_ref * 3 + 2] += grad_result[2];
            }

        } else if (dihe_ref < 0) {
            /* Third atom: placed in plane */
            if ((size_t)angl_ref < n_atoms && (size_t)dist_ref < n_atoms) {
                const float *ref1 = &coords[dist_ref * 3];
                const float *ref2 = &coords[angl_ref * 3];

                float grad_ref1[3], grad_ref2[3];
                nerf_place_in_plane_backward(
                    ref1, ref2,
                    distances[idx], angles[idx],
                    grad_result,
                    grad_ref1, grad_ref2,
                    &grad_distances[idx], &grad_angles[idx]
                );
                grad_dihedrals[idx] = 0.0f;

                /* Propagate gradients to reference atoms */
                grad_coords[dist_ref * 3 + 0] += grad_ref1[0];
                grad_coords[dist_ref * 3 + 1] += grad_ref1[1];
                grad_coords[dist_ref * 3 + 2] += grad_ref1[2];
                grad_coords[angl_ref * 3 + 0] += grad_ref2[0];
                grad_coords[angl_ref * 3 + 1] += grad_ref2[1];
                grad_coords[angl_ref * 3 + 2] += grad_ref2[2];
            } else {
                grad_distances[idx] = 0.0f;
                grad_angles[idx] = 0.0f;
                grad_dihedrals[idx] = 0.0f;
            }

        } else {
            /* Full NERF placement */
            if ((size_t)dihe_ref < n_atoms &&
                (size_t)angl_ref < n_atoms &&
                (size_t)dist_ref < n_atoms) {

                const float *p1 = &coords[dihe_ref * 3];
                const float *p2 = &coords[angl_ref * 3];
                const float *p3 = &coords[dist_ref * 3];

                nerf_place_atom_backward(
                    p1, p2, p3,
                    distances[idx], angles[idx], dihedrals[idx],
                    grad_result,
                    grad_a, grad_b, grad_c,
                    &grad_distances[idx], &grad_angles[idx], &grad_dihedrals[idx]
                );

                /* Propagate gradients to reference atoms */
                grad_coords[dihe_ref * 3 + 0] += grad_a[0];
                grad_coords[dihe_ref * 3 + 1] += grad_a[1];
                grad_coords[dihe_ref * 3 + 2] += grad_a[2];
                grad_coords[angl_ref * 3 + 0] += grad_b[0];
                grad_coords[angl_ref * 3 + 1] += grad_b[1];
                grad_coords[angl_ref * 3 + 2] += grad_b[2];
                grad_coords[dist_ref * 3 + 0] += grad_c[0];
                grad_coords[dist_ref * 3 + 1] += grad_c[1];
                grad_coords[dist_ref * 3 + 2] += grad_c[2];
            } else {
                grad_distances[idx] = 0.0f;
                grad_angles[idx] = 0.0f;
                grad_dihedrals[idx] = 0.0f;
            }
        }
    }
}


/* ========================================================================= */
/* Level-parallel NERF functions (OpenMP)                                    */
/* ========================================================================= */


/* Helper: single-entry NERF placement with anchor support */
static inline void nerf_place_single_entry_anchored(
    float *coords, size_t n_atoms,
    const int64_t *indices,
    const float *distances, const float *angles, const float *dihedrals,
    size_t i,
    const float *anchor_coords, const int32_t *component_ids
) {
    int64_t atom_idx = indices[i * 4 + 0];
    int64_t dist_ref = indices[i * 4 + 1];
    int64_t angl_ref = indices[i * 4 + 2];
    int64_t dihe_ref = indices[i * 4 + 3];

#ifdef CIFFY_INTERNAL_BOUNDS_CHECK
    if (atom_idx < 0 || (size_t)atom_idx >= n_atoms) {
        return;
    }
#endif
    float *result = &coords[atom_idx * 3];

    /* Get anchors for this component if available */
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
            result[0] = anchor0[0];
            result[1] = anchor0[1];
            result[2] = anchor0[2];
        } else {
            result[0] = 0.0f;
            result[1] = 0.0f;
            result[2] = 0.0f;
        }

    } else if (angl_ref < 0) {
        /* Second atom: place along anchor direction if available */
#ifdef CIFFY_INTERNAL_BOUNDS_CHECK
        if ((size_t)dist_ref >= n_atoms) {
            result[0] = result[1] = result[2] = 0.0f;
        } else
#endif
        {
            const float *ref = &coords[dist_ref * 3];
            if (anchor1 != NULL) {
                nerf_place_along_direction_impl(ref, anchor1, distances[i], result);
            } else {
                nerf_place_along_x_impl(ref, distances[i], result);
            }
        }

    } else if (dihe_ref < 0) {
        /* Third atom: place in anchored plane if available */
#ifdef CIFFY_INTERNAL_BOUNDS_CHECK
        if ((size_t)dist_ref >= n_atoms || (size_t)angl_ref >= n_atoms) {
            result[0] = result[1] = result[2] = 0.0f;
        } else
#endif
        {
            const float *ref1 = &coords[dist_ref * 3];
            const float *ref2 = &coords[angl_ref * 3];
            if (anchor2 != NULL) {
                nerf_place_in_plane_anchored_impl(ref1, ref2, anchor2, distances[i], angles[i], result);
            } else {
                nerf_place_in_plane_impl(ref1, ref2, distances[i], angles[i], result);
            }
        }

    } else {
        /* Full NERF placement (no anchor needed) */
#ifdef CIFFY_INTERNAL_BOUNDS_CHECK
        if ((size_t)dihe_ref >= n_atoms ||
            (size_t)angl_ref >= n_atoms ||
            (size_t)dist_ref >= n_atoms) {
            result[0] = result[1] = result[2] = 0.0f;
        } else
#endif
        {
            const float *p1 = &coords[dihe_ref * 3];
            const float *p2 = &coords[angl_ref * 3];
            const float *p3 = &coords[dist_ref * 3];
            nerf_place_atom_impl(p1, p2, p3, distances[i], angles[i], dihedrals[i], result);
        }
    }
}


void batch_nerf_reconstruct_leveled(
    float *coords, size_t n_atoms,
    const int64_t *indices, size_t n_entries,
    const float *distances, const float *angles, const float *dihedrals,
    const int32_t *level_offsets, int n_levels
) {
    (void)n_entries;  /* Used implicitly via level_offsets */

    /* Single parallel region with barriers between levels.
     * This avoids the overhead of spawning threads for each level. */
#ifdef _OPENMP
    #pragma omp parallel
    {
#endif
        for (int level = 0; level < n_levels; level++) {
            int start = level_offsets[level];
            int end = level_offsets[level + 1];

#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (int i = start; i < end; i++) {
                nerf_place_single_entry(coords, n_atoms, indices, distances, angles, dihedrals, (size_t)i);
            }
            /* Implicit barrier at end of omp for ensures level completes before next */
        }
#ifdef _OPENMP
    }
#endif
}


void batch_nerf_reconstruct_backward_leveled(
    const float *coords, size_t n_atoms,
    const int64_t *indices, size_t n_entries,
    const float *distances, const float *angles, const float *dihedrals,
    float *grad_coords,
    float *grad_distances, float *grad_angles, float *grad_dihedrals,
    const int32_t *level_offsets, int n_levels
) {
    (void)n_entries;  /* Used implicitly via level_offsets */

    /* Single parallel region with barriers between levels.
     * Process levels in REVERSE order since later atoms depend on earlier ones. */
#ifdef _OPENMP
    #pragma omp parallel
    {
#endif
        for (int level = n_levels - 1; level >= 0; level--) {
            int start = level_offsets[level];
            int end = level_offsets[level + 1];

#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (int i = start; i < end; i++) {
            size_t idx = (size_t)i;
            int64_t atom_idx = indices[idx * 4 + 0];
            int64_t dist_ref = indices[idx * 4 + 1];
            int64_t angl_ref = indices[idx * 4 + 2];
            int64_t dihe_ref = indices[idx * 4 + 3];

            if (atom_idx < 0 || (size_t)atom_idx >= n_atoms) {
                grad_distances[idx] = 0.0f;
                grad_angles[idx] = 0.0f;
                grad_dihedrals[idx] = 0.0f;
                continue;
            }

            const float *grad_result = &grad_coords[atom_idx * 3];

            if (dist_ref < 0) {
                /* First atom at origin: no gradients to propagate */
                grad_distances[idx] = 0.0f;
                grad_angles[idx] = 0.0f;
                grad_dihedrals[idx] = 0.0f;

            } else if (angl_ref < 0) {
                /* Second atom: placed along +X from distance reference */
                grad_distances[idx] = grad_result[0];
                grad_angles[idx] = 0.0f;
                grad_dihedrals[idx] = 0.0f;

                if ((size_t)dist_ref < n_atoms) {
                    /* Atomic add for thread safety */
#ifdef _OPENMP
                    #pragma omp atomic
#endif
                    grad_coords[dist_ref * 3 + 0] += grad_result[0];
#ifdef _OPENMP
                    #pragma omp atomic
#endif
                    grad_coords[dist_ref * 3 + 1] += grad_result[1];
#ifdef _OPENMP
                    #pragma omp atomic
#endif
                    grad_coords[dist_ref * 3 + 2] += grad_result[2];
                }

            } else if (dihe_ref < 0) {
                /* Third atom: placed in plane */
                if ((size_t)angl_ref < n_atoms && (size_t)dist_ref < n_atoms) {
                    const float *ref1 = &coords[dist_ref * 3];
                    const float *ref2 = &coords[angl_ref * 3];

                    float grad_ref1[3], grad_ref2[3];
                    nerf_place_in_plane_backward(
                        ref1, ref2,
                        distances[idx], angles[idx],
                        grad_result,
                        grad_ref1, grad_ref2,
                        &grad_distances[idx], &grad_angles[idx]
                    );
                    grad_dihedrals[idx] = 0.0f;

                    /* Atomic adds for thread safety */
#ifdef _OPENMP
                    #pragma omp atomic
#endif
                    grad_coords[dist_ref * 3 + 0] += grad_ref1[0];
#ifdef _OPENMP
                    #pragma omp atomic
#endif
                    grad_coords[dist_ref * 3 + 1] += grad_ref1[1];
#ifdef _OPENMP
                    #pragma omp atomic
#endif
                    grad_coords[dist_ref * 3 + 2] += grad_ref1[2];
#ifdef _OPENMP
                    #pragma omp atomic
#endif
                    grad_coords[angl_ref * 3 + 0] += grad_ref2[0];
#ifdef _OPENMP
                    #pragma omp atomic
#endif
                    grad_coords[angl_ref * 3 + 1] += grad_ref2[1];
#ifdef _OPENMP
                    #pragma omp atomic
#endif
                    grad_coords[angl_ref * 3 + 2] += grad_ref2[2];
                } else {
                    grad_distances[idx] = 0.0f;
                    grad_angles[idx] = 0.0f;
                    grad_dihedrals[idx] = 0.0f;
                }

            } else {
                /* Full NERF placement */
                if ((size_t)dihe_ref < n_atoms &&
                    (size_t)angl_ref < n_atoms &&
                    (size_t)dist_ref < n_atoms) {

                    const float *p1 = &coords[dihe_ref * 3];
                    const float *p2 = &coords[angl_ref * 3];
                    const float *p3 = &coords[dist_ref * 3];

                    float grad_a[3], grad_b[3], grad_c[3];
                    nerf_place_atom_backward(
                        p1, p2, p3,
                        distances[idx], angles[idx], dihedrals[idx],
                        grad_result,
                        grad_a, grad_b, grad_c,
                        &grad_distances[idx], &grad_angles[idx], &grad_dihedrals[idx]
                    );

                    /* Atomic adds for thread safety */
#ifdef _OPENMP
                    #pragma omp atomic
#endif
                    grad_coords[dihe_ref * 3 + 0] += grad_a[0];
#ifdef _OPENMP
                    #pragma omp atomic
#endif
                    grad_coords[dihe_ref * 3 + 1] += grad_a[1];
#ifdef _OPENMP
                    #pragma omp atomic
#endif
                    grad_coords[dihe_ref * 3 + 2] += grad_a[2];
#ifdef _OPENMP
                    #pragma omp atomic
#endif
                    grad_coords[angl_ref * 3 + 0] += grad_b[0];
#ifdef _OPENMP
                    #pragma omp atomic
#endif
                    grad_coords[angl_ref * 3 + 1] += grad_b[1];
#ifdef _OPENMP
                    #pragma omp atomic
#endif
                    grad_coords[angl_ref * 3 + 2] += grad_b[2];
#ifdef _OPENMP
                    #pragma omp atomic
#endif
                    grad_coords[dist_ref * 3 + 0] += grad_c[0];
#ifdef _OPENMP
                    #pragma omp atomic
#endif
                    grad_coords[dist_ref * 3 + 1] += grad_c[1];
#ifdef _OPENMP
                    #pragma omp atomic
#endif
                    grad_coords[dist_ref * 3 + 2] += grad_c[2];
                } else {
                    grad_distances[idx] = 0.0f;
                    grad_angles[idx] = 0.0f;
                    grad_dihedrals[idx] = 0.0f;
                }
            }
            }
            /* Implicit barrier at end of omp for ensures level completes before next */
        }
#ifdef _OPENMP
    }
#endif
}


/* ========================================================================= */
/* Anchored NERF functions                                                   */
/* ========================================================================= */


void batch_nerf_reconstruct_leveled_anchored(
    float *coords, size_t n_atoms,
    const int64_t *indices, size_t n_entries,
    const float *distances, const float *angles, const float *dihedrals,
    const int32_t *level_offsets, int n_levels,
    const float *anchor_coords, const int32_t *component_ids
) {
    (void)n_entries;  /* Used implicitly via level_offsets */

#ifdef _OPENMP
    #pragma omp parallel
    {
#endif
        for (int level = 0; level < n_levels; level++) {
            int start = level_offsets[level];
            int end = level_offsets[level + 1];

#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (int i = start; i < end; i++) {
                nerf_place_single_entry_anchored(coords, n_atoms, indices,
                    distances, angles, dihedrals, (size_t)i,
                    anchor_coords, component_ids);
            }
        }
#ifdef _OPENMP
    }
#endif
}


void batch_nerf_reconstruct_backward_leveled_anchored(
    const float *coords, size_t n_atoms,
    const int64_t *indices, size_t n_entries,
    const float *distances, const float *angles, const float *dihedrals,
    float *grad_coords,
    float *grad_distances, float *grad_angles, float *grad_dihedrals,
    const int32_t *level_offsets, int n_levels,
    const float *anchor_coords, const int32_t *component_ids
) {
    (void)n_entries;

    /* Get anchors helper */
    #define GET_ANCHORS(i) \
        const float *anchor0 = NULL, *anchor1 = NULL, *anchor2 = NULL; \
        if (anchor_coords != NULL && component_ids != NULL) { \
            int32_t comp_id = component_ids[i]; \
            anchor0 = &anchor_coords[comp_id * 9 + 0]; \
            anchor1 = &anchor_coords[comp_id * 9 + 3]; \
            anchor2 = &anchor_coords[comp_id * 9 + 6]; \
        }

#ifdef _OPENMP
    #pragma omp parallel
    {
#endif
        for (int level = n_levels - 1; level >= 0; level--) {
            int start = level_offsets[level];
            int end = level_offsets[level + 1];

#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (int i = start; i < end; i++) {
                size_t idx = (size_t)i;
                int64_t atom_idx = indices[idx * 4 + 0];
                int64_t dist_ref = indices[idx * 4 + 1];
                int64_t angl_ref = indices[idx * 4 + 2];
                int64_t dihe_ref = indices[idx * 4 + 3];

                if (atom_idx < 0 || (size_t)atom_idx >= n_atoms) {
                    grad_distances[idx] = 0.0f;
                    grad_angles[idx] = 0.0f;
                    grad_dihedrals[idx] = 0.0f;
                    continue;
                }

                const float *grad_result = &grad_coords[atom_idx * 3];
                GET_ANCHORS(idx);

                if (dist_ref < 0) {
                    /* First atom: no gradients to propagate (anchor is frozen) */
                    grad_distances[idx] = 0.0f;
                    grad_angles[idx] = 0.0f;
                    grad_dihedrals[idx] = 0.0f;

                } else if (angl_ref < 0) {
                    /* Second atom */
                    if ((size_t)dist_ref < n_atoms) {
                        if (anchor1 != NULL) {
                            const float *ref = &coords[dist_ref * 3];
                            float grad_ref[3];
                            nerf_place_along_direction_backward_impl(
                                ref, anchor1, distances[idx],
                                grad_result, grad_ref, &grad_distances[idx]
                            );
                            grad_angles[idx] = 0.0f;
                            grad_dihedrals[idx] = 0.0f;
#ifdef _OPENMP
                            #pragma omp atomic
#endif
                            grad_coords[dist_ref * 3 + 0] += grad_ref[0];
#ifdef _OPENMP
                            #pragma omp atomic
#endif
                            grad_coords[dist_ref * 3 + 1] += grad_ref[1];
#ifdef _OPENMP
                            #pragma omp atomic
#endif
                            grad_coords[dist_ref * 3 + 2] += grad_ref[2];
                        } else {
                            /* Canonical +X case */
                            grad_distances[idx] = grad_result[0];
                            grad_angles[idx] = 0.0f;
                            grad_dihedrals[idx] = 0.0f;
#ifdef _OPENMP
                            #pragma omp atomic
#endif
                            grad_coords[dist_ref * 3 + 0] += grad_result[0];
#ifdef _OPENMP
                            #pragma omp atomic
#endif
                            grad_coords[dist_ref * 3 + 1] += grad_result[1];
#ifdef _OPENMP
                            #pragma omp atomic
#endif
                            grad_coords[dist_ref * 3 + 2] += grad_result[2];
                        }
                    }

                } else if (dihe_ref < 0) {
                    /* Third atom */
                    if ((size_t)angl_ref < n_atoms && (size_t)dist_ref < n_atoms) {
                        const float *ref1 = &coords[dist_ref * 3];
                        const float *ref2 = &coords[angl_ref * 3];
                        float grad_ref1[3], grad_ref2[3];

                        if (anchor2 != NULL) {
                            nerf_place_in_plane_anchored_backward_impl(
                                ref1, ref2, anchor2,
                                distances[idx], angles[idx],
                                grad_result,
                                grad_ref1, grad_ref2,
                                &grad_distances[idx], &grad_angles[idx]
                            );
                        } else {
                            nerf_place_in_plane_backward_impl(
                                ref1, ref2,
                                distances[idx], angles[idx],
                                grad_result,
                                grad_ref1, grad_ref2,
                                &grad_distances[idx], &grad_angles[idx]
                            );
                        }
                        grad_dihedrals[idx] = 0.0f;

#ifdef _OPENMP
                        #pragma omp atomic
#endif
                        grad_coords[dist_ref * 3 + 0] += grad_ref1[0];
#ifdef _OPENMP
                        #pragma omp atomic
#endif
                        grad_coords[dist_ref * 3 + 1] += grad_ref1[1];
#ifdef _OPENMP
                        #pragma omp atomic
#endif
                        grad_coords[dist_ref * 3 + 2] += grad_ref1[2];
#ifdef _OPENMP
                        #pragma omp atomic
#endif
                        grad_coords[angl_ref * 3 + 0] += grad_ref2[0];
#ifdef _OPENMP
                        #pragma omp atomic
#endif
                        grad_coords[angl_ref * 3 + 1] += grad_ref2[1];
#ifdef _OPENMP
                        #pragma omp atomic
#endif
                        grad_coords[angl_ref * 3 + 2] += grad_ref2[2];
                    } else {
                        grad_distances[idx] = 0.0f;
                        grad_angles[idx] = 0.0f;
                        grad_dihedrals[idx] = 0.0f;
                    }

                } else {
                    /* Full NERF (same as non-anchored) */
                    if ((size_t)dihe_ref < n_atoms &&
                        (size_t)angl_ref < n_atoms &&
                        (size_t)dist_ref < n_atoms) {

                        const float *p1 = &coords[dihe_ref * 3];
                        const float *p2 = &coords[angl_ref * 3];
                        const float *p3 = &coords[dist_ref * 3];

                        float grad_a[3], grad_b[3], grad_c[3];
                        nerf_place_atom_backward_impl(
                            p1, p2, p3,
                            distances[idx], angles[idx], dihedrals[idx],
                            grad_result,
                            grad_a, grad_b, grad_c,
                            &grad_distances[idx], &grad_angles[idx], &grad_dihedrals[idx]
                        );

#ifdef _OPENMP
                        #pragma omp atomic
#endif
                        grad_coords[dihe_ref * 3 + 0] += grad_a[0];
#ifdef _OPENMP
                        #pragma omp atomic
#endif
                        grad_coords[dihe_ref * 3 + 1] += grad_a[1];
#ifdef _OPENMP
                        #pragma omp atomic
#endif
                        grad_coords[dihe_ref * 3 + 2] += grad_a[2];
#ifdef _OPENMP
                        #pragma omp atomic
#endif
                        grad_coords[angl_ref * 3 + 0] += grad_b[0];
#ifdef _OPENMP
                        #pragma omp atomic
#endif
                        grad_coords[angl_ref * 3 + 1] += grad_b[1];
#ifdef _OPENMP
                        #pragma omp atomic
#endif
                        grad_coords[angl_ref * 3 + 2] += grad_b[2];
#ifdef _OPENMP
                        #pragma omp atomic
#endif
                        grad_coords[dist_ref * 3 + 0] += grad_c[0];
#ifdef _OPENMP
                        #pragma omp atomic
#endif
                        grad_coords[dist_ref * 3 + 1] += grad_c[1];
#ifdef _OPENMP
                        #pragma omp atomic
#endif
                        grad_coords[dist_ref * 3 + 2] += grad_c[2];
                    } else {
                        grad_distances[idx] = 0.0f;
                        grad_angles[idx] = 0.0f;
                        grad_dihedrals[idx] = 0.0f;
                    }
                }
            }
        }
#ifdef _OPENMP
    }
#endif

    #undef GET_ANCHORS
}
