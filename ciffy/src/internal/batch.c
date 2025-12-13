/**
 * @file batch.c
 * @brief Batch operations for internal coordinate conversion.
 */

#include "batch.h"
#include "geometry.h"


void batch_cartesian_to_internal(
    const float *coords, size_t n_atoms,
    const int64_t *indices, size_t n_entries,
    float *distances, float *angles, float *dihedrals
) {
    (void)n_atoms;  /* Used for bounds checking if needed */

    for (size_t i = 0; i < n_entries; i++) {
        int64_t atom_idx = indices[i * 4 + 0];
        int64_t dist_ref = indices[i * 4 + 1];
        int64_t angl_ref = indices[i * 4 + 2];
        int64_t dihe_ref = indices[i * 4 + 3];

        const float *atom = &coords[atom_idx * 3];

        /* Bond length */
        if (dist_ref >= 0) {
            const float *ref1 = &coords[dist_ref * 3];
            distances[i] = compute_distance(atom, ref1);
        } else {
            distances[i] = 0.0f;
        }

        /* Bond angle */
        if (angl_ref >= 0 && dist_ref >= 0) {
            const float *ref1 = &coords[dist_ref * 3];
            const float *ref2 = &coords[angl_ref * 3];
            /* Angle at dist_ref between atom and angl_ref */
            angles[i] = compute_angle(atom, ref1, ref2);
        } else {
            angles[i] = 0.0f;
        }

        /* Dihedral angle */
        if (dihe_ref >= 0 && angl_ref >= 0 && dist_ref >= 0) {
            const float *ref1 = &coords[dist_ref * 3];
            const float *ref2 = &coords[angl_ref * 3];
            const float *ref3 = &coords[dihe_ref * 3];
            /* Dihedral: dihe_ref - angl_ref - dist_ref - atom */
            dihedrals[i] = compute_dihedral(ref3, ref2, ref1, atom);
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
    (void)n_atoms;  /* Used for bounds checking if needed */

    for (size_t i = 0; i < n_entries; i++) {
        int64_t atom_idx = indices[i * 4 + 0];
        int64_t dist_ref = indices[i * 4 + 1];
        int64_t angl_ref = indices[i * 4 + 2];
        int64_t dihe_ref = indices[i * 4 + 3];

        float *result = &coords[atom_idx * 3];

        if (dist_ref < 0) {
            /* First atom: place at origin */
            result[0] = 0.0f;
            result[1] = 0.0f;
            result[2] = 0.0f;

        } else if (angl_ref < 0) {
            /* Second atom: place along +X from distance reference */
            const float *ref = &coords[dist_ref * 3];
            nerf_place_along_x(ref, distances[i], result);

        } else if (dihe_ref < 0) {
            /* Third atom: place in plane */
            const float *ref1 = &coords[dist_ref * 3];
            const float *ref2 = &coords[angl_ref * 3];
            nerf_place_in_plane(ref1, ref2, distances[i], angles[i], result);

        } else {
            /* Full NERF placement */
            const float *p1 = &coords[dihe_ref * 3];
            const float *p2 = &coords[angl_ref * 3];
            const float *p3 = &coords[dist_ref * 3];
            nerf_place_atom(p1, p2, p3, distances[i], angles[i], dihedrals[i], result);
        }
    }
}
