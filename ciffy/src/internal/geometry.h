/**
 * @file geometry.h
 * @brief Core geometry functions for internal coordinates.
 *
 * Provides optimized implementations of:
 * - Distance computation between two 3D points
 * - Bond angle computation between three points
 * - Dihedral angle computation between four points
 * - NERF atom placement algorithm
 */

#ifndef CIFFY_INTERNAL_GEOMETRY_H
#define CIFFY_INTERNAL_GEOMETRY_H

/**
 * Compute Euclidean distance between two 3D points.
 *
 * @param a First point (3 floats).
 * @param b Second point (3 floats).
 * @return Distance in same units as input coordinates.
 */
float compute_distance(const float *a, const float *b);

/**
 * Compute bond angle at vertex B for points A-B-C.
 *
 * Returns the angle between vectors (A-B) and (C-B).
 *
 * @param a First point (3 floats).
 * @param b Vertex point (3 floats).
 * @param c Third point (3 floats).
 * @return Angle in radians [0, pi].
 */
float compute_angle(const float *a, const float *b, const float *c);

/**
 * Compute dihedral (torsion) angle for four points A-B-C-D.
 *
 * The dihedral angle is the angle between planes (A,B,C) and (B,C,D).
 * Uses atan2 for numerical stability and correct quadrant.
 *
 * @param a First point (3 floats).
 * @param b Second point (3 floats).
 * @param c Third point (3 floats).
 * @param d Fourth point (3 floats).
 * @return Dihedral angle in radians [-pi, pi].
 */
float compute_dihedral(const float *a, const float *b,
                       const float *c, const float *d);

/**
 * Place an atom using the NERF algorithm.
 *
 * Places a new atom D such that:
 * - |D - c| = distance
 * - angle(b, c, D) = angle
 * - dihedral(a, b, c, D) = dihedral
 *
 * @param a Dihedral reference position (3 floats).
 * @param b Angle reference position (3 floats).
 * @param c Distance reference position - bonded to new atom (3 floats).
 * @param distance Bond length from c to new atom.
 * @param angle Bond angle at c (b-c-new) in radians.
 * @param dihedral Dihedral angle (a-b-c-new) in radians.
 * @param result Output position for new atom (3 floats).
 */
void nerf_place_atom(const float *a, const float *b, const float *c,
                     float distance, float angle, float dihedral,
                     float *result);

/**
 * Place second atom along +X axis from reference.
 *
 * Used for the second atom in a chain when no angle reference exists.
 *
 * @param ref Reference position (3 floats).
 * @param distance Distance from reference.
 * @param result Output position (3 floats).
 */
void nerf_place_along_x(const float *ref, float distance, float *result);

/**
 * Place third atom using distance and angle only.
 *
 * Used for the third atom in a chain when no dihedral reference exists.
 * Places the atom in a plane with the first two atoms.
 *
 * @param ref1 Distance reference (parent) (3 floats).
 * @param ref2 Angle reference (3 floats).
 * @param distance Bond length to ref1.
 * @param angle Bond angle at ref1 (ref2-ref1-new) in radians.
 * @param result Output position (3 floats).
 */
void nerf_place_in_plane(const float *ref1, const float *ref2,
                         float distance, float angle, float *result);

#endif /* CIFFY_INTERNAL_GEOMETRY_H */
