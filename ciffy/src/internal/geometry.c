/**
 * @file geometry.c
 * @brief Core geometry functions for internal coordinates (CPU wrappers).
 *
 * This file provides the public API for geometry functions. The actual
 * implementations are in geometry_impl.h, which is shared between CPU
 * and CUDA builds.
 */

#include "geometry.h"
#include "geometry_impl.h"


/* ========================================================================= */
/* Forward functions                                                         */
/* ========================================================================= */


float compute_distance(const float *a, const float *b) {
    return compute_distance_impl(a, b);
}


float compute_angle(const float *a, const float *b, const float *c) {
    return compute_angle_impl(a, b, c);
}


float compute_dihedral(const float *a, const float *b,
                       const float *c, const float *d) {
    return compute_dihedral_impl(a, b, c, d);
}


void nerf_place_atom(const float *a, const float *b, const float *c,
                     float distance, float angle, float dihedral,
                     float *result) {
    nerf_place_atom_impl(a, b, c, distance, angle, dihedral, result);
}


void nerf_place_along_x(const float *ref, float distance, float *result) {
    nerf_place_along_x_impl(ref, distance, result);
}


void nerf_place_in_plane(const float *ref1, const float *ref2,
                         float distance, float angle, float *result) {
    nerf_place_in_plane_impl(ref1, ref2, distance, angle, result);
}


/* ========================================================================= */
/* Backward (gradient) functions for automatic differentiation              */
/* ========================================================================= */


void compute_distance_backward(
    const float *a, const float *b,
    float distance, float grad_output,
    float *grad_a, float *grad_b
) {
    compute_distance_backward_impl(a, b, distance, grad_output, grad_a, grad_b);
}


void compute_angle_backward(
    const float *a, const float *b, const float *c,
    float angle, float grad_output,
    float *grad_a, float *grad_b, float *grad_c
) {
    compute_angle_backward_impl(a, b, c, angle, grad_output, grad_a, grad_b, grad_c);
}


void compute_dihedral_backward(
    const float *a, const float *b, const float *c, const float *d,
    float grad_output,
    float *grad_a, float *grad_b, float *grad_c, float *grad_d
) {
    compute_dihedral_backward_impl(a, b, c, d, grad_output, grad_a, grad_b, grad_c, grad_d);
}


void nerf_place_atom_backward(
    const float *a, const float *b, const float *c,
    float distance, float angle, float dihedral,
    const float *grad_result,
    float *grad_a, float *grad_b, float *grad_c,
    float *grad_distance, float *grad_angle, float *grad_dihedral
) {
    nerf_place_atom_backward_impl(
        a, b, c, distance, angle, dihedral, grad_result,
        grad_a, grad_b, grad_c, grad_distance, grad_angle, grad_dihedral
    );
}


void nerf_place_in_plane_backward(
    const float *ref1, const float *ref2,
    float distance, float angle,
    const float *grad_result,
    float *grad_ref1, float *grad_ref2,
    float *grad_distance, float *grad_angle
) {
    nerf_place_in_plane_backward_impl(
        ref1, ref2, distance, angle, grad_result,
        grad_ref1, grad_ref2, grad_distance, grad_angle
    );
}
