/**
 * @file primitives.h
 * @brief Vector primitive operations with backward passes for autodiff.
 *
 * Each primitive has a forward function and a corresponding backward function
 * that computes gradients using the chain rule. These primitives can be
 * composed to build complex geometric operations with correct gradients.
 *
 * Naming convention:
 * - Forward: vec_<op>(inputs..., output)
 * - Backward: vec_<op>_backward(inputs..., grad_output, grad_inputs...)
 */

#ifndef CIFFY_PRIMITIVES_H
#define CIFFY_PRIMITIVES_H

#include "cuda_compat.h"

/* Small epsilon for numerical stability (alias for backwards compatibility) */
#define PRIM_EPS CIFFY_EPS

/* ========================================================================= */
/* Forward primitives                                                        */
/* ========================================================================= */

/**
 * Vector subtraction: out = a - b
 */
CIFFY_HOST_DEVICE static inline void vec_sub(const float *a, const float *b, float *out) {
    out[0] = a[0] - b[0];
    out[1] = a[1] - b[1];
    out[2] = a[2] - b[2];
}

/**
 * Vector addition: out = a + b
 */
CIFFY_HOST_DEVICE static inline void vec_add(const float *a, const float *b, float *out) {
    out[0] = a[0] + b[0];
    out[1] = a[1] + b[1];
    out[2] = a[2] + b[2];
}

/**
 * Scalar-vector multiply: out = s * v
 */
CIFFY_HOST_DEVICE static inline void vec_scale(float s, const float *v, float *out) {
    out[0] = s * v[0];
    out[1] = s * v[1];
    out[2] = s * v[2];
}

/**
 * Dot product: returns a . b
 */
CIFFY_HOST_DEVICE static inline float vec_dot(const float *a, const float *b) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

/**
 * Cross product: out = a x b
 */
CIFFY_HOST_DEVICE static inline void vec_cross(const float *a, const float *b, float *out) {
    out[0] = a[1]*b[2] - a[2]*b[1];
    out[1] = a[2]*b[0] - a[0]*b[2];
    out[2] = a[0]*b[1] - a[1]*b[0];
}

/**
 * Vector norm: returns |v|
 */
CIFFY_HOST_DEVICE static inline float vec_norm(const float *v) {
    return CIFFY_SQRTF(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
}

/**
 * Normalize vector: out = v / |v|
 * Returns the norm for use in backward pass.
 */
CIFFY_HOST_DEVICE static inline float vec_normalize(const float *v, float *out) {
    float n = vec_norm(v) + PRIM_EPS;
    out[0] = v[0] / n;
    out[1] = v[1] / n;
    out[2] = v[2] / n;
    return n;
}

/**
 * Copy vector: out = v
 */
CIFFY_HOST_DEVICE static inline void vec_copy(const float *v, float *out) {
    out[0] = v[0];
    out[1] = v[1];
    out[2] = v[2];
}

/**
 * Zero vector: out = 0
 */
CIFFY_HOST_DEVICE static inline void vec_zero(float *out) {
    out[0] = 0.0f;
    out[1] = 0.0f;
    out[2] = 0.0f;
}

/**
 * Accumulate: out += v
 */
CIFFY_HOST_DEVICE static inline void vec_acc(const float *v, float *out) {
    out[0] += v[0];
    out[1] += v[1];
    out[2] += v[2];
}

/**
 * Scaled accumulate: out += s * v
 */
CIFFY_HOST_DEVICE static inline void vec_acc_scaled(float s, const float *v, float *out) {
    out[0] += s * v[0];
    out[1] += s * v[1];
    out[2] += s * v[2];
}

/* ========================================================================= */
/* Backward primitives                                                       */
/* ========================================================================= */

/**
 * Backward for vec_sub: out = a - b
 * grad_a += grad_out
 * grad_b -= grad_out
 */
CIFFY_HOST_DEVICE static inline void vec_sub_backward(
    const float *grad_out,
    float *grad_a, float *grad_b
) {
    grad_a[0] += grad_out[0];
    grad_a[1] += grad_out[1];
    grad_a[2] += grad_out[2];
    grad_b[0] -= grad_out[0];
    grad_b[1] -= grad_out[1];
    grad_b[2] -= grad_out[2];
}

/**
 * Backward for vec_add: out = a + b
 * grad_a += grad_out
 * grad_b += grad_out
 */
CIFFY_HOST_DEVICE static inline void vec_add_backward(
    const float *grad_out,
    float *grad_a, float *grad_b
) {
    grad_a[0] += grad_out[0];
    grad_a[1] += grad_out[1];
    grad_a[2] += grad_out[2];
    grad_b[0] += grad_out[0];
    grad_b[1] += grad_out[1];
    grad_b[2] += grad_out[2];
}

/**
 * Backward for vec_scale: out = s * v
 * grad_s += grad_out . v
 * grad_v += s * grad_out
 */
CIFFY_HOST_DEVICE static inline void vec_scale_backward(
    float s, const float *v,
    const float *grad_out,
    float *grad_s, float *grad_v
) {
    *grad_s += vec_dot(grad_out, v);
    grad_v[0] += s * grad_out[0];
    grad_v[1] += s * grad_out[1];
    grad_v[2] += s * grad_out[2];
}

/**
 * Backward for vec_dot: out = a . b
 * grad_a += grad_out * b
 * grad_b += grad_out * a
 */
CIFFY_HOST_DEVICE static inline void vec_dot_backward(
    const float *a, const float *b,
    float grad_out,
    float *grad_a, float *grad_b
) {
    grad_a[0] += grad_out * b[0];
    grad_a[1] += grad_out * b[1];
    grad_a[2] += grad_out * b[2];
    grad_b[0] += grad_out * a[0];
    grad_b[1] += grad_out * a[1];
    grad_b[2] += grad_out * a[2];
}

/**
 * Backward for vec_cross: out = a x b
 * grad_a += b x grad_out
 * grad_b += grad_out x a
 */
CIFFY_HOST_DEVICE static inline void vec_cross_backward(
    const float *a, const float *b,
    const float *grad_out,
    float *grad_a, float *grad_b
) {
    /* grad_a += b x grad_out */
    grad_a[0] += b[1]*grad_out[2] - b[2]*grad_out[1];
    grad_a[1] += b[2]*grad_out[0] - b[0]*grad_out[2];
    grad_a[2] += b[0]*grad_out[1] - b[1]*grad_out[0];

    /* grad_b += grad_out x a */
    grad_b[0] += grad_out[1]*a[2] - grad_out[2]*a[1];
    grad_b[1] += grad_out[2]*a[0] - grad_out[0]*a[2];
    grad_b[2] += grad_out[0]*a[1] - grad_out[1]*a[0];
}

/**
 * Backward for vec_normalize: out = v / |v|
 * grad_v += (grad_out - out * (out . grad_out)) / norm
 *
 * @param v_hat The normalized vector (output of forward pass)
 * @param norm The norm from forward pass (before adding epsilon)
 */
CIFFY_HOST_DEVICE static inline void vec_normalize_backward(
    const float *v_hat, float norm,
    const float *grad_out,
    float *grad_v
) {
    float dot = vec_dot(v_hat, grad_out);
    float inv_norm = 1.0f / norm;
    grad_v[0] += (grad_out[0] - v_hat[0] * dot) * inv_norm;
    grad_v[1] += (grad_out[1] - v_hat[1] * dot) * inv_norm;
    grad_v[2] += (grad_out[2] - v_hat[2] * dot) * inv_norm;
}

/* ========================================================================= */
/* Composite operations                                                      */
/* ========================================================================= */

/**
 * Linear combination: out = s1*v1 + s2*v2 + s3*v3
 */
CIFFY_HOST_DEVICE static inline void vec_lincomb3(
    float s1, const float *v1,
    float s2, const float *v2,
    float s3, const float *v3,
    float *out
) {
    out[0] = s1*v1[0] + s2*v2[0] + s3*v3[0];
    out[1] = s1*v1[1] + s2*v2[1] + s3*v3[1];
    out[2] = s1*v1[2] + s2*v2[2] + s3*v3[2];
}

/**
 * Backward for vec_lincomb3: out = s1*v1 + s2*v2 + s3*v3
 */
CIFFY_HOST_DEVICE static inline void vec_lincomb3_backward(
    float s1, const float *v1,
    float s2, const float *v2,
    float s3, const float *v3,
    const float *grad_out,
    float *grad_s1, float *grad_v1,
    float *grad_s2, float *grad_v2,
    float *grad_s3, float *grad_v3
) {
    /* grad_si += grad_out . vi */
    *grad_s1 += vec_dot(grad_out, v1);
    *grad_s2 += vec_dot(grad_out, v2);
    *grad_s3 += vec_dot(grad_out, v3);

    /* grad_vi += si * grad_out */
    vec_acc_scaled(s1, grad_out, grad_v1);
    vec_acc_scaled(s2, grad_out, grad_v2);
    vec_acc_scaled(s3, grad_out, grad_v3);
}

#endif /* CIFFY_PRIMITIVES_H */
