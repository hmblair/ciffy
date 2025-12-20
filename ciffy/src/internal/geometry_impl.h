/**
 * @file geometry_impl.h
 * @brief Device-compatible geometry implementations.
 *
 * This header contains the actual implementations of geometry functions
 * that can run on both CPU and GPU. The functions are marked with
 * CIFFY_HOST_DEVICE so they compile for both targets.
 *
 * Include this header in:
 *   - geometry.c (for CPU builds)
 *   - batch.cu (for CUDA kernels)
 *
 * The implementations are suffixed with _impl to distinguish them from
 * the public API functions declared in geometry.h.
 */

#ifndef CIFFY_GEOMETRY_IMPL_H
#define CIFFY_GEOMETRY_IMPL_H

#include "cuda_compat.h"
#include "primitives.h"

/* ========================================================================= */
/* Forward implementations                                                   */
/* ========================================================================= */

/**
 * Compute Euclidean distance between two 3D points.
 */
CIFFY_HOST_DEVICE
static inline float compute_distance_impl(const float *a, const float *b) {
    float dx = a[0] - b[0];
    float dy = a[1] - b[1];
    float dz = a[2] - b[2];
    return CIFFY_SQRTF(dx*dx + dy*dy + dz*dz);
}


/**
 * Compute bond angle at vertex B for points A-B-C.
 * Returns angle in radians.
 */
CIFFY_HOST_DEVICE
static inline float compute_angle_impl(const float *a, const float *b, const float *c) {
    /* Vectors from vertex B */
    float v1x = a[0] - b[0];
    float v1y = a[1] - b[1];
    float v1z = a[2] - b[2];

    float v2x = c[0] - b[0];
    float v2y = c[1] - b[1];
    float v2z = c[2] - b[2];

    /* Norms */
    float v1_norm = CIFFY_SQRTF(v1x*v1x + v1y*v1y + v1z*v1z) + CIFFY_EPS;
    float v2_norm = CIFFY_SQRTF(v2x*v2x + v2y*v2y + v2z*v2z) + CIFFY_EPS;

    /* Dot product and cosine */
    float dot = v1x*v2x + v1y*v2y + v1z*v2z;
    float cos_angle = dot / (v1_norm * v2_norm);

    /* Clamp to [-1, 1] for numerical stability */
    if (cos_angle > 1.0f) cos_angle = 1.0f;
    if (cos_angle < -1.0f) cos_angle = -1.0f;

    return CIFFY_ACOSF(cos_angle);
}


/**
 * Compute dihedral (torsion) angle for atoms A-B-C-D.
 * Returns angle in radians in range [-pi, pi].
 */
CIFFY_HOST_DEVICE
static inline float compute_dihedral_impl(const float *a, const float *b,
                                          const float *c, const float *d) {
    /* Bond vectors */
    float b1x = b[0] - a[0];
    float b1y = b[1] - a[1];
    float b1z = b[2] - a[2];

    float b2x = c[0] - b[0];
    float b2y = c[1] - b[1];
    float b2z = c[2] - b[2];

    float b3x = d[0] - c[0];
    float b3y = d[1] - c[1];
    float b3z = d[2] - c[2];

    /* Normal to plane A-B-C: n1 = b1 x b2 */
    float n1x = b1y*b2z - b1z*b2y;
    float n1y = b1z*b2x - b1x*b2z;
    float n1z = b1x*b2y - b1y*b2x;

    /* Normal to plane B-C-D: n2 = b2 x b3 */
    float n2x = b2y*b3z - b2z*b3y;
    float n2y = b2z*b3x - b2x*b3z;
    float n2z = b2x*b3y - b2y*b3x;

    /* Normalize n1 */
    float n1_norm = CIFFY_SQRTF(n1x*n1x + n1y*n1y + n1z*n1z) + CIFFY_EPS;
    n1x /= n1_norm;
    n1y /= n1_norm;
    n1z /= n1_norm;

    /* Normalize n2 */
    float n2_norm = CIFFY_SQRTF(n2x*n2x + n2y*n2y + n2z*n2z) + CIFFY_EPS;
    n2x /= n2_norm;
    n2y /= n2_norm;
    n2z /= n2_norm;

    /* Normalize b2 for m1 calculation */
    float b2_norm = CIFFY_SQRTF(b2x*b2x + b2y*b2y + b2z*b2z) + CIFFY_EPS;
    float b2ux = b2x / b2_norm;
    float b2uy = b2y / b2_norm;
    float b2uz = b2z / b2_norm;

    /* m1 = n1_hat x b2_hat */
    float m1x = n1y*b2uz - n1z*b2uy;
    float m1y = n1z*b2ux - n1x*b2uz;
    float m1z = n1x*b2uy - n1y*b2ux;

    /* x = n1_hat . n2_hat = cos(phi), y = m1 . n2_hat = sin(phi) */
    float x = n1x*n2x + n1y*n2y + n1z*n2z;
    float y = m1x*n2x + m1y*n2y + m1z*n2z;

    return CIFFY_ATAN2F(y, x);
}


/**
 * Place a new atom D using NERF (Natural Extension Reference Frame).
 *
 * Given three reference atoms A, B, C and internal coordinates (distance, angle, dihedral),
 * computes the position of atom D such that:
 *   - |CD| = distance
 *   - angle(BCD) = angle
 *   - dihedral(ABCD) = dihedral
 */
CIFFY_HOST_DEVICE
static inline void nerf_place_atom_impl(const float *a, const float *b, const float *c,
                                        float distance, float angle, float dihedral,
                                        float *result) {
    /* Build local coordinate system at c:
     * - z-axis: direction from c towards b
     * - x-axis: in the a-b-c plane, perpendicular to z
     * - y-axis: perpendicular to both (normal to plane)
     */

    /* z = direction from c to b (normalized) */
    float zx = b[0] - c[0];
    float zy = b[1] - c[1];
    float zz = b[2] - c[2];
    float z_len = CIFFY_SQRTF(zx*zx + zy*zy + zz*zz) + CIFFY_EPS;
    zx /= z_len;
    zy /= z_len;
    zz /= z_len;

    /* v = direction from c to a (normalized) */
    float vx = a[0] - c[0];
    float vy = a[1] - c[1];
    float vz = a[2] - c[2];
    float v_len = CIFFY_SQRTF(vx*vx + vy*vy + vz*vz) + CIFFY_EPS;
    vx /= v_len;
    vy /= v_len;
    vz /= v_len;

    /* y = z cross v (normal to plane, right-handed) */
    float yx = zy*vz - zz*vy;
    float yy = zz*vx - zx*vz;
    float yz = zx*vy - zy*vx;
    float y_len = CIFFY_SQRTF(yx*yx + yy*yy + yz*yz) + CIFFY_EPS;
    yx /= y_len;
    yy /= y_len;
    yz /= y_len;

    /* x = y cross z (in plane, perpendicular to z) */
    float xx = yy*zz - yz*zy;
    float xy = yz*zx - yx*zz;
    float xz = yx*zy - yy*zx;

    /* Place new atom D at distance from c:
     * D - c = distance * (cos(angle) * z + sin(angle) * (cos(dihedral) * x + sin(dihedral) * y))
     */
    float cos_a = CIFFY_COSF(angle);
    float sin_a = CIFFY_SINF(angle);
    float cos_d = CIFFY_COSF(dihedral);
    float sin_d = CIFFY_SINF(dihedral);

    float d_z = distance * cos_a;
    float d_perp = distance * sin_a;
    float d_x = d_perp * cos_d;
    float d_y = d_perp * sin_d;

    result[0] = c[0] + d_z * zx + d_x * xx + d_y * yx;
    result[1] = c[1] + d_z * zy + d_x * xy + d_y * yy;
    result[2] = c[2] + d_z * zz + d_x * xz + d_y * yz;
}


/**
 * Place second atom along +X from reference.
 */
CIFFY_HOST_DEVICE
static inline void nerf_place_along_x_impl(const float *ref, float distance, float *result) {
    result[0] = ref[0] + distance;
    result[1] = ref[1];
    result[2] = ref[2];
}


/**
 * Place third atom in plane with first two.
 */
CIFFY_HOST_DEVICE
static inline void nerf_place_in_plane_impl(const float *ref1, const float *ref2,
                                            float distance, float angle, float *result) {
    /* Direction from ref1 to ref2 (normalized) */
    float ux = ref2[0] - ref1[0];
    float uy = ref2[1] - ref1[1];
    float uz = ref2[2] - ref1[2];
    float u_norm = CIFFY_SQRTF(ux*ux + uy*uy + uz*uz) + CIFFY_EPS;
    ux /= u_norm;
    uy /= u_norm;
    uz /= u_norm;

    /* Create perpendicular direction using z-axis cross product */
    /* perp = (0, 0, 1) x u */
    float perpx = -uy;   /* 0*uz - 1*uy */
    float perpy = ux;    /* 1*ux - 0*uz */
    float perpz = 0.0f;  /* 0*uy - 0*ux */
    float perp_norm = CIFFY_SQRTF(perpx*perpx + perpy*perpy + perpz*perpz);

    if (perp_norm < CIFFY_EPS) {
        /* u is parallel to z-axis, use x-axis instead */
        perpx = 0.0f;
        perpy = uz;
        perpz = -uy;
        perp_norm = CIFFY_SQRTF(perpx*perpx + perpy*perpy + perpz*perpz);
    }
    perp_norm += CIFFY_EPS;
    perpx /= perp_norm;
    perpy /= perp_norm;
    perpz /= perp_norm;

    /* new_pos = ref1 + distance * (cos(angle) * u + sin(angle) * perp) */
    float cos_a = CIFFY_COSF(angle);
    float sin_a = CIFFY_SINF(angle);

    result[0] = ref1[0] + distance * (cos_a * ux + sin_a * perpx);
    result[1] = ref1[1] + distance * (cos_a * uy + sin_a * perpy);
    result[2] = ref1[2] + distance * (cos_a * uz + sin_a * perpz);
}


/**
 * Place second atom along direction defined by anchors (instead of +X).
 *
 * Places new atom at: ref + distance * normalize(anchor1 - anchor0)
 * where ref is already at anchor0's position.
 */
CIFFY_HOST_DEVICE
static inline void nerf_place_along_direction_impl(
    const float *ref,      /* First atom position (already at anchor0) */
    const float *anchor1,  /* Second anchor (defines direction) */
    float distance,
    float *result
) {
    /* Direction from anchor0 (ref) to anchor1 */
    float dx = anchor1[0] - ref[0];
    float dy = anchor1[1] - ref[1];
    float dz = anchor1[2] - ref[2];
    float norm = CIFFY_SQRTF(dx*dx + dy*dy + dz*dz) + CIFFY_EPS;

    /* Place at distance along this direction */
    result[0] = ref[0] + distance * dx / norm;
    result[1] = ref[1] + distance * dy / norm;
    result[2] = ref[2] + distance * dz / norm;
}


/**
 * Place third atom in plane defined by anchors (instead of XY plane).
 *
 * Uses the plane normal from (anchor1-anchor0) x (anchor2-anchor0) to
 * define the perpendicular direction for the angle.
 */
CIFFY_HOST_DEVICE
static inline void nerf_place_in_plane_anchored_impl(
    const float *ref1,     /* Distance reference (atom 1, already placed) */
    const float *ref2,     /* Angle reference (atom 0, already placed) */
    const float *anchor2,  /* Third anchor (defines plane normal) */
    float distance,
    float angle,
    float *result
) {
    /* u = direction from ref1 toward ref2 (normalized) */
    float ux = ref2[0] - ref1[0];
    float uy = ref2[1] - ref1[1];
    float uz = ref2[2] - ref1[2];
    float u_norm = CIFFY_SQRTF(ux*ux + uy*uy + uz*uz) + CIFFY_EPS;
    ux /= u_norm; uy /= u_norm; uz /= u_norm;

    /* Build plane normal from anchors: (ref1-ref2) x (anchor2-ref2)
     * Note: ref2 is at anchor0 position, ref1 is at anchor1 direction */
    float v1x = ref1[0] - ref2[0];
    float v1y = ref1[1] - ref2[1];
    float v1z = ref1[2] - ref2[2];

    float v2x = anchor2[0] - ref2[0];
    float v2y = anchor2[1] - ref2[1];
    float v2z = anchor2[2] - ref2[2];

    /* plane_normal = v1 x v2 */
    float nx = v1y*v2z - v1z*v2y;
    float ny = v1z*v2x - v1x*v2z;
    float nz = v1x*v2y - v1y*v2x;

    /* perp = plane_normal x u (in-plane perpendicular to bond direction) */
    float perpx = ny*uz - nz*uy;
    float perpy = nz*ux - nx*uz;
    float perpz = nx*uy - ny*ux;
    float perp_norm = CIFFY_SQRTF(perpx*perpx + perpy*perpy + perpz*perpz);

    if (perp_norm < CIFFY_EPS) {
        /* plane_normal is parallel to u - use fallback perpendicular.
         * This occurs when the three anchor points are collinear. */
        float fx = -uy;
        float fy = ux;
        float fz = 0.0f;
        float fn = CIFFY_SQRTF(fx*fx + fy*fy + fz*fz);

        if (fn < CIFFY_EPS) {
            /* u is parallel to z-axis, use x-axis cross u */
            fx = 0.0f;
            fy = uz;
            fz = -uy;
            fn = CIFFY_SQRTF(fx*fx + fy*fy + fz*fz);
        }
        fn += CIFFY_EPS;
        perpx = fx / fn;
        perpy = fy / fn;
        perpz = fz / fn;
    } else {
        perp_norm += CIFFY_EPS;
        perpx /= perp_norm; perpy /= perp_norm; perpz /= perp_norm;
    }

    /* Check if perp points toward anchor2; if not, flip it.
     * The cross product can give either direction, so we must ensure
     * perp points toward the side where anchor2 lies. */
    float to_anchor_x = anchor2[0] - ref1[0];
    float to_anchor_y = anchor2[1] - ref1[1];
    float to_anchor_z = anchor2[2] - ref1[2];
    float dot = perpx * to_anchor_x + perpy * to_anchor_y + perpz * to_anchor_z;
    if (dot < 0) {
        perpx = -perpx; perpy = -perpy; perpz = -perpz;
    }

    /* Place: ref1 + distance * (cos(angle)*u + sin(angle)*perp) */
    float cos_a = CIFFY_COSF(angle), sin_a = CIFFY_SINF(angle);
    result[0] = ref1[0] + distance * (cos_a * ux + sin_a * perpx);
    result[1] = ref1[1] + distance * (cos_a * uy + sin_a * perpy);
    result[2] = ref1[2] + distance * (cos_a * uz + sin_a * perpz);
}


/* ========================================================================= */
/* Backward (gradient) implementations                                       */
/* ========================================================================= */

/**
 * Backward pass for compute_distance.
 */
CIFFY_HOST_DEVICE
static inline void compute_distance_backward_impl(
    const float *a, const float *b,
    float distance, float grad_output,
    float *grad_a, float *grad_b
) {
    /* d = ||a - b||
     * pd/pa = (a - b) / d
     * pd/pb = (b - a) / d
     */
    float inv_d = 1.0f / (distance + CIFFY_EPS);
    float scale = grad_output * inv_d;

    float dx = a[0] - b[0];
    float dy = a[1] - b[1];
    float dz = a[2] - b[2];

    grad_a[0] = scale * dx;
    grad_a[1] = scale * dy;
    grad_a[2] = scale * dz;

    grad_b[0] = -scale * dx;
    grad_b[1] = -scale * dy;
    grad_b[2] = -scale * dz;
}


/**
 * Backward pass for compute_angle.
 */
CIFFY_HOST_DEVICE
static inline void compute_angle_backward_impl(
    const float *a, const float *b, const float *c,
    float angle, float grad_output,
    float *grad_a, float *grad_b, float *grad_c
) {
    float v1x = a[0] - b[0];
    float v1y = a[1] - b[1];
    float v1z = a[2] - b[2];

    float v2x = c[0] - b[0];
    float v2y = c[1] - b[1];
    float v2z = c[2] - b[2];

    float n1 = CIFFY_SQRTF(v1x*v1x + v1y*v1y + v1z*v1z) + CIFFY_EPS;
    float n2 = CIFFY_SQRTF(v2x*v2x + v2y*v2y + v2z*v2z) + CIFFY_EPS;

    /* Unit vectors */
    float u1x = v1x / n1, u1y = v1y / n1, u1z = v1z / n1;
    float u2x = v2x / n2, u2y = v2y / n2, u2z = v2z / n2;

    float cos_theta = u1x*u2x + u1y*u2y + u1z*u2z;
    if (cos_theta > 1.0f) cos_theta = 1.0f;
    if (cos_theta < -1.0f) cos_theta = -1.0f;

    float sin_theta = CIFFY_SINF(angle);
    if (sin_theta < CIFFY_EPS) sin_theta = CIFFY_EPS;

    float d_theta_d_cos = -1.0f / sin_theta;

    float dcos_dv1x = (u2x - cos_theta * u1x) / n1;
    float dcos_dv1y = (u2y - cos_theta * u1y) / n1;
    float dcos_dv1z = (u2z - cos_theta * u1z) / n1;

    float dcos_dv2x = (u1x - cos_theta * u2x) / n2;
    float dcos_dv2y = (u1y - cos_theta * u2y) / n2;
    float dcos_dv2z = (u1z - cos_theta * u2z) / n2;

    float scale = grad_output * d_theta_d_cos;

    grad_a[0] = scale * dcos_dv1x;
    grad_a[1] = scale * dcos_dv1y;
    grad_a[2] = scale * dcos_dv1z;

    grad_c[0] = scale * dcos_dv2x;
    grad_c[1] = scale * dcos_dv2y;
    grad_c[2] = scale * dcos_dv2z;

    grad_b[0] = -grad_a[0] - grad_c[0];
    grad_b[1] = -grad_a[1] - grad_c[1];
    grad_b[2] = -grad_a[2] - grad_c[2];
}


/**
 * Backward pass for compute_dihedral.
 */
CIFFY_HOST_DEVICE
static inline void compute_dihedral_backward_impl(
    const float *a, const float *b, const float *c, const float *d,
    float grad_output,
    float *grad_a, float *grad_b, float *grad_c, float *grad_d
) {
    /* Forward pass (save intermediates) */
    float b1x = b[0] - a[0], b1y = b[1] - a[1], b1z = b[2] - a[2];
    float b2x = c[0] - b[0], b2y = c[1] - b[1], b2z = c[2] - b[2];
    float b3x = d[0] - c[0], b3y = d[1] - c[1], b3z = d[2] - c[2];

    float n1x = b1y*b2z - b1z*b2y;
    float n1y = b1z*b2x - b1x*b2z;
    float n1z = b1x*b2y - b1y*b2x;

    float n2x = b2y*b3z - b2z*b3y;
    float n2y = b2z*b3x - b2x*b3z;
    float n2z = b2x*b3y - b2y*b3x;

    float n1_norm = CIFFY_SQRTF(n1x*n1x + n1y*n1y + n1z*n1z) + CIFFY_EPS;
    float n2_norm = CIFFY_SQRTF(n2x*n2x + n2y*n2y + n2z*n2z) + CIFFY_EPS;
    float b2_norm = CIFFY_SQRTF(b2x*b2x + b2y*b2y + b2z*b2z) + CIFFY_EPS;

    float n1hx = n1x/n1_norm, n1hy = n1y/n1_norm, n1hz = n1z/n1_norm;
    float n2hx = n2x/n2_norm, n2hy = n2y/n2_norm, n2hz = n2z/n2_norm;
    float b2hx = b2x/b2_norm, b2hy = b2y/b2_norm, b2hz = b2z/b2_norm;

    float m1x = n1hy*b2hz - n1hz*b2hy;
    float m1y = n1hz*b2hx - n1hx*b2hz;
    float m1z = n1hx*b2hy - n1hy*b2hx;

    float x = n1hx*n2hx + n1hy*n2hy + n1hz*n2hz;
    float y = m1x*n2hx + m1y*n2hy + m1z*n2hz;

    /* Backward pass for atan2(y, x) */
    float denom = x*x + y*y;
    float grad_y, grad_x;

    if (denom < CIFFY_EPS * CIFFY_EPS) {
        /* Near singularity: both x and y are near zero.
         * This happens with collinear atoms (undefined dihedral).
         * Return zero gradients - no meaningful direction to optimize. */
        grad_y = 0.0f;
        grad_x = 0.0f;
    } else {
        denom += CIFFY_EPS;
        grad_y = grad_output * x / denom;
        grad_x = grad_output * (-y) / denom;
    }

    float gn1h_x = grad_x * n2hx;
    float gn1h_y = grad_x * n2hy;
    float gn1h_z = grad_x * n2hz;
    float gn2h_x = grad_x * n1hx;
    float gn2h_y = grad_x * n1hy;
    float gn2h_z = grad_x * n1hz;

    float gm1x = grad_y * n2hx;
    float gm1y = grad_y * n2hy;
    float gm1z = grad_y * n2hz;
    gn2h_x += grad_y * m1x;
    gn2h_y += grad_y * m1y;
    gn2h_z += grad_y * m1z;

    gn1h_x += b2hy*gm1z - b2hz*gm1y;
    gn1h_y += b2hz*gm1x - b2hx*gm1z;
    gn1h_z += b2hx*gm1y - b2hy*gm1x;

    float gb2h_x = gm1y*n1hz - gm1z*n1hy;
    float gb2h_y = gm1z*n1hx - gm1x*n1hz;
    float gb2h_z = gm1x*n1hy - gm1y*n1hx;

    float b2h_dot_gb2h = b2hx*gb2h_x + b2hy*gb2h_y + b2hz*gb2h_z;
    float gb2_x = (gb2h_x - b2hx * b2h_dot_gb2h) / b2_norm;
    float gb2_y = (gb2h_y - b2hy * b2h_dot_gb2h) / b2_norm;
    float gb2_z = (gb2h_z - b2hz * b2h_dot_gb2h) / b2_norm;

    float n2h_dot_gn2h = n2hx*gn2h_x + n2hy*gn2h_y + n2hz*gn2h_z;
    float gn2x = (gn2h_x - n2hx * n2h_dot_gn2h) / n2_norm;
    float gn2y = (gn2h_y - n2hy * n2h_dot_gn2h) / n2_norm;
    float gn2z = (gn2h_z - n2hz * n2h_dot_gn2h) / n2_norm;

    float n1h_dot_gn1h = n1hx*gn1h_x + n1hy*gn1h_y + n1hz*gn1h_z;
    float gn1x = (gn1h_x - n1hx * n1h_dot_gn1h) / n1_norm;
    float gn1y = (gn1h_y - n1hy * n1h_dot_gn1h) / n1_norm;
    float gn1z = (gn1h_z - n1hz * n1h_dot_gn1h) / n1_norm;

    gb2_x += b3y*gn2z - b3z*gn2y;
    gb2_y += b3z*gn2x - b3x*gn2z;
    gb2_z += b3x*gn2y - b3y*gn2x;

    float gb3_x = gn2y*b2z - gn2z*b2y;
    float gb3_y = gn2z*b2x - gn2x*b2z;
    float gb3_z = gn2x*b2y - gn2y*b2x;

    float gb1_x = b2y*gn1z - b2z*gn1y;
    float gb1_y = b2z*gn1x - b2x*gn1z;
    float gb1_z = b2x*gn1y - b2y*gn1x;

    gb2_x += gn1y*b1z - gn1z*b1y;
    gb2_y += gn1z*b1x - gn1x*b1z;
    gb2_z += gn1x*b1y - gn1y*b1x;

    grad_a[0] = -gb1_x;
    grad_a[1] = -gb1_y;
    grad_a[2] = -gb1_z;

    grad_b[0] = gb1_x - gb2_x;
    grad_b[1] = gb1_y - gb2_y;
    grad_b[2] = gb1_z - gb2_z;

    grad_c[0] = gb2_x - gb3_x;
    grad_c[1] = gb2_y - gb3_y;
    grad_c[2] = gb2_z - gb3_z;

    grad_d[0] = gb3_x;
    grad_d[1] = gb3_y;
    grad_d[2] = gb3_z;
}


/**
 * Backward pass for nerf_place_atom.
 */
CIFFY_HOST_DEVICE
static inline void nerf_place_atom_backward_impl(
    const float *a, const float *b, const float *c,
    float distance, float angle, float dihedral,
    const float *grad_result,
    float *grad_a, float *grad_b, float *grad_c,
    float *grad_distance, float *grad_angle, float *grad_dihedral
) {
    /* Forward pass: save all intermediates */
    float z_raw[3], z[3], v_raw[3], v[3], y_raw[3], y[3], x[3];
    float z_norm, v_norm, y_norm;

    vec_sub(b, c, z_raw);
    z_norm = vec_normalize(z_raw, z);

    vec_sub(a, c, v_raw);
    v_norm = vec_normalize(v_raw, v);

    vec_cross(z, v, y_raw);
    y_norm = vec_normalize(y_raw, y);

    vec_cross(y, z, x);

    float cos_a = CIFFY_COSF(angle), sin_a = CIFFY_SINF(angle);
    float cos_d = CIFFY_COSF(dihedral), sin_d = CIFFY_SINF(dihedral);

    float d_z = distance * cos_a;
    float d_perp = distance * sin_a;
    float d_x = d_perp * cos_d;
    float d_y = d_perp * sin_d;

    /* Backward pass: reverse order */
    vec_zero(grad_a);
    vec_zero(grad_b);
    vec_zero(grad_c);
    *grad_distance = 0.0f;
    *grad_angle = 0.0f;
    *grad_dihedral = 0.0f;

    float grad_z[3] = {0}, grad_v[3] = {0}, grad_y[3] = {0}, grad_x[3] = {0};
    float grad_z_raw[3] = {0}, grad_v_raw[3] = {0}, grad_y_raw[3] = {0};
    float grad_d_z = 0, grad_d_x = 0, grad_d_y = 0, grad_d_perp = 0;

    vec_acc(grad_result, grad_c);

    vec_lincomb3_backward(
        d_z, z, d_x, x, d_y, y,
        grad_result,
        &grad_d_z, grad_z,
        &grad_d_x, grad_x,
        &grad_d_y, grad_y
    );

    grad_d_perp = grad_d_x * cos_d + grad_d_y * sin_d;
    *grad_dihedral = d_perp * (-sin_d * grad_d_x + cos_d * grad_d_y);

    *grad_distance = grad_d_z * cos_a + grad_d_perp * sin_a;
    *grad_angle = distance * (-sin_a * grad_d_z + cos_a * grad_d_perp);

    vec_cross_backward(y, z, grad_x, grad_y, grad_z);
    vec_normalize_backward(y, y_norm, grad_y, grad_y_raw);
    vec_cross_backward(z, v, grad_y_raw, grad_z, grad_v);
    vec_normalize_backward(v, v_norm, grad_v, grad_v_raw);
    vec_normalize_backward(z, z_norm, grad_z, grad_z_raw);
    vec_sub_backward(grad_v_raw, grad_a, grad_c);
    vec_sub_backward(grad_z_raw, grad_b, grad_c);
}


/**
 * Backward pass for nerf_place_in_plane.
 */
CIFFY_HOST_DEVICE
static inline void nerf_place_in_plane_backward_impl(
    const float *ref1, const float *ref2,
    float distance, float angle,
    const float *grad_result,
    float *grad_ref1, float *grad_ref2,
    float *grad_distance, float *grad_angle
) {
    /* Forward pass: save all intermediates */
    float u_raw[3], u[3];
    vec_sub(ref2, ref1, u_raw);
    float u_norm = vec_normalize(u_raw, u);

    float perp_raw[3] = {-u[1], u[0], 0.0f};
    float perp_norm_val = vec_norm(perp_raw);

    float perp[3];
    int use_fallback = (perp_norm_val < PRIM_EPS);
    if (use_fallback) {
        perp_raw[0] = 0.0f;
        perp_raw[1] = u[2];
        perp_raw[2] = -u[1];
        perp_norm_val = vec_norm(perp_raw);
    }
    perp_norm_val += PRIM_EPS;
    perp[0] = perp_raw[0] / perp_norm_val;
    perp[1] = perp_raw[1] / perp_norm_val;
    perp[2] = perp_raw[2] / perp_norm_val;

    float cos_a = CIFFY_COSF(angle), sin_a = CIFFY_SINF(angle);

    /* Backward pass */
    vec_zero(grad_ref1);
    vec_zero(grad_ref2);
    *grad_distance = 0.0f;
    *grad_angle = 0.0f;

    float grad_u[3] = {0}, grad_perp[3] = {0};
    float grad_u_raw[3] = {0}, grad_perp_raw[3] = {0};

    float disp[3];
    float zeros[3] = {0, 0, 0};
    vec_lincomb3(cos_a, u, sin_a, perp, 0.0f, zeros, disp);

    vec_acc(grad_result, grad_ref1);
    *grad_distance = vec_dot(grad_result, disp);

    float grad_disp[3];
    vec_scale(distance, grad_result, grad_disp);

    float grad_cos_a = vec_dot(grad_disp, u);
    float grad_sin_a = vec_dot(grad_disp, perp);

    vec_acc_scaled(cos_a, grad_disp, grad_u);
    vec_acc_scaled(sin_a, grad_disp, grad_perp);

    *grad_angle = -sin_a * grad_cos_a + cos_a * grad_sin_a;

    vec_normalize_backward(perp, perp_norm_val, grad_perp, grad_perp_raw);

    if (!use_fallback) {
        grad_u[0] += grad_perp_raw[1];
        grad_u[1] += -grad_perp_raw[0];
    } else {
        grad_u[1] += -grad_perp_raw[2];
        grad_u[2] += grad_perp_raw[1];
    }

    vec_normalize_backward(u, u_norm, grad_u, grad_u_raw);
    vec_sub_backward(grad_u_raw, grad_ref2, grad_ref1);
}


/**
 * Backward pass for nerf_place_along_direction.
 *
 * Note: No gradients flow to anchor1 (frozen reference).
 */
CIFFY_HOST_DEVICE
static inline void nerf_place_along_direction_backward_impl(
    const float *ref,
    const float *anchor1,
    float distance,
    const float *grad_result,
    float *grad_ref,
    float *grad_distance
) {
    /* Forward: result = ref + distance * d_hat where d_hat = (anchor1 - ref) / |anchor1 - ref| */
    float dx = anchor1[0] - ref[0];
    float dy = anchor1[1] - ref[1];
    float dz = anchor1[2] - ref[2];
    float norm = CIFFY_SQRTF(dx*dx + dy*dy + dz*dz) + CIFFY_EPS;
    float dhx = dx / norm, dhy = dy / norm, dhz = dz / norm;

    /* grad_distance = grad_result . d_hat */
    *grad_distance = grad_result[0] * dhx + grad_result[1] * dhy + grad_result[2] * dhz;

    /* grad_ref = grad_result (from the ref term in result = ref + ...) */
    /* Plus contribution from d_hat depending on ref, but anchor is frozen so we simplify */
    grad_ref[0] = grad_result[0];
    grad_ref[1] = grad_result[1];
    grad_ref[2] = grad_result[2];
}


/**
 * Backward pass for nerf_place_in_plane_anchored.
 *
 * Note: No gradients flow to anchor2 (frozen reference).
 */
CIFFY_HOST_DEVICE
static inline void nerf_place_in_plane_anchored_backward_impl(
    const float *ref1,
    const float *ref2,
    const float *anchor2,
    float distance,
    float angle,
    const float *grad_result,
    float *grad_ref1,
    float *grad_ref2,
    float *grad_distance,
    float *grad_angle
) {
    /* This is similar to nerf_place_in_plane_backward but with anchored plane normal.
     * For simplicity, we compute gradients only for distance and angle,
     * and pass gradient through to ref1/ref2. anchor2 is frozen. */

    float ux = ref2[0] - ref1[0];
    float uy = ref2[1] - ref1[1];
    float uz = ref2[2] - ref1[2];
    float u_norm = CIFFY_SQRTF(ux*ux + uy*uy + uz*uz) + CIFFY_EPS;
    ux /= u_norm; uy /= u_norm; uz /= u_norm;

    float v1x = ref1[0] - ref2[0];
    float v1y = ref1[1] - ref2[1];
    float v1z = ref1[2] - ref2[2];

    float v2x = anchor2[0] - ref2[0];
    float v2y = anchor2[1] - ref2[1];
    float v2z = anchor2[2] - ref2[2];

    float nx = v1y*v2z - v1z*v2y;
    float ny = v1z*v2x - v1x*v2z;
    float nz = v1x*v2y - v1y*v2x;

    float perpx = ny*uz - nz*uy;
    float perpy = nz*ux - nx*uz;
    float perpz = nx*uy - ny*ux;
    float perp_norm = CIFFY_SQRTF(perpx*perpx + perpy*perpy + perpz*perpz) + CIFFY_EPS;
    perpx /= perp_norm; perpy /= perp_norm; perpz /= perp_norm;

    float cos_a = CIFFY_COSF(angle), sin_a = CIFFY_SINF(angle);

    /* disp = cos_a * u + sin_a * perp */
    float dispx = cos_a * ux + sin_a * perpx;
    float dispy = cos_a * uy + sin_a * perpy;
    float dispz = cos_a * uz + sin_a * perpz;

    /* grad_distance = grad_result . disp */
    *grad_distance = grad_result[0] * dispx + grad_result[1] * dispy + grad_result[2] * dispz;

    /* grad_angle: result depends on angle via cos_a, sin_a */
    /* d(result)/d(angle) = distance * (-sin_a * u + cos_a * perp) */
    float da_dispx = -sin_a * ux + cos_a * perpx;
    float da_dispy = -sin_a * uy + cos_a * perpy;
    float da_dispz = -sin_a * uz + cos_a * perpz;
    *grad_angle = distance * (grad_result[0] * da_dispx + grad_result[1] * da_dispy + grad_result[2] * da_dispz);

    /* Gradients to ref1, ref2 - simplified: pass through from result = ref1 + ... */
    grad_ref1[0] = grad_result[0];
    grad_ref1[1] = grad_result[1];
    grad_ref1[2] = grad_result[2];

    grad_ref2[0] = 0.0f;
    grad_ref2[1] = 0.0f;
    grad_ref2[2] = 0.0f;
}


#endif /* CIFFY_GEOMETRY_IMPL_H */
