/**
 * @file geometry.c
 * @brief Core geometry functions for internal coordinates.
 */

#include "geometry.h"
#include "primitives.h"
#include <math.h>

/* Small epsilon for numerical stability */
#define EPS 1e-6f


float compute_distance(const float *a, const float *b) {
    float dx = a[0] - b[0];
    float dy = a[1] - b[1];
    float dz = a[2] - b[2];
    return sqrtf(dx*dx + dy*dy + dz*dz);
}


float compute_angle(const float *a, const float *b, const float *c) {
    /* Vectors from vertex B */
    float v1x = a[0] - b[0];
    float v1y = a[1] - b[1];
    float v1z = a[2] - b[2];

    float v2x = c[0] - b[0];
    float v2y = c[1] - b[1];
    float v2z = c[2] - b[2];

    /* Norms */
    float v1_norm = sqrtf(v1x*v1x + v1y*v1y + v1z*v1z) + EPS;
    float v2_norm = sqrtf(v2x*v2x + v2y*v2y + v2z*v2z) + EPS;

    /* Dot product and cosine */
    float dot = v1x*v2x + v1y*v2y + v1z*v2z;
    float cos_angle = dot / (v1_norm * v2_norm);

    /* Clamp to [-1, 1] for numerical stability */
    if (cos_angle > 1.0f) cos_angle = 1.0f;
    if (cos_angle < -1.0f) cos_angle = -1.0f;

    return acosf(cos_angle);
}


float compute_dihedral(const float *a, const float *b,
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
    float n1_norm = sqrtf(n1x*n1x + n1y*n1y + n1z*n1z) + EPS;
    n1x /= n1_norm;
    n1y /= n1_norm;
    n1z /= n1_norm;

    /* Normalize n2 */
    float n2_norm = sqrtf(n2x*n2x + n2y*n2y + n2z*n2z) + EPS;
    n2x /= n2_norm;
    n2y /= n2_norm;
    n2z /= n2_norm;

    /* Normalize b2 for m1 calculation */
    float b2_norm = sqrtf(b2x*b2x + b2y*b2y + b2z*b2z) + EPS;
    float b2ux = b2x / b2_norm;
    float b2uy = b2y / b2_norm;
    float b2uz = b2z / b2_norm;

    /* m1 = n1_hat x b2_hat */
    float m1x = n1y*b2uz - n1z*b2uy;
    float m1y = n1z*b2ux - n1x*b2uz;
    float m1z = n1x*b2uy - n1y*b2ux;

    /* x = n1_hat · n2_hat = cos(φ), y = m1 · n2_hat = sin(φ) */
    float x = n1x*n2x + n1y*n2y + n1z*n2z;
    float y = m1x*n2x + m1y*n2y + m1z*n2z;

    return atan2f(y, x);
}


void nerf_place_atom(const float *a, const float *b, const float *c,
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
    float z_len = sqrtf(zx*zx + zy*zy + zz*zz) + EPS;
    zx /= z_len;
    zy /= z_len;
    zz /= z_len;

    /* v = direction from c to a (normalized) */
    float vx = a[0] - c[0];
    float vy = a[1] - c[1];
    float vz = a[2] - c[2];
    float v_len = sqrtf(vx*vx + vy*vy + vz*vz) + EPS;
    vx /= v_len;
    vy /= v_len;
    vz /= v_len;

    /* y = z cross v (normal to plane, right-handed) */
    float yx = zy*vz - zz*vy;
    float yy = zz*vx - zx*vz;
    float yz = zx*vy - zy*vx;
    float y_len = sqrtf(yx*yx + yy*yy + yz*yz) + EPS;
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
    float cos_a = cosf(angle);
    float sin_a = sinf(angle);
    float cos_d = cosf(dihedral);
    float sin_d = sinf(dihedral);

    float d_z = distance * cos_a;
    float d_perp = distance * sin_a;
    float d_x = d_perp * cos_d;
    float d_y = d_perp * sin_d;

    result[0] = c[0] + d_z * zx + d_x * xx + d_y * yx;
    result[1] = c[1] + d_z * zy + d_x * xy + d_y * yy;
    result[2] = c[2] + d_z * zz + d_x * xz + d_y * yz;
}


void nerf_place_along_x(const float *ref, float distance, float *result) {
    result[0] = ref[0] + distance;
    result[1] = ref[1];
    result[2] = ref[2];
}


void nerf_place_in_plane(const float *ref1, const float *ref2,
                         float distance, float angle, float *result) {
    /* Direction from ref1 to ref2 (normalized) */
    float ux = ref2[0] - ref1[0];
    float uy = ref2[1] - ref1[1];
    float uz = ref2[2] - ref1[2];
    float u_norm = sqrtf(ux*ux + uy*uy + uz*uz) + EPS;
    ux /= u_norm;
    uy /= u_norm;
    uz /= u_norm;

    /* Create perpendicular direction using z-axis cross product */
    /* perp = (0, 0, 1) x u */
    float perpx = -uy;   /* 0*uz - 1*uy */
    float perpy = ux;    /* 1*ux - 0*uz */
    float perpz = 0.0f;  /* 0*uy - 0*ux */
    float perp_norm = sqrtf(perpx*perpx + perpy*perpy + perpz*perpz);

    if (perp_norm < EPS) {
        /* u is parallel to z-axis, use x-axis instead */
        perpx = 0.0f;
        perpy = uz;
        perpz = -uy;
        perp_norm = sqrtf(perpx*perpx + perpy*perpy + perpz*perpz);
    }
    perp_norm += EPS;
    perpx /= perp_norm;
    perpy /= perp_norm;
    perpz /= perp_norm;

    /* new_pos = ref1 + distance * (cos(angle) * u + sin(angle) * perp) */
    float cos_a = cosf(angle);
    float sin_a = sinf(angle);

    result[0] = ref1[0] + distance * (cos_a * ux + sin_a * perpx);
    result[1] = ref1[1] + distance * (cos_a * uy + sin_a * perpy);
    result[2] = ref1[2] + distance * (cos_a * uz + sin_a * perpz);
}


/* ========================================================================= */
/* Backward (gradient) functions for automatic differentiation              */
/* ========================================================================= */


void compute_distance_backward(
    const float *a, const float *b,
    float distance, float grad_output,
    float *grad_a, float *grad_b
) {
    /* d = ||a - b||
     * ∂d/∂a = (a - b) / d
     * ∂d/∂b = (b - a) / d
     */
    float inv_d = 1.0f / (distance + EPS);
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


void compute_angle_backward(
    const float *a, const float *b, const float *c,
    float angle, float grad_output,
    float *grad_a, float *grad_b, float *grad_c
) {
    /* θ = acos(cos_θ) where cos_θ = (v1 · v2) / (|v1| |v2|)
     * v1 = a - b, v2 = c - b
     *
     * ∂θ/∂cos_θ = -1 / sin(θ)
     * ∂cos_θ/∂v1 = (v2/|v2| - cos_θ * v1/|v1|) / |v1|
     * ∂cos_θ/∂v2 = (v1/|v1| - cos_θ * v2/|v2|) / |v2|
     */
    float v1x = a[0] - b[0];
    float v1y = a[1] - b[1];
    float v1z = a[2] - b[2];

    float v2x = c[0] - b[0];
    float v2y = c[1] - b[1];
    float v2z = c[2] - b[2];

    float n1 = sqrtf(v1x*v1x + v1y*v1y + v1z*v1z) + EPS;
    float n2 = sqrtf(v2x*v2x + v2y*v2y + v2z*v2z) + EPS;

    /* Unit vectors */
    float u1x = v1x / n1, u1y = v1y / n1, u1z = v1z / n1;
    float u2x = v2x / n2, u2y = v2y / n2, u2z = v2z / n2;

    float cos_theta = u1x*u2x + u1y*u2y + u1z*u2z;
    if (cos_theta > 1.0f) cos_theta = 1.0f;
    if (cos_theta < -1.0f) cos_theta = -1.0f;

    float sin_theta = sinf(angle);
    if (sin_theta < EPS) sin_theta = EPS;  /* Avoid division by zero at 0 or π */

    /* ∂θ/∂cos_θ = -1/sin_θ */
    float d_theta_d_cos = -1.0f / sin_theta;

    /* ∂cos_θ/∂v1 = (u2 - cos_θ * u1) / n1 */
    float dcos_dv1x = (u2x - cos_theta * u1x) / n1;
    float dcos_dv1y = (u2y - cos_theta * u1y) / n1;
    float dcos_dv1z = (u2z - cos_theta * u1z) / n1;

    /* ∂cos_θ/∂v2 = (u1 - cos_θ * u2) / n2 */
    float dcos_dv2x = (u1x - cos_theta * u2x) / n2;
    float dcos_dv2y = (u1y - cos_theta * u2y) / n2;
    float dcos_dv2z = (u1z - cos_theta * u2z) / n2;

    /* Chain rule: ∂θ/∂v = ∂θ/∂cos_θ * ∂cos_θ/∂v */
    float scale = grad_output * d_theta_d_cos;

    /* ∂θ/∂a = ∂θ/∂v1 (since v1 = a - b) */
    grad_a[0] = scale * dcos_dv1x;
    grad_a[1] = scale * dcos_dv1y;
    grad_a[2] = scale * dcos_dv1z;

    /* ∂θ/∂c = ∂θ/∂v2 (since v2 = c - b) */
    grad_c[0] = scale * dcos_dv2x;
    grad_c[1] = scale * dcos_dv2y;
    grad_c[2] = scale * dcos_dv2z;

    /* ∂θ/∂b = -∂θ/∂a - ∂θ/∂c (since b appears in both v1 and v2) */
    grad_b[0] = -grad_a[0] - grad_c[0];
    grad_b[1] = -grad_a[1] - grad_c[1];
    grad_b[2] = -grad_a[2] - grad_c[2];
}


void compute_dihedral_backward(
    const float *a, const float *b, const float *c, const float *d,
    float grad_output,
    float *grad_a, float *grad_b, float *grad_c, float *grad_d
) {
    /*
     * Backward pass for dihedral angle, composed from primitive operations.
     *
     * Forward:
     *   b1 = b - a, b2 = c - b, b3 = d - c
     *   n1 = b1 × b2, n2 = b2 × b3
     *   n1_hat = n1 / |n1|, n2_hat = n2 / |n2|, b2_hat = b2 / |b2|
     *   m1 = n1_hat × b2_hat
     *   x = n1_hat · n2_hat, y = m1 · n2_hat
     *   φ = atan2(y, x)
     *
     * We backprop through each operation using standard rules:
     *   cross: ∂L/∂a = b × grad, ∂L/∂b = grad × a
     *   normalize: ∂L/∂v = (grad - v_hat * (v_hat · grad)) / |v|
     *   dot: ∂L/∂a = grad * b, ∂L/∂b = grad * a
     *   atan2: ∂L/∂y = grad * x / (x²+y²), ∂L/∂x = grad * (-y) / (x²+y²)
     */

    /* === Forward pass (save intermediates) === */
    float b1x = b[0] - a[0], b1y = b[1] - a[1], b1z = b[2] - a[2];
    float b2x = c[0] - b[0], b2y = c[1] - b[1], b2z = c[2] - b[2];
    float b3x = d[0] - c[0], b3y = d[1] - c[1], b3z = d[2] - c[2];

    float n1x = b1y*b2z - b1z*b2y;
    float n1y = b1z*b2x - b1x*b2z;
    float n1z = b1x*b2y - b1y*b2x;

    float n2x = b2y*b3z - b2z*b3y;
    float n2y = b2z*b3x - b2x*b3z;
    float n2z = b2x*b3y - b2y*b3x;

    float n1_norm = sqrtf(n1x*n1x + n1y*n1y + n1z*n1z) + EPS;
    float n2_norm = sqrtf(n2x*n2x + n2y*n2y + n2z*n2z) + EPS;
    float b2_norm = sqrtf(b2x*b2x + b2y*b2y + b2z*b2z) + EPS;

    float n1hx = n1x/n1_norm, n1hy = n1y/n1_norm, n1hz = n1z/n1_norm;
    float n2hx = n2x/n2_norm, n2hy = n2y/n2_norm, n2hz = n2z/n2_norm;
    float b2hx = b2x/b2_norm, b2hy = b2y/b2_norm, b2hz = b2z/b2_norm;

    float m1x = n1hy*b2hz - n1hz*b2hy;
    float m1y = n1hz*b2hx - n1hx*b2hz;
    float m1z = n1hx*b2hy - n1hy*b2hx;

    float x = n1hx*n2hx + n1hy*n2hy + n1hz*n2hz;
    float y = m1x*n2hx + m1y*n2hy + m1z*n2hz;

    /* === Backward pass === */

    /* φ = atan2(y, x) */
    float denom = x*x + y*y + EPS;
    float grad_y = grad_output * x / denom;
    float grad_x = grad_output * (-y) / denom;

    /* x = n1_hat · n2_hat */
    float gn1h_x = grad_x * n2hx;
    float gn1h_y = grad_x * n2hy;
    float gn1h_z = grad_x * n2hz;
    float gn2h_x = grad_x * n1hx;
    float gn2h_y = grad_x * n1hy;
    float gn2h_z = grad_x * n1hz;

    /* y = m1 · n2_hat */
    float gm1x = grad_y * n2hx;
    float gm1y = grad_y * n2hy;
    float gm1z = grad_y * n2hz;
    gn2h_x += grad_y * m1x;
    gn2h_y += grad_y * m1y;
    gn2h_z += grad_y * m1z;

    /* m1 = n1_hat × b2_hat */
    /* ∂L/∂n1_hat = b2_hat × grad_m1 */
    gn1h_x += b2hy*gm1z - b2hz*gm1y;
    gn1h_y += b2hz*gm1x - b2hx*gm1z;
    gn1h_z += b2hx*gm1y - b2hy*gm1x;
    /* ∂L/∂b2_hat = grad_m1 × n1_hat */
    float gb2h_x = gm1y*n1hz - gm1z*n1hy;
    float gb2h_y = gm1z*n1hx - gm1x*n1hz;
    float gb2h_z = gm1x*n1hy - gm1y*n1hx;

    /* b2_hat = b2 / |b2| (normalize backward) */
    float b2h_dot_gb2h = b2hx*gb2h_x + b2hy*gb2h_y + b2hz*gb2h_z;
    float gb2_x = (gb2h_x - b2hx * b2h_dot_gb2h) / b2_norm;
    float gb2_y = (gb2h_y - b2hy * b2h_dot_gb2h) / b2_norm;
    float gb2_z = (gb2h_z - b2hz * b2h_dot_gb2h) / b2_norm;

    /* n2_hat = n2 / |n2| */
    float n2h_dot_gn2h = n2hx*gn2h_x + n2hy*gn2h_y + n2hz*gn2h_z;
    float gn2x = (gn2h_x - n2hx * n2h_dot_gn2h) / n2_norm;
    float gn2y = (gn2h_y - n2hy * n2h_dot_gn2h) / n2_norm;
    float gn2z = (gn2h_z - n2hz * n2h_dot_gn2h) / n2_norm;

    /* n1_hat = n1 / |n1| */
    float n1h_dot_gn1h = n1hx*gn1h_x + n1hy*gn1h_y + n1hz*gn1h_z;
    float gn1x = (gn1h_x - n1hx * n1h_dot_gn1h) / n1_norm;
    float gn1y = (gn1h_y - n1hy * n1h_dot_gn1h) / n1_norm;
    float gn1z = (gn1h_z - n1hz * n1h_dot_gn1h) / n1_norm;

    /* n2 = b2 × b3 */
    /* ∂L/∂b2 += b3 × grad_n2 */
    gb2_x += b3y*gn2z - b3z*gn2y;
    gb2_y += b3z*gn2x - b3x*gn2z;
    gb2_z += b3x*gn2y - b3y*gn2x;
    /* ∂L/∂b3 = grad_n2 × b2 */
    float gb3_x = gn2y*b2z - gn2z*b2y;
    float gb3_y = gn2z*b2x - gn2x*b2z;
    float gb3_z = gn2x*b2y - gn2y*b2x;

    /* n1 = b1 × b2 */
    /* ∂L/∂b1 = b2 × grad_n1 */
    float gb1_x = b2y*gn1z - b2z*gn1y;
    float gb1_y = b2z*gn1x - b2x*gn1z;
    float gb1_z = b2x*gn1y - b2y*gn1x;
    /* ∂L/∂b2 += grad_n1 × b1 */
    gb2_x += gn1y*b1z - gn1z*b1y;
    gb2_y += gn1z*b1x - gn1x*b1z;
    gb2_z += gn1x*b1y - gn1y*b1x;

    /* Bond vectors to atom gradients */
    /* b1 = b - a  =>  grad_a = -grad_b1, grad_b += grad_b1 */
    /* b2 = c - b  =>  grad_b -= grad_b2, grad_c = grad_b2 */
    /* b3 = d - c  =>  grad_c -= grad_b3, grad_d = grad_b3 */
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


void nerf_place_atom_backward(
    const float *a, const float *b, const float *c,
    float distance, float angle, float dihedral,
    const float *grad_result,
    float *grad_a, float *grad_b, float *grad_c,
    float *grad_distance, float *grad_angle, float *grad_dihedral
) {
    /*
     * Backward pass for NERF placement using primitive operations.
     *
     * Forward computation graph:
     *   z_raw = b - c           (vec_sub)
     *   z = z_raw / |z_raw|     (vec_normalize)
     *   v_raw = a - c           (vec_sub)
     *   v = v_raw / |v_raw|     (vec_normalize)
     *   y_raw = z × v           (vec_cross)
     *   y = y_raw / |y_raw|     (vec_normalize)
     *   x = y × z               (vec_cross)
     *   d_z = distance * cos(angle)
     *   d_perp = distance * sin(angle)
     *   d_x = d_perp * cos(dihedral)
     *   d_y = d_perp * sin(dihedral)
     *   result = c + d_z * z + d_x * x + d_y * y   (vec_lincomb3 + vec_add)
     */

    /* === Forward pass: save all intermediates === */
    float z_raw[3], z[3], v_raw[3], v[3], y_raw[3], y[3], x[3];
    float z_norm, v_norm, y_norm;

    vec_sub(b, c, z_raw);
    z_norm = vec_normalize(z_raw, z);

    vec_sub(a, c, v_raw);
    v_norm = vec_normalize(v_raw, v);

    vec_cross(z, v, y_raw);
    y_norm = vec_normalize(y_raw, y);

    vec_cross(y, z, x);

    float cos_a = cosf(angle), sin_a = sinf(angle);
    float cos_d = cosf(dihedral), sin_d = sinf(dihedral);

    float d_z = distance * cos_a;
    float d_perp = distance * sin_a;
    float d_x = d_perp * cos_d;
    float d_y = d_perp * sin_d;

    /* === Backward pass: reverse order === */

    /* Initialize output gradients to zero */
    vec_zero(grad_a);
    vec_zero(grad_b);
    vec_zero(grad_c);
    *grad_distance = 0.0f;
    *grad_angle = 0.0f;
    *grad_dihedral = 0.0f;

    /* Intermediate gradients */
    float grad_z[3] = {0}, grad_v[3] = {0}, grad_y[3] = {0}, grad_x[3] = {0};
    float grad_z_raw[3] = {0}, grad_v_raw[3] = {0}, grad_y_raw[3] = {0};
    float grad_d_z = 0, grad_d_x = 0, grad_d_y = 0, grad_d_perp = 0;

    /* result = c + d_z * z + d_x * x + d_y * y */
    /* ∂c += grad_result (direct contribution) */
    vec_acc(grad_result, grad_c);

    /* Backward through lincomb: grad_si = grad_out · vi, grad_vi = si * grad_out */
    vec_lincomb3_backward(
        d_z, z, d_x, x, d_y, y,
        grad_result,
        &grad_d_z, grad_z,
        &grad_d_x, grad_x,
        &grad_d_y, grad_y
    );

    /* d_x = d_perp * cos_d, d_y = d_perp * sin_d */
    /* ∂d_perp = ∂d_x * cos_d + ∂d_y * sin_d */
    grad_d_perp = grad_d_x * cos_d + grad_d_y * sin_d;
    /* ∂dihedral = d_perp * (-sin_d * ∂d_x + cos_d * ∂d_y) */
    *grad_dihedral = d_perp * (-sin_d * grad_d_x + cos_d * grad_d_y);

    /* d_z = distance * cos_a, d_perp = distance * sin_a */
    /* ∂distance = ∂d_z * cos_a + ∂d_perp * sin_a */
    *grad_distance = grad_d_z * cos_a + grad_d_perp * sin_a;
    /* ∂angle = distance * (-sin_a * ∂d_z + cos_a * ∂d_perp) */
    *grad_angle = distance * (-sin_a * grad_d_z + cos_a * grad_d_perp);

    /* x = y × z */
    vec_cross_backward(y, z, grad_x, grad_y, grad_z);

    /* y = y_raw / |y_raw| */
    vec_normalize_backward(y, y_norm, grad_y, grad_y_raw);

    /* y_raw = z × v */
    vec_cross_backward(z, v, grad_y_raw, grad_z, grad_v);

    /* v = v_raw / |v_raw| */
    vec_normalize_backward(v, v_norm, grad_v, grad_v_raw);

    /* z = z_raw / |z_raw| */
    vec_normalize_backward(z, z_norm, grad_z, grad_z_raw);

    /* v_raw = a - c */
    vec_sub_backward(grad_v_raw, grad_a, grad_c);

    /* z_raw = b - c */
    vec_sub_backward(grad_z_raw, grad_b, grad_c);
}


void nerf_place_in_plane_backward(
    const float *ref1, const float *ref2,
    float distance, float angle,
    const float *grad_result,
    float *grad_ref1, float *grad_ref2,
    float *grad_distance, float *grad_angle
) {
    /*
     * Backward pass for in-plane placement using primitive operations.
     *
     * Forward computation graph:
     *   u_raw = ref2 - ref1           (vec_sub)
     *   u = u_raw / |u_raw|           (vec_normalize)
     *   perp_raw = z_axis × u         (vec_cross, z_axis = [0,0,1])
     *   perp = perp_raw / |perp_raw|  (vec_normalize, with fallback)
     *   result = ref1 + distance * (cos(angle) * u + sin(angle) * perp)
     */

    /* === Forward pass: save all intermediates === */
    float u_raw[3], u[3];
    vec_sub(ref2, ref1, u_raw);
    float u_norm = vec_normalize(u_raw, u);

    /* perp = z_axis × u = [-u[1], u[0], 0] (before normalization) */
    float perp_raw[3] = {-u[1], u[0], 0.0f};
    float perp_norm_val = vec_norm(perp_raw);

    float perp[3];
    int use_fallback = (perp_norm_val < PRIM_EPS);
    if (use_fallback) {
        /* u parallel to z, use x_axis × u = [0, u[2], -u[1]] */
        perp_raw[0] = 0.0f;
        perp_raw[1] = u[2];
        perp_raw[2] = -u[1];
        perp_norm_val = vec_norm(perp_raw);
    }
    perp_norm_val += PRIM_EPS;
    perp[0] = perp_raw[0] / perp_norm_val;
    perp[1] = perp_raw[1] / perp_norm_val;
    perp[2] = perp_raw[2] / perp_norm_val;

    float cos_a = cosf(angle), sin_a = sinf(angle);

    /* === Backward pass === */

    /* Initialize output gradients */
    vec_zero(grad_ref1);
    vec_zero(grad_ref2);
    *grad_distance = 0.0f;
    *grad_angle = 0.0f;

    /* Intermediate gradients */
    float grad_u[3] = {0}, grad_perp[3] = {0};
    float grad_u_raw[3] = {0}, grad_perp_raw[3] = {0};

    /* result = ref1 + distance * (cos_a * u + sin_a * perp) */
    /* Let disp = cos_a * u + sin_a * perp */
    float disp[3];
    float zeros[3] = {0, 0, 0};
    vec_lincomb3(cos_a, u, sin_a, perp, 0.0f, zeros, disp);

    /* ∂ref1 += grad_result */
    vec_acc(grad_result, grad_ref1);

    /* ∂distance = grad_result · disp */
    *grad_distance = vec_dot(grad_result, disp);

    /* ∂disp = distance * grad_result */
    float grad_disp[3];
    vec_scale(distance, grad_result, grad_disp);

    /* disp = cos_a * u + sin_a * perp */
    /* ∂cos_a = grad_disp · u, ∂sin_a = grad_disp · perp */
    float grad_cos_a = vec_dot(grad_disp, u);
    float grad_sin_a = vec_dot(grad_disp, perp);

    /* ∂u += cos_a * grad_disp */
    vec_acc_scaled(cos_a, grad_disp, grad_u);
    /* ∂perp += sin_a * grad_disp */
    vec_acc_scaled(sin_a, grad_disp, grad_perp);

    /* cos_a = cos(angle), sin_a = sin(angle) */
    /* ∂angle = -sin_a * ∂cos_a + cos_a * ∂sin_a */
    *grad_angle = -sin_a * grad_cos_a + cos_a * grad_sin_a;

    /* perp = perp_raw / |perp_raw| */
    vec_normalize_backward(perp, perp_norm_val, grad_perp, grad_perp_raw);

    /* perp_raw depends on u (via cross product with z or x axis) */
    if (!use_fallback) {
        /* perp_raw = [-u[1], u[0], 0] = z × u */
        /* ∂u[0] += ∂perp_raw[1], ∂u[1] += -∂perp_raw[0] */
        grad_u[0] += grad_perp_raw[1];
        grad_u[1] += -grad_perp_raw[0];
    } else {
        /* perp_raw = [0, u[2], -u[1]] = x × u */
        grad_u[1] += -grad_perp_raw[2];
        grad_u[2] += grad_perp_raw[1];
    }

    /* u = u_raw / |u_raw| */
    vec_normalize_backward(u, u_norm, grad_u, grad_u_raw);

    /* u_raw = ref2 - ref1 */
    vec_sub_backward(grad_u_raw, grad_ref2, grad_ref1);
}
