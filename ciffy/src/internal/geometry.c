/**
 * @file geometry.c
 * @brief Core geometry functions for internal coordinates.
 */

#include "geometry.h"
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

    /* m1 = n1 x b2_unit */
    float m1x = n1y*b2uz - n1z*b2uy;
    float m1y = n1z*b2ux - n1x*b2uz;
    float m1z = n1x*b2uy - n1y*b2ux;

    /* atan2(y, x) where y = n2 . m1, x = n2 . n1 */
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
