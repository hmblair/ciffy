/**
 * @file internal_module.h
 * @brief Python bindings for internal coordinate functions.
 */

#ifndef CIFFY_INTERNAL_MODULE_H
#define CIFFY_INTERNAL_MODULE_H

#include "../pyutils.h"

/**
 * Convert Cartesian coordinates to internal coordinates.
 * Python: _cartesian_to_internal(coords, indices) -> (distances, angles, dihedrals)
 */
PyObject *py_cartesian_to_internal(PyObject *self, PyObject *args);

/**
 * Reconstruct Cartesian coordinates from internal coordinates using NERF.
 * Python: _nerf_reconstruct(indices, distances, angles, dihedrals, n_atoms) -> coords
 */
PyObject *py_nerf_reconstruct(PyObject *self, PyObject *args);

#endif /* CIFFY_INTERNAL_MODULE_H */
