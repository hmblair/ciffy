/**
 * @file internal_module.c
 * @brief Python bindings for internal coordinate functions.
 *
 * Provides Python-callable functions for converting between
 * Cartesian and internal coordinates using the NumPy C API.
 */

#include "../pyutils.h"
#include "batch.h"


/**
 * Convert Cartesian coordinates to internal coordinates.
 *
 * Python signature:
 *   _cartesian_to_internal(coords, indices) -> (distances, angles, dihedrals)
 *
 * Args:
 *   coords: (N, 3) float32 array of Cartesian coordinates.
 *   indices: (M, 4) int64 array of Z-matrix indices.
 *
 * Returns:
 *   Tuple of (distances, angles, dihedrals), each (M,) float32.
 */
PyObject *py_cartesian_to_internal(PyObject *self, PyObject *args) {
    (void)self;

    PyObject *py_coords, *py_indices;
    if (!PyArg_ParseTuple(args, "OO", &py_coords, &py_indices)) {
        return NULL;
    }

    /* Validate input arrays */
    PyArrayObject *coords_arr = validate_float32_2d(py_coords, 3, "coords");
    if (coords_arr == NULL) return NULL;

    PyArrayObject *indices_arr = validate_int64_2d(py_indices, 4, "indices");
    if (indices_arr == NULL) return NULL;

    /* Get sizes */
    npy_intp n_atoms = PyArray_DIM(coords_arr, 0);
    npy_intp n_entries = PyArray_DIM(indices_arr, 0);

    /* Get data pointers */
    const float *coords = (const float *)PyArray_DATA(coords_arr);
    const int64_t *indices = (const int64_t *)PyArray_DATA(indices_arr);

    /* Allocate output arrays */
    npy_intp dims[1] = {n_entries};
    PyObject *py_distances = PyArray_SimpleNew(1, dims, NPY_FLOAT32);
    PyObject *py_angles = PyArray_SimpleNew(1, dims, NPY_FLOAT32);
    PyObject *py_dihedrals = PyArray_SimpleNew(1, dims, NPY_FLOAT32);

    if (py_distances == NULL || py_angles == NULL || py_dihedrals == NULL) {
        Py_XDECREF(py_distances);
        Py_XDECREF(py_angles);
        Py_XDECREF(py_dihedrals);
        return PyErr_NoMemory();
    }

    float *distances = (float *)PyArray_DATA((PyArrayObject *)py_distances);
    float *angles = (float *)PyArray_DATA((PyArrayObject *)py_angles);
    float *dihedrals = (float *)PyArray_DATA((PyArrayObject *)py_dihedrals);

    /* Call batch function */
    batch_cartesian_to_internal(
        coords, (size_t)n_atoms,
        indices, (size_t)n_entries,
        distances, angles, dihedrals
    );

    /* Build result tuple */
    PyObject *result = PyTuple_Pack(3, py_distances, py_angles, py_dihedrals);
    Py_DECREF(py_distances);
    Py_DECREF(py_angles);
    Py_DECREF(py_dihedrals);

    return result;
}


/**
 * Reconstruct Cartesian coordinates from internal coordinates using NERF.
 *
 * Python signature:
 *   _nerf_reconstruct(indices, distances, angles, dihedrals, n_atoms) -> coords
 *
 * Args:
 *   indices: (M, 4) int64 array of Z-matrix indices.
 *   distances: (M,) float32 array of bond lengths.
 *   angles: (M,) float32 array of bond angles in radians.
 *   dihedrals: (M,) float32 array of dihedral angles in radians.
 *   n_atoms: Total number of atoms (int).
 *
 * Returns:
 *   coords: (N, 3) float32 array of Cartesian coordinates.
 */
PyObject *py_nerf_reconstruct(PyObject *self, PyObject *args) {
    (void)self;

    PyObject *py_indices, *py_distances, *py_angles, *py_dihedrals;
    int n_atoms;

    if (!PyArg_ParseTuple(args, "OOOOi",
                          &py_indices, &py_distances, &py_angles,
                          &py_dihedrals, &n_atoms)) {
        return NULL;
    }

    /* Validate input arrays */
    PyArrayObject *indices_arr = validate_int64_2d(py_indices, 4, "indices");
    if (indices_arr == NULL) return NULL;

    PyArrayObject *distances_arr = validate_float32_1d(py_distances, "distances");
    if (distances_arr == NULL) return NULL;

    PyArrayObject *angles_arr = validate_float32_1d(py_angles, "angles");
    if (angles_arr == NULL) return NULL;

    PyArrayObject *dihedrals_arr = validate_float32_1d(py_dihedrals, "dihedrals");
    if (dihedrals_arr == NULL) return NULL;

    /* Verify array length consistency */
    npy_intp n_entries = PyArray_DIM(indices_arr, 0);
    if (PyArray_DIM(distances_arr, 0) != n_entries ||
        PyArray_DIM(angles_arr, 0) != n_entries ||
        PyArray_DIM(dihedrals_arr, 0) != n_entries) {
        PyErr_SetString(PyExc_ValueError,
            "distances, angles, and dihedrals must have same length as indices");
        return NULL;
    }

    /* Get data pointers */
    const int64_t *indices = (const int64_t *)PyArray_DATA(indices_arr);
    const float *distances = (const float *)PyArray_DATA(distances_arr);
    const float *angles = (const float *)PyArray_DATA(angles_arr);
    const float *dihedrals = (const float *)PyArray_DATA(dihedrals_arr);

    /* Allocate output array (initialized to zero) */
    npy_intp dims[2] = {n_atoms, 3};
    PyObject *py_coords = PyArray_ZEROS(2, dims, NPY_FLOAT32, 0);
    if (py_coords == NULL) {
        return PyErr_NoMemory();
    }

    float *coords = (float *)PyArray_DATA((PyArrayObject *)py_coords);

    /* Call batch function */
    batch_nerf_reconstruct(
        coords, (size_t)n_atoms,
        indices, (size_t)n_entries,
        distances, angles, dihedrals
    );

    return py_coords;
}
