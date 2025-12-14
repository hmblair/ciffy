/**
 * @file internal_module.c
 * @brief Python bindings for internal coordinate functions.
 *
 * Provides Python-callable functions for converting between
 * Cartesian and internal coordinates using the NumPy C API.
 */

#include "../pyutils.h"
#include "batch.h"
#include "graph.h"

/* Helpers to normalize array-like inputs (NumPy, Torch tensor, etc.) to
 * contiguous NumPy arrays with shape checks.
 * These are local to this file to avoid spreading Python API surface.
 */
static PyArrayObject *require_array_2d(
    PyObject *obj, int typenum, npy_intp cols, const char *name
) {
    PyArrayObject *arr = (PyArrayObject *)PyArray_FROM_OTF(
        obj, typenum, NPY_ARRAY_IN_ARRAY
    );
    if (arr == NULL) {
        return NULL;
    }
    if (PyArray_NDIM(arr) != 2 || PyArray_DIM(arr, 1) != cols) {
        Py_DECREF(arr);
        PyErr_Format(PyExc_ValueError, "%s must have shape (N, %ld)", name, (long)cols);
        return NULL;
    }
    return arr;
}

static PyArrayObject *require_array_1d(
    PyObject *obj, int typenum, const char *name
) {
    PyArrayObject *arr = (PyArrayObject *)PyArray_FROM_OTF(
        obj, typenum, NPY_ARRAY_IN_ARRAY
    );
    if (arr == NULL) {
        return NULL;
    }
    if (PyArray_NDIM(arr) != 1) {
        Py_DECREF(arr);
        PyErr_Format(PyExc_ValueError, "%s must be 1D", name);
        return NULL;
    }
    return arr;
}
/* Helper to clean up up to four arrays */
static void decref_arrays(PyArrayObject *a, PyArrayObject *b,
                          PyArrayObject *c, PyArrayObject *d) {
    Py_XDECREF(a);
    Py_XDECREF(b);
    Py_XDECREF(c);
    Py_XDECREF(d);
}


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

    /* Accept any array-like input (NumPy array, Torch tensor, etc.) */
    PyArrayObject *coords_arr = require_array_2d(py_coords, NPY_FLOAT32, 3, "coords");
    if (coords_arr == NULL) {
        return NULL;
    }

    PyArrayObject *indices_arr = require_array_2d(py_indices, NPY_INT64, 4, "indices");
    if (indices_arr == NULL) {
        Py_DECREF(coords_arr);
        return NULL;
    }

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

    Py_DECREF(coords_arr);
    Py_DECREF(indices_arr);

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
    PyArrayObject *indices_arr = require_array_2d(py_indices, NPY_INT64, 4, "indices");
    if (indices_arr == NULL) return NULL;

    PyArrayObject *distances_arr = require_array_1d(py_distances, NPY_FLOAT32, "distances");
    if (distances_arr == NULL) {
        decref_arrays(indices_arr, NULL, NULL, NULL);
        return NULL;
    }

    PyArrayObject *angles_arr = require_array_1d(py_angles, NPY_FLOAT32, "angles");
    if (angles_arr == NULL) {
        decref_arrays(indices_arr, distances_arr, NULL, NULL);
        return NULL;
    }

    PyArrayObject *dihedrals_arr = require_array_1d(py_dihedrals, NPY_FLOAT32, "dihedrals");
    if (dihedrals_arr == NULL) {
        decref_arrays(indices_arr, distances_arr, angles_arr, NULL);
        return NULL;
    }

    /* Verify array length consistency */
    npy_intp n_entries = PyArray_DIM(indices_arr, 0);
    if (PyArray_DIM(distances_arr, 0) != n_entries ||
        PyArray_DIM(angles_arr, 0) != n_entries ||
        PyArray_DIM(dihedrals_arr, 0) != n_entries) {
        PyErr_SetString(PyExc_ValueError,
            "distances, angles, and dihedrals must have same length as indices");
        decref_arrays(indices_arr, distances_arr, angles_arr, dihedrals_arr);
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
        Py_DECREF(indices_arr);
        Py_DECREF(distances_arr);
        Py_DECREF(angles_arr);
        Py_DECREF(dihedrals_arr);
        return PyErr_NoMemory();
    }

    float *coords = (float *)PyArray_DATA((PyArrayObject *)py_coords);

    /* Call batch function */
    batch_nerf_reconstruct(
        coords, (size_t)n_atoms,
        indices, (size_t)n_entries,
        distances, angles, dihedrals
    );

    Py_DECREF(indices_arr);
    Py_DECREF(distances_arr);
    Py_DECREF(angles_arr);
    Py_DECREF(dihedrals_arr);

    return py_coords;
}


/**
 * Build bond graph edge list from polymer arrays.
 *
 * Python signature:
 *   _build_bond_graph(atoms, sequence, res_sizes, chain_lengths) -> edges
 *
 * Args:
 *   atoms: (N,) int32 array of atom values.
 *   sequence: (R,) int32 array of residue type indices.
 *   res_sizes: (R,) int32 array of atoms per residue.
 *   chain_lengths: (C,) int32 array of residues per chain.
 *
 * Returns:
 *   edges: (E, 2) int64 array of [atom_i, atom_j] pairs (symmetric).
 */
PyObject *py_build_bond_graph(PyObject *self, PyObject *args) {
    (void)self;

    PyObject *py_atoms, *py_sequence, *py_res_sizes, *py_chain_lengths;

    if (!PyArg_ParseTuple(args, "OOOO",
                          &py_atoms, &py_sequence,
                          &py_res_sizes, &py_chain_lengths)) {
        return NULL;
    }

    /* Validate input arrays */
    PyArrayObject *atoms_arr = require_array_1d(py_atoms, NPY_INT32, "atoms");
    if (atoms_arr == NULL) return NULL;

    PyArrayObject *sequence_arr = require_array_1d(py_sequence, NPY_INT32, "sequence");
    if (sequence_arr == NULL) {
        Py_DECREF(atoms_arr);
        return NULL;
    }

    PyArrayObject *res_sizes_arr = require_array_1d(py_res_sizes, NPY_INT32, "res_sizes");
    if (res_sizes_arr == NULL) {
        Py_DECREF(atoms_arr);
        Py_DECREF(sequence_arr);
        return NULL;
    }

    PyArrayObject *chain_lengths_arr = require_array_1d(py_chain_lengths, NPY_INT32, "chain_lengths");
    if (chain_lengths_arr == NULL) {
        Py_DECREF(atoms_arr);
        Py_DECREF(sequence_arr);
        Py_DECREF(res_sizes_arr);
        return NULL;
    }

    /* Get sizes */
    npy_intp n_atoms = PyArray_DIM(atoms_arr, 0);
    npy_intp n_residues = PyArray_DIM(sequence_arr, 0);
    npy_intp n_chains = PyArray_DIM(chain_lengths_arr, 0);

    /* Verify res_sizes length matches sequence */
    if (PyArray_DIM(res_sizes_arr, 0) != n_residues) {
        PyErr_SetString(PyExc_ValueError,
            "res_sizes must have same length as sequence");
        Py_DECREF(atoms_arr);
        Py_DECREF(sequence_arr);
        Py_DECREF(res_sizes_arr);
        Py_DECREF(chain_lengths_arr);
        return NULL;
    }

    /* Get data pointers */
    const int32_t *atoms = (const int32_t *)PyArray_DATA(atoms_arr);
    const int32_t *sequence = (const int32_t *)PyArray_DATA(sequence_arr);
    const int32_t *res_sizes = (const int32_t *)PyArray_DATA(res_sizes_arr);
    const int32_t *chain_lengths = (const int32_t *)PyArray_DATA(chain_lengths_arr);

    /* Estimate max edges for allocation */
    int64_t max_edges = estimate_max_edges(sequence, n_residues);

    /* Allocate output array */
    npy_intp dims[2] = {max_edges, 2};
    PyObject *py_edges = PyArray_SimpleNew(2, dims, NPY_INT64);
    if (py_edges == NULL) {
        Py_DECREF(atoms_arr);
        Py_DECREF(sequence_arr);
        Py_DECREF(res_sizes_arr);
        Py_DECREF(chain_lengths_arr);
        return PyErr_NoMemory();
    }

    int64_t *edges = (int64_t *)PyArray_DATA((PyArrayObject *)py_edges);

    /* Build bond graph */
    int64_t edge_count = build_bond_graph_c(
        atoms, sequence, res_sizes, chain_lengths,
        n_atoms, n_residues, n_chains,
        edges, max_edges
    );

    Py_DECREF(atoms_arr);
    Py_DECREF(sequence_arr);
    Py_DECREF(res_sizes_arr);
    Py_DECREF(chain_lengths_arr);

    if (edge_count < 0) {
        Py_DECREF(py_edges);
        return PyErr_NoMemory();
    }

    /* Resize output array to actual size */
    if (edge_count < max_edges) {
        npy_intp new_dims[2] = {edge_count, 2};
        PyArray_Dims new_shape = {new_dims, 2};
        PyObject *resized = PyArray_Resize((PyArrayObject *)py_edges, &new_shape, 0, NPY_CORDER);
        if (resized == NULL) {
            /* Resize failed, but original array is still valid */
            PyErr_Clear();
        }
    }

    return py_edges;
}
