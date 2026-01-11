/**
 * @file bindings.c
 * @brief Python bindings for graph functions.
 *
 * Provides Python-callable functions for building and traversing
 * molecular bond graphs using the NumPy C API.
 */

#include "../pyutils.h"
#include "graph.h"

/* Helpers to normalize array-like inputs (NumPy, Torch tensor, etc.) to
 * contiguous NumPy arrays with shape checks.
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


/**
 * Convert edge list to CSR format.
 *
 * Python signature:
 *   _edges_to_csr(edges, n_atoms) -> (offsets, neighbors)
 *
 * Args:
 *   edges: (E, 2) int64 array of symmetric edges.
 *   n_atoms: Total number of atoms (int).
 *
 * Returns:
 *   Tuple of (offsets, neighbors):
 *     offsets: (n_atoms+1,) int64 array of CSR offsets.
 *     neighbors: (E,) int64 array of neighbor indices.
 */
PyObject *py_edges_to_csr(PyObject *self, PyObject *args) {
    (void)self;

    PyObject *py_edges;
    int n_atoms;

    if (!PyArg_ParseTuple(args, "Oi", &py_edges, &n_atoms)) {
        return NULL;
    }

    /* Validate edges array */
    PyArrayObject *edges_arr = require_array_2d(py_edges, NPY_INT64, 2, "edges");
    if (edges_arr == NULL) return NULL;

    npy_intp n_edges = PyArray_DIM(edges_arr, 0);
    const int64_t *edges = (const int64_t *)PyArray_DATA(edges_arr);

    /* Validate parameters */
    if (n_atoms <= 0) {
        Py_DECREF(edges_arr);
        PyErr_SetString(PyExc_ValueError, "n_atoms must be positive");
        return NULL;
    }

    /* Allocate output arrays */
    npy_intp offset_dims[1] = {n_atoms + 1};
    npy_intp neighbor_dims[1] = {n_edges};

    PyObject *py_offsets = PyArray_SimpleNew(1, offset_dims, NPY_INT64);
    PyObject *py_neighbors = PyArray_SimpleNew(1, neighbor_dims, NPY_INT64);

    if (py_offsets == NULL || py_neighbors == NULL) {
        Py_XDECREF(py_offsets);
        Py_XDECREF(py_neighbors);
        Py_DECREF(edges_arr);
        return PyErr_NoMemory();
    }

    int64_t *offsets = (int64_t *)PyArray_DATA((PyArrayObject *)py_offsets);
    int64_t *neighbors = (int64_t *)PyArray_DATA((PyArrayObject *)py_neighbors);

    /* Convert to CSR */
    int result = edges_to_csr(edges, n_edges, n_atoms, offsets, neighbors);

    Py_DECREF(edges_arr);

    if (result < 0) {
        Py_DECREF(py_offsets);
        Py_DECREF(py_neighbors);
        return PyErr_NoMemory();
    }

    /* Build result tuple */
    PyObject *tuple = PyTuple_Pack(2, py_offsets, py_neighbors);
    Py_DECREF(py_offsets);
    Py_DECREF(py_neighbors);

    return tuple;
}


/**
 * Find connected components in CSR graph.
 *
 * Python signature:
 *   _find_connected_components(offsets, neighbors, n_atoms) -> (atoms, component_offsets, n_components)
 *
 * Args:
 *   offsets: (n_atoms+1,) int64 array of CSR offsets.
 *   neighbors: (E,) int64 array of neighbor indices.
 *   n_atoms: Total number of atoms (int).
 *
 * Returns:
 *   Tuple of (atoms, component_offsets, n_components):
 *     atoms: (n_atoms,) int64 array of atom indices grouped by component.
 *     component_offsets: (n_components+1,) int64 offsets into atoms array.
 *     n_components: int number of components found.
 */
PyObject *py_find_connected_components(PyObject *self, PyObject *args) {
    (void)self;

    PyObject *py_offsets, *py_neighbors;
    int n_atoms;

    if (!PyArg_ParseTuple(args, "OOi", &py_offsets, &py_neighbors, &n_atoms)) {
        return NULL;
    }

    /* Validate arrays */
    PyArrayObject *offsets_arr = require_array_1d(py_offsets, NPY_INT64, "offsets");
    if (offsets_arr == NULL) return NULL;

    PyArrayObject *neighbors_arr = require_array_1d(py_neighbors, NPY_INT64, "neighbors");
    if (neighbors_arr == NULL) {
        Py_DECREF(offsets_arr);
        return NULL;
    }

    const int64_t *offsets = (const int64_t *)PyArray_DATA(offsets_arr);
    const int64_t *neighbors = (const int64_t *)PyArray_DATA(neighbors_arr);

    /* Allocate output arrays */
    npy_intp atoms_dims[1] = {n_atoms};
    npy_intp offsets_dims[1] = {n_atoms + 1};  /* Max n_atoms components + 1 */
    PyObject *py_out_atoms = PyArray_SimpleNew(1, atoms_dims, NPY_INT64);
    PyObject *py_component_offsets = PyArray_SimpleNew(1, offsets_dims, NPY_INT64);

    if (py_out_atoms == NULL || py_component_offsets == NULL) {
        Py_XDECREF(py_out_atoms);
        Py_XDECREF(py_component_offsets);
        Py_DECREF(offsets_arr);
        Py_DECREF(neighbors_arr);
        return PyErr_NoMemory();
    }

    int64_t *out_atoms = (int64_t *)PyArray_DATA((PyArrayObject *)py_out_atoms);
    int64_t *component_offsets = (int64_t *)PyArray_DATA((PyArrayObject *)py_component_offsets);

    /* Find connected components */
    int64_t n_components = find_connected_components_c(
        offsets, neighbors, n_atoms, out_atoms, component_offsets
    );

    Py_DECREF(offsets_arr);
    Py_DECREF(neighbors_arr);

    if (n_components < 0) {
        Py_DECREF(py_out_atoms);
        Py_DECREF(py_component_offsets);
        return PyErr_NoMemory();
    }

    /* Resize component_offsets to actual size (n_components + 1) */
    if (n_components + 1 < n_atoms + 1) {
        npy_intp new_dims[1] = {n_components + 1};
        PyArray_Dims new_shape = {new_dims, 1};

        PyObject *resized = PyArray_Resize((PyArrayObject *)py_component_offsets, &new_shape, 0, NPY_CORDER);
        if (resized == NULL) PyErr_Clear();
    }

    /* Build result tuple */
    PyObject *py_n_components = PyLong_FromLongLong(n_components);
    PyObject *tuple = PyTuple_Pack(3, py_out_atoms, py_component_offsets, py_n_components);
    Py_DECREF(py_out_atoms);
    Py_DECREF(py_component_offsets);
    Py_DECREF(py_n_components);

    return tuple;
}
