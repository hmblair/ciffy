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
#include "geometry.h"

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
 *   _cartesian_to_internal(coords, indices) -> internal
 *
 * Args:
 *   coords: (N, 3) float32 array of Cartesian coordinates.
 *   indices: (M, 4) int64 array of Z-matrix indices.
 *
 * Returns:
 *   internal: (M, 3) float32 array where each row is [distance, angle, dihedral].
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

    /* Allocate output array: (n_entries, 3) for [distance, angle, dihedral] */
    npy_intp dims[2] = {n_entries, 3};
    PyObject *py_internal = PyArray_SimpleNew(2, dims, NPY_FLOAT32);

    if (py_internal == NULL) {
        Py_DECREF(coords_arr);
        Py_DECREF(indices_arr);
        return PyErr_NoMemory();
    }

    float *internal = (float *)PyArray_DATA((PyArrayObject *)py_internal);

    /* Call batch function */
    batch_cartesian_to_internal(
        coords, (size_t)n_atoms,
        indices, (size_t)n_entries,
        internal
    );

    Py_DECREF(coords_arr);
    Py_DECREF(indices_arr);

    return py_internal;
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
 * Backward pass for cartesian_to_internal.
 *
 * Python signature:
 *   _cartesian_to_internal_backward(coords, indices, internal, grad_internal) -> grad_coords
 *
 * Args:
 *   coords: (N, 3) float32 array of Cartesian coordinates.
 *   indices: (M, 4) int64 array of Z-matrix indices.
 *   internal: (M, 3) float32 array of internal coordinates from forward pass.
 *   grad_internal: (M, 3) float32 array of upstream gradients.
 *
 * Returns:
 *   grad_coords: (N, 3) float32 array of gradients w.r.t. coords.
 */
PyObject *py_cartesian_to_internal_backward(PyObject *self, PyObject *args) {
    (void)self;

    PyObject *py_coords, *py_indices, *py_internal, *py_grad_internal;

    if (!PyArg_ParseTuple(args, "OOOO",
                          &py_coords, &py_indices, &py_internal, &py_grad_internal)) {
        return NULL;
    }

    /* Validate input arrays */
    PyArrayObject *coords_arr = require_array_2d(py_coords, NPY_FLOAT32, 3, "coords");
    if (coords_arr == NULL) return NULL;

    PyArrayObject *indices_arr = require_array_2d(py_indices, NPY_INT64, 4, "indices");
    if (indices_arr == NULL) {
        Py_DECREF(coords_arr);
        return NULL;
    }

    PyArrayObject *internal_arr = require_array_2d(py_internal, NPY_FLOAT32, 3, "internal");
    if (internal_arr == NULL) {
        Py_DECREF(coords_arr);
        Py_DECREF(indices_arr);
        return NULL;
    }

    PyArrayObject *grad_internal_arr = require_array_2d(py_grad_internal, NPY_FLOAT32, 3, "grad_internal");
    if (grad_internal_arr == NULL) {
        Py_DECREF(coords_arr);
        Py_DECREF(indices_arr);
        Py_DECREF(internal_arr);
        return NULL;
    }

    npy_intp n_atoms = PyArray_DIM(coords_arr, 0);
    npy_intp n_entries = PyArray_DIM(indices_arr, 0);

    /* Verify array length consistency */
    if (PyArray_DIM(internal_arr, 0) != n_entries ||
        PyArray_DIM(grad_internal_arr, 0) != n_entries) {
        PyErr_SetString(PyExc_ValueError,
            "internal and grad_internal must have same number of rows as indices");
        Py_DECREF(coords_arr);
        Py_DECREF(indices_arr);
        Py_DECREF(internal_arr);
        Py_DECREF(grad_internal_arr);
        return NULL;
    }

    /* Allocate output gradient array (initialized to zero) */
    npy_intp dims[2] = {n_atoms, 3};
    PyObject *py_grad_coords = PyArray_ZEROS(2, dims, NPY_FLOAT32, 0);
    if (py_grad_coords == NULL) {
        Py_DECREF(coords_arr);
        Py_DECREF(indices_arr);
        Py_DECREF(internal_arr);
        Py_DECREF(grad_internal_arr);
        return PyErr_NoMemory();
    }

    /* Get data pointers */
    const float *coords = (const float *)PyArray_DATA(coords_arr);
    const int64_t *indices = (const int64_t *)PyArray_DATA(indices_arr);
    const float *internal = (const float *)PyArray_DATA(internal_arr);
    const float *grad_internal = (const float *)PyArray_DATA(grad_internal_arr);
    float *grad_coords = (float *)PyArray_DATA((PyArrayObject *)py_grad_coords);

    /* Call batch backward function */
    batch_cartesian_to_internal_backward(
        coords, (size_t)n_atoms,
        indices, (size_t)n_entries,
        internal,
        grad_internal,
        grad_coords
    );

    /* Clean up input arrays */
    Py_DECREF(coords_arr);
    Py_DECREF(indices_arr);
    Py_DECREF(internal_arr);
    Py_DECREF(grad_internal_arr);

    return py_grad_coords;
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


/**
 * Level-parallel NERF reconstruction with anchor coordinates.
 *
 * Python signature:
 *   _nerf_reconstruct_leveled_anchored(indices, internal, n_atoms, level_offsets,
 *       anchor_coords, component_ids) -> coords
 *
 * Args:
 *   indices: (M, 4) int64 array of Z-matrix indices (sorted by level).
 *   internal: (M, 3) float32 array of internal coordinates.
 *             Each row: [distance, angle, dihedral].
 *   n_atoms: Total number of atoms (int).
 *   level_offsets: (n_levels+1,) int32 array of CSR-style offsets.
 *   anchor_coords: (n_components, 3, 3) float32 array of anchor positions.
 *   component_ids: (M,) int32 array of component IDs per entry.
 *
 * Returns:
 *   coords: (N, 3) float32 array of Cartesian coordinates.
 */
PyObject *py_nerf_reconstruct_leveled_anchored(PyObject *self, PyObject *args) {
    (void)self;

    PyObject *py_indices, *py_internal;
    PyObject *py_level_offsets, *py_anchor_coords, *py_component_ids;
    int n_atoms;

    if (!PyArg_ParseTuple(args, "OOiOOO",
                          &py_indices, &py_internal, &n_atoms, &py_level_offsets,
                          &py_anchor_coords, &py_component_ids)) {
        return NULL;
    }

    /* Validate input arrays */
    PyArrayObject *indices_arr = require_array_2d(py_indices, NPY_INT64, 4, "indices");
    if (indices_arr == NULL) return NULL;

    PyArrayObject *internal_arr = require_array_2d(py_internal, NPY_FLOAT32, 3, "internal");
    if (internal_arr == NULL) {
        Py_DECREF(indices_arr);
        return NULL;
    }

    PyArrayObject *level_offsets_arr = require_array_1d(py_level_offsets, NPY_INT32, "level_offsets");
    if (level_offsets_arr == NULL) {
        Py_DECREF(indices_arr);
        Py_DECREF(internal_arr);
        return NULL;
    }

    /* anchor_coords: (n_components, 3, 3) -> (n_components, 9) flattened */
    PyArrayObject *anchor_coords_arr = (PyArrayObject *)PyArray_FROM_OTF(
        py_anchor_coords, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY
    );
    if (anchor_coords_arr == NULL) {
        Py_DECREF(level_offsets_arr);
        Py_DECREF(indices_arr);
        Py_DECREF(internal_arr);
        return NULL;
    }

    PyArrayObject *component_ids_arr = require_array_1d(py_component_ids, NPY_INT32, "component_ids");
    if (component_ids_arr == NULL) {
        Py_DECREF(anchor_coords_arr);
        Py_DECREF(level_offsets_arr);
        Py_DECREF(indices_arr);
        Py_DECREF(internal_arr);
        return NULL;
    }

    /* Verify array length consistency */
    npy_intp n_entries = PyArray_DIM(indices_arr, 0);
    if (PyArray_DIM(internal_arr, 0) != n_entries ||
        PyArray_DIM(component_ids_arr, 0) != n_entries) {
        PyErr_SetString(PyExc_ValueError,
            "internal and component_ids must have same number of rows as indices");
        Py_DECREF(component_ids_arr);
        Py_DECREF(anchor_coords_arr);
        Py_DECREF(level_offsets_arr);
        Py_DECREF(indices_arr);
        Py_DECREF(internal_arr);
        return NULL;
    }

    /* n_components comes from anchor_coords shape for bounds checking */
    npy_intp n_components = PyArray_DIM(anchor_coords_arr, 0);

    /* Get data pointers */
    const int64_t *indices = (const int64_t *)PyArray_DATA(indices_arr);
    const float *internal = (const float *)PyArray_DATA(internal_arr);
    const int32_t *level_offsets = (const int32_t *)PyArray_DATA(level_offsets_arr);
    const float *anchor_coords = (const float *)PyArray_DATA(anchor_coords_arr);
    const int32_t *component_ids = (const int32_t *)PyArray_DATA(component_ids_arr);

    /* Allocate output array (initialized to zero) */
    npy_intp dims[2] = {n_atoms, 3};
    PyObject *py_coords = PyArray_ZEROS(2, dims, NPY_FLOAT32, 0);
    if (py_coords == NULL) {
        Py_DECREF(component_ids_arr);
        Py_DECREF(anchor_coords_arr);
        Py_DECREF(level_offsets_arr);
        Py_DECREF(indices_arr);
        Py_DECREF(internal_arr);
        return PyErr_NoMemory();
    }

    float *coords = (float *)PyArray_DATA((PyArrayObject *)py_coords);

    /* Call batch function */
    batch_nerf_reconstruct_leveled_anchored(
        coords, (size_t)n_atoms,
        indices, (size_t)n_entries,
        internal,
        level_offsets, (int)n_components,
        anchor_coords, component_ids
    );

    Py_DECREF(component_ids_arr);
    Py_DECREF(anchor_coords_arr);
    Py_DECREF(level_offsets_arr);
    Py_DECREF(indices_arr);
    Py_DECREF(internal_arr);

    return py_coords;
}


/**
 * Level-parallel backward pass for anchored NERF reconstruction.
 *
 * Python signature:
 *   _nerf_reconstruct_backward_leveled_anchored(
 *       coords, indices, internal, grad_coords,
 *       level_offsets, anchor_coords, component_ids
 *   ) -> grad_internal
 *
 * Args:
 *   coords: (N, 3) float32 array of reconstructed Cartesian coordinates.
 *   indices: (M, 4) int64 array of Z-matrix indices.
 *   internal: (M, 3) float32 array of internal coordinates from forward pass.
 *   grad_coords: (N, 3) float32 array of upstream gradients (modified in place).
 *   level_offsets: (n_levels+1,) int32 array of CSR-style offsets.
 *   anchor_coords: (n_components, 3, 3) float32 array of anchor positions.
 *   component_ids: (M,) int32 array of component IDs per entry.
 *
 * Returns:
 *   grad_internal: (M, 3) float32 array of gradients w.r.t. internal coordinates.
 */
PyObject *py_nerf_reconstruct_backward_leveled_anchored(PyObject *self, PyObject *args) {
    (void)self;

    PyObject *py_coords, *py_indices, *py_internal;
    PyObject *py_grad_coords, *py_level_offsets, *py_anchor_coords, *py_component_ids;

    if (!PyArg_ParseTuple(args, "OOOOOOO",
                          &py_coords, &py_indices, &py_internal,
                          &py_grad_coords, &py_level_offsets,
                          &py_anchor_coords, &py_component_ids)) {
        return NULL;
    }

    /* Validate input arrays */
    PyArrayObject *coords_arr = require_array_2d(py_coords, NPY_FLOAT32, 3, "coords");
    if (coords_arr == NULL) return NULL;

    PyArrayObject *indices_arr = require_array_2d(py_indices, NPY_INT64, 4, "indices");
    if (indices_arr == NULL) {
        Py_DECREF(coords_arr);
        return NULL;
    }

    PyArrayObject *internal_arr = require_array_2d(py_internal, NPY_FLOAT32, 3, "internal");
    if (internal_arr == NULL) {
        Py_DECREF(coords_arr);
        Py_DECREF(indices_arr);
        return NULL;
    }

    /* grad_coords needs to be writable */
    PyArrayObject *grad_coords_arr = (PyArrayObject *)PyArray_FROM_OTF(
        py_grad_coords, NPY_FLOAT32, NPY_ARRAY_INOUT_ARRAY2
    );
    if (grad_coords_arr == NULL) {
        Py_DECREF(coords_arr);
        Py_DECREF(indices_arr);
        Py_DECREF(internal_arr);
        return NULL;
    }

    PyArrayObject *level_offsets_arr = require_array_1d(py_level_offsets, NPY_INT32, "level_offsets");
    if (level_offsets_arr == NULL) {
        PyArray_ResolveWritebackIfCopy(grad_coords_arr);
        Py_DECREF(grad_coords_arr);
        Py_DECREF(coords_arr);
        Py_DECREF(indices_arr);
        Py_DECREF(internal_arr);
        return NULL;
    }

    PyArrayObject *anchor_coords_arr = (PyArrayObject *)PyArray_FROM_OTF(
        py_anchor_coords, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY
    );
    if (anchor_coords_arr == NULL) {
        Py_DECREF(level_offsets_arr);
        PyArray_ResolveWritebackIfCopy(grad_coords_arr);
        Py_DECREF(grad_coords_arr);
        Py_DECREF(coords_arr);
        Py_DECREF(indices_arr);
        Py_DECREF(internal_arr);
        return NULL;
    }

    PyArrayObject *component_ids_arr = require_array_1d(py_component_ids, NPY_INT32, "component_ids");
    if (component_ids_arr == NULL) {
        Py_DECREF(anchor_coords_arr);
        Py_DECREF(level_offsets_arr);
        PyArray_ResolveWritebackIfCopy(grad_coords_arr);
        Py_DECREF(grad_coords_arr);
        Py_DECREF(coords_arr);
        Py_DECREF(indices_arr);
        Py_DECREF(internal_arr);
        return NULL;
    }

    npy_intp n_atoms = PyArray_DIM(coords_arr, 0);
    npy_intp n_entries = PyArray_DIM(indices_arr, 0);
    /* n_components comes from anchor_coords shape for bounds checking */
    npy_intp n_components = PyArray_DIM(anchor_coords_arr, 0);

    /* Verify array length consistency */
    if (PyArray_DIM(internal_arr, 0) != n_entries ||
        PyArray_DIM(component_ids_arr, 0) != n_entries) {
        PyErr_SetString(PyExc_ValueError,
            "internal and component_ids must have same number of rows as indices");
        Py_DECREF(component_ids_arr);
        Py_DECREF(anchor_coords_arr);
        Py_DECREF(level_offsets_arr);
        PyArray_ResolveWritebackIfCopy(grad_coords_arr);
        Py_DECREF(grad_coords_arr);
        Py_DECREF(coords_arr);
        Py_DECREF(indices_arr);
        Py_DECREF(internal_arr);
        return NULL;
    }

    /* Allocate output gradient array: (n_entries, 3) */
    npy_intp dims[2] = {n_entries, 3};
    PyObject *py_grad_internal = PyArray_SimpleNew(2, dims, NPY_FLOAT32);
    if (py_grad_internal == NULL) {
        Py_DECREF(component_ids_arr);
        Py_DECREF(anchor_coords_arr);
        Py_DECREF(level_offsets_arr);
        PyArray_ResolveWritebackIfCopy(grad_coords_arr);
        Py_DECREF(grad_coords_arr);
        Py_DECREF(coords_arr);
        Py_DECREF(indices_arr);
        Py_DECREF(internal_arr);
        return PyErr_NoMemory();
    }

    /* Get data pointers */
    const float *coords = (const float *)PyArray_DATA(coords_arr);
    const int64_t *indices = (const int64_t *)PyArray_DATA(indices_arr);
    const float *internal = (const float *)PyArray_DATA(internal_arr);
    float *grad_coords = (float *)PyArray_DATA(grad_coords_arr);
    const int32_t *level_offsets = (const int32_t *)PyArray_DATA(level_offsets_arr);
    const float *anchor_coords = (const float *)PyArray_DATA(anchor_coords_arr);
    const int32_t *component_ids = (const int32_t *)PyArray_DATA(component_ids_arr);
    float *grad_internal = (float *)PyArray_DATA((PyArrayObject *)py_grad_internal);

    /* Call batch backward function */
    batch_nerf_reconstruct_backward_leveled_anchored(
        coords, (size_t)n_atoms,
        indices, (size_t)n_entries,
        internal,
        grad_coords,
        grad_internal,
        level_offsets, (int)n_components,
        anchor_coords, component_ids
    );

    /* Clean up input arrays */
    Py_DECREF(component_ids_arr);
    Py_DECREF(anchor_coords_arr);
    Py_DECREF(level_offsets_arr);
    PyArray_ResolveWritebackIfCopy(grad_coords_arr);
    Py_DECREF(grad_coords_arr);
    Py_DECREF(coords_arr);
    Py_DECREF(indices_arr);
    Py_DECREF(internal_arr);

    return py_grad_internal;
}


/**
 * Place a single atom using NERF algorithm.
 *
 * Python signature:
 *   _nerf_place_atom(a, b, c, distance, angle, dihedral) -> result
 *
 * Args:
 *   a: (3,) float32 array - dihedral reference atom
 *   b: (3,) float32 array - angle reference atom
 *   c: (3,) float32 array - distance reference atom (bonded to new atom)
 *   distance: float - bond length to new atom
 *   angle: float - bond angle (radians)
 *   dihedral: float - dihedral angle (radians)
 *
 * Returns:
 *   result: (3,) float32 array - position of new atom
 */
PyObject *py_nerf_place_atom(PyObject *self, PyObject *args) {
    (void)self;

    PyObject *py_a, *py_b, *py_c;
    float distance, angle, dihedral;

    if (!PyArg_ParseTuple(args, "OOOfff", &py_a, &py_b, &py_c,
                          &distance, &angle, &dihedral)) {
        return NULL;
    }

    /* Convert inputs to contiguous float32 arrays */
    PyArrayObject *a_arr = (PyArrayObject *)PyArray_FROM_OTF(
        py_a, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *b_arr = (PyArrayObject *)PyArray_FROM_OTF(
        py_b, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *c_arr = (PyArrayObject *)PyArray_FROM_OTF(
        py_c, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);

    if (a_arr == NULL || b_arr == NULL || c_arr == NULL) {
        Py_XDECREF(a_arr);
        Py_XDECREF(b_arr);
        Py_XDECREF(c_arr);
        return NULL;
    }

    /* Verify shapes are (3,) */
    if (PyArray_NDIM(a_arr) != 1 || PyArray_DIM(a_arr, 0) != 3 ||
        PyArray_NDIM(b_arr) != 1 || PyArray_DIM(b_arr, 0) != 3 ||
        PyArray_NDIM(c_arr) != 1 || PyArray_DIM(c_arr, 0) != 3) {
        Py_DECREF(a_arr);
        Py_DECREF(b_arr);
        Py_DECREF(c_arr);
        PyErr_SetString(PyExc_ValueError, "a, b, c must each have shape (3,)");
        return NULL;
    }

    /* Get data pointers */
    const float *a = (const float *)PyArray_DATA(a_arr);
    const float *b = (const float *)PyArray_DATA(b_arr);
    const float *c = (const float *)PyArray_DATA(c_arr);

    /* Allocate output */
    npy_intp dims[1] = {3};
    PyObject *py_result = PyArray_SimpleNew(1, dims, NPY_FLOAT32);
    if (py_result == NULL) {
        Py_DECREF(a_arr);
        Py_DECREF(b_arr);
        Py_DECREF(c_arr);
        return PyErr_NoMemory();
    }

    float *result = (float *)PyArray_DATA((PyArrayObject *)py_result);

    /* Call NERF placement */
    nerf_place_atom(a, b, c, distance, angle, dihedral, result);

    /* Clean up */
    Py_DECREF(a_arr);
    Py_DECREF(b_arr);
    Py_DECREF(c_arr);

    return py_result;
}


/**
 * Convert Cartesian to internal coordinates using parent array.
 *
 * Python signature:
 *   _cartesian_to_internal_parent(coords, parent) -> internal
 *
 * Args:
 *   coords: (N, 3) float32 array of Cartesian coordinates.
 *   parent: (N,) int64 array where parent[k] is parent of atom k (-1 for roots).
 *
 * Returns:
 *   internal: (N, 3) float32 array where each row is [distance, angle, dihedral].
 */
PyObject *py_cartesian_to_internal_parent(PyObject *self, PyObject *args) {
    (void)self;

    PyObject *py_coords, *py_parent;
    if (!PyArg_ParseTuple(args, "OO", &py_coords, &py_parent)) {
        return NULL;
    }

    PyArrayObject *coords_arr = require_array_2d(py_coords, NPY_FLOAT32, 3, "coords");
    if (coords_arr == NULL) return NULL;

    PyArrayObject *parent_arr = require_array_1d(py_parent, NPY_INT64, "parent");
    if (parent_arr == NULL) {
        Py_DECREF(coords_arr);
        return NULL;
    }

    npy_intp n_atoms = PyArray_DIM(coords_arr, 0);

    /* Verify sizes match */
    if (PyArray_DIM(parent_arr, 0) != n_atoms) {
        PyErr_SetString(PyExc_ValueError, "coords and parent must have same length");
        Py_DECREF(coords_arr);
        Py_DECREF(parent_arr);
        return NULL;
    }

    /* Get data pointers */
    const float *coords = (const float *)PyArray_DATA(coords_arr);
    const int64_t *parent = (const int64_t *)PyArray_DATA(parent_arr);

    /* Allocate output */
    npy_intp dims[2] = {n_atoms, 3};
    PyObject *py_internal = PyArray_SimpleNew(2, dims, NPY_FLOAT32);
    if (py_internal == NULL) {
        Py_DECREF(coords_arr);
        Py_DECREF(parent_arr);
        return PyErr_NoMemory();
    }

    float *internal = (float *)PyArray_DATA((PyArrayObject *)py_internal);

    /* Call batch function */
    batch_cartesian_to_internal_parent(coords, (size_t)n_atoms, parent, internal);

    /* Clean up */
    Py_DECREF(coords_arr);
    Py_DECREF(parent_arr);

    return py_internal;
}


/**
 * NERF reconstruction using parent array from spanning tree.
 *
 * Python signature:
 *   _nerf_reconstruct_parent(parent, level, internal, level_offsets,
 *       level_atoms, n_levels, fixed_coords=None, component_id=None) -> coords
 *
 * Args:
 *   parent: (N,) int64 array where parent[k] is parent of atom k.
 *   level: (N,) int32 array of depth levels.
 *   internal: (N, 3) float32 array of [distance, angle, dihedral] per atom.
 *   level_offsets: (n_levels+1,) int32 CSR-style offsets for level groups.
 *   level_atoms: (N,) int64 atoms sorted by level.
 *   n_levels: int, number of levels.
 *   fixed_coords: Optional (N, 3) float32 original coordinates. Atoms at
 *                 levels 0-2 are copied from here (required for accuracy).
 *   component_id: Optional (N,) int32 component index for each atom. If
 *                 provided, enables component-parallel processing which is
 *                 faster when n_components << n_levels.
 *
 * Returns:
 *   coords: (N, 3) float32 array of reconstructed coordinates.
 */
PyObject *py_nerf_reconstruct_parent(PyObject *self, PyObject *args) {
    (void)self;

    PyObject *py_parent, *py_level, *py_internal;
    PyObject *py_level_offsets, *py_level_atoms;
    int n_levels;
    PyObject *py_fixed_coords = Py_None;
    PyObject *py_component_id = Py_None;

    if (!PyArg_ParseTuple(args, "OOOOOi|OO",
                          &py_parent, &py_level, &py_internal,
                          &py_level_offsets, &py_level_atoms, &n_levels,
                          &py_fixed_coords, &py_component_id)) {
        return NULL;
    }

    /* Parse required arrays */
    PyArrayObject *parent_arr = require_array_1d(py_parent, NPY_INT64, "parent");
    if (parent_arr == NULL) return NULL;

    PyArrayObject *level_arr = require_array_1d(py_level, NPY_INT32, "level");
    if (level_arr == NULL) {
        Py_DECREF(parent_arr);
        return NULL;
    }

    PyArrayObject *internal_arr = require_array_2d(py_internal, NPY_FLOAT32, 3, "internal");
    if (internal_arr == NULL) {
        Py_DECREF(parent_arr);
        Py_DECREF(level_arr);
        return NULL;
    }

    PyArrayObject *level_offsets_arr = require_array_1d(py_level_offsets, NPY_INT32, "level_offsets");
    if (level_offsets_arr == NULL) {
        Py_DECREF(parent_arr);
        Py_DECREF(level_arr);
        Py_DECREF(internal_arr);
        return NULL;
    }

    PyArrayObject *level_atoms_arr = require_array_1d(py_level_atoms, NPY_INT64, "level_atoms");
    if (level_atoms_arr == NULL) {
        Py_DECREF(parent_arr);
        Py_DECREF(level_arr);
        Py_DECREF(internal_arr);
        Py_DECREF(level_offsets_arr);
        return NULL;
    }

    npy_intp n_atoms = PyArray_DIM(parent_arr, 0);

    /* Verify sizes */
    if (PyArray_DIM(level_arr, 0) != n_atoms ||
        PyArray_DIM(internal_arr, 0) != n_atoms ||
        PyArray_DIM(level_atoms_arr, 0) != n_atoms) {
        PyErr_SetString(PyExc_ValueError,
            "parent, level, internal, level_atoms must have same length");
        Py_DECREF(parent_arr);
        Py_DECREF(level_arr);
        Py_DECREF(internal_arr);
        Py_DECREF(level_offsets_arr);
        Py_DECREF(level_atoms_arr);
        return NULL;
    }

    /* Parse optional fixed_coords */
    PyArrayObject *fixed_arr = NULL;
    if (py_fixed_coords != Py_None) {
        fixed_arr = require_array_2d(py_fixed_coords, NPY_FLOAT32, 3, "fixed_coords");
        if (fixed_arr == NULL) {
            Py_DECREF(parent_arr);
            Py_DECREF(level_arr);
            Py_DECREF(internal_arr);
            Py_DECREF(level_offsets_arr);
            Py_DECREF(level_atoms_arr);
            return NULL;
        }
        if (PyArray_DIM(fixed_arr, 0) != n_atoms) {
            PyErr_SetString(PyExc_ValueError, "fixed_coords must have same length as parent");
            Py_DECREF(fixed_arr);
            Py_DECREF(parent_arr);
            Py_DECREF(level_arr);
            Py_DECREF(internal_arr);
            Py_DECREF(level_offsets_arr);
            Py_DECREF(level_atoms_arr);
            return NULL;
        }
    }

    /* Parse optional component_id for component-parallel processing */
    PyArrayObject *component_id_arr = NULL;
    int n_components = 0;
    if (py_component_id != Py_None) {
        component_id_arr = require_array_1d(py_component_id, NPY_INT32, "component_id");
        if (component_id_arr == NULL) {
            Py_XDECREF(fixed_arr);
            Py_DECREF(parent_arr);
            Py_DECREF(level_arr);
            Py_DECREF(internal_arr);
            Py_DECREF(level_offsets_arr);
            Py_DECREF(level_atoms_arr);
            return NULL;
        }
        if (PyArray_DIM(component_id_arr, 0) != n_atoms) {
            PyErr_SetString(PyExc_ValueError, "component_id must have same length as parent");
            Py_DECREF(component_id_arr);
            Py_XDECREF(fixed_arr);
            Py_DECREF(parent_arr);
            Py_DECREF(level_arr);
            Py_DECREF(internal_arr);
            Py_DECREF(level_offsets_arr);
            Py_DECREF(level_atoms_arr);
            return NULL;
        }
        /* Count components (max component_id + 1) */
        const int32_t *comp_data = (const int32_t *)PyArray_DATA(component_id_arr);
        for (npy_intp i = 0; i < n_atoms; i++) {
            if (comp_data[i] >= n_components) {
                n_components = comp_data[i] + 1;
            }
        }
    }

    /* Get data pointers */
    const int64_t *parent = (const int64_t *)PyArray_DATA(parent_arr);
    const int32_t *level = (const int32_t *)PyArray_DATA(level_arr);
    const float *internal = (const float *)PyArray_DATA(internal_arr);
    const int32_t *level_offsets = (const int32_t *)PyArray_DATA(level_offsets_arr);
    const int64_t *level_atoms = (const int64_t *)PyArray_DATA(level_atoms_arr);
    const float *fixed_coords = fixed_arr ? (const float *)PyArray_DATA(fixed_arr) : NULL;
    const int32_t *component_id = component_id_arr ? (const int32_t *)PyArray_DATA(component_id_arr) : NULL;

    /* Allocate output */
    npy_intp dims[2] = {n_atoms, 3};
    PyObject *py_coords = PyArray_SimpleNew(2, dims, NPY_FLOAT32);
    if (py_coords == NULL) {
        Py_XDECREF(fixed_arr);
        Py_DECREF(parent_arr);
        Py_DECREF(level_arr);
        Py_DECREF(internal_arr);
        Py_DECREF(level_offsets_arr);
        Py_DECREF(level_atoms_arr);
        return PyErr_NoMemory();
    }

    float *coords = (float *)PyArray_DATA((PyArrayObject *)py_coords);

    /* Call batch function */
    batch_nerf_reconstruct_parent(
        coords, (size_t)n_atoms,
        parent, level, internal,
        level_offsets, level_atoms, n_levels,
        fixed_coords,
        NULL,  /* No anchor coords */
        component_id, n_components  /* Component-parallel if available */
    );

    /* Clean up */
    Py_XDECREF(component_id_arr);
    Py_XDECREF(fixed_arr);
    Py_DECREF(parent_arr);
    Py_DECREF(level_arr);
    Py_DECREF(internal_arr);
    Py_DECREF(level_offsets_arr);
    Py_DECREF(level_atoms_arr);

    return py_coords;
}
