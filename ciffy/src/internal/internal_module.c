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
 * Build Z-matrix from CSR graph for a single chain.
 *
 * Python signature:
 *   _build_zmatrix_from_csr(offsets, neighbors, n_atoms, chain_start, chain_size, root) -> zmatrix
 *
 * Args:
 *   offsets: (n_atoms+1,) int64 array of CSR offsets.
 *   neighbors: (E,) int64 array of neighbor indices.
 *   n_atoms: Total number of atoms (int).
 *   chain_start: First atom index for this chain (int).
 *   chain_size: Number of atoms in this chain (int).
 *   root: Root atom index for BFS (int).
 *
 * Returns:
 *   zmatrix: (M, 4) int64 array [atom_idx, dist_ref, ang_ref, dih_ref].
 */
PyObject *py_build_zmatrix_from_csr(PyObject *self, PyObject *args) {
    (void)self;

    PyObject *py_offsets, *py_neighbors;
    int n_atoms, chain_start, chain_size, root;

    if (!PyArg_ParseTuple(args, "OOiiii",
                          &py_offsets, &py_neighbors,
                          &n_atoms, &chain_start, &chain_size, &root)) {
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

    /* Validate parameters */
    if (chain_size < 0 || chain_start < 0 || root < 0 || n_atoms <= 0) {
        Py_DECREF(offsets_arr);
        Py_DECREF(neighbors_arr);
        PyErr_SetString(PyExc_ValueError, "Invalid chain parameters");
        return NULL;
    }

    /* Allocate output array */
    npy_intp dims[2] = {chain_size, 4};
    PyObject *py_zmatrix = PyArray_SimpleNew(2, dims, NPY_INT64);
    if (py_zmatrix == NULL) {
        Py_DECREF(offsets_arr);
        Py_DECREF(neighbors_arr);
        return PyErr_NoMemory();
    }

    int64_t *zmatrix = (int64_t *)PyArray_DATA((PyArrayObject *)py_zmatrix);

    /* Build Z-matrix */
    int64_t n_entries = build_zmatrix_from_csr(
        offsets, neighbors, n_atoms,
        chain_start, chain_size, root,
        zmatrix
    );

    Py_DECREF(offsets_arr);
    Py_DECREF(neighbors_arr);

    if (n_entries < 0) {
        Py_DECREF(py_zmatrix);
        return PyErr_NoMemory();
    }

    /* Resize output if needed (some atoms may be unreachable) */
    if (n_entries < chain_size) {
        npy_intp new_dims[2] = {n_entries, 4};
        PyArray_Dims new_shape = {new_dims, 2};
        PyObject *resized = PyArray_Resize((PyArrayObject *)py_zmatrix, &new_shape, 0, NPY_CORDER);
        if (resized == NULL) {
            /* Resize failed, but original array is still valid */
            PyErr_Clear();
        }
    }

    return py_zmatrix;
}


/**
 * Build Z-matrix for all chains in parallel using OpenMP.
 *
 * Python signature:
 *   _build_zmatrix_parallel(offsets, neighbors, n_atoms, chain_starts, chain_sizes, roots) -> (zmatrix, counts)
 *
 * Args:
 *   offsets: (n_atoms+1,) int64 array of CSR offsets.
 *   neighbors: (E,) int64 array of neighbor indices.
 *   n_atoms: Total number of atoms (int).
 *   chain_starts: (n_chains,) int64 array of first atom index per chain.
 *   chain_sizes: (n_chains,) int64 array of atoms per chain.
 *   roots: (n_chains,) int64 array of root atom index per chain.
 *
 * Returns:
 *   Tuple of (zmatrix, counts):
 *     zmatrix: (total_atoms, 4) int64 Z-matrix entries.
 *     counts: (n_chains,) int64 entries written per chain.
 */
PyObject *py_build_zmatrix_parallel(PyObject *self, PyObject *args) {
    (void)self;

    PyObject *py_offsets, *py_neighbors, *py_chain_starts, *py_chain_sizes, *py_roots;
    int n_atoms;

    if (!PyArg_ParseTuple(args, "OOiOOO",
                          &py_offsets, &py_neighbors, &n_atoms,
                          &py_chain_starts, &py_chain_sizes, &py_roots)) {
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

    PyArrayObject *chain_starts_arr = require_array_1d(py_chain_starts, NPY_INT64, "chain_starts");
    if (chain_starts_arr == NULL) {
        Py_DECREF(offsets_arr);
        Py_DECREF(neighbors_arr);
        return NULL;
    }

    PyArrayObject *chain_sizes_arr = require_array_1d(py_chain_sizes, NPY_INT64, "chain_sizes");
    if (chain_sizes_arr == NULL) {
        Py_DECREF(offsets_arr);
        Py_DECREF(neighbors_arr);
        Py_DECREF(chain_starts_arr);
        return NULL;
    }

    PyArrayObject *roots_arr = require_array_1d(py_roots, NPY_INT64, "roots");
    if (roots_arr == NULL) {
        Py_DECREF(offsets_arr);
        Py_DECREF(neighbors_arr);
        Py_DECREF(chain_starts_arr);
        Py_DECREF(chain_sizes_arr);
        return NULL;
    }

    /* Get array sizes */
    npy_intp n_chains = PyArray_DIM(chain_starts_arr, 0);

    /* Verify all chain arrays have same length */
    if (PyArray_DIM(chain_sizes_arr, 0) != n_chains ||
        PyArray_DIM(roots_arr, 0) != n_chains) {
        PyErr_SetString(PyExc_ValueError,
            "chain_starts, chain_sizes, and roots must have same length");
        Py_DECREF(offsets_arr);
        Py_DECREF(neighbors_arr);
        Py_DECREF(chain_starts_arr);
        Py_DECREF(chain_sizes_arr);
        Py_DECREF(roots_arr);
        return NULL;
    }

    /* Get data pointers */
    const int64_t *offsets = (const int64_t *)PyArray_DATA(offsets_arr);
    const int64_t *neighbors = (const int64_t *)PyArray_DATA(neighbors_arr);
    const int64_t *chain_starts = (const int64_t *)PyArray_DATA(chain_starts_arr);
    const int64_t *chain_sizes = (const int64_t *)PyArray_DATA(chain_sizes_arr);
    const int64_t *roots = (const int64_t *)PyArray_DATA(roots_arr);

    /* Compute total output size */
    int64_t total_size = 0;
    for (npy_intp i = 0; i < n_chains; i++) {
        total_size += chain_sizes[i];
    }

    /* Allocate output arrays */
    npy_intp zmat_dims[2] = {total_size, 4};
    npy_intp counts_dims[1] = {n_chains};

    PyObject *py_zmatrix = PyArray_SimpleNew(2, zmat_dims, NPY_INT64);
    PyObject *py_counts = PyArray_SimpleNew(1, counts_dims, NPY_INT64);

    if (py_zmatrix == NULL || py_counts == NULL) {
        Py_XDECREF(py_zmatrix);
        Py_XDECREF(py_counts);
        Py_DECREF(offsets_arr);
        Py_DECREF(neighbors_arr);
        Py_DECREF(chain_starts_arr);
        Py_DECREF(chain_sizes_arr);
        Py_DECREF(roots_arr);
        return PyErr_NoMemory();
    }

    int64_t *zmatrix = (int64_t *)PyArray_DATA((PyArrayObject *)py_zmatrix);
    int64_t *counts = (int64_t *)PyArray_DATA((PyArrayObject *)py_counts);

    /* Build Z-matrices in parallel */
    int64_t result = build_zmatrix_parallel(
        offsets, neighbors, n_atoms,
        chain_starts, chain_sizes, roots,
        n_chains, zmatrix, counts
    );

    Py_DECREF(offsets_arr);
    Py_DECREF(neighbors_arr);
    Py_DECREF(chain_starts_arr);
    Py_DECREF(chain_sizes_arr);
    Py_DECREF(roots_arr);

    if (result < 0) {
        Py_DECREF(py_zmatrix);
        Py_DECREF(py_counts);
        return PyErr_NoMemory();
    }

    /* Build result tuple */
    PyObject *tuple = PyTuple_Pack(2, py_zmatrix, py_counts);
    Py_DECREF(py_zmatrix);
    Py_DECREF(py_counts);

    return tuple;
}


/**
 * Backward pass for cartesian_to_internal.
 *
 * Python signature:
 *   _cartesian_to_internal_backward(
 *       coords, indices, distances, angles,
 *       grad_distances, grad_angles, grad_dihedrals
 *   ) -> grad_coords
 */
PyObject *py_cartesian_to_internal_backward(PyObject *self, PyObject *args) {
    (void)self;

    PyObject *py_coords, *py_indices, *py_distances, *py_angles;
    PyObject *py_grad_distances, *py_grad_angles, *py_grad_dihedrals;

    if (!PyArg_ParseTuple(args, "OOOOOOO",
                          &py_coords, &py_indices, &py_distances, &py_angles,
                          &py_grad_distances, &py_grad_angles, &py_grad_dihedrals)) {
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

    PyArrayObject *distances_arr = require_array_1d(py_distances, NPY_FLOAT32, "distances");
    if (distances_arr == NULL) {
        decref_arrays(coords_arr, indices_arr, NULL, NULL);
        return NULL;
    }

    PyArrayObject *angles_arr = require_array_1d(py_angles, NPY_FLOAT32, "angles");
    if (angles_arr == NULL) {
        decref_arrays(coords_arr, indices_arr, distances_arr, NULL);
        return NULL;
    }

    PyArrayObject *grad_distances_arr = require_array_1d(py_grad_distances, NPY_FLOAT32, "grad_distances");
    if (grad_distances_arr == NULL) {
        decref_arrays(coords_arr, indices_arr, distances_arr, angles_arr);
        return NULL;
    }

    PyArrayObject *grad_angles_arr = require_array_1d(py_grad_angles, NPY_FLOAT32, "grad_angles");
    if (grad_angles_arr == NULL) {
        Py_DECREF(grad_distances_arr);
        decref_arrays(coords_arr, indices_arr, distances_arr, angles_arr);
        return NULL;
    }

    PyArrayObject *grad_dihedrals_arr = require_array_1d(py_grad_dihedrals, NPY_FLOAT32, "grad_dihedrals");
    if (grad_dihedrals_arr == NULL) {
        Py_DECREF(grad_distances_arr);
        Py_DECREF(grad_angles_arr);
        decref_arrays(coords_arr, indices_arr, distances_arr, angles_arr);
        return NULL;
    }

    npy_intp n_atoms = PyArray_DIM(coords_arr, 0);
    npy_intp n_entries = PyArray_DIM(indices_arr, 0);

    /* Allocate output gradient array (initialized to zero) */
    npy_intp dims[2] = {n_atoms, 3};
    PyObject *py_grad_coords = PyArray_ZEROS(2, dims, NPY_FLOAT32, 0);
    if (py_grad_coords == NULL) {
        Py_DECREF(grad_distances_arr);
        Py_DECREF(grad_angles_arr);
        Py_DECREF(grad_dihedrals_arr);
        decref_arrays(coords_arr, indices_arr, distances_arr, angles_arr);
        return PyErr_NoMemory();
    }

    /* Get data pointers */
    const float *coords = (const float *)PyArray_DATA(coords_arr);
    const int64_t *indices = (const int64_t *)PyArray_DATA(indices_arr);
    const float *distances = (const float *)PyArray_DATA(distances_arr);
    const float *angles = (const float *)PyArray_DATA(angles_arr);
    const float *grad_distances = (const float *)PyArray_DATA(grad_distances_arr);
    const float *grad_angles = (const float *)PyArray_DATA(grad_angles_arr);
    const float *grad_dihedrals = (const float *)PyArray_DATA(grad_dihedrals_arr);
    float *grad_coords = (float *)PyArray_DATA((PyArrayObject *)py_grad_coords);

    /* Call batch backward function */
    batch_cartesian_to_internal_backward(
        coords, (size_t)n_atoms,
        indices, (size_t)n_entries,
        distances, angles,
        grad_distances, grad_angles, grad_dihedrals,
        grad_coords
    );

    /* Clean up input arrays */
    Py_DECREF(grad_distances_arr);
    Py_DECREF(grad_angles_arr);
    Py_DECREF(grad_dihedrals_arr);
    decref_arrays(coords_arr, indices_arr, distances_arr, angles_arr);

    return py_grad_coords;
}


/**
 * Backward pass for nerf_reconstruct.
 *
 * Python signature:
 *   _nerf_reconstruct_backward(
 *       coords, indices, distances, angles, dihedrals, grad_coords
 *   ) -> (grad_distances, grad_angles, grad_dihedrals)
 */
PyObject *py_nerf_reconstruct_backward(PyObject *self, PyObject *args) {
    (void)self;

    PyObject *py_coords, *py_indices, *py_distances, *py_angles, *py_dihedrals, *py_grad_coords;

    if (!PyArg_ParseTuple(args, "OOOOOO",
                          &py_coords, &py_indices, &py_distances, &py_angles,
                          &py_dihedrals, &py_grad_coords)) {
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

    PyArrayObject *distances_arr = require_array_1d(py_distances, NPY_FLOAT32, "distances");
    if (distances_arr == NULL) {
        decref_arrays(coords_arr, indices_arr, NULL, NULL);
        return NULL;
    }

    PyArrayObject *angles_arr = require_array_1d(py_angles, NPY_FLOAT32, "angles");
    if (angles_arr == NULL) {
        decref_arrays(coords_arr, indices_arr, distances_arr, NULL);
        return NULL;
    }

    PyArrayObject *dihedrals_arr = require_array_1d(py_dihedrals, NPY_FLOAT32, "dihedrals");
    if (dihedrals_arr == NULL) {
        decref_arrays(coords_arr, indices_arr, distances_arr, angles_arr);
        return NULL;
    }

    /* grad_coords needs to be writable - make a copy */
    PyArrayObject *grad_coords_arr = (PyArrayObject *)PyArray_FROM_OTF(
        py_grad_coords, NPY_FLOAT32, NPY_ARRAY_INOUT_ARRAY2
    );
    if (grad_coords_arr == NULL) {
        Py_DECREF(dihedrals_arr);
        decref_arrays(coords_arr, indices_arr, distances_arr, angles_arr);
        return NULL;
    }

    npy_intp n_atoms = PyArray_DIM(coords_arr, 0);
    npy_intp n_entries = PyArray_DIM(indices_arr, 0);

    /* Allocate output gradient arrays */
    npy_intp dims[1] = {n_entries};
    PyObject *py_grad_distances = PyArray_SimpleNew(1, dims, NPY_FLOAT32);
    PyObject *py_grad_angles = PyArray_SimpleNew(1, dims, NPY_FLOAT32);
    PyObject *py_grad_dihedrals = PyArray_SimpleNew(1, dims, NPY_FLOAT32);

    if (py_grad_distances == NULL || py_grad_angles == NULL || py_grad_dihedrals == NULL) {
        Py_XDECREF(py_grad_distances);
        Py_XDECREF(py_grad_angles);
        Py_XDECREF(py_grad_dihedrals);
        Py_DECREF(grad_coords_arr);
        Py_DECREF(dihedrals_arr);
        decref_arrays(coords_arr, indices_arr, distances_arr, angles_arr);
        return PyErr_NoMemory();
    }

    /* Get data pointers */
    const float *coords = (const float *)PyArray_DATA(coords_arr);
    const int64_t *indices = (const int64_t *)PyArray_DATA(indices_arr);
    const float *distances = (const float *)PyArray_DATA(distances_arr);
    const float *angles = (const float *)PyArray_DATA(angles_arr);
    const float *dihedrals = (const float *)PyArray_DATA(dihedrals_arr);
    float *grad_coords = (float *)PyArray_DATA(grad_coords_arr);
    float *grad_distances = (float *)PyArray_DATA((PyArrayObject *)py_grad_distances);
    float *grad_angles = (float *)PyArray_DATA((PyArrayObject *)py_grad_angles);
    float *grad_dihedrals_out = (float *)PyArray_DATA((PyArrayObject *)py_grad_dihedrals);

    /* Call batch backward function */
    batch_nerf_reconstruct_backward(
        coords, (size_t)n_atoms,
        indices, (size_t)n_entries,
        distances, angles, dihedrals,
        grad_coords,
        grad_distances, grad_angles, grad_dihedrals_out
    );

    /* Clean up input arrays */
    PyArray_ResolveWritebackIfCopy(grad_coords_arr);
    Py_DECREF(grad_coords_arr);
    Py_DECREF(dihedrals_arr);
    decref_arrays(coords_arr, indices_arr, distances_arr, angles_arr);

    /* Build result tuple */
    PyObject *result = PyTuple_Pack(3, py_grad_distances, py_grad_angles, py_grad_dihedrals);
    Py_DECREF(py_grad_distances);
    Py_DECREF(py_grad_angles);
    Py_DECREF(py_grad_dihedrals);

    return result;
}


/**
 * Find connected components in CSR graph.
 *
 * Python signature:
 *   _find_connected_components(offsets, neighbors, n_atoms) -> (roots, sizes, n_components)
 *
 * Args:
 *   offsets: (n_atoms+1,) int64 array of CSR offsets.
 *   neighbors: (E,) int64 array of neighbor indices.
 *   n_atoms: Total number of atoms (int).
 *
 * Returns:
 *   Tuple of (roots, sizes, n_components):
 *     roots: (n_components,) int64 array of root atom per component.
 *     sizes: (n_components,) int64 array of component sizes.
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

    /* Allocate output arrays (worst case: n_atoms components) */
    npy_intp dims[1] = {n_atoms};
    PyObject *py_roots = PyArray_SimpleNew(1, dims, NPY_INT64);
    PyObject *py_sizes = PyArray_SimpleNew(1, dims, NPY_INT64);

    if (py_roots == NULL || py_sizes == NULL) {
        Py_XDECREF(py_roots);
        Py_XDECREF(py_sizes);
        Py_DECREF(offsets_arr);
        Py_DECREF(neighbors_arr);
        return PyErr_NoMemory();
    }

    int64_t *roots = (int64_t *)PyArray_DATA((PyArrayObject *)py_roots);
    int64_t *sizes = (int64_t *)PyArray_DATA((PyArrayObject *)py_sizes);

    /* Find connected components */
    int64_t n_components = find_connected_components_c(
        offsets, neighbors, n_atoms, roots, sizes
    );

    Py_DECREF(offsets_arr);
    Py_DECREF(neighbors_arr);

    if (n_components < 0) {
        Py_DECREF(py_roots);
        Py_DECREF(py_sizes);
        return PyErr_NoMemory();
    }

    /* Resize arrays to actual size */
    if (n_components < n_atoms) {
        npy_intp new_dims[1] = {n_components};
        PyArray_Dims new_shape = {new_dims, 1};

        PyObject *resized_roots = PyArray_Resize((PyArrayObject *)py_roots, &new_shape, 0, NPY_CORDER);
        if (resized_roots == NULL) PyErr_Clear();

        PyObject *resized_sizes = PyArray_Resize((PyArrayObject *)py_sizes, &new_shape, 0, NPY_CORDER);
        if (resized_sizes == NULL) PyErr_Clear();
    }

    /* Build result tuple */
    PyObject *py_n_components = PyLong_FromLongLong(n_components);
    PyObject *tuple = PyTuple_Pack(3, py_roots, py_sizes, py_n_components);
    Py_DECREF(py_roots);
    Py_DECREF(py_sizes);
    Py_DECREF(py_n_components);

    return tuple;
}
