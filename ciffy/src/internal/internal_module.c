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
 * Build Z-matrix for all chains in parallel using OpenMP.
 *
 * Python signature:
 *   _build_zmatrix_parallel(offsets, neighbors, n_atoms, chain_starts, chain_sizes, roots,
 *                           atoms=None, sequence=None, res_sizes=None) -> (zmatrix, dihedral_types, levels, counts)
 *
 * Args:
 *   offsets: (n_atoms+1,) int64 array of CSR offsets.
 *   neighbors: (E,) int64 array of neighbor indices.
 *   n_atoms: Total number of atoms (int).
 *   chain_starts: (n_chains,) int64 array of first atom index per chain.
 *   chain_sizes: (n_chains,) int64 array of atoms per chain.
 *   roots: (n_chains,) int64 array of root atom index per chain.
 *   atoms: (n_atoms,) int32 array of atom types (optional, for dihedral-aware mode).
 *   sequence: (n_residues,) int32 array of residue types (optional).
 *   res_sizes: (n_residues,) int32 array of atoms per residue (optional).
 *
 * Returns:
 *   Tuple of (zmatrix, dihedral_types, levels, counts):
 *     zmatrix: (total_atoms, 4) int64 Z-matrix entries.
 *     dihedral_types: (total_atoms,) int8 dihedral type per entry (-1 if not named dihedral).
 *     levels: (total_atoms,) int32 BFS level per entry.
 *     counts: (n_chains,) int64 entries written per chain.
 */
PyObject *py_build_zmatrix_parallel(PyObject *self, PyObject *args) {
    (void)self;

    PyObject *py_offsets, *py_neighbors, *py_chain_starts, *py_chain_sizes, *py_roots;
    PyObject *py_atoms = Py_None, *py_sequence = Py_None, *py_res_sizes = Py_None;
    int n_atoms;

    if (!PyArg_ParseTuple(args, "OOiOOO|OOO",
                          &py_offsets, &py_neighbors, &n_atoms,
                          &py_chain_starts, &py_chain_sizes, &py_roots,
                          &py_atoms, &py_sequence, &py_res_sizes)) {
        return NULL;
    }

    /* Validate required arrays */
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

    /* Handle optional dihedral-aware parameters */
    PyArrayObject *atoms_arr = NULL;
    PyArrayObject *sequence_arr = NULL;
    PyArrayObject *res_sizes_arr = NULL;
    int dihedral_aware = 0;

    if (py_atoms != Py_None && py_sequence != Py_None && py_res_sizes != Py_None) {
        atoms_arr = require_array_1d(py_atoms, NPY_INT32, "atoms");
        if (atoms_arr == NULL) {
            Py_DECREF(offsets_arr);
            Py_DECREF(neighbors_arr);
            Py_DECREF(chain_starts_arr);
            Py_DECREF(chain_sizes_arr);
            Py_DECREF(roots_arr);
            return NULL;
        }

        sequence_arr = require_array_1d(py_sequence, NPY_INT32, "sequence");
        if (sequence_arr == NULL) {
            Py_DECREF(offsets_arr);
            Py_DECREF(neighbors_arr);
            Py_DECREF(chain_starts_arr);
            Py_DECREF(chain_sizes_arr);
            Py_DECREF(roots_arr);
            Py_DECREF(atoms_arr);
            return NULL;
        }

        res_sizes_arr = require_array_1d(py_res_sizes, NPY_INT32, "res_sizes");
        if (res_sizes_arr == NULL) {
            Py_DECREF(offsets_arr);
            Py_DECREF(neighbors_arr);
            Py_DECREF(chain_starts_arr);
            Py_DECREF(chain_sizes_arr);
            Py_DECREF(roots_arr);
            Py_DECREF(atoms_arr);
            Py_DECREF(sequence_arr);
            return NULL;
        }

        dihedral_aware = 1;
    }

    /* Get data pointers */
    const int64_t *offsets = (const int64_t *)PyArray_DATA(offsets_arr);
    const int64_t *neighbors = (const int64_t *)PyArray_DATA(neighbors_arr);
    const int64_t *chain_starts = (const int64_t *)PyArray_DATA(chain_starts_arr);
    const int64_t *chain_sizes = (const int64_t *)PyArray_DATA(chain_sizes_arr);
    const int64_t *roots = (const int64_t *)PyArray_DATA(roots_arr);

    const int32_t *atoms = dihedral_aware ? (const int32_t *)PyArray_DATA(atoms_arr) : NULL;
    const int32_t *sequence = dihedral_aware ? (const int32_t *)PyArray_DATA(sequence_arr) : NULL;
    const int32_t *res_sizes = dihedral_aware ? (const int32_t *)PyArray_DATA(res_sizes_arr) : NULL;
    npy_intp n_residues = dihedral_aware ? PyArray_DIM(sequence_arr, 0) : 0;

    /* Compute total output size */
    int64_t total_size = 0;
    for (npy_intp i = 0; i < n_chains; i++) {
        total_size += chain_sizes[i];
    }

    /* Allocate working arrays for dihedral-aware mode */
    int64_t *residue_starts = NULL;
    int64_t *chain_res_starts = NULL;

    if (dihedral_aware) {
        /* Compute residue_starts (cumulative sum of res_sizes) */
        residue_starts = (int64_t *)malloc((size_t)(n_residues + 1) * sizeof(int64_t));
        chain_res_starts = (int64_t *)malloc((size_t)(n_chains + 1) * sizeof(int64_t));

        if (!residue_starts || !chain_res_starts) {
            free(residue_starts);
            free(chain_res_starts);
            Py_DECREF(offsets_arr);
            Py_DECREF(neighbors_arr);
            Py_DECREF(chain_starts_arr);
            Py_DECREF(chain_sizes_arr);
            Py_DECREF(roots_arr);
            Py_XDECREF(atoms_arr);
            Py_XDECREF(sequence_arr);
            Py_XDECREF(res_sizes_arr);
            return PyErr_NoMemory();
        }

        residue_starts[0] = 0;
        for (npy_intp r = 0; r < n_residues; r++) {
            residue_starts[r + 1] = residue_starts[r] + res_sizes[r];
        }

        /* Compute chain_res_starts from chain_sizes and residue_starts */
        /* Each chain's residue count is derived from (chain_atom_end - chain_atom_start) */
        /* For now, use residue indices by counting residues in each chain */
        chain_res_starts[0] = 0;
        npy_intp res_idx = 0;
        for (npy_intp c = 0; c < n_chains; c++) {
            int64_t chain_end = chain_starts[c] + chain_sizes[c];
            while (res_idx < n_residues && residue_starts[res_idx + 1] <= chain_end) {
                res_idx++;
            }
            chain_res_starts[c + 1] = res_idx;
        }
    }

    /* Allocate output arrays */
    npy_intp zmat_dims[2] = {total_size, 4};
    npy_intp dih_dims[1] = {total_size};
    npy_intp level_dims[1] = {total_size};
    npy_intp counts_dims[1] = {n_chains};

    PyObject *py_zmatrix = PyArray_SimpleNew(2, zmat_dims, NPY_INT64);
    PyObject *py_dihedral_types = PyArray_SimpleNew(1, dih_dims, NPY_INT8);
    PyObject *py_levels = PyArray_SimpleNew(1, level_dims, NPY_INT32);
    PyObject *py_counts = PyArray_SimpleNew(1, counts_dims, NPY_INT64);

    if (py_zmatrix == NULL || py_dihedral_types == NULL || py_levels == NULL || py_counts == NULL) {
        Py_XDECREF(py_zmatrix);
        Py_XDECREF(py_dihedral_types);
        Py_XDECREF(py_levels);
        Py_XDECREF(py_counts);
        free(residue_starts);
        free(chain_res_starts);
        Py_DECREF(offsets_arr);
        Py_DECREF(neighbors_arr);
        Py_DECREF(chain_starts_arr);
        Py_DECREF(chain_sizes_arr);
        Py_DECREF(roots_arr);
        Py_XDECREF(atoms_arr);
        Py_XDECREF(sequence_arr);
        Py_XDECREF(res_sizes_arr);
        return PyErr_NoMemory();
    }

    int64_t *zmatrix = (int64_t *)PyArray_DATA((PyArrayObject *)py_zmatrix);
    int8_t *dihedral_types = (int8_t *)PyArray_DATA((PyArrayObject *)py_dihedral_types);
    int32_t *levels = (int32_t *)PyArray_DATA((PyArrayObject *)py_levels);
    int64_t *counts = (int64_t *)PyArray_DATA((PyArrayObject *)py_counts);

    /* Build Z-matrices in parallel */
    int64_t result = build_zmatrix_parallel(
        offsets, neighbors, n_atoms,
        chain_starts, chain_sizes, roots,
        n_chains,
        atoms, sequence, residue_starts, n_residues, chain_res_starts,
        zmatrix, dihedral_types, levels, counts
    );

    /* Cleanup */
    free(residue_starts);
    free(chain_res_starts);
    Py_DECREF(offsets_arr);
    Py_DECREF(neighbors_arr);
    Py_DECREF(chain_starts_arr);
    Py_DECREF(chain_sizes_arr);
    Py_DECREF(roots_arr);
    Py_XDECREF(atoms_arr);
    Py_XDECREF(sequence_arr);
    Py_XDECREF(res_sizes_arr);

    if (result < 0) {
        Py_DECREF(py_zmatrix);
        Py_DECREF(py_dihedral_types);
        Py_DECREF(py_levels);
        Py_DECREF(py_counts);
        return PyErr_NoMemory();
    }

    /* Build result tuple */
    PyObject *tuple = PyTuple_Pack(4, py_zmatrix, py_dihedral_types, py_levels, py_counts);
    Py_DECREF(py_zmatrix);
    Py_DECREF(py_dihedral_types);
    Py_DECREF(py_levels);
    Py_DECREF(py_counts);

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
