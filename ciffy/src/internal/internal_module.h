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

/**
 * Build bond graph edge list from polymer arrays.
 * Python: _build_bond_graph(atoms, sequence, res_sizes, chain_lengths) -> edges
 */
PyObject *py_build_bond_graph(PyObject *self, PyObject *args);

/**
 * Convert edge list to CSR format.
 * Python: _edges_to_csr(edges, n_atoms) -> (offsets, neighbors)
 */
PyObject *py_edges_to_csr(PyObject *self, PyObject *args);

/**
 * Build Z-matrix for all chains in parallel.
 * Python: _build_zmatrix_parallel(offsets, neighbors, n_atoms, chain_starts, chain_sizes, roots) -> (zmatrix, counts)
 */
PyObject *py_build_zmatrix_parallel(PyObject *self, PyObject *args);

/**
 * Backward pass for cartesian_to_internal.
 * Python: _cartesian_to_internal_backward(coords, indices, distances, angles,
 *             grad_distances, grad_angles, grad_dihedrals) -> grad_coords
 */
PyObject *py_cartesian_to_internal_backward(PyObject *self, PyObject *args);

/**
 * Backward pass for nerf_reconstruct.
 * Python: _nerf_reconstruct_backward(coords, indices, distances, angles, dihedrals,
 *             grad_coords) -> (grad_distances, grad_angles, grad_dihedrals)
 */
PyObject *py_nerf_reconstruct_backward(PyObject *self, PyObject *args);

/**
 * Find connected components in CSR graph.
 * Python: _find_connected_components(offsets, neighbors, n_atoms) -> (atoms, component_offsets, n_components)
 */
PyObject *py_find_connected_components(PyObject *self, PyObject *args);

/**
 * Level-parallel NERF reconstruction.
 * Python: _nerf_reconstruct_leveled(indices, distances, angles, dihedrals,
 *             n_atoms, level_offsets) -> coords
 */
PyObject *py_nerf_reconstruct_leveled(PyObject *self, PyObject *args);

/**
 * Level-parallel backward pass for NERF reconstruction.
 * Python: _nerf_reconstruct_backward_leveled(coords, indices, distances,
 *             angles, dihedrals, grad_coords, level_offsets)
 *             -> (grad_distances, grad_angles, grad_dihedrals)
 */
PyObject *py_nerf_reconstruct_backward_leveled(PyObject *self, PyObject *args);

/**
 * Level-parallel NERF reconstruction with anchor coordinates.
 * Python: _nerf_reconstruct_leveled_anchored(indices, distances, angles, dihedrals,
 *             n_atoms, level_offsets, anchor_coords, component_ids) -> coords
 */
PyObject *py_nerf_reconstruct_leveled_anchored(PyObject *self, PyObject *args);

/**
 * Level-parallel backward pass for anchored NERF reconstruction.
 * Python: _nerf_reconstruct_backward_leveled_anchored(coords, indices, distances,
 *             angles, dihedrals, grad_coords, level_offsets, anchor_coords, component_ids)
 *             -> (grad_distances, grad_angles, grad_dihedrals)
 */
PyObject *py_nerf_reconstruct_backward_leveled_anchored(PyObject *self, PyObject *args);

#endif /* CIFFY_INTERNAL_MODULE_H */
