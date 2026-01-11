/**
 * @file bindings.h
 * @brief Python bindings for graph functions.
 */

#ifndef CIFFY_GRAPH_BINDINGS_H
#define CIFFY_GRAPH_BINDINGS_H

#include "../pyutils.h"

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
 * Find connected components in CSR graph.
 * Python: _find_connected_components(offsets, neighbors, n_atoms) -> (atoms, component_offsets, n_components)
 */
PyObject *py_find_connected_components(PyObject *self, PyObject *args);

#endif /* CIFFY_GRAPH_BINDINGS_H */
