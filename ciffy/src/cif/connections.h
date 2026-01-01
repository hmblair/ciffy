/**
 * @file connections.h
 * @brief Parse _struct_conn block for hydrogen bonds, covalent bonds, etc.
 *
 * This module provides optional parsing of molecular connections from
 * the _struct_conn mmCIF category. It builds a temporary hash table
 * during atom parsing to resolve (chain, seq, atom) references to
 * global atom indices.
 */

#ifndef _CIFFY_CONNECTIONS_H
#define _CIFFY_CONNECTIONS_H

#include "parser.h"
#include "atom_hash.h"

/**
 * @brief Build atom lookup hash from _atom_site block.
 *
 * Creates a hash table mapping (chain_id, seq_id, atom_id) -> atom_index.
 * Must be called before _parse_connections().
 *
 * @param block The _atom_site block (must have lines precomputed)
 * @param n_atoms Number of atoms
 * @param ctx Error context
 * @return Initialized hash table, or {NULL,0,0} on error
 */
AtomHash _build_atom_lookup(mmBlock *block, int n_atoms, CifErrorContext *ctx);

/**
 * @brief Parse _struct_conn block and resolve to atom indices.
 *
 * Parses connection records and uses the atom hash to convert
 * (chain, seq, atom) references to global atom indices.
 *
 * @param cif Output structure (connections, conn_types, n_connections populated)
 * @param conn_block The _struct_conn block
 * @param atom_hash Hash from _build_atom_lookup()
 * @param ctx Error context
 * @return CIF_OK on success, error code on failure
 */
CifError _parse_connections(mmCIF *cif, mmBlock *conn_block,
                            const AtomHash *atom_hash, CifErrorContext *ctx);

#endif /* _CIFFY_CONNECTIONS_H */
