#ifndef _CIFFY_PARSER_H
#define _CIFFY_PARSER_H

/**
 * @file parser.h
 * @brief mmCIF-specific parsing structures and functions.
 *
 * Provides the main data structures for representing parsed mmCIF data
 * and functions for extracting molecular structure information.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#include "io.h"

/**
 * @brief Parsed mmCIF molecular structure data.
 *
 * Contains all extracted information from an mmCIF file including
 * coordinates, atom types, residue sequences, and chain organization.
 *
 * Note: Named struct for forward declaration compatibility with registry.h.
 */
typedef struct mmCIF {

    char *id;               /**< PDB identifier (e.g., "4V5D") */
    char **names;           /**< Chain names array */
    char **strands;         /**< Strand IDs array */
    char **descriptions;    /**< Chain descriptions (unused) */

    int models;             /**< Number of models in structure */
    int chains;             /**< Number of chains */
    int residues;           /**< Total number of residues */
    int atoms;              /**< Total number of atoms (per model) */

    int polymer;            /**< Count of polymeric atoms */
    int nonpoly;            /**< Count of non-polymeric atoms */

    float *coordinates;     /**< Atom coordinates [atoms * 3] as x,y,z triplets */
    int   *types;           /**< Atom type indices [atoms] */
    int   *elements;        /**< Element type indices [atoms] */
    int   *is_nonpoly;      /**< Non-polymer mask [atoms], temp during parse */
    int   write_dest;       /**< Current write destination for batch callbacks */

    int *sequence;          /**< Residue type indices [residues] */
    int *res_per_chain;     /**< Residues per chain [chains] */
    int *atoms_per_chain;   /**< Atoms per chain [chains] */
    int *atoms_per_res;     /**< Atoms per residue [residues] */
    int *molecule_types;    /**< Molecule type per chain [chains] (from _entity_poly.type) */

} mmCIF;

/**
 * @brief Collection of parsed mmCIF blocks.
 *
 * Blocks are stored in an array indexed by BlockId (defined in registry.h).
 * Access blocks using: blocks->b[BLOCK_ATOM], blocks->b[BLOCK_POLY], etc.
 *
 * Note: Named struct for forward declaration compatibility with registry.h.
 */
typedef struct mmBlockList {
    mmBlock b[BLOCK_COUNT];  /**< Array of blocks indexed by BlockId */
} mmBlockList;

/**
 * @brief Extract the PDB identifier from the file header.
 *
 * Parses the "data_XXXX" line at the start of an mmCIF file.
 *
 * @param cursor Parse cursor (position advanced past header)
 * @param ctx Error context, populated on failure
 * @return Allocated PDB ID string, or NULL on error
 */
char *_get_id(ParseCursor *cursor, CifErrorContext *ctx);

/**
 * @brief Populate an mmCIF structure from parsed blocks.
 *
 * Extracts all molecular data from the parsed blocks including
 * coordinates, sequences, atom types, and chain organization.
 *
 * @param cif Output structure to populate
 * @param blocks Parsed block collection
 * @param ctx Error context, populated on failure
 * @return CIF_OK on success, error code on failure
 */
CifError _fill_cif(mmCIF *cif, mmBlockList *blocks, bool metadata_only, CifErrorContext *ctx);


/* ─────────────────────────────────────────────────────────────────────────────
 * Block Parsing API
 * Functions for reading and managing mmCIF blocks.
 * ───────────────────────────────────────────────────────────────────────────── */

/**
 * @brief Parse a single mmCIF block.
 *
 * Reads block header, counts attributes, and for multi-entry blocks,
 * calculates line width and entry count.
 *
 * @param cursor Parse cursor (position advanced past block)
 * @param ctx Error context for allocation failures
 * @return Parsed block structure (check category for NULL on error)
 */
mmBlock _read_block(ParseCursor *cursor, CifErrorContext *ctx);

/**
 * @brief Free resources associated with a block.
 *
 * @param block Block to free (fields are set to NULL)
 */
void _free_block(mmBlock *block);

/**
 * @brief Skip past a multi-line attribute value.
 *
 * Multi-line values start and end with ';' on their own line.
 *
 * @param cursor Parse cursor (position advanced past value)
 * @return true on success, false if unterminated (exceeded max lines)
 */
bool _skip_multiline_attr(ParseCursor *cursor);

/**
 * @brief Advance to the next block (skip to section end marker).
 *
 * Skips until finding a line starting with '#' or reaching end of buffer.
 *
 * @param cursor Parse cursor (position advanced to next block)
 */
void _next_block(ParseCursor *cursor);

/**
 * @brief Store a block if it's needed, otherwise free it.
 *
 * Routes blocks to appropriate slots in the block list based on category.
 *
 * @param block Block to store or free
 * @param blocks Block list to store in
 */
void _store_or_free_block(mmBlock *block, mmBlockList *blocks);

/**
 * @brief Free all blocks in a block list.
 *
 * @param blocks Block list to free
 */
void _free_block_list(mmBlockList *blocks);

#endif /* _CIFFY_PARSER_H */
