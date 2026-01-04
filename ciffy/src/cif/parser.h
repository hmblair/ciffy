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
#include "registry.h"  /* For FieldSkipMask */

/**
 * @brief Filter options for partial loading.
 *
 * Allows loading only specific chains, molecule types, or models.
 * All filter fields are optional - NULL/0 means no filtering.
 */
typedef struct LoadFilter {
    int *molecule_types;    /**< Array of molecule type values to include (-1 terminated), NULL = all */
    char **chain_names;     /**< NULL-terminated array of chain names to include, NULL = all */
    int mol_type_count;     /**< Number of molecule types in filter */
    int chain_count;        /**< Number of chain names in filter */
    int model;              /**< Model number to load (1-indexed), 0 = default (first) */
    bool connections;       /**< If true, parse _struct_conn for hydrogen bonds etc. */
    char alt_loc;           /**< Alt conformation to keep ('A', 'B', etc.), '\0' = keep all */
} LoadFilter;

/**
 * @brief Connection type enum (from _struct_conn.conn_type_id).
 */
typedef enum {
    CONN_TYPE_UNKNOWN = 0,
    CONN_TYPE_HYDROG  = 1,  /**< Hydrogen bond (base pairs, etc.) */
    CONN_TYPE_COVALE  = 2,  /**< Covalent bond (to/from modified residues) */
    CONN_TYPE_METALC  = 3,  /**< Metal coordination */
    CONN_TYPE_DISULF  = 4,  /**< Disulfide bridge */
} ConnType;

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
    float *bfactors;        /**< B-factors/temperature factors [atoms] */
    int   *types;           /**< Atom type indices [atoms] */
    int   *elements;        /**< Element type indices [atoms] */
    int   *is_nonpoly;      /**< Non-polymer mask [atoms], temp during parse */
    int   write_dest;       /**< Current write destination for batch callbacks */

    float resolution;       /**< Resolution in Angstroms (from _refine.ls_d_res_high), -1 if not available */
    char *deposit_date;     /**< Initial deposition date (YYYY-MM-DD), NULL if not available */

    int *sequence;          /**< Residue type indices [residues] */
    int *res_per_chain;     /**< Residues per chain [chains] */
    int *atoms_per_chain;   /**< Atoms per chain [chains] */
    int *atoms_per_res;     /**< Atoms per residue [residues] */
    int *molecule_types;    /**< Molecule type per chain [chains] (from _entity_poly.type) */

    /* Partial loading support */
    int *chain_mask;        /**< Per-chain inclusion mask [chains], 1=include, 0=exclude, NULL=all */
    int *is_excluded;       /**< Per-atom exclusion mask [atoms], 1=exclude, 0=include, NULL=all */
    int excluded_count;     /**< Count of excluded atoms */
    int original_chains;    /**< Original chain count before filtering */
    int original_atoms;     /**< Original atom count before filtering */
    int target_model;       /**< Target model to load (1-indexed), set from filter */

    /* Connection data from _struct_conn (optional, requires connections=true) */
    int *connections;       /**< Atom index pairs [n_connections * 2] as (atom1, atom2) */
    int *conn_types;        /**< Connection type per connection [n_connections] */
    int n_connections;      /**< Number of connections */

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
 * @param skip_mask Bitmask of fields to skip (use SKIP_NONE for all, SKIP_METADATA for metadata only)
 * @param filter Optional filter for partial loading (NULL = load all)
 * @param ctx Error context, populated on failure
 * @return CIF_OK on success, error code on failure
 */
CifError _fill_cif(mmCIF *cif, mmBlockList *blocks, FieldSkipMask skip_mask,
                   const LoadFilter *filter, CifErrorContext *ctx);

/**
 * @brief Build a chain inclusion mask based on filter criteria.
 *
 * Creates a boolean mask indicating which chains to include based on
 * molecule_types and/or chain_names filters. Returns count of included chains.
 *
 * @param cif Structure with chains and molecule_types already parsed
 * @param filter Filter criteria
 * @param ctx Error context, populated on failure
 * @return Number of included chains, or -1 on error
 */
int _build_chain_mask(mmCIF *cif, const LoadFilter *filter, CifErrorContext *ctx);

/**
 * @brief Free filter resources.
 *
 * @param filter Filter to free (fields are set to NULL)
 */
void _free_filter(LoadFilter *filter);

/**
 * @brief Create a LoadFilter with default values.
 *
 * Default values:
 *   - alt_loc = 'A' (keep first alternate conformation)
 *   - model = 1 (first model)
 *   - All other fields = 0/NULL (no filtering)
 *
 * @return LoadFilter with sensible defaults
 */
LoadFilter _default_filter(void);


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
