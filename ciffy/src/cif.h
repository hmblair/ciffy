#ifndef _CIFFY_CIF_H
#define _CIFFY_CIF_H

/**
 * @file cif.h
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
 */
typedef struct {

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

    int *sequence;          /**< Residue type indices [residues] */
    int *res_per_chain;     /**< Residues per chain [chains] */
    int *atoms_per_chain;   /**< Atoms per chain [chains] */
    int *atoms_per_res;     /**< Atoms per residue [residues] */

} mmCIF;

/**
 * @brief Collection of parsed mmCIF blocks.
 *
 * Groups the relevant blocks extracted from an mmCIF file
 * for subsequent data extraction.
 */
typedef struct {

    mmBlock atom;       /**< _atom_site block (coordinates) */
    mmBlock poly;       /**< _pdbx_poly_seq_scheme block (sequence) */
    mmBlock nonpoly;    /**< _pdbx_nonpoly_scheme block */
    mmBlock conn;       /**< _struct_conn block (connectivity) */
    mmBlock chain;      /**< _struct_asym block (chain info) */
    mmBlock entity;     /**< Entity information block */

} mmBlockList;

/**
 * @brief Extract the PDB identifier from the file header.
 *
 * Parses the "data_XXXX" line at the start of an mmCIF file.
 *
 * @param buffer Start of the file buffer
 * @param ctx Error context, populated on failure
 * @return Allocated PDB ID string, or NULL on error
 */
char *_get_id(char *buffer, CifErrorContext *ctx);

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
CifError _fill_cif(mmCIF *cif, mmBlockList *blocks, CifErrorContext *ctx);

#endif /* _CIFFY_CIF_H */
