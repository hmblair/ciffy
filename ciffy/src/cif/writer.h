#ifndef _CIFFY_WRITER_H
#define _CIFFY_WRITER_H

/**
 * @file writer.h
 * @brief CIF file writing declarations.
 *
 * Provides functions for serializing mmCIF structures back to CIF format.
 * The writer produces output compatible with the reader, enabling round-trip
 * operations (load -> modify -> save).
 */

#include <stdio.h>
#include "parser.h"
#include "../error.h"

/**
 * @brief Write mmCIF structure to a file.
 *
 * Serializes the mmCIF structure to standard mmCIF format, including:
 *   - data_XXXX header
 *   - _struct_asym block (chain definitions)
 *   - _pdbx_poly_seq_scheme block (polymer sequence)
 *   - _atom_site block (coordinates and atom types)
 *
 * @param cif Structure to write
 * @param filename Output file path
 * @param ctx Error context for failure reporting
 * @return CIF_OK on success, error code on failure
 */
CifError _write_cif(const mmCIF *cif, const char *filename, CifErrorContext *ctx);

/**
 * @brief Write mmCIF structure to an open file handle.
 *
 * Lower-level function that writes to an already-open file.
 * The file is not closed by this function.
 *
 * @param cif Structure to write
 * @param file Open file handle (must be writable)
 * @param ctx Error context for failure reporting
 * @return CIF_OK on success, error code on failure
 */
CifError _write_cif_file(const mmCIF *cif, FILE *file, CifErrorContext *ctx);

#endif /* _CIFFY_WRITER_H */
