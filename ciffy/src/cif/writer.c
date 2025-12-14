/**
 * @file writer.c
 * @brief CIF file writing implementation.
 *
 * Serializes mmCIF structures to standard mmCIF format.
 * The output includes all data loaded by the reader:
 *   1. data_XXXX header
 *   2. _struct_asym block (chain definitions)
 *   3. _pdbx_poly_seq_scheme block (polymer sequence)
 *   4. _atom_site block (coordinates and atom types)
 */

#include "writer.h"
#include "../hash/reverse.h"
#include "../log.h"

#include <unistd.h>  /* for isatty */

/* Extern declaration for ion lookup (defined in parser.c via hash/ion.c) */
struct _LOOKUP;
extern struct _LOOKUP * _lookup_ion(const char *str, size_t len);


/* ============================================================================
 * HELPER MACROS
 * ============================================================================ */

/**
 * @brief fprintf with error checking.
 *
 * Returns CIF_ERR_IO if write fails.
 */
#define CIF_FPRINTF(file, ctx, ...) do { \
    if (fprintf(file, __VA_ARGS__) < 0) { \
        LOG_ERROR("Write failed"); \
        CIF_SET_ERROR(ctx, CIF_ERR_IO, "Write failed"); \
        return CIF_ERR_IO; \
    } \
} while(0)

/**
 * @brief Bounds check with error reporting.
 *
 * Returns CIF_ERR_BOUNDS if val >= max.
 */
#define CIF_CHECK_BOUNDS(val, max, name, ctx) do { \
    if ((val) >= (max)) { \
        LOG_ERROR("%s index %d exceeds count %d", (name), (val), (max)); \
        CIF_SET_ERROR((ctx), CIF_ERR_BOUNDS, "%s index %d exceeds count %d", (name), (val), (max)); \
        return CIF_ERR_BOUNDS; \
    } \
} while(0)

/**
 * @brief Chain name validation with error reporting.
 *
 * Returns CIF_ERR_PARSE if name is NULL.
 */
#define CIF_CHECK_CHAIN_NAME(name, idx, ctx) do { \
    if ((name) == NULL) { \
        LOG_ERROR("NULL chain name at index %d", (idx)); \
        CIF_SET_ERROR((ctx), CIF_ERR_PARSE, "NULL chain name at index %d", (idx)); \
        return CIF_ERR_PARSE; \
    } \
} while(0)

/**
 * @brief Get strand name, defaulting to "?" if NULL or empty.
 */
static inline const char *_safe_strand(const char *s) {
    return (s && s[0]) ? s : "?";
}

/** Maximum atom name length (e.g., "C2'" plus quotes and null) */
#define MAX_ATOM_NAME_BUF 32

/**
 * @brief Format atom name for CIF output, quoting if contains prime (').
 *
 * CIF requires strings containing special characters like ' to be quoted.
 * This function writes the formatted name to buffer, returning the buffer.
 *
 * @param name Atom name (e.g., "C2'" or "C2")
 * @param buffer Output buffer (must be at least MAX_ATOM_NAME_BUF bytes)
 * @return Pointer to buffer with formatted name
 */
static inline const char *_format_atom_name(const char *name, char *buffer) {
    /* Check if name contains a prime (') character */
    int needs_quote = 0;
    for (const char *p = name; *p; p++) {
        if (*p == '\'') {
            needs_quote = 1;
            break;
        }
    }

    if (needs_quote) {
        /* Quote the name: C2' becomes "C2'" */
        int written = snprintf(buffer, MAX_ATOM_NAME_BUF, "\"%s\"", name);
        if (written >= MAX_ATOM_NAME_BUF) {
            LOG_WARNING("Atom name truncated: %s", name);
        }
        return buffer;
    }

    return name;
}


/* ============================================================================
 * INTERNAL: Block Writers
 * Each function writes a specific mmCIF block to the file.
 * ============================================================================ */

/**
 * @brief Write the data_ header line.
 */
static CifError _write_header(FILE *file, const char *id, CifErrorContext *ctx) {
    CIF_FPRINTF(file, ctx, "data_%s\n", id);
    CIF_FPRINTF(file, ctx, "#\n");
    return CIF_OK;
}


/**
 * @brief Get the _entity.type string for a molecule type.
 *
 * Maps molecule type enum values to CIF entity type strings.
 */
static const char *_entity_type_string(int mol_type, int res_per_chain) {
    /* Molecule type enum values from types/molecule.py */
    enum { LIGAND = 8, ION = 9, WATER = 10 };

    if (mol_type == WATER) return "water";
    if (mol_type == LIGAND || mol_type == ION) return "non-polymer";
    if (res_per_chain > 0) return "polymer";
    return "non-polymer";
}

/**
 * @brief Write the _entity block (entity type definitions).
 *
 * Defines whether each entity is a polymer, non-polymer, or water.
 * Each chain is treated as its own entity with entity_id = chain_index + 1.
 */
static CifError _write_entity(FILE *file, const mmCIF *cif, CifErrorContext *ctx) {
    CIF_FPRINTF(file, ctx, "loop_\n");
    CIF_FPRINTF(file, ctx, "_entity.id\n");
    CIF_FPRINTF(file, ctx, "_entity.type\n");

    for (int i = 0; i < cif->chains; i++) {
        int mol_type = cif->molecule_types ? cif->molecule_types[i] : -1;
        const char *type = _entity_type_string(mol_type, cif->res_per_chain[i]);
        CIF_FPRINTF(file, ctx, "%d %s\n", i + 1, type);
    }

    CIF_FPRINTF(file, ctx, "#\n");
    return CIF_OK;
}


/**
 * @brief Write the _struct_asym block (chain definitions).
 *
 * Writes ALL chains (both polymer and non-polymer) with their entity mapping.
 * Each chain maps to its own entity with entity_id = chain_index + 1.
 */
static CifError _write_struct_asym(FILE *file, const mmCIF *cif, CifErrorContext *ctx) {
    CIF_FPRINTF(file, ctx, "loop_\n");
    CIF_FPRINTF(file, ctx, "_struct_asym.id\n");
    CIF_FPRINTF(file, ctx, "_struct_asym.pdbx_strand_id\n");
    CIF_FPRINTF(file, ctx, "_struct_asym.entity_id\n");

    for (int i = 0; i < cif->chains; i++) {
        const char *name = cif->names[i];
        CIF_CHECK_CHAIN_NAME(name, i, ctx);
        CIF_FPRINTF(file, ctx, "%-4.4s %-4.4s %d\n", name, _safe_strand(cif->strands[i]), i + 1);
    }

    CIF_FPRINTF(file, ctx, "#\n");
    return CIF_OK;
}


/**
 * @brief Write the _entity_poly block (polymer type per entity).
 *
 * Maps each polymer chain to its molecule type (RNA, protein, etc).
 * Each chain is treated as its own entity with entity_id = chain_index + 1.
 * Non-polymer chains (res_per_chain == 0) are skipped.
 */
static CifError _write_entity_poly(FILE *file, const mmCIF *cif, CifErrorContext *ctx) {
    /* Skip if no molecule_types array provided */
    if (cif->molecule_types == NULL) {
        LOG_DEBUG("No molecule_types array, skipping _entity_poly block");
        return CIF_OK;
    }

    /* Count polymer chains first */
    int poly_chains = 0;
    for (int i = 0; i < cif->chains; i++) {
        if (cif->res_per_chain[i] > 0) {
            poly_chains++;
        }
    }

    /* Skip block if no polymer chains */
    if (poly_chains == 0) {
        LOG_DEBUG("No polymer chains, skipping _entity_poly block");
        return CIF_OK;
    }

    CIF_FPRINTF(file, ctx, "loop_\n");
    CIF_FPRINTF(file, ctx, "_entity_poly.entity_id\n");
    CIF_FPRINTF(file, ctx, "_entity_poly.type\n");
    CIF_FPRINTF(file, ctx, "_entity_poly.pdbx_strand_id\n");

    for (int i = 0; i < cif->chains; i++) {
        /* Skip non-polymer chains (no residues) */
        if (cif->res_per_chain[i] == 0) {
            continue;
        }

        int mol_type = cif->molecule_types[i];
        const char *type_str = molecule_type_name(mol_type);
        const char *strand = _safe_strand(cif->strands[i]);

        /* Quote type strings that contain special characters */
        if (strchr(type_str, '(') || strchr(type_str, '/') || strchr(type_str, ' ')) {
            CIF_FPRINTF(file, ctx, "%d '%s' %s\n", i + 1, type_str, strand);
        } else {
            CIF_FPRINTF(file, ctx, "%d %s %s\n", i + 1, type_str, strand);
        }
    }

    CIF_FPRINTF(file, ctx, "#\n");
    return CIF_OK;
}


/**
 * @brief Write the _pdbx_entity_nonpoly block (non-polymer entity comp_ids).
 *
 * For ION entities, infers comp_id from the element symbol of the first atom.
 * This enables proper round-trip of ION molecule types.
 */
static CifError _write_entity_nonpoly(FILE *file, const mmCIF *cif, CifErrorContext *ctx) {
    /* Skip if no molecule_types array provided */
    if (cif->molecule_types == NULL) {
        LOG_DEBUG("No molecule_types array, skipping _pdbx_entity_nonpoly block");
        return CIF_OK;
    }

    enum { LIGAND = 8, ION = 9, WATER = 10 };

    /* Count non-polymer entities that need comp_id */
    int nonpoly_count = 0;
    for (int i = 0; i < cif->chains; i++) {
        int mol_type = cif->molecule_types[i];
        if (mol_type == LIGAND || mol_type == ION) {
            nonpoly_count++;
        }
    }

    /* Skip block if no non-polymer entities */
    if (nonpoly_count == 0) {
        LOG_DEBUG("No non-polymer entities, skipping _pdbx_entity_nonpoly block");
        return CIF_OK;
    }

    CIF_FPRINTF(file, ctx, "loop_\n");
    CIF_FPRINTF(file, ctx, "_pdbx_entity_nonpoly.entity_id\n");
    CIF_FPRINTF(file, ctx, "_pdbx_entity_nonpoly.comp_id\n");

    /* Use atoms_per_chain to compute first atom of each chain */
    int atom_idx = 0;

    for (int chain = 0; chain < cif->chains; chain++) {
        int chain_first_atom = atom_idx;
        int n_atoms = cif->atoms_per_chain[chain];
        atom_idx += n_atoms;

        int mol_type = cif->molecule_types[chain];
        if (mol_type != LIGAND && mol_type != ION) {
            continue;
        }

        /* Get element of first atom in chain */
        if (n_atoms == 0 || chain_first_atom >= cif->atoms) {
            LOG_WARNING("Chain %d has no atoms, cannot infer comp_id", chain);
            continue;
        }

        int elem_idx = cif->elements[chain_first_atom];
        const char *elem = element_name(elem_idx);

        /* For ION type, use element as comp_id (enables round-trip) */
        /* For LIGAND type, also write element (won't become ION on reload) */
        CIF_FPRINTF(file, ctx, "%d %s\n", chain + 1, elem);

        if (mol_type == ION) {
            size_t elem_len = strlen(elem);
            struct _LOOKUP *ion_lookup = _lookup_ion(elem, elem_len);
            if (ion_lookup == NULL) {
                LOG_DEBUG("Unknown ion element %s for chain %d, will become LIGAND on reload", elem, chain);
            }
        }
    }

    CIF_FPRINTF(file, ctx, "#\n");
    return CIF_OK;
}


/**
 * @brief Write the _pdbx_poly_seq_scheme block (polymer sequence).
 *
 * Only writes residues that have atoms (atoms_per_res > 0).
 * This ensures consistency with the atom_site block.
 */
static CifError _write_poly_seq(FILE *file, const mmCIF *cif, CifErrorContext *ctx) {
    CIF_FPRINTF(file, ctx, "loop_\n");
    CIF_FPRINTF(file, ctx, "_pdbx_poly_seq_scheme.asym_id\n");
    CIF_FPRINTF(file, ctx, "_pdbx_poly_seq_scheme.mon_id\n");
    CIF_FPRINTF(file, ctx, "_pdbx_poly_seq_scheme.pdb_strand_id\n");
    CIF_FPRINTF(file, ctx, "_pdbx_poly_seq_scheme.seq_id\n");

    int res_idx = 0;
    int skipped_count = 0;

    for (int chain = 0; chain < cif->chains; chain++) {
        const char *chain_name = cif->names[chain];
        const char *strand = _safe_strand(cif->strands[chain]);
        CIF_CHECK_CHAIN_NAME(chain_name, chain, ctx);

        int output_seq_id = 1;  /* Track output sequence number (restarts per chain) */

        for (int res = 0; res < cif->res_per_chain[chain]; res++) {
            CIF_CHECK_BOUNDS(res_idx, cif->residues, "Residue", ctx);

            /* Skip residues with no polymer atoms (e.g., HETATM-only residues) */
            if (cif->atoms_per_res[res_idx] == 0) {
                skipped_count++;
                res_idx++;
                continue;
            }

            /* residue_name() logs warning automatically for unknown indices */
            const char *res_name = residue_name(cif->sequence[res_idx]);

            CIF_FPRINTF(file, ctx, "%-4.4s %-4.4s %-4.4s %-6d\n",
                chain_name, res_name, strand, output_seq_id);
            output_seq_id++;
            res_idx++;
        }
    }

    if (skipped_count > 0) {
        LOG_DEBUG("Skipped %d residues with no polymer atoms", skipped_count);
    }

    CIF_FPRINTF(file, ctx, "#\n");
    return CIF_OK;
}


/**
 * @brief Write the _atom_site block (coordinates and atom types).
 */
static CifError _write_atom_site(FILE *file, const mmCIF *cif, CifErrorContext *ctx) {
    /* Write block header */
    CIF_FPRINTF(file, ctx, "loop_\n");
    CIF_FPRINTF(file, ctx, "_atom_site.group_PDB\n");
    CIF_FPRINTF(file, ctx, "_atom_site.id\n");
    CIF_FPRINTF(file, ctx, "_atom_site.type_symbol\n");
    CIF_FPRINTF(file, ctx, "_atom_site.label_atom_id\n");
    CIF_FPRINTF(file, ctx, "_atom_site.label_alt_id\n");
    CIF_FPRINTF(file, ctx, "_atom_site.label_comp_id\n");
    CIF_FPRINTF(file, ctx, "_atom_site.label_asym_id\n");
    CIF_FPRINTF(file, ctx, "_atom_site.label_seq_id\n");
    CIF_FPRINTF(file, ctx, "_atom_site.Cartn_x\n");
    CIF_FPRINTF(file, ctx, "_atom_site.Cartn_y\n");
    CIF_FPRINTF(file, ctx, "_atom_site.Cartn_z\n");
    CIF_FPRINTF(file, ctx, "_atom_site.pdbx_PDB_model_num\n");

    LOG_INFO("Writing %d atoms (%d polymer, %d non-polymer)",
             cif->atoms, cif->polymer, cif->nonpoly);

    /* Track position within structure */
    int atom_idx = 0;
    int res_idx = 0;
    int serial = 1;

    /* Iterate through chains */
    for (int chain = 0; chain < cif->chains; chain++) {
        const char *chain_name = cif->names[chain];
        CIF_CHECK_CHAIN_NAME(chain_name, chain, ctx);

        LOG_DEBUG("Writing chain %s with %d residues",
                  chain_name, cif->res_per_chain[chain]);

        int output_seq_id = 1;  /* Track output sequence number (restarts per chain) */

        /* Iterate through residues in this chain */
        for (int res = 0; res < cif->res_per_chain[chain]; res++) {
            CIF_CHECK_BOUNDS(res_idx, cif->residues, "Residue", ctx);

            int atoms_in_res = cif->atoms_per_res[res_idx];

            /* Skip residues with no polymer atoms (e.g., HETATM-only residues) */
            if (atoms_in_res == 0) {
                res_idx++;
                continue;
            }

            /* residue_name() logs warning automatically for unknown indices */
            const char *res_name = residue_name(cif->sequence[res_idx]);

            /* Iterate through atoms in this residue */
            for (int a = 0; a < atoms_in_res; a++) {
                CIF_CHECK_BOUNDS(atom_idx, cif->atoms, "Atom", ctx);

                /* Determine if polymer or non-polymer atom */
                const char *group = (atom_idx < cif->polymer) ? "ATOM" : "HETATM";

                /* element_name() logs warning automatically for unknown indices */
                const char *elem = element_name(cif->elements[atom_idx]);

                /* atom_info() logs warning automatically for unknown indices */
                const AtomInfo *ainfo = atom_info(cif->types[atom_idx]);

                /* Format atom name, quoting if it contains a prime (') */
                char atom_buf[MAX_ATOM_NAME_BUF];
                const char *atom_name = _format_atom_name(ainfo->atom, atom_buf);

                /* Get coordinates - bounds check atom_idx first to prevent overflow */
                if (atom_idx < 0 || atom_idx >= cif->atoms) {
                    LOG_ERROR("Atom index %d out of bounds [0, %d)", atom_idx, cif->atoms);
                    CIF_SET_ERROR(ctx, CIF_ERR_BOUNDS,
                        "Atom index %d out of bounds [0, %d)", atom_idx, cif->atoms);
                    return CIF_ERR_BOUNDS;
                }
                /* Safe: atom_idx < atoms, so 3*atom_idx < 3*atoms */
                int coord_idx = 3 * atom_idx;
                float x = cif->coordinates[coord_idx + 0];
                float y = cif->coordinates[coord_idx + 1];
                float z = cif->coordinates[coord_idx + 2];

                /* Sequence ID: use output_seq_id for polymer, '.' for non-polymer */
                /* NOTE: Use LEFT-justified fields (%-Nd) to avoid leading spaces that merge
                 * with field delimiters. The parser uses whitespace to find field boundaries,
                 * so right-justified fields like %5d break offset computation. All fields
                 * must have consistent width for the fixed-line-width parser to work.
                 * Field widths: serial(7), element(2), atom(6 for quoted), res(4), chain(4), seq(6), coord(10) */
                if (atom_idx < cif->polymer) {
                    CIF_FPRINTF(file, ctx, "%-6s %-7d %-2.2s %-6s . %-4.4s %-4.4s %-6d %-10.3f %-10.3f %-10.3f 1\n",
                        group, serial, elem, atom_name,
                        res_name, chain_name, output_seq_id, x, y, z);
                } else {
                    CIF_FPRINTF(file, ctx, "%-6s %-7d %-2.2s %-6s . %-4.4s %-4.4s %-6s %-10.3f %-10.3f %-10.3f 1\n",
                        group, serial, elem, atom_name,
                        res_name, chain_name, ".", x, y, z);
                }

                atom_idx++;
                serial++;
            }
            output_seq_id++;
            res_idx++;
        }
    }

    LOG_INFO("Wrote %d atoms to file", serial - 1);

    CIF_FPRINTF(file, ctx, "#\n");
    return CIF_OK;
}


/* ============================================================================
 * PUBLIC INTERFACE
 * ============================================================================ */

CifError _write_cif_file(const mmCIF *cif, FILE *file, CifErrorContext *ctx) {
    CifError err;

    /* Validate basic inputs */
    if (cif == NULL) {
        LOG_ERROR("Cannot write NULL mmCIF structure");
        CIF_SET_ERROR(ctx, CIF_ERR_PARSE, "Cannot write NULL mmCIF structure");
        return CIF_ERR_PARSE;
    }
    if (file == NULL) {
        LOG_ERROR("Cannot write to NULL file handle");
        CIF_SET_ERROR(ctx, CIF_ERR_IO, "Cannot write to NULL file handle");
        return CIF_ERR_IO;
    }
    if (cif->id == NULL) {
        LOG_ERROR("mmCIF structure has no ID");
        CIF_SET_ERROR(ctx, CIF_ERR_PARSE, "mmCIF structure has no ID");
        return CIF_ERR_PARSE;
    }

    /* Validate required arrays for non-empty structures */
    if (cif->chains > 0) {
        if (cif->names == NULL) {
            LOG_ERROR("chains=%d but names array is NULL", cif->chains);
            CIF_SET_ERROR(ctx, CIF_ERR_PARSE, "Missing chain names array");
            return CIF_ERR_PARSE;
        }
        if (cif->strands == NULL) {
            LOG_ERROR("chains=%d but strands array is NULL", cif->chains);
            CIF_SET_ERROR(ctx, CIF_ERR_PARSE, "Missing strands array");
            return CIF_ERR_PARSE;
        }
        if (cif->res_per_chain == NULL) {
            LOG_ERROR("chains=%d but res_per_chain array is NULL", cif->chains);
            CIF_SET_ERROR(ctx, CIF_ERR_PARSE, "Missing res_per_chain array");
            return CIF_ERR_PARSE;
        }
    }

    if (cif->residues > 0) {
        if (cif->sequence == NULL) {
            LOG_ERROR("residues=%d but sequence array is NULL", cif->residues);
            CIF_SET_ERROR(ctx, CIF_ERR_PARSE, "Missing sequence array");
            return CIF_ERR_PARSE;
        }
        if (cif->atoms_per_res == NULL) {
            LOG_ERROR("residues=%d but atoms_per_res array is NULL", cif->residues);
            CIF_SET_ERROR(ctx, CIF_ERR_PARSE, "Missing atoms_per_res array");
            return CIF_ERR_PARSE;
        }
    }

    if (cif->atoms > 0) {
        if (cif->coordinates == NULL) {
            LOG_ERROR("atoms=%d but coordinates array is NULL", cif->atoms);
            CIF_SET_ERROR(ctx, CIF_ERR_PARSE, "Missing coordinates array");
            return CIF_ERR_PARSE;
        }
        if (cif->elements == NULL) {
            LOG_ERROR("atoms=%d but elements array is NULL", cif->atoms);
            CIF_SET_ERROR(ctx, CIF_ERR_PARSE, "Missing elements array");
            return CIF_ERR_PARSE;
        }
        if (cif->types == NULL) {
            LOG_ERROR("atoms=%d but types array is NULL", cif->atoms);
            CIF_SET_ERROR(ctx, CIF_ERR_PARSE, "Missing types array");
            return CIF_ERR_PARSE;
        }
    }

    LOG_DEBUG("Validated structure: %d chains, %d residues, %d atoms",
              cif->chains, cif->residues, cif->atoms);

    /* Write each block in order */
    err = _write_header(file, cif->id, ctx);
    if (err != CIF_OK) return err;

    err = _write_entity(file, cif, ctx);
    if (err != CIF_OK) return err;

    err = _write_struct_asym(file, cif, ctx);
    if (err != CIF_OK) return err;

    err = _write_entity_poly(file, cif, ctx);
    if (err != CIF_OK) return err;

    err = _write_entity_nonpoly(file, cif, ctx);
    if (err != CIF_OK) return err;

    err = _write_poly_seq(file, cif, ctx);
    if (err != CIF_OK) return err;

    err = _write_atom_site(file, cif, ctx);
    if (err != CIF_OK) return err;

    return CIF_OK;
}


CifError _write_cif(const mmCIF *cif, const char *filename, CifErrorContext *ctx) {
    LOG_INFO("Writing CIF file: %s", filename);

    /* Open file for writing */
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        LOG_ERROR("Failed to open file for writing: %s", filename);
        CIF_SET_ERROR(ctx, CIF_ERR_IO, "Failed to open file for writing: %s", filename);
        return CIF_ERR_IO;
    }

    /* Write the structure */
    CifError err = _write_cif_file(cif, file, ctx);

    /* Close file */
    if (fclose(file) != 0 && err == CIF_OK) {
        LOG_ERROR("Failed to close file: %s", filename);
        CIF_SET_ERROR(ctx, CIF_ERR_IO, "Failed to close file: %s", filename);
        return CIF_ERR_IO;
    }

    if (err == CIF_OK) {
        LOG_INFO("Successfully wrote CIF file: %s", filename);
    }

    return err;
}
