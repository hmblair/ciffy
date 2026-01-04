/**
 * @file schema.h
 * @brief Shared mmCIF attribute name constants.
 *
 * Defines attribute names used by both the parser (registry.c) and
 * writer (writer.c) to ensure round-trip compatibility. If an attribute
 * name is changed here, both reading and writing will update together.
 *
 * Attribute names are defined WITHOUT the category prefix (e.g., "Cartn_x"
 * not "_atom_site.Cartn_x"). The category is added by the caller.
 */

#ifndef _CIFFY_SCHEMA_H
#define _CIFFY_SCHEMA_H


/* ============================================================================
 * _atom_site ATTRIBUTES
 * Atom-level data: coordinates, types, elements, B-factors.
 * ============================================================================ */

#define SCHEMA_ATOM_GROUP_PDB       "group_PDB"
#define SCHEMA_ATOM_ID              "id"
#define SCHEMA_ATOM_TYPE_SYMBOL     "type_symbol"
#define SCHEMA_ATOM_LABEL_ATOM_ID   "label_atom_id"
#define SCHEMA_ATOM_LABEL_ALT_ID    "label_alt_id"
#define SCHEMA_ATOM_LABEL_COMP_ID   "label_comp_id"
#define SCHEMA_ATOM_LABEL_ASYM_ID   "label_asym_id"
#define SCHEMA_ATOM_LABEL_SEQ_ID    "label_seq_id"
#define SCHEMA_ATOM_CARTN_X         "Cartn_x"
#define SCHEMA_ATOM_CARTN_Y         "Cartn_y"
#define SCHEMA_ATOM_CARTN_Z         "Cartn_z"
#define SCHEMA_ATOM_B_ISO           "B_iso_or_equiv"
#define SCHEMA_ATOM_MODEL_NUM       "pdbx_PDB_model_num"


/* ============================================================================
 * _struct_asym ATTRIBUTES
 * Chain definitions.
 * ============================================================================ */

#define SCHEMA_ASYM_ID              "id"
#define SCHEMA_ASYM_STRAND_ID       "pdbx_strand_id"
#define SCHEMA_ASYM_ENTITY_ID       "entity_id"


/* ============================================================================
 * _entity ATTRIBUTES
 * Entity type definitions.
 * ============================================================================ */

#define SCHEMA_ENTITY_ID            "id"
#define SCHEMA_ENTITY_TYPE          "type"


/* ============================================================================
 * _entity_poly ATTRIBUTES
 * Polymer type per entity.
 * ============================================================================ */

#define SCHEMA_ENTITY_POLY_ID       "entity_id"
#define SCHEMA_ENTITY_POLY_TYPE     "type"
#define SCHEMA_ENTITY_POLY_STRAND   "pdbx_strand_id"


/* ============================================================================
 * _pdbx_entity_nonpoly ATTRIBUTES
 * Non-polymer entity comp_ids.
 * ============================================================================ */

#define SCHEMA_NONPOLY_ENTITY_ID    "entity_id"
#define SCHEMA_NONPOLY_COMP_ID      "comp_id"


/* ============================================================================
 * _pdbx_poly_seq_scheme ATTRIBUTES
 * Polymer sequence information.
 * ============================================================================ */

#define SCHEMA_SEQ_ASYM_ID          "asym_id"
#define SCHEMA_SEQ_MON_ID           "mon_id"
#define SCHEMA_SEQ_STRAND_ID        "pdb_strand_id"
#define SCHEMA_SEQ_SEQ_ID           "seq_id"


/* ============================================================================
 * _refine ATTRIBUTES
 * Refinement statistics.
 * ============================================================================ */

#define SCHEMA_REFINE_RESOLUTION    "ls_d_res_high"


/* ============================================================================
 * _database_PDB_rev ATTRIBUTES
 * Database revision information.
 * ============================================================================ */

#define SCHEMA_REV_DATE             "recvd_initial_deposition_date"


#endif /* _CIFFY_SCHEMA_H */
