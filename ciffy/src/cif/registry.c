/**
 * @file registry.c
 * @brief Block and field registry implementation.
 *
 * Contains the declarative definitions for mmCIF blocks and fields,
 * plus the topological sort algorithm for computing parse order.
 */

#include "registry.h"
#include "parser.h"
#include "../log.h"
#include "../profile.h"

#include <stddef.h>
#include <stdlib.h>
#include <string.h>


/* ============================================================================
 * BLOCK DEFINITIONS
 * Auto-generated from BLOCK_LIST macro in registry.h.
 * ============================================================================ */

static const BlockDef BLOCKS[] = {
    #define X(name, category, required) { BLOCK_##name, category, required },
    BLOCK_LIST
    #undef X
};

_Static_assert(sizeof(BLOCKS) / sizeof(BLOCKS[0]) == BLOCK_COUNT,
               "BLOCKS array size must match BLOCK_COUNT");


/* ============================================================================
 * ATTRIBUTE NAME CONSTANTS
 * Used in field definitions below.
 * ============================================================================ */

static const char *ATTR_MODEL[]         = { "pdbx_PDB_model_num", NULL };
static const char *ATTR_CHAIN_ID[]      = { "id", NULL };
static const char *ATTR_RES_PER_CHAIN[] = { "asym_id", NULL };
static const char *ATTR_STRAND_ID[]     = { "pdb_strand_id", NULL };
static const char *ATTR_RESIDUE_NAME[]  = { "mon_id", NULL };

/* Batch-parsed field attributes */
static const char *ATTR_COORDS[]   = { "Cartn_x", "Cartn_y", "Cartn_z", NULL };
static const char *ATTR_ELEMENT[]  = { "type_symbol", NULL };
static const char *ATTR_ATOM_TYPE[] = { "label_comp_id", "label_atom_id", NULL };


/* ============================================================================
 * DEPENDENCY ARRAYS
 * Terminated with -1 sentinel.
 * ============================================================================ */

static const FieldId DEP_MODELS[]   = { FIELD_MODELS, -1 };
static const FieldId DEP_CHAINS[]   = { FIELD_CHAINS, -1 };
static const FieldId DEP_RESIDUES[] = { FIELD_RESIDUES, -1 };


/* ============================================================================
 * FORWARD DECLARATIONS
 * Helper functions for field parsing operations.
 * ============================================================================ */

/* These functions are defined in parser.c - we declare them here for use */
extern int _count_unique(mmBlock *block, const char *attr, CifErrorContext *ctx);
extern char **_get_unique(mmBlock *block, const char *attr, int *size, CifErrorContext *ctx);
extern int *_count_sizes_by_group(mmBlock *block, const char *attr, int *size, CifErrorContext *ctx);
extern int *_parse_via_lookup(mmBlock *block, HashTable func, const char *attr, CifErrorContext *ctx);

/* Hash lookup function - defined in hash/residue.c, included by parser.c */
extern struct _LOOKUP *_lookup_residue(const char *str, size_t len);

/* Hash lookup functions for batch parsing - defined in parser.c via includes */
extern struct _LOOKUP *_lookup_element(const char *str, size_t len);
extern struct _LOOKUP *_lookup_atom(const char *str, size_t len);
extern struct _LOOKUP *_lookup_molecule(const char *str, size_t len);
extern struct _LOOKUP *_lookup_entity(const char *str, size_t len);
extern struct _LOOKUP *_lookup_ion(const char *str, size_t len);

/* Parser functions used by molecule_types */
extern CifError _precompute_lines(mmBlock *block, CifErrorContext *ctx);
extern void _free_lines(mmBlock *block);
extern int _parse_int_inline(mmBlock *block, int line, int index);


/* ============================================================================
 * BATCH ROW CALLBACKS
 * Per-row parsing functions for batch-parsed fields.
 * ============================================================================ */

/**
 * @brief Parse coordinates for a single row with two-pointer placement.
 * attr_indices: [0]=x, [1]=y, [2]=z
 * Uses cif->write_dest (set by batch loop) as destination.
 */
static void _batch_coords(mmCIF *cif, mmBlock *block, int row,
                          const int *idx, char *scratch) {
    (void)scratch;
    _parse_coords_inline(block, row, idx, &cif->coordinates[3 * cif->write_dest]);
}

/**
 * @brief Parse element type for a single row with two-pointer placement.
 * attr_indices: [0]=type_symbol
 */
static void _batch_elements(mmCIF *cif, mmBlock *block, int row,
                            const int *idx, char *scratch) {
    (void)scratch;
    cif->elements[cif->write_dest] = _lookup_element_fast(block, row, idx[0], _lookup_element);
}

/**
 * @brief Parse atom type for a single row with two-pointer placement.
 * attr_indices: [0]=label_comp_id, [1]=label_atom_id
 */
static void _batch_types(mmCIF *cif, mmBlock *block, int row,
                         const int *idx, char *scratch) {
    cif->types[cif->write_dest] = _lookup_atom_type_fast(block, row, idx[0], idx[1],
                                                          _lookup_atom, scratch);
}


/* ============================================================================
 * OP_COMPUTE PARSE FUNCTIONS
 * Custom computation functions for fields that need special handling.
 * ============================================================================ */

/**
 * Parse molecule types from _entity, _entity_poly, and _pdbx_entity_nonpoly.
 *
 * Classification hierarchy:
 * 1. Parse _entity.type for base classification:
 *    - "water" -> WATER (10)
 *    - "branched" -> POLYSACCHARIDE (5)
 *    - "non-polymer" -> LIGAND (8, may be refined to ION)
 *    - "polymer" -> UNKNOWN (will be refined by _entity_poly)
 *
 * 2. Parse _pdbx_entity_nonpoly.comp_id to refine non-polymers:
 *    - Known ion comp_ids (MG, CA, ZN, etc.) -> ION (9)
 *
 * 3. Parse _entity_poly.type for polymer classification:
 *    - "polyribonucleotide" -> RNA (1)
 *    - "polypeptide(L)" -> PROTEIN (0)
 *    - etc.
 *
 * 4. Map chains via _struct_asym.entity_id
 */
static CifError _parse_molecule_types(mmCIF *cif, mmBlockList *blocks,
                                      const void *def, CifErrorContext *ctx) {
    (void)def;

    /* Allocate molecule_types array */
    cif->molecule_types = calloc((size_t)cif->chains, sizeof(int));
    if (!cif->molecule_types) {
        CIF_SET_ERROR(ctx, CIF_ERR_ALLOC, "Failed to allocate molecule_types");
        return CIF_ERR_ALLOC;
    }

    /* Default to UNKNOWN (12) for all chains */
    for (int i = 0; i < cif->chains; i++) {
        cif->molecule_types[i] = 12;  /* Molecule.UNKNOWN */
    }

    /* Get block pointers */
    mmBlock *entity = &blocks->b[BLOCK_ENTITY];
    mmBlock *entity_poly = &blocks->b[BLOCK_ENTITY_POLY];
    mmBlock *entity_nonpoly = &blocks->b[BLOCK_ENTITY_NONPOLY];
    mmBlock *chain_block = &blocks->b[BLOCK_CHAIN];

    /* Build entity_id -> molecule_type map (sized for number of chains + 1) */
    int entity_map_size = cif->chains + 1;  /* +1 for 1-indexed entity_ids */
    int *entity_map = calloc((size_t)entity_map_size, sizeof(int));
    if (!entity_map) {
        CIF_SET_ERROR(ctx, CIF_ERR_ALLOC, "Failed to allocate entity_map");
        return CIF_ERR_ALLOC;
    }
    for (int i = 0; i < entity_map_size; i++) entity_map[i] = 12;  /* UNKNOWN */

    /* Get attribute index for struct_asym.entity_id */
    int sa_entity_idx = _get_attr_index(chain_block, "entity_id", ctx);
    if (sa_entity_idx < 0) {
        LOG_WARNING("_struct_asym missing entity_id attribute");
        free(entity_map);
        return CIF_OK;
    }

    /* ========================================================================
     * STEP 1: Parse _entity.type for base classification
     * ======================================================================== */
    if (entity->category != NULL) {
        CifError err = _precompute_lines(entity, ctx);
        if (err != CIF_OK) { free(entity_map); return err; }

        int e_id_idx = _get_attr_index(entity, "id", ctx);
        int e_type_idx = _get_attr_index(entity, "type", ctx);

        if (e_id_idx >= 0 && e_type_idx >= 0) {
            for (int row = 0; row < entity->size; row++) {
                int entity_id = _parse_int_inline(entity, row, e_id_idx);
                if (entity_id < 0 || entity_id >= entity_map_size) continue;

                size_t type_len;
                const char *type_ptr = _get_field_ptr(entity, row, e_type_idx, &type_len);
                if (!type_ptr || type_len == 0) continue;

                /* Copy to buffer and null-terminate (required for gperf strcmp) */
                char type_buf[32];
                if (type_len >= sizeof(type_buf)) continue;
                const char *src = type_ptr;
                size_t src_len = type_len;
                _strip_outer_quotes(&src, &src_len);
                memcpy(type_buf, src, src_len);
                type_buf[src_len] = '\0';

                /* Look up entity type */
                struct _LOOKUP *result = _lookup_entity(type_buf, src_len);
                if (result) {
                    entity_map[entity_id] = result->value;
                    LOG_DEBUG("Entity %d: _entity.type='%s' -> %d",
                              entity_id, type_buf, result->value);
                }
            }
        }
        _free_lines(entity);
    }

    /* ========================================================================
     * STEP 2: Parse _pdbx_entity_nonpoly.comp_id to refine ION vs LIGAND
     * ======================================================================== */
    if (entity_nonpoly->category != NULL) {
        CifError err = _precompute_lines(entity_nonpoly, ctx);
        if (err != CIF_OK) { free(entity_map); return err; }

        int enp_entity_idx = _get_attr_index(entity_nonpoly, "entity_id", ctx);
        int enp_comp_idx = _get_attr_index(entity_nonpoly, "comp_id", ctx);

        if (enp_entity_idx >= 0 && enp_comp_idx >= 0) {
            for (int row = 0; row < entity_nonpoly->size; row++) {
                int entity_id = _parse_int_inline(entity_nonpoly, row, enp_entity_idx);
                if (entity_id < 0 || entity_id >= entity_map_size) continue;

                size_t comp_len;
                const char *comp_ptr = _get_field_ptr(entity_nonpoly, row, enp_comp_idx, &comp_len);
                if (!comp_ptr || comp_len == 0) continue;

                /* Copy to buffer and null-terminate (required for gperf strcmp) */
                char comp_buf[16];
                if (comp_len >= sizeof(comp_buf)) continue;
                const char *src = comp_ptr;
                size_t src_len = comp_len;
                _strip_outer_quotes(&src, &src_len);
                memcpy(comp_buf, src, src_len);
                comp_buf[src_len] = '\0';

                /* Check if this comp_id is a known ion */
                struct _LOOKUP *result = _lookup_ion(comp_buf, src_len);
                if (result) {
                    entity_map[entity_id] = result->value;  /* ION (9) */
                    LOG_DEBUG("Entity %d: comp_id='%s' -> ION", entity_id, comp_buf);
                }
            }
        }
        _free_lines(entity_nonpoly);
    }

    /* ========================================================================
     * STEP 3: Parse _entity_poly.type for polymer classification
     * ======================================================================== */
    if (entity_poly->category != NULL) {
        CifError err = _precompute_lines(entity_poly, ctx);
        if (err != CIF_OK) { free(entity_map); return err; }

        int ep_entity_idx = _get_attr_index(entity_poly, "entity_id", ctx);
        int ep_type_idx = _get_attr_index(entity_poly, "type", ctx);

        if (ep_entity_idx >= 0 && ep_type_idx >= 0) {
            for (int row = 0; row < entity_poly->size; row++) {
                int entity_id = _parse_int_inline(entity_poly, row, ep_entity_idx);
                if (entity_id < 0 || entity_id >= entity_map_size) continue;

                size_t type_len;
                const char *type_ptr = _get_field_ptr(entity_poly, row, ep_type_idx, &type_len);
                if (!type_ptr || type_len == 0) continue;

                /* Strip quotes */
                char type_buf[64];
                if (type_len >= sizeof(type_buf)) continue;
                const char *src = type_ptr;
                size_t src_len = type_len;
                _strip_outer_quotes(&src, &src_len);
                memcpy(type_buf, src, src_len);
                type_buf[src_len] = '\0';

                /* Look up molecule type via hash table */
                struct _LOOKUP *result = _lookup_molecule(type_buf, src_len);
                int mol_type = result ? result->value : 11;  /* OTHER if not found */

                entity_map[entity_id] = mol_type;
                LOG_DEBUG("Entity %d: _entity_poly.type='%s' -> %d",
                          entity_id, type_buf, mol_type);
            }
        }
        _free_lines(entity_poly);
    }

    /* ========================================================================
     * STEP 4: Map chains to molecule types via entity_id
     * ======================================================================== */
    for (int chain = 0; chain < cif->chains; chain++) {
        int entity_id = _parse_int_inline(chain_block, chain, sa_entity_idx);
        if (entity_id >= 0 && entity_id < entity_map_size) {
            cif->molecule_types[chain] = entity_map[entity_id];
        }
    }

    free(entity_map);
    LOG_DEBUG("Molecule types parsed for %d chains", cif->chains);
    return CIF_OK;
}


/* ============================================================================
 * FIELD DEFINITIONS
 * Declarative specification of fields and their dependencies.
 *
 * Fields are organized by dependency level:
 *   Level 0: Leaf fields (no dependencies)
 *   Level 1: Depend on leaf fields
 *
 * The topological sort will compute the actual execution order.
 *
 * Batch-parsed fields (batchable=true) are grouped by source block and
 * parsed in a single pass for cache efficiency. Each has a batch_row_func
 * callback that is called once per row.
 * ============================================================================ */

/* IMPORTANT: FIELDS[] must be indexed by FieldId enum value.
 * The array order must match the enum order in registry.h.
 *
 * Field format:
 *   { id, name, source_block, operation, attrs, depends_on, parse_func,
 *     batchable, batch_row_func,
 *     storage_offset, storage_type,
 *     size_source, element_size, elements_per_item,
 *     py_export, py_name }
 */
static const FieldDef FIELDS[] = {
    /* FIELD_MODELS = 0 - internal only, not exported to Python */
    { FIELD_MODELS, "models", BLOCK_ATOM, OP_COUNT_UNIQUE,
      ATTR_MODEL, NULL, NULL, false, NULL,
      offsetof(mmCIF, models), STORAGE_INT,
      SIZE_NONE, 0, 0,
      PY_NONE, NULL },

    /* FIELD_CHAINS = 1 - internal only, not exported to Python */
    { FIELD_CHAINS, "chains", BLOCK_CHAIN, OP_BLOCK_SIZE,
      NULL, NULL, NULL, false, NULL,
      offsetof(mmCIF, chains), STORAGE_INT,
      SIZE_NONE, 0, 0,
      PY_NONE, NULL },

    /* FIELD_RESIDUES = 2 - internal only, not exported to Python */
    { FIELD_RESIDUES, "residues", BLOCK_POLY, OP_BLOCK_SIZE,
      NULL, NULL, NULL, false, NULL,
      offsetof(mmCIF, residues), STORAGE_INT,
      SIZE_NONE, 0, 0,
      PY_NONE, NULL },

    /* FIELD_ATOMS = 3 - atoms = atom_site.size / models, internal only */
    { FIELD_ATOMS, "atoms", BLOCK_ATOM, OP_COMPUTE,
      NULL, DEP_MODELS, NULL, false, NULL,
      offsetof(mmCIF, atoms), STORAGE_INT,
      SIZE_NONE, 0, 0,
      PY_NONE, NULL },

    /* FIELD_NAMES = 4 - allocated by _get_unique, exported as "chain_names" */
    { FIELD_NAMES, "names", BLOCK_CHAIN, OP_GET_UNIQUE,
      ATTR_CHAIN_ID, DEP_CHAINS, NULL, false, NULL,
      offsetof(mmCIF, names), STORAGE_STR_ARRAY,
      SIZE_NONE, 0, 0,
      PY_STR_LIST, "chain_names" },

    /* FIELD_STRANDS = 5 - allocated by _get_unique, exported as "strand_names" */
    { FIELD_STRANDS, "strands", BLOCK_POLY, OP_GET_UNIQUE,
      ATTR_STRAND_ID, DEP_CHAINS, NULL, false, NULL,
      offsetof(mmCIF, strands), STORAGE_STR_ARRAY,
      SIZE_NONE, 0, 0,
      PY_STR_LIST, "strand_names" },

    /* FIELD_SEQUENCE = 6 - allocated by _parse_via_lookup, exported as "residues" */
    { FIELD_SEQUENCE, "sequence", BLOCK_POLY, OP_LOOKUP,
      ATTR_RESIDUE_NAME, DEP_RESIDUES, NULL, false, NULL,
      offsetof(mmCIF, sequence), STORAGE_INT_PTR,
      SIZE_NONE, 0, 0,
      PY_1D_INT, "residues" },

    /* FIELD_COORDS = 7 - batch parsed, auto-allocated */
    { FIELD_COORDS, "coordinates", BLOCK_ATOM, OP_COMPUTE,
      ATTR_COORDS, DEP_MODELS, NULL, true, _batch_coords,
      offsetof(mmCIF, coordinates), STORAGE_FLOAT_PTR,
      SIZE_ATOMS, sizeof(float), 3,
      PY_2D_FLOAT, NULL },

    /* FIELD_TYPES = 8 - batch parsed, auto-allocated, exported as "atoms" */
    { FIELD_TYPES, "types", BLOCK_ATOM, OP_COMPUTE,
      ATTR_ATOM_TYPE, DEP_MODELS, NULL, true, _batch_types,
      offsetof(mmCIF, types), STORAGE_INT_PTR,
      SIZE_ATOMS, sizeof(int), 1,
      PY_1D_INT, "atoms" },

    /* FIELD_ELEMENTS = 9 - batch parsed, auto-allocated */
    { FIELD_ELEMENTS, "elements", BLOCK_ATOM, OP_COMPUTE,
      ATTR_ELEMENT, DEP_MODELS, NULL, true, _batch_elements,
      offsetof(mmCIF, elements), STORAGE_INT_PTR,
      SIZE_ATOMS, sizeof(int), 1,
      PY_1D_INT, NULL },

    /* FIELD_RES_PER_CHAIN = 10 - allocated by _count_sizes_by_group */
    { FIELD_RES_PER_CHAIN, "res_per_chain", BLOCK_POLY, OP_COUNT_BY_GROUP,
      ATTR_RES_PER_CHAIN, DEP_CHAINS, NULL, false, NULL,
      offsetof(mmCIF, res_per_chain), STORAGE_INT_PTR,
      SIZE_NONE, 0, 0,
      PY_1D_INT, NULL },

    /* FIELD_ATOMS_PER_RES = 11 - computed externally via _count_atoms_per_residue */
    { FIELD_ATOMS_PER_RES, "atoms_per_res", BLOCK_ATOM, OP_COMPUTE,
      NULL, NULL, NULL, false, NULL,
      offsetof(mmCIF, atoms_per_res), STORAGE_INT_PTR,
      SIZE_NONE, 0, 0,
      PY_1D_INT, NULL },

    /* FIELD_MOL_TYPES = 12 - allocated in parse_func */
    { FIELD_MOL_TYPES, "molecule_types", BLOCK_ENTITY_POLY, OP_COMPUTE,
      NULL, DEP_CHAINS, _parse_molecule_types, false, NULL,
      offsetof(mmCIF, molecule_types), STORAGE_INT_PTR,
      SIZE_NONE, 0, 0,
      PY_1D_INT, NULL },

    /* FIELD_DESCRIPTIONS = 13 - optional, parsed separately in module.c */
    { FIELD_DESCRIPTIONS, "descriptions", BLOCK_ENTITY, OP_COMPUTE,
      NULL, DEP_CHAINS, NULL, false, NULL,
      offsetof(mmCIF, descriptions), STORAGE_STR_ARRAY,
      SIZE_NONE, 0, 0,
      PY_NONE, NULL },  /* PY_NONE: not auto-exported, handled in module.c */
};

_Static_assert(sizeof(FIELDS) / sizeof(FIELDS[0]) == FIELD_COUNT,
               "FIELDS array size must match FIELD_COUNT");


/* ============================================================================
 * REGISTRY API
 * ============================================================================ */

const BlockDef *_get_blocks(void) {
    return BLOCKS;
}

const FieldDef *_get_fields(void) {
    return FIELDS;
}


/* ============================================================================
 * TOPOLOGICAL SORT
 * Computes field execution order from dependencies.
 * ============================================================================ */

/**
 * DFS visitor for topological sort.
 */
static CifError _topo_visit(FieldId fid, bool *visited, bool *in_stack,
                            ParsePlan *plan, CifErrorContext *ctx) {
    if (in_stack[fid]) {
        CIF_SET_ERROR(ctx, CIF_ERR_PARSE,
            "Circular dependency detected at field '%s' (id=%d)",
            FIELDS[fid].name, fid);
        return CIF_ERR_PARSE;
    }
    if (visited[fid]) {
        return CIF_OK;
    }

    in_stack[fid] = true;

    const FieldId *deps = FIELDS[fid].depends_on;
    if (deps != NULL) {
        for (int i = 0; deps[i] != (FieldId)-1; i++) {
            CifError err = _topo_visit(deps[i], visited, in_stack, plan, ctx);
            if (err != CIF_OK) return err;
        }
    }

    in_stack[fid] = false;
    visited[fid] = true;
    plan->order[plan->count++] = fid;

    return CIF_OK;
}

CifError _plan_parse(ParsePlan *plan, CifErrorContext *ctx) {
    bool visited[FIELD_COUNT] = {false};
    bool in_stack[FIELD_COUNT] = {false};
    plan->count = 0;

    LOG_DEBUG("Computing parse order via topological sort (%d fields)", FIELD_COUNT);

    for (int i = 0; i < FIELD_COUNT; i++) {
        if (!visited[i]) {
            CifError err = _topo_visit((FieldId)i, visited, in_stack, plan, ctx);
            if (err != CIF_OK) return err;
        }
    }

    LOG_DEBUG("Parse order computed: %d fields in order", plan->count);
    return CIF_OK;
}


/* ============================================================================
 * BLOCK UTILITIES
 * ============================================================================ */

mmBlock *_get_block_by_id(mmBlockList *blocks, BlockId id) {
    if (id < 0 || id >= BLOCK_COUNT) return NULL;
    return &blocks->b[id];
}

CifError _validate_blocks_registry(mmBlockList *blocks, CifErrorContext *ctx) {
    for (int i = 0; i < BLOCK_COUNT; i++) {
        if (!BLOCKS[i].required) continue;

        mmBlock *block = _get_block_by_id(blocks, BLOCKS[i].id);
        if (block == NULL || block->category == NULL) {
            LOG_ERROR("Missing required block '%s'", BLOCKS[i].category);
            CIF_SET_ERROR(ctx, CIF_ERR_BLOCK,
                "Missing required %s block", BLOCKS[i].category);
            return CIF_ERR_BLOCK;
        }
    }
    return CIF_OK;
}


/* ============================================================================
 * OPERATION IMPLEMENTATIONS
 * Each _op_* function handles one type of field parsing operation.
 * ============================================================================ */

/**
 * OP_BLOCK_SIZE: Assign block size to an integer field.
 */
static CifError _op_block_size(mmCIF *cif, mmBlock *block, const FieldDef *def,
                               CifErrorContext *ctx) {
    (void)ctx;

    int value = block->size;
    LOG_DEBUG("OP_BLOCK_SIZE: %s = %d", def->name, value);

    _store_int(cif, def, value);
    return CIF_OK;
}

/**
 * OP_COUNT_UNIQUE: Count unique values in an attribute.
 */
static CifError _op_count_unique(mmCIF *cif, mmBlock *block, const FieldDef *def,
                                  CifErrorContext *ctx) {
    if (def->attrs == NULL || def->attrs[0] == NULL) {
        CIF_SET_ERROR(ctx, CIF_ERR_PARSE, "OP_COUNT_UNIQUE requires attribute");
        return CIF_ERR_PARSE;
    }

    int count = _count_unique(block, def->attrs[0], ctx);
    if (count < 0) return ctx->code;

    /* Validate non-zero for count fields */
    if (count == 0) {
        CIF_SET_ERROR(ctx, CIF_ERR_PARSE, "Invalid %s count: 0", def->name);
        return CIF_ERR_PARSE;
    }

    LOG_DEBUG("OP_COUNT_UNIQUE: %s = %d (attr=%s)", def->name, count, def->attrs[0]);

    _store_int(cif, def, count);
    return CIF_OK;
}

/**
 * OP_GET_UNIQUE: Extract unique strings from an attribute.
 */
static CifError _op_get_unique(mmCIF *cif, mmBlock *block, const FieldDef *def,
                               CifErrorContext *ctx) {
    if (def->attrs == NULL || def->attrs[0] == NULL) {
        CIF_SET_ERROR(ctx, CIF_ERR_PARSE, "OP_GET_UNIQUE requires attribute");
        return CIF_ERR_PARSE;
    }

    int size = cif->chains;  /* Pre-allocate based on chain count */
    char **result = _get_unique(block, def->attrs[0], &size, ctx);
    if (result == NULL) return ctx->code;

    LOG_DEBUG("OP_GET_UNIQUE: %s = %d unique values (attr=%s)", def->name, size, def->attrs[0]);

    _store_ptr(cif, def, result);
    return CIF_OK;
}

/**
 * OP_COUNT_BY_GROUP: Count items grouped by attribute value changes.
 */
static CifError _op_count_by_group(mmCIF *cif, mmBlock *block, const FieldDef *def,
                                    CifErrorContext *ctx) {
    if (def->attrs == NULL || def->attrs[0] == NULL) {
        CIF_SET_ERROR(ctx, CIF_ERR_PARSE, "OP_COUNT_BY_GROUP requires attribute");
        return CIF_ERR_PARSE;
    }

    int size = cif->chains;
    int *result = _count_sizes_by_group(block, def->attrs[0], &size, ctx);
    if (result == NULL) return ctx->code;

    LOG_DEBUG("OP_COUNT_BY_GROUP: %s (attr=%s)", def->name, def->attrs[0]);

    _store_ptr(cif, def, result);
    return CIF_OK;
}

/**
 * OP_LOOKUP: Parse values via hash table lookup.
 *
 * Note: Currently only supports residue lookup. To support other lookup
 * types, add a lookup_func field to FieldDef.
 */
static CifError _op_lookup(mmCIF *cif, mmBlock *block, const FieldDef *def,
                           CifErrorContext *ctx) {
    if (def->attrs == NULL || def->attrs[0] == NULL) {
        CIF_SET_ERROR(ctx, CIF_ERR_PARSE, "OP_LOOKUP requires attribute");
        return CIF_ERR_PARSE;
    }

    /*
     * TODO: Add lookup_func pointer to FieldDef struct to support different
     * hash tables. Currently hardcoded to _lookup_residue, but could support:
     *   - _lookup_residue: residue name -> index (current)
     *   - _lookup_element: element symbol -> atomic number
     *   - _lookup_atom: atom name -> atom type index
     * Would require: adding `LookupFunc lookup_func` to FieldDef, updating
     * FIELDS[] definitions, and using def->lookup_func here instead.
     */
    int *result = _parse_via_lookup(block, _lookup_residue, def->attrs[0], ctx);
    if (result == NULL) return ctx->code;

    LOG_DEBUG("OP_LOOKUP: %s (attr=%s)", def->name, def->attrs[0]);

    _store_ptr(cif, def, result);
    return CIF_OK;
}

/**
 * OP_COMPUTE: Custom computation for atoms field.
 */
static CifError _op_compute_atoms(mmCIF *cif, mmBlock *block,
                                   const FieldDef *def, CifErrorContext *ctx) {
    /* Validate block size */
    if (block->size == 0) {
        LOG_ERROR("Empty _atom_site block");
        CIF_SET_ERROR(ctx, CIF_ERR_BLOCK, "No atoms in structure");
        return CIF_ERR_BLOCK;
    }

    /* Adjust for multi-model structures (use first model only) */
    int atom_count = block->size;
    if (cif->models > 1) {
        if (block->size % cif->models != 0) {
            LOG_WARNING("Atom count %d not evenly divisible by model count %d",
                        block->size, cif->models);
        }
        atom_count = block->size / cif->models;
        /* Note: We modify block->size here for subsequent operations */
        block->size = atom_count;
    }

    _store_int(cif, def, atom_count);

    LOG_DEBUG("OP_COMPUTE: atoms = %d (from %d total / %d models)",
              atom_count, block->size * cif->models, cif->models);

    return CIF_OK;
}


/* ============================================================================
 * EXECUTE PLAN
 * Dispatch operations based on field definitions.
 * ============================================================================ */

/**
 * Execute a single field operation.
 */
static CifError _execute_field(mmCIF *cif, mmBlockList *blocks,
                               const FieldDef *def, CifErrorContext *ctx) {
    /* Skip batchable fields - they're handled by _execute_batch_group() */
    if (def->batchable) {
        return CIF_OK;
    }

    mmBlock *block = _get_block_by_id(blocks, def->source_block);
    if (block == NULL) {
        LOG_WARNING("No block for field %s", def->name);
        return CIF_OK;
    }

    switch (def->operation) {
        case OP_BLOCK_SIZE:
            return _op_block_size(cif, block, def, ctx);

        case OP_COUNT_UNIQUE:
            return _op_count_unique(cif, block, def, ctx);

        case OP_GET_UNIQUE:
            return _op_get_unique(cif, block, def, ctx);

        case OP_COUNT_BY_GROUP:
            return _op_count_by_group(cif, block, def, ctx);

        case OP_LOOKUP:
            return _op_lookup(cif, block, def, ctx);

        case OP_COMPUTE:
            /* If field has a parse_func, call it */
            if (def->parse_func != NULL) {
                return def->parse_func(cif, blocks, def, ctx);
            }
            /* Handle FIELD_ATOMS compute */
            if (def->id == FIELD_ATOMS) {
                return _op_compute_atoms(cif, block, def, ctx);
            }
            /* Skip other OP_COMPUTE fields (batch-parsed or external) */
            return CIF_OK;

        case OP_PARSE_FLOAT:
            /* Float parsing now handled via batch system */
            return CIF_OK;

        default:
            LOG_WARNING("Unknown operation %d for field %s", def->operation, def->name);
            return CIF_OK;
    }
}

CifError _execute_plan(mmCIF *cif, mmBlockList *blocks,
                       const ParsePlan *plan, CifErrorContext *ctx) {
    LOG_DEBUG("Executing parse plan (%d fields)", plan->count);

    for (int i = 0; i < plan->count; i++) {
        FieldId fid = plan->order[i];
        const FieldDef *def = &FIELDS[fid];

        CifError err = _execute_field(cif, blocks, def, ctx);
        if (err != CIF_OK) {
            LOG_ERROR("Failed to parse field '%s'", def->name);
            return err;
        }
    }

    LOG_DEBUG("Parse plan execution complete");
    return CIF_OK;
}


/* ============================================================================
 * BATCH EXECUTION
 * Runtime batch grouping and single-pass iteration.
 * ============================================================================ */

/**
 * Count number of attributes in a NULL-terminated array.
 */
static int _count_attrs(const char **attrs) {
    if (attrs == NULL) return 0;
    int count = 0;
    while (attrs[count] != NULL) count++;
    return count;
}

void _compute_batch_groups(BatchGroup *groups, int *group_count, int max_groups) {
    *group_count = 0;

    /* Group batchable fields by source block */
    for (int i = 0; i < FIELD_COUNT; i++) {
        const FieldDef *def = &FIELDS[i];
        if (!def->batchable || def->batch_row_func == NULL) continue;

        /* Find existing group for this block, or create new one */
        BatchGroup *group = NULL;
        for (int g = 0; g < *group_count; g++) {
            if (groups[g].block_id == def->source_block) {
                group = &groups[g];
                break;
            }
        }

        if (group == NULL) {
            if (*group_count >= max_groups) {
                LOG_WARNING("Max batch groups exceeded, some fields won't be batched");
                continue;
            }
            group = &groups[(*group_count)++];
            group->block_id = def->source_block;
            group->field_count = 0;
            group->attr_count = 0;
        }

        if (group->field_count >= MAX_BATCH_FIELDS) {
            LOG_WARNING("Max fields per batch exceeded for block %d", def->source_block);
            continue;
        }

        /* Add field to group */
        int field_idx = group->field_count++;
        group->fields[field_idx] = def->id;

        /* Add this field's attributes to the group's attr list */
        int field_attr_count = _count_attrs(def->attrs);

        for (int a = 0; a < field_attr_count && group->attr_count < MAX_BATCH_ATTRS; a++) {
            group->attrs[group->attr_count] = def->attrs[a];
            group->attr_map[field_idx][a] = group->attr_count;
            group->attr_count++;
        }

        LOG_DEBUG("Batch group %d: added field '%s' with %d attrs (total attrs: %d)",
                  (int)(group - groups), def->name, field_attr_count, group->attr_count);
    }

    LOG_DEBUG("Computed %d batch groups", *group_count);
}

/* ============================================================================
 * FUSED BATCH PARSING MACROS
 *
 * These macros enable easy extension of the fused batch loop.
 * All macros assume these variables are in scope:
 *   - line_start: char* to current row
 *   - offsets: const int* column offsets
 *
 * To add a new field (e.g., B-factor):
 *   1. Add attribute index lookup in _execute_batch_group
 *   2. Add one line in the fused loop: BATCH_FLOAT(bfactor[dest], bfactor_idx);
 * ============================================================================ */

/**
 * @brief Parse a float field.
 * @param dest  Destination lvalue (e.g., coords[3*dest + 0])
 * @param idx   Column index in offsets array
 */
#define BATCH_FLOAT(dest, idx) do { \
    char *_p = line_start + offsets[idx]; \
    while (*_p == ' ') _p++; \
    (dest) = _fast_parse_float(_p); \
} while(0)

/**
 * @brief Parse an integer field.
 * @param dest  Destination lvalue
 * @param idx   Column index in offsets array
 */
#define BATCH_INT(dest, idx) do { \
    char *_p = line_start + offsets[idx]; \
    while (*_p == ' ') _p++; \
    (dest) = atoi(_p); \
} while(0)

/**
 * @brief Parse a field and perform hash table lookup.
 * @param dest   Destination lvalue (receives lookup value or PARSE_FAIL)
 * @param idx    Column index in offsets array
 * @param table  Hash lookup function (e.g., _lookup_element)
 * @param buf    Scratch buffer (must be MAX_INLINE_BUFFER size)
 */
#define BATCH_LOOKUP(dest, idx, table, buf) do { \
    char *_p = line_start + offsets[idx]; \
    while (*_p == ' ') _p++; \
    char *_end = _p; \
    while (*_end != ' ' && *_end != '\n' && *_end != '\0') _end++; \
    size_t _len = (size_t)(_end - _p); \
    if (_len > 0 && _len < MAX_INLINE_BUFFER) { \
        const char *_src = _p; \
        size_t _src_len = _len; \
        _strip_outer_quotes(&_src, &_src_len); \
        memcpy(buf, _src, _src_len); \
        (buf)[_src_len] = '\0'; \
        struct _LOOKUP *_r = table(buf, _src_len); \
        if (_r) { \
            (dest) = _r->value; \
        } else { \
            LOG_WARNING("Unknown element '%s' at line %d", buf, block->data.line + row); \
            (dest) = PARSE_FAIL; \
        } \
    } else { \
        (dest) = PARSE_FAIL; \
    } \
} while(0)

/**
 * @brief Parse two fields, combine with separator, and perform hash lookup.
 * @param dest   Destination lvalue (receives lookup value or PARSE_FAIL)
 * @param idx1   Column index for first field (e.g., comp_id)
 * @param idx2   Column index for second field (e.g., atom_id)
 * @param sep    Separator character (e.g., '_')
 * @param table  Hash lookup function (e.g., _lookup_atom)
 * @param buf    Scratch buffer (must be MAX_INLINE_BUFFER size)
 */
#define BATCH_LOOKUP2(dest, idx1, idx2, sep, table, buf) do { \
    /* First field */ \
    char *_p1 = line_start + offsets[idx1]; \
    while (*_p1 == ' ') _p1++; \
    char *_end1 = _p1; \
    while (*_end1 != ' ' && *_end1 != '\n' && *_end1 != '\0') _end1++; \
    size_t _len1 = (size_t)(_end1 - _p1); \
    /* Second field */ \
    char *_p2 = line_start + offsets[idx2]; \
    while (*_p2 == ' ') _p2++; \
    char *_end2 = _p2; \
    while (*_end2 != ' ' && *_end2 != '\n' && *_end2 != '\0') _end2++; \
    size_t _len2 = (size_t)(_end2 - _p2); \
    /* Combine and lookup */ \
    if (_len1 > 0 && _len2 > 0 && _len1 + 1 + _len2 + 1 < MAX_INLINE_BUFFER) { \
        _strip_outer_quotes((const char **)&_p1, &_len1); \
        _strip_outer_quotes((const char **)&_p2, &_len2); \
        memcpy(buf, _p1, _len1); \
        (buf)[_len1] = (sep); \
        memcpy((buf) + _len1 + 1, _p2, _len2); \
        size_t _total = _len1 + 1 + _len2; \
        (buf)[_total] = '\0'; \
        struct _LOOKUP *_r = table(buf, _total); \
        if (_r) { \
            (dest) = _r->value; \
        } else { \
            LOG_WARNING("Unknown atom '%s' at line %d", buf, block->data.line + row); \
            (dest) = PARSE_FAIL; \
        } \
    } else { \
        (dest) = PARSE_FAIL; \
    } \
} while(0)


/**
 * @brief Fused batch processing for ATOM block fields.
 *
 * Processes coords, elements, and types in a single tight loop,
 * eliminating per-field function call overhead. Uses two-pointer
 * placement for polymer/non-polymer separation.
 *
 * To add new fields (e.g., B-factor, occupancy):
 *   1. Add attribute name to ATTR_xxx array at top of file
 *   2. Add field index lookup in _execute_batch_group
 *   3. Add one BATCH_xxx macro call in the loop below
 */
static CifError _batch_atom_fields_fused(mmCIF *cif, mmBlock *block,
                                          const int *coord_idx,
                                          int elem_idx, int comp_idx, int atom_idx,
                                          CifErrorContext *ctx) {
    (void)ctx;

    char scratch[MAX_INLINE_BUFFER];
    char elem_buf[MAX_INLINE_BUFFER];
    float *coords = cif->coordinates;
    int *elements = cif->elements;
    int *types = cif->types;
    int *is_nonpoly = cif->is_nonpoly;
    const int *offsets = block->offsets;
    char **lines = block->lines;

    int poly_idx = 0;
    int nonpoly_idx = cif->polymer;

    for (int row = 0; row < block->size; row++) {
        int dest = is_nonpoly[row] ? nonpoly_idx++ : poly_idx++;
        char *line_start = lines[row];

        /* Coordinates (x, y, z) */
        BATCH_FLOAT(coords[3 * dest + 0], coord_idx[0]);
        BATCH_FLOAT(coords[3 * dest + 1], coord_idx[1]);
        BATCH_FLOAT(coords[3 * dest + 2], coord_idx[2]);

        /* Element symbol -> element index */
        BATCH_LOOKUP(elements[dest], elem_idx, _lookup_element, elem_buf);

        /* Residue_Atom -> atom type index */
        BATCH_LOOKUP2(types[dest], comp_idx, atom_idx, '_', _lookup_atom, scratch);
    }

    return CIF_OK;
}


CifError _execute_batch_group(mmCIF *cif, mmBlockList *blocks,
                               const BatchGroup *group, CifErrorContext *ctx) {
    mmBlock *block = _get_block_by_id(blocks, group->block_id);
    if (block == NULL || block->size == 0) {
        LOG_WARNING("Empty or missing block for batch group");
        return CIF_OK;
    }

    LOG_DEBUG("Executing batch group for block %d: %d fields, %d rows",
              group->block_id, group->field_count, block->size);

    /* Pre-compute all attribute indices */
    int attr_indices[MAX_BATCH_ATTRS];
    for (int a = 0; a < group->attr_count; a++) {
        attr_indices[a] = _get_attr_index(block, group->attrs[a], ctx);
        if (attr_indices[a] == BAD_IX) {
            CIF_SET_ERROR(ctx, CIF_ERR_ATTR,
                "Missing batch attribute '%s'", group->attrs[a]);
            return CIF_ERR_ATTR;
        }
    }

    /* Use fused loop for ATOM block when is_nonpoly is available */
    if (group->block_id == BLOCK_ATOM && cif->is_nonpoly != NULL) {
        /* Find attribute indices for coords, elements, types */
        int coord_idx[3] = {-1, -1, -1};
        int elem_idx = -1, comp_idx = -1, atom_idx = -1;

        for (int f = 0; f < group->field_count; f++) {
            FieldId fid = group->fields[f];
            if (fid == FIELD_COORDS) {
                coord_idx[0] = attr_indices[group->attr_map[f][0]];
                coord_idx[1] = attr_indices[group->attr_map[f][1]];
                coord_idx[2] = attr_indices[group->attr_map[f][2]];
            } else if (fid == FIELD_ELEMENTS) {
                elem_idx = attr_indices[group->attr_map[f][0]];
            } else if (fid == FIELD_TYPES) {
                comp_idx = attr_indices[group->attr_map[f][0]];
                atom_idx = attr_indices[group->attr_map[f][1]];
            }
        }

        LOG_DEBUG("Using fused batch loop for ATOM block");
        return _batch_atom_fields_fused(cif, block, coord_idx,
                                        elem_idx, comp_idx, atom_idx, ctx);
    }

    /* Scratch buffer for combined lookups */
    char scratch[MAX_INLINE_BUFFER];

    /* Pre-compute field indices outside the row loop */
    int all_field_indices[MAX_BATCH_FIELDS][MAX_BATCH_ATTRS];
    for (int f = 0; f < group->field_count; f++) {
        const FieldDef *def = &FIELDS[group->fields[f]];
        int field_attr_count = _count_attrs(def->attrs);
        for (int a = 0; a < field_attr_count; a++) {
            all_field_indices[f][a] = attr_indices[group->attr_map[f][a]];
        }
    }

    /* Generic path: Single pass over all rows */
    for (int row = 0; row < block->size; row++) {
        cif->write_dest = row;

        /* Call each field's batch callback */
        for (int f = 0; f < group->field_count; f++) {
            FieldId fid = group->fields[f];
            const FieldDef *def = &FIELDS[fid];

#ifdef CIFFY_PROFILE
            struct timespec _t_start, _t_end;
            clock_gettime(CLOCK_MONOTONIC, &_t_start);
#endif
            def->batch_row_func(cif, block, row, all_field_indices[f], scratch);

#ifdef CIFFY_PROFILE
            clock_gettime(CLOCK_MONOTONIC, &_t_end);
            double elapsed = (_t_end.tv_sec - _t_start.tv_sec) +
                           (_t_end.tv_nsec - _t_start.tv_nsec) / 1e9;
            switch (fid) {
                case FIELD_COORDS:   g_profile.batch_coords += elapsed; break;
                case FIELD_ELEMENTS: g_profile.batch_elements += elapsed; break;
                case FIELD_TYPES:    g_profile.batch_types += elapsed; break;
                default: break;
            }
#endif
        }
    }

    LOG_DEBUG("Batch group execution complete");
    return CIF_OK;
}

bool _field_executed(FieldId fid, const bool *executed) {
    return executed[fid];
}


/* ============================================================================
 * STORAGE AND ALLOCATION
 * Generic functions for storing values and allocating arrays.
 * ============================================================================ */

void _store_int(mmCIF *cif, const FieldDef *def, int value) {
    if (def->storage_type != STORAGE_INT) {
        LOG_WARNING("_store_int called on non-int field '%s'", def->name);
        return;
    }
    int *dest = (int *)((char *)cif + def->storage_offset);
    *dest = value;
    LOG_DEBUG("Stored %s = %d", def->name, value);
}

void _store_ptr(mmCIF *cif, const FieldDef *def, void *ptr) {
    void **dest = (void **)((char *)cif + def->storage_offset);

    switch (def->storage_type) {
        case STORAGE_INT_PTR:
        case STORAGE_FLOAT_PTR:
        case STORAGE_STR_ARRAY:
            *dest = ptr;
            LOG_DEBUG("Stored %s = %p", def->name, ptr);
            break;
        default:
            LOG_WARNING("_store_ptr called on incompatible field '%s'", def->name);
            break;
    }
}

int _get_alloc_size(const mmCIF *cif, const FieldDef *def) {
    if (def->size_source == SIZE_NONE || def->element_size == 0) {
        return 0;
    }

    int count = 0;
    switch (def->size_source) {
        case SIZE_ATOMS:    count = cif->atoms;    break;
        case SIZE_CHAINS:   count = cif->chains;   break;
        case SIZE_RESIDUES: count = cif->residues; break;
        default:            return 0;
    }

    return count * def->elements_per_item;
}

CifError _allocate_field_arrays(mmCIF *cif, CifErrorContext *ctx) {
    LOG_DEBUG("Allocating field arrays");

    for (int i = 0; i < FIELD_COUNT; i++) {
        const FieldDef *def = &FIELDS[i];

        if (def->size_source == SIZE_NONE || def->element_size == 0) {
            continue;
        }

        int count = _get_alloc_size(cif, def);
        if (count <= 0) {
            LOG_WARNING("Invalid allocation size for field '%s'", def->name);
            continue;
        }

        void *ptr = calloc((size_t)count, def->element_size);
        if (ptr == NULL) {
            CIF_SET_ERROR(ctx, CIF_ERR_ALLOC,
                "Failed to allocate %s array (%d elements)", def->name, count);
            return CIF_ERR_ALLOC;
        }

        _store_ptr(cif, def, ptr);
        LOG_DEBUG("Allocated %s: %d elements of size %zu",
                  def->name, count, def->element_size);
    }

    return CIF_OK;
}
