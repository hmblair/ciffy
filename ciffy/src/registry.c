/**
 * @file registry.c
 * @brief Block and field registry implementation.
 *
 * Contains the declarative definitions for mmCIF blocks and fields,
 * plus the topological sort algorithm for computing parse order.
 */

#include "registry.h"
#include "parser.h"
#include "log.h"

#include <stddef.h>
#include <stdlib.h>
#include <string.h>


/* ============================================================================
 * BLOCK DEFINITIONS
 * Static table of mmCIF blocks to parse.
 * ============================================================================ */

static const BlockDef BLOCKS[] = {
    { BLOCK_ATOM,        "_atom_site.",            true  },
    { BLOCK_POLY,        "_pdbx_poly_seq_scheme.", true  },
    { BLOCK_CHAIN,       "_struct_asym.",          true  },
    { BLOCK_NONPOLY,     "_pdbx_nonpoly_scheme.",  false },
    { BLOCK_CONN,        "_struct_conn.",          false },
    { BLOCK_ENTITY_POLY, "_entity_poly.",          false }, /* molecule types */
    { BLOCK_ENTITY,      "_entity.",               false }, /* entity types (polymer/non-polymer) */
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


/* ============================================================================
 * BATCH ROW CALLBACKS
 * Per-row parsing functions for batch-parsed fields.
 * ============================================================================ */

/**
 * @brief Parse coordinates for a single row.
 * attr_indices: [0]=x, [1]=y, [2]=z
 */
static void _batch_coords(mmCIF *cif, mmBlock *block, int row,
                          const int *idx, char *scratch) {
    (void)scratch;
    cif->coordinates[3 * row + 0] = _parse_float_inline(block, row, idx[0]);
    cif->coordinates[3 * row + 1] = _parse_float_inline(block, row, idx[1]);
    cif->coordinates[3 * row + 2] = _parse_float_inline(block, row, idx[2]);
}

/**
 * @brief Parse element type for a single row.
 * attr_indices: [0]=type_symbol
 */
static void _batch_elements(mmCIF *cif, mmBlock *block, int row,
                            const int *idx, char *scratch) {
    (void)scratch;
    cif->elements[row] = _lookup_inline(block, row, idx[0], _lookup_element);
}

/**
 * @brief Parse atom type for a single row.
 * attr_indices: [0]=label_comp_id, [1]=label_atom_id
 */
static void _batch_types(mmCIF *cif, mmBlock *block, int row,
                         const int *idx, char *scratch) {
    cif->types[row] = _lookup_double_inline(block, row, idx[0], idx[1],
                                            _lookup_atom, scratch);
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
 * The array order must match the enum order in registry.h. */
static const FieldDef FIELDS[] = {
    /* FIELD_MODELS = 0 */
    { FIELD_MODELS,   "models",   BLOCK_ATOM,  OP_COUNT_UNIQUE,
      ATTR_MODEL, NULL, NULL, false, NULL },

    /* FIELD_CHAINS = 1 */
    { FIELD_CHAINS,   "chains",   BLOCK_CHAIN, OP_BLOCK_SIZE,
      NULL, NULL, NULL, false, NULL },

    /* FIELD_RESIDUES = 2 */
    { FIELD_RESIDUES, "residues", BLOCK_POLY,  OP_BLOCK_SIZE,
      NULL, NULL, NULL, false, NULL },

    /* FIELD_ATOMS = 3 - atoms = atom_site.size / models */
    { FIELD_ATOMS, "atoms", BLOCK_ATOM, OP_COMPUTE,
      NULL, DEP_MODELS, NULL, false, NULL },

    /* FIELD_NAMES = 4 */
    { FIELD_NAMES, "names", BLOCK_CHAIN, OP_GET_UNIQUE,
      ATTR_CHAIN_ID, DEP_CHAINS, NULL, false, NULL },

    /* FIELD_STRANDS = 5 */
    { FIELD_STRANDS, "strands", BLOCK_POLY, OP_GET_UNIQUE,
      ATTR_STRAND_ID, DEP_CHAINS, NULL, false, NULL },

    /* FIELD_SEQUENCE = 6 */
    { FIELD_SEQUENCE, "sequence", BLOCK_POLY, OP_LOOKUP,
      ATTR_RESIDUE_NAME, DEP_RESIDUES, NULL, false, NULL },

    /* FIELD_COORDS = 7 - batch parsed */
    { FIELD_COORDS, "coordinates", BLOCK_ATOM, OP_COMPUTE,
      ATTR_COORDS, DEP_MODELS, NULL, true, _batch_coords },

    /* FIELD_TYPES = 8 - batch parsed */
    { FIELD_TYPES, "types", BLOCK_ATOM, OP_COMPUTE,
      ATTR_ATOM_TYPE, DEP_MODELS, NULL, true, _batch_types },

    /* FIELD_ELEMENTS = 9 - batch parsed */
    { FIELD_ELEMENTS, "elements", BLOCK_ATOM, OP_COMPUTE,
      ATTR_ELEMENT, DEP_MODELS, NULL, true, _batch_elements },

    /* FIELD_RES_PER_CHAIN = 10 */
    { FIELD_RES_PER_CHAIN, "res_per_chain", BLOCK_POLY, OP_COUNT_BY_GROUP,
      ATTR_RES_PER_CHAIN, DEP_CHAINS, NULL, false, NULL },

    /* FIELD_ATOMS_PER_RES = 11 - computed via _count_atoms_per_residue */
    { FIELD_ATOMS_PER_RES, "atoms_per_res", BLOCK_ATOM, OP_COMPUTE,
      NULL, NULL, NULL, false, NULL },

    /* FIELD_MOL_TYPES = 12 - parsed directly in _fill_cif() */
    { FIELD_MOL_TYPES, "molecule_types", BLOCK_ENTITY_POLY, OP_COMPUTE,
      NULL, NULL, NULL, false, NULL },
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
    switch (id) {
        case BLOCK_ATOM:        return &blocks->atom;
        case BLOCK_POLY:        return &blocks->poly;
        case BLOCK_CHAIN:       return &blocks->chain;
        case BLOCK_NONPOLY:     return &blocks->nonpoly;
        case BLOCK_CONN:        return &blocks->conn;
        case BLOCK_ENTITY_POLY: return &blocks->entity_poly;
        case BLOCK_ENTITY:      return &blocks->entity;
        default:                return NULL;
    }
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

    switch (def->id) {
        case FIELD_CHAINS:   cif->chains = value;   break;
        case FIELD_RESIDUES: cif->residues = value; break;
        default:
            LOG_WARNING("OP_BLOCK_SIZE: Unknown field %d", def->id);
            break;
    }

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

    LOG_DEBUG("OP_COUNT_UNIQUE: %s = %d (attr=%s)", def->name, count, def->attrs[0]);

    switch (def->id) {
        case FIELD_MODELS:
            if (count == 0) {
                CIF_SET_ERROR(ctx, CIF_ERR_PARSE, "Invalid model count: 0");
                return CIF_ERR_PARSE;
            }
            cif->models = count;
            break;
        default:
            LOG_WARNING("OP_COUNT_UNIQUE: Unknown field %d", def->id);
            break;
    }

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

    switch (def->id) {
        case FIELD_NAMES:   cif->names = result;   break;
        case FIELD_STRANDS: cif->strands = result; break;
        default:
            LOG_WARNING("OP_GET_UNIQUE: Unknown field %d, freeing result", def->id);
            for (int i = 0; i < size; i++) free(result[i]);
            free(result);
            break;
    }

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

    switch (def->id) {
        case FIELD_RES_PER_CHAIN: cif->res_per_chain = result; break;
        default:
            LOG_WARNING("OP_COUNT_BY_GROUP: Unknown field %d, freeing result", def->id);
            free(result);
            break;
    }

    return CIF_OK;
}

/**
 * OP_LOOKUP: Parse values via hash table lookup.
 */
static CifError _op_lookup(mmCIF *cif, mmBlock *block, const FieldDef *def,
                           CifErrorContext *ctx) {
    if (def->attrs == NULL || def->attrs[0] == NULL) {
        CIF_SET_ERROR(ctx, CIF_ERR_PARSE, "OP_LOOKUP requires attribute");
        return CIF_ERR_PARSE;
    }

    /* For now, only FIELD_SEQUENCE uses OP_LOOKUP with residue lookup */
    if (def->id != FIELD_SEQUENCE) {
        LOG_WARNING("OP_LOOKUP: Unsupported field %d", def->id);
        return CIF_OK;
    }

    int *result = _parse_via_lookup(block, _lookup_residue, def->attrs[0], ctx);
    if (result == NULL) return ctx->code;

    LOG_DEBUG("OP_LOOKUP: %s (attr=%s)", def->name, def->attrs[0]);
    cif->sequence = result;

    return CIF_OK;
}

/**
 * OP_COMPUTE: Custom computation for atoms field.
 */
static CifError _op_compute_atoms(mmCIF *cif, mmBlock *block, CifErrorContext *ctx) {
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
    cif->atoms = atom_count;

    LOG_DEBUG("OP_COMPUTE: atoms = %d (from %d total / %d models)",
              cif->atoms, block->size * cif->models, cif->models);

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
            /* Handle FIELD_ATOMS compute */
            if (def->id == FIELD_ATOMS) {
                return _op_compute_atoms(cif, block, ctx);
            }
            /* Skip other OP_COMPUTE fields without batch callbacks */
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

    /* Scratch buffer for combined lookups */
    char scratch[MAX_INLINE_BUFFER];

    /* Single pass over all rows */
    for (int row = 0; row < block->size; row++) {
        /* Call each field's batch callback */
        for (int f = 0; f < group->field_count; f++) {
            FieldId fid = group->fields[f];
            const FieldDef *def = &FIELDS[fid];

            /* Build attr indices for this field from the attr_map */
            int field_indices[MAX_BATCH_ATTRS];
            int field_attr_count = _count_attrs(def->attrs);
            for (int a = 0; a < field_attr_count; a++) {
                field_indices[a] = attr_indices[group->attr_map[f][a]];
            }

            def->batch_row_func(cif, block, row, field_indices, scratch);
        }
    }

    LOG_DEBUG("Batch group execution complete");
    return CIF_OK;
}

bool _field_executed(FieldId fid, const bool *executed) {
    return executed[fid];
}
