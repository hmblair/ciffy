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


/* ============================================================================
 * BLOCK DEFINITIONS
 * Static table of mmCIF blocks to parse.
 * ============================================================================ */

static const BlockDef BLOCKS[] = {
    { BLOCK_ATOM,    "_atom_site.",            true  },
    { BLOCK_POLY,    "_pdbx_poly_seq_scheme.", true  },
    { BLOCK_CHAIN,   "_struct_asym.",          true  },
    { BLOCK_NONPOLY, "_pdbx_nonpoly_scheme.",  false },
    { BLOCK_CONN,    "_struct_conn.",          false },
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
static const char *ATTR_COORDS[]        = { "Cartn_x", "Cartn_y", "Cartn_z", NULL };
static const char *ATTR_ELEMENT[]       = { "type_symbol", NULL };
static const char *ATTR_ATOM_TYPE[]     = { "label_comp_id", "label_atom_id", NULL };


/* ============================================================================
 * DEPENDENCY ARRAYS
 * Terminated with -1 sentinel.
 * ============================================================================ */

static const FieldId DEP_MODELS[]   = { FIELD_MODELS, -1 };
static const FieldId DEP_CHAINS[]   = { FIELD_CHAINS, -1 };
static const FieldId DEP_ATOMS[]    = { FIELD_ATOMS, -1 };
static const FieldId DEP_RESIDUES[] = { FIELD_RESIDUES, -1 };


/* ============================================================================
 * FIELD DEFINITIONS
 * Declarative specification of fields and their dependencies.
 *
 * Fields are organized by dependency level:
 *   Level 0: Leaf fields (no dependencies)
 *   Level 1: Depend on leaf fields
 *   Level 2+: Deeper dependencies
 *
 * The topological sort will compute the actual execution order.
 * ============================================================================ */

static const FieldDef FIELDS[] = {
    /* ── Level 0: Leaf fields (no dependencies) ─────────────────────────────── */

    { FIELD_CHAINS,   "chains",   BLOCK_CHAIN, OP_BLOCK_SIZE,
      NULL, NULL, NULL },

    { FIELD_RESIDUES, "residues", BLOCK_POLY,  OP_BLOCK_SIZE,
      NULL, NULL, NULL },

    { FIELD_MODELS,   "models",   BLOCK_ATOM,  OP_COUNT_UNIQUE,
      ATTR_MODEL, NULL, NULL },

    /* ── Level 1: Depends on models ─────────────────────────────────────────── */

    { FIELD_ATOMS, "atoms", BLOCK_ATOM, OP_COMPUTE,
      NULL, DEP_MODELS, NULL },  /* atoms = atom_site.size / models */

    /* ── Level 1: Depends on chains ─────────────────────────────────────────── */

    { FIELD_NAMES, "names", BLOCK_CHAIN, OP_GET_UNIQUE,
      ATTR_CHAIN_ID, DEP_CHAINS, NULL },

    { FIELD_RES_PER_CHAIN, "res_per_chain", BLOCK_POLY, OP_COUNT_BY_GROUP,
      ATTR_RES_PER_CHAIN, DEP_CHAINS, NULL },

    { FIELD_STRANDS, "strands", BLOCK_POLY, OP_GET_UNIQUE,
      ATTR_STRAND_ID, DEP_CHAINS, NULL },

    /* ── Level 1: Depends on residues ───────────────────────────────────────── */

    { FIELD_SEQUENCE, "sequence", BLOCK_POLY, OP_LOOKUP,
      ATTR_RESIDUE_NAME, DEP_RESIDUES, NULL },

    /* ── Level 2: Depends on atoms ──────────────────────────────────────────── */

    { FIELD_COORDS, "coordinates", BLOCK_ATOM, OP_PARSE_FLOAT,
      ATTR_COORDS, DEP_ATOMS, NULL },

    { FIELD_ELEMENTS, "elements", BLOCK_ATOM, OP_LOOKUP,
      ATTR_ELEMENT, DEP_ATOMS, NULL },

    { FIELD_TYPES, "types", BLOCK_ATOM, OP_LOOKUP,
      ATTR_ATOM_TYPE, DEP_ATOMS, NULL },

    { FIELD_ATOMS_PER_RES, "atoms_per_res", BLOCK_ATOM, OP_COMPUTE,
      NULL, DEP_ATOMS, NULL },  /* computed from atom parsing */
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
 *
 * @param fid Field to visit
 * @param visited Array tracking visited fields
 * @param in_stack Array tracking fields in current recursion stack (cycle detection)
 * @param plan Output plan being built
 * @param ctx Error context for cycle detection
 * @return CIF_OK or CIF_ERR_PARSE on cycle
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
        return CIF_OK;  /* Already processed */
    }

    in_stack[fid] = true;

    /* Visit dependencies first */
    const FieldId *deps = FIELDS[fid].depends_on;
    if (deps != NULL) {
        for (int i = 0; deps[i] != -1; i++) {
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
        case BLOCK_ATOM:    return &blocks->atom;
        case BLOCK_POLY:    return &blocks->poly;
        case BLOCK_CHAIN:   return &blocks->chain;
        case BLOCK_NONPOLY: return &blocks->nonpoly;
        case BLOCK_CONN:    return &blocks->conn;
        default:            return NULL;
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
 * EXECUTE PLAN (STUB)
 *
 * This is a placeholder for Batch 3. Currently the existing _fill_cif()
 * implementation is used. This will be replaced with the registry-driven
 * execution in Batch 3.
 * ============================================================================ */

CifError _execute_plan(mmCIF *cif, mmBlockList *blocks,
                       const ParsePlan *plan, CifErrorContext *ctx) {
    (void)cif;
    (void)blocks;
    (void)plan;

    LOG_WARNING("_execute_plan() is a stub - using legacy _fill_cif() instead");
    CIF_SET_ERROR(ctx, CIF_ERR_PARSE,
        "_execute_plan() not yet implemented - use _fill_cif()");
    return CIF_ERR_PARSE;
}
