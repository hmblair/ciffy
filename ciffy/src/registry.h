/**
 * @file registry.h
 * @brief Declarative block and field registry for mmCIF parsing.
 *
 * This header defines the registry system that allows declarative specification
 * of mmCIF blocks, fields, and their dependencies. The parsing order is computed
 * via topological sort based on declared field dependencies.
 *
 * ============================================================================
 * HOW TO ADD A NEW BLOCK
 * ============================================================================
 *
 * Example: Adding _entity block
 *
 * 1. Add to BlockId enum (registry.h):
 *
 *        BLOCK_ENTITY,    // _entity - molecular entities
 *
 * 2. Add to BLOCKS[] array (registry.c):
 *
 *        { BLOCK_ENTITY, "_entity.", false },  // false = optional
 *
 * 3. Add slot to mmBlockList struct (parser.h):
 *
 *        mmBlock entity;
 *
 * 4. Add case to _get_block_by_id() (registry.c):
 *
 *        case BLOCK_ENTITY: return &blocks->entity;
 *
 * ============================================================================
 * HOW TO ADD A NEW METADATA FIELD
 * ============================================================================
 *
 * Example: Adding entity_count field
 *
 * 1. Add to FieldId enum (registry.h):
 *
 *        FIELD_ENTITY_COUNT,  // cif->entity_count
 *
 * 2. Add attribute constant (registry.c, if needed):
 *
 *        static const char *ATTR_ENTITY_ID[] = { "id", NULL };
 *
 * 3. Add to FIELDS[] array (registry.c):
 *
 *        { FIELD_ENTITY_COUNT, "entity_count", BLOCK_ENTITY, OP_COUNT_UNIQUE,
 *          ATTR_ENTITY_ID, NULL, NULL },
 *
 *    Or with dependencies:
 *
 *        { FIELD_ENTITY_COUNT, "entity_count", BLOCK_ENTITY, OP_COUNT_UNIQUE,
 *          ATTR_ENTITY_ID, DEP_CHAINS, NULL },  // runs after FIELD_CHAINS
 *
 * 4. Add case to appropriate _op_* function (registry.c):
 *
 *        case FIELD_ENTITY_COUNT: cif->entity_count = count; break;
 *
 * 5. Add storage to mmCIF struct (parser.h):
 *
 *        int entity_count;
 *
 * 6. Update _c_to_py() to export to Python (module.c)
 *
 * ============================================================================
 * AVAILABLE OPERATIONS (ParseOp)
 * ============================================================================
 *
 * OP_BLOCK_SIZE    - field = block.size (int)
 * OP_COUNT_UNIQUE  - field = count of unique consecutive values (int)
 * OP_GET_UNIQUE    - field = array of unique strings (char**)
 * OP_COUNT_BY_GROUP- field = count of items per group (int*)
 * OP_LOOKUP        - field = hash table lookup results (int*)
 * OP_PARSE_FLOAT   - field = parsed float values (float*) [batch-only]
 * OP_COMPUTE       - field = custom computation via parse_func
 *
 * ============================================================================
 * DEPENDENCY SYSTEM
 * ============================================================================
 *
 * Fields can declare dependencies on other fields. The topological sort
 * ensures dependencies are parsed before dependents.
 *
 * Declare dependency arrays (registry.c):
 *
 *     static const FieldId DEP_ENTITY[] = { FIELD_ENTITY_COUNT, -1 };
 *
 * Use in field definition:
 *
 *     { FIELD_FOO, "foo", BLOCK_X, OP_Y, ATTR_Z, DEP_ENTITY, NULL },
 *
 * Circular dependencies are detected and reported as errors.
 */

#ifndef _CIFFY_REGISTRY_H
#define _CIFFY_REGISTRY_H

#include <stdbool.h>
#include "error.h"
#include "io.h"  /* For mmBlock */

/* ============================================================================
 * BLOCK REGISTRY
 * Blocks are independent - just parsed and counted, no dependencies.
 * ============================================================================ */

/**
 * @brief Block identifier enum.
 *
 * Each value corresponds to an mmCIF category block.
 */
typedef enum {
    BLOCK_ATOM,        /**< _atom_site - atomic coordinates */
    BLOCK_POLY,        /**< _pdbx_poly_seq_scheme - polymer sequence */
    BLOCK_CHAIN,       /**< _struct_asym - chain definitions */
    BLOCK_NONPOLY,     /**< _pdbx_nonpoly_scheme - non-polymer entities */
    BLOCK_CONN,        /**< _struct_conn - connectivity/bonds */
    BLOCK_ENTITY_POLY, /**< _entity_poly - polymer entity types (RNA/DNA/protein) */
    BLOCK_ENTITY,      /**< _entity - molecular entity types (polymer/non-polymer) */
    BLOCK_COUNT        /**< Total number of block types */
} BlockId;

/**
 * @brief Block definition structure.
 *
 * Declares a block's category prefix and whether it's required.
 */
typedef struct {
    BlockId      id;         /**< Block identifier */
    const char  *category;   /**< mmCIF category prefix (e.g., "_atom_site.") */
    bool         required;   /**< Whether block must exist for valid parse */
} BlockDef;


/* ============================================================================
 * FIELD REGISTRY
 * Fields declare dependencies on other fields. Parsing order is computed
 * via topological sort.
 * ============================================================================ */

/**
 * @brief Field identifier enum.
 *
 * Each value corresponds to a field in the mmCIF output structure.
 */
typedef enum {
    FIELD_MODELS,        /**< cif->models - model count */
    FIELD_CHAINS,        /**< cif->chains - chain count */
    FIELD_RESIDUES,      /**< cif->residues - residue count */
    FIELD_ATOMS,         /**< cif->atoms - atom count per model */
    FIELD_NAMES,         /**< cif->names - chain name strings */
    FIELD_STRANDS,       /**< cif->strands - strand ID strings */
    FIELD_SEQUENCE,      /**< cif->sequence - residue type indices */
    FIELD_COORDS,        /**< cif->coordinates - x,y,z positions */
    FIELD_TYPES,         /**< cif->types - atom type indices */
    FIELD_ELEMENTS,      /**< cif->elements - element type indices */
    FIELD_RES_PER_CHAIN, /**< cif->res_per_chain - residue counts per chain */
    FIELD_ATOMS_PER_RES, /**< cif->atoms_per_res - atom counts per residue */
    FIELD_MOL_TYPES,     /**< cif->molecule_types - molecule type per chain */
    FIELD_COUNT          /**< Total number of field types */
} FieldId;

/**
 * @brief Parsing operation enum.
 *
 * Specifies how a field's value should be computed from block data.
 */
typedef enum {
    OP_BLOCK_SIZE,       /**< field = block.size */
    OP_COUNT_UNIQUE,     /**< field = count unique values in attribute */
    OP_GET_UNIQUE,       /**< field = extract unique strings from attribute */
    OP_LOOKUP,           /**< field = hash table lookup on attribute(s) */
    OP_PARSE_FLOAT,      /**< field = parse floats from attribute(s) */
    OP_COUNT_BY_GROUP,   /**< field = count items per group */
    OP_COMPUTE,          /**< field = custom computation via function pointer */
} ParseOp;

/* Forward declarations - defined in parser.h */
typedef struct mmCIF mmCIF;
typedef struct mmBlockList mmBlockList;

/**
 * @brief Custom computation function signature.
 */
typedef CifError (*ParseFunc)(mmCIF *cif, mmBlockList *blocks,
                              const void *def, CifErrorContext *ctx);

/**
 * @brief Field definition structure.
 *
 * Declares a field's source block, operation, required attributes,
 * and dependencies on other fields.
 */
typedef struct {
    FieldId      id;            /**< Field identifier */
    const char  *name;          /**< Name for debugging/logging */
    BlockId      source_block;  /**< Which block provides the data */
    ParseOp      operation;     /**< How to compute this field */
    const char **attrs;         /**< Required attribute names (NULL-terminated) */
    const FieldId *depends_on;  /**< Field dependencies (-1 terminated) */
    ParseFunc    parse_func;    /**< Custom function for OP_COMPUTE */
} FieldDef;


/* ============================================================================
 * PARSE PLAN
 * Result of topological sort - execution order for fields.
 * ============================================================================ */

/**
 * @brief Computed parsing order from dependency resolution.
 *
 * Contains the topologically sorted field execution order.
 */
typedef struct {
    FieldId order[FIELD_COUNT]; /**< Fields in execution order */
    int     count;              /**< Number of fields to execute */
} ParsePlan;


/* ============================================================================
 * REGISTRY API
 * Functions for accessing the registry and computing parse plans.
 * ============================================================================ */

/**
 * @brief Get the block definitions array.
 *
 * @return Pointer to static BLOCKS[] array
 */
const BlockDef *_get_blocks(void);

/**
 * @brief Get the field definitions array.
 *
 * @return Pointer to static FIELDS[] array
 */
const FieldDef *_get_fields(void);

/**
 * @brief Compute parsing order via topological sort.
 *
 * Resolves field dependencies and produces an execution order.
 * Detects circular dependencies.
 *
 * @param plan Output parse plan to populate
 * @param ctx Error context, populated on circular dependency
 * @return CIF_OK on success, CIF_ERR_PARSE on cycle
 */
CifError _plan_parse(ParsePlan *plan, CifErrorContext *ctx);

/**
 * @brief Execute a computed parse plan.
 *
 * Parses fields in dependency order using registered operations.
 *
 * @param cif Output structure to populate
 * @param blocks Parsed block collection
 * @param plan Pre-computed parse plan
 * @param ctx Error context for failures
 * @return CIF_OK on success, error code on failure
 */
CifError _execute_plan(mmCIF *cif, mmBlockList *blocks,
                       const ParsePlan *plan, CifErrorContext *ctx);

/**
 * @brief Validate that required blocks exist.
 *
 * Checks each block marked as required in the registry.
 *
 * @param blocks Block collection to validate
 * @param ctx Error context for missing blocks
 * @return CIF_OK if all required blocks present, CIF_ERR_PARSE otherwise
 */
CifError _validate_blocks_registry(mmBlockList *blocks, CifErrorContext *ctx);

/**
 * @brief Get a block from the block list by ID.
 *
 * @param blocks Block collection
 * @param id Block identifier
 * @return Pointer to the block, or NULL if invalid ID
 */
mmBlock *_get_block_by_id(mmBlockList *blocks, BlockId id);

#endif /* _CIFFY_REGISTRY_H */
