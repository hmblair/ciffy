/**
 * @file registry.h
 * @brief Declarative block and field registry for mmCIF parsing.
 *
 * This header defines the registry system that allows declarative specification
 * of mmCIF blocks, fields, and their dependencies. The parsing order is computed
 * via topological sort based on declared field dependencies.
 *
 * Storage mapping and automatic allocation are handled via FieldDef metadata,
 * eliminating the need for switch statements when adding new fields.
 *
 * ============================================================================
 * HOW TO ADD A NEW BLOCK
 * ============================================================================
 *
 * 1. Add to BLOCK_LIST macro (io.h):
 *
 *        X(ENTITY, "_entity.", false)  // false = optional
 *
 *    This auto-generates:
 *    - BLOCK_ENTITY enum value
 *    - Entry in BLOCKS[] array
 *    - Slot in mmBlockList.b[] array
 *
 * ============================================================================
 * HOW TO ADD A NEW METADATA FIELD (int)
 * ============================================================================
 *
 * Example: Adding entity_count field
 *
 * 1. Add to FieldId enum (registry.h):
 *
 *        FIELD_ENTITY_COUNT,  // cif->entity_count
 *
 * 2. Add storage to mmCIF struct (parser.h):
 *
 *        int entity_count;
 *
 * 3. Add to FIELDS[] array (registry.c) with storage info:
 *
 *        { FIELD_ENTITY_COUNT, "entity_count", BLOCK_ENTITY, OP_COUNT_UNIQUE,
 *          ATTR_ENTITY_ID, NULL, NULL, false, NULL,
 *          offsetof(mmCIF, entity_count), STORAGE_INT,
 *          SIZE_NONE, 0, 0 },
 *
 * 4. Update _c_to_py() to export to Python (module.c)
 *
 * That's it! No switch statements needed - storage_offset handles assignment.
 *
 * ============================================================================
 * HOW TO ADD A NEW BATCH-PARSED FIELD (array)
 * ============================================================================
 *
 * Batch fields are parsed together in a single pass over the block data.
 * Arrays are automatically allocated based on size_source metadata.
 *
 * Example: Adding a new per-atom field (e.g., b_factor)
 *
 * 1. Add to FieldId enum (registry.h):
 *
 *        FIELD_B_FACTOR,  // cif->b_factors - B-factor per atom
 *
 * 2. Add storage to mmCIF struct (parser.h):
 *
 *        float *b_factors;
 *
 * 3. Add attribute constant (registry.c):
 *
 *        static const char *ATTR_B_FACTOR[] = { "B_iso_or_equiv", NULL };
 *
 * 4. Add batch row callback (registry.c):
 *
 *        static void _batch_b_factor(mmCIF *cif, mmBlock *block,
 *                                    int row, const int *idx, char *scratch) {
 *            (void)scratch;
 *            cif->b_factors[row] = _parse_float_inline(block, row, idx[0]);
 *        }
 *
 * 5. Add to FIELDS[] array with storage AND allocation info (registry.c):
 *
 *        { FIELD_B_FACTOR, "b_factors", BLOCK_ATOM, OP_COMPUTE,
 *          ATTR_B_FACTOR, DEP_ATOMS, NULL, true, _batch_b_factor,
 *          offsetof(mmCIF, b_factors), STORAGE_FLOAT_PTR,
 *          SIZE_ATOMS, sizeof(float), 1 },
 *
 * 6. Update _c_to_py() to export to Python (module.c)
 *
 * The array is automatically allocated by _allocate_field_arrays() based on:
 *   - size_source: SIZE_ATOMS means size = cif->atoms
 *   - element_size: sizeof(float)
 *   - elements_per_item: 1 (use 3 for xyz coordinates)
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
 *     { FIELD_FOO, "foo", ..., DEP_ENTITY, ... },
 *
 * Circular dependencies are detected and reported as errors.
 */

#ifndef _CIFFY_REGISTRY_H
#define _CIFFY_REGISTRY_H

#include <stdbool.h>
#include "../error.h"
#include "io.h"  /* For mmBlock */

/* ============================================================================
 * BLOCK REGISTRY
 * Blocks are independent - just parsed and counted, no dependencies.
 *
 * Note: BlockId enum and BLOCK_LIST macro are defined in io.h since they're
 * needed by both parser.h (for mmBlockList) and registry.h.
 * ============================================================================ */

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
    FIELD_DESCRIPTIONS,  /**< cif->descriptions - entity description per chain (optional) */
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

/**
 * @brief Storage type for field values.
 *
 * Specifies the C type of the storage location in mmCIF struct.
 * Used by generic storage functions to correctly assign values.
 */
typedef enum {
    STORAGE_NONE = 0,    /**< No automatic storage (custom handling) */
    STORAGE_INT,         /**< int field (e.g., cif->chains) */
    STORAGE_INT_PTR,     /**< int* field (e.g., cif->sequence) */
    STORAGE_FLOAT_PTR,   /**< float* field (e.g., cif->coordinates) */
    STORAGE_STR_ARRAY,   /**< char** field (e.g., cif->names) */
} StorageType;

/**
 * @brief Size source for array allocation.
 *
 * Specifies which count field determines the array size.
 */
typedef enum {
    SIZE_NONE = 0,       /**< No allocation needed */
    SIZE_ATOMS,          /**< Size = cif->atoms */
    SIZE_CHAINS,         /**< Size = cif->chains */
    SIZE_RESIDUES,       /**< Size = cif->residues */
} SizeSource;

/**
 * @brief Python export type for automatic dict generation.
 *
 * Specifies how to convert a field to a Python object.
 */
typedef enum {
    PY_NONE = 0,         /**< Not exported to Python */
    PY_INT,              /**< Scalar int -> Python int */
    PY_STRING,           /**< char* -> Python str */
    PY_1D_INT,           /**< int* -> 1D numpy int32 array */
    PY_1D_FLOAT,         /**< float* -> 1D numpy float32 array */
    PY_2D_FLOAT,         /**< float* -> 2D numpy array [size, elements_per_item] */
    PY_STR_LIST,         /**< char** -> Python list of str */
} PyExportType;

/* Forward declarations - defined in parser.h */
typedef struct mmCIF mmCIF;
typedef struct mmBlockList mmBlockList;

/**
 * @brief Custom computation function signature.
 */
typedef CifError (*ParseFunc)(mmCIF *cif, mmBlockList *blocks,
                              const void *def, CifErrorContext *ctx);

/**
 * @brief Batch row parsing callback signature.
 *
 * Called once per row during batch iteration. Implementations should parse
 * their specific field(s) from the block row and store results in cif.
 *
 * @param cif Output structure to populate
 * @param block Source block being iterated (non-const for inline parser functions)
 * @param row Current row index (0-based)
 * @param attr_indices Pre-computed attribute column indices
 * @param scratch Scratch buffer for combined lookups (MAX_INLINE_BUFFER size)
 */
typedef void (*BatchRowFunc)(mmCIF *cif, mmBlock *block, int row,
                             const int *attr_indices, char *scratch);

/**
 * @brief Field definition structure.
 *
 * Declares a field's source block, operation, required attributes,
 * and dependencies on other fields. For batch-parsed fields, also
 * includes the per-row callback.
 *
 * Storage mapping fields allow generic value assignment without
 * switch statements on field ID. Allocation fields enable automatic
 * array allocation before batch parsing.
 */
typedef struct {
    FieldId      id;            /**< Field identifier */
    const char  *name;          /**< Name for debugging/logging */
    BlockId      source_block;  /**< Which block provides the data */
    ParseOp      operation;     /**< How to compute this field */
    const char **attrs;         /**< Required attribute names (NULL-terminated) */
    const FieldId *depends_on;  /**< Field dependencies (-1 terminated) */
    ParseFunc    parse_func;    /**< Custom function for OP_COMPUTE */

    /* Batch parsing support */
    bool         batchable;     /**< Can be batched with other fields from same block */
    BatchRowFunc batch_row_func;/**< Per-row callback for batch iteration */

    /* Storage mapping - eliminates switch statements */
    size_t       storage_offset;/**< offsetof(mmCIF, field) for direct assignment */
    StorageType  storage_type;  /**< Type of storage location */

    /* Allocation support - enables automatic array allocation */
    SizeSource   size_source;   /**< What count determines array size */
    size_t       element_size;  /**< sizeof(element), 0 if no allocation */
    int          elements_per_item; /**< Elements per item (3 for coords, 1 for others) */

    /* Python export - enables automatic dict generation */
    PyExportType py_export;     /**< How to convert to Python object */
    const char  *py_name;       /**< Key name in returned dict (NULL = use 'name') */
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
 * BATCH EXECUTION
 * Runtime batch grouping and execution for fields from the same block.
 * ============================================================================ */

/**
 * @brief Maximum number of fields that can be batched together.
 */
#define MAX_BATCH_FIELDS 16

/**
 * @brief Maximum number of attributes across all fields in a batch.
 */
#define MAX_BATCH_ATTRS 32

/**
 * @brief Batch group for fields from the same block.
 *
 * Groups batchable fields together for single-pass iteration.
 */
typedef struct {
    BlockId      block_id;                    /**< Source block for all fields */
    int          field_count;                 /**< Number of fields in batch */
    FieldId      fields[MAX_BATCH_FIELDS];    /**< Field IDs in this batch */
    int          attr_count;                  /**< Total unique attributes */
    const char  *attrs[MAX_BATCH_ATTRS];      /**< Unique attribute names */
    int          attr_map[MAX_BATCH_FIELDS][MAX_BATCH_ATTRS]; /**< Per-field attr indices */
} BatchGroup;


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

/**
 * @brief Compute batch groups from batchable fields.
 *
 * Groups batchable fields by source block for efficient single-pass parsing.
 * Called once before plan execution.
 *
 * @param groups Output array of batch groups (caller provides storage)
 * @param group_count Output: number of batch groups created
 * @param max_groups Maximum groups allowed
 */
void _compute_batch_groups(BatchGroup *groups, int *group_count, int max_groups);

/**
 * @brief Execute a batch group.
 *
 * Pre-computes attribute indices and iterates over all rows, calling each
 * field's batch_row_func for efficient single-pass parsing.
 *
 * @param cif Output structure to populate
 * @param blocks Block collection
 * @param group Batch group to execute
 * @param ctx Error context
 * @return CIF_OK on success, error code on failure
 */
CifError _execute_batch_group(mmCIF *cif, mmBlockList *blocks,
                               const BatchGroup *group, CifErrorContext *ctx);

/**
 * @brief Check if a field has been executed via batch.
 *
 * Used by _execute_plan to skip fields already parsed in a batch.
 *
 * @param fid Field identifier
 * @param executed Bitmap of executed fields
 * @return true if field was already executed
 */
bool _field_executed(FieldId fid, const bool *executed);


/* ============================================================================
 * STORAGE AND ALLOCATION API
 * Generic functions for storing values and allocating arrays.
 * ============================================================================ */

/**
 * @brief Store an integer value into mmCIF using field definition.
 *
 * Uses storage_offset and storage_type to assign value directly
 * without switch statements.
 *
 * @param cif Output structure
 * @param def Field definition with storage info
 * @param value Integer value to store
 */
void _store_int(mmCIF *cif, const FieldDef *def, int value);

/**
 * @brief Store a pointer value into mmCIF using field definition.
 *
 * Uses storage_offset and storage_type to assign pointer directly.
 * Works for int*, float*, and char**.
 *
 * @param cif Output structure
 * @param def Field definition with storage info
 * @param ptr Pointer value to store
 */
void _store_ptr(mmCIF *cif, const FieldDef *def, void *ptr);

/**
 * @brief Get the size for array allocation based on field definition.
 *
 * @param cif Structure with count fields already populated
 * @param def Field definition with size_source
 * @return Array size (count * elements_per_item), or 0 if no allocation
 */
int _get_alloc_size(const mmCIF *cif, const FieldDef *def);

/**
 * @brief Allocate arrays for all fields that require allocation.
 *
 * Iterates through FIELDS[] and allocates arrays based on size_source,
 * element_size, and elements_per_item. Must be called after count
 * fields (atoms, chains, residues) are populated.
 *
 * @param cif Structure with count fields already populated
 * @param ctx Error context
 * @return CIF_OK on success, CIF_ERR_ALLOC on failure
 */
CifError _allocate_field_arrays(mmCIF *cif, CifErrorContext *ctx);

#endif /* _CIFFY_REGISTRY_H */
