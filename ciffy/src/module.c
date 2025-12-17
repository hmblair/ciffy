/**
 * @file module.c
 * @brief Python C extension entry point for ciffy.
 *
 * Provides the _load function that reads mmCIF files and returns
 * parsed molecular structure data as Python/NumPy objects.
 */

/* Define CIFFY_MAIN_MODULE before including headers so pyutils.h knows to import numpy */
#define CIFFY_MAIN_MODULE
#include "module.h"
#include "log.h"
#include "profile.h"
#include "internal/internal_module.h"

#ifdef CIFFY_PROFILE
/* Global profile instance for timing data */
CifProfile g_profile = {0};
#endif


/**
 * @brief Convert CifError to appropriate Python exception.
 *
 * Maps internal error codes to Python exception types and sets
 * the Python error state with detailed message.
 *
 * @param ctx Error context with code and message
 * @param filename Filename for context in error message
 * @return NULL (always, for convenient return)
 */
static PyObject *_set_py_error(CifErrorContext *ctx, const char *filename) {
    switch (ctx->code) {
        case CIF_ERR_ALLOC:
            return PyErr_NoMemory();

        case CIF_ERR_IO:
            return PyErr_Format(PyExc_IOError,
                "I/O error reading '%s': %s", filename, ctx->message);

        case CIF_ERR_PARSE:
            return PyErr_Format(PyExc_ValueError,
                "Parse error in '%s': %s", filename, ctx->message);

        case CIF_ERR_ATTR:
            return PyErr_Format(PyExc_KeyError,
                "Missing attribute in '%s': %s", filename, ctx->message);

        case CIF_ERR_BLOCK:
            return PyErr_Format(PyExc_ValueError,
                "Missing required block in '%s': %s", filename, ctx->message);

        case CIF_ERR_BOUNDS:
            return PyErr_Format(PyExc_IndexError,
                "Index out of bounds in '%s': %s", filename, ctx->message);

        case CIF_ERR_OVERFLOW:
            return PyErr_Format(PyExc_OverflowError,
                "Buffer overflow prevented in '%s': %s", filename, ctx->message);

        case CIF_ERR_LOOKUP:
            return PyErr_Format(PyExc_ValueError,
                "Unknown token in '%s': %s", filename, ctx->message);

        default:
            return PyErr_Format(PyExc_RuntimeError,
                "Unknown error in '%s': %s", filename, ctx->message);
    }
}


/**
 * @brief Create a 1D NumPy int64 array from int data.
 *
 * Converts int32 data to int64 for Python compatibility (indexing, etc).
 * Sets NPY_ARRAY_OWNDATA so NumPy frees the memory when the array
 * is garbage collected.
 */
static PyObject *_init_1d_arr_int(int size, int *data) {
    /* Allocate int64 array */
    int64_t *data64 = malloc(size * sizeof(int64_t));
    if (data64 == NULL) {
        free(data);
        return PyErr_NoMemory();
    }

    /* Copy int32 -> int64 */
    for (int i = 0; i < size; i++) {
        data64[i] = data[i];
    }
    free(data);

    npy_intp dims[1] = {size};
    PyObject *arr = PyArray_SimpleNewFromData(1, dims, NPY_INT64, data64);
    if (arr == NULL) {
        free(data64);
        PyErr_SetString(PyExc_MemoryError, "Failed to create NumPy array");
        return NULL;
    }
    PyArray_ENABLEFLAGS((PyArrayObject *)arr, NPY_ARRAY_OWNDATA);
    return arr;
}


/**
 * @brief Create a 2D NumPy array from float data.
 *
 * Sets NPY_ARRAY_OWNDATA so NumPy frees the memory when the array
 * is garbage collected.
 */
static PyObject *_init_2d_arr_float(int size1, int size2, float *data) {
    npy_intp dims[2] = {size1, size2};
    PyObject *arr = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT, data);
    if (arr == NULL) {
        free(data);
        PyErr_SetString(PyExc_MemoryError, "Failed to create NumPy array");
        return NULL;
    }
    PyArray_ENABLEFLAGS((PyArrayObject *)arr, NPY_ARRAY_OWNDATA);
    return arr;
}


/**
 * @brief Get the size for a field's array based on its size_source.
 */
static int _get_py_size(const mmCIF *cif, const FieldDef *def) {
    switch (def->size_source) {
        case SIZE_ATOMS:    return cif->atoms;
        case SIZE_CHAINS:   return cif->chains;
        case SIZE_RESIDUES: return cif->residues;
        default:            return 0;
    }
}

/**
 * @brief Get the size for fields not using size_source metadata.
 *
 * Some fields (e.g., sequence, res_per_chain) are allocated via
 * custom functions and don't use the size_source system.
 */
static int _get_py_size_fallback(const mmCIF *cif, const FieldDef *def) {
    switch (def->id) {
        case FIELD_SEQUENCE:
        case FIELD_ATOMS_PER_RES:
            return cif->residues;
        case FIELD_NAMES:
        case FIELD_STRANDS:
        case FIELD_RES_PER_CHAIN:
        case FIELD_MOL_TYPES:
            return cif->chains;
        default:
            return _get_py_size(cif, def);
    }
}

/**
 * @brief Export a single field to a Python object.
 *
 * Converts the field data to the appropriate Python type based on py_export.
 *
 * @param cif Parsed mmCIF data
 * @param def Field definition with py_export type
 * @return New Python object, or NULL on error
 */
static PyObject *_export_field(const mmCIF *cif, const FieldDef *def) {
    /* Get pointer to the field data using storage_offset */
    const char *base = (const char *)cif;
    int size = _get_py_size_fallback(cif, def);

    switch (def->py_export) {
        case PY_INT: {
            int value = *(const int *)(base + def->storage_offset);
            return _c_int_to_py_int(value);
        }

        case PY_STRING: {
            char *str = *(char **)(base + def->storage_offset);
            return _c_str_to_py_str(str);
        }

        case PY_1D_INT: {
            int *data = *(int **)(base + def->storage_offset);
            if (data == NULL) Py_RETURN_NONE;  /* metadata_only mode */
            return _init_1d_arr_int(size, data);
        }

        case PY_1D_FLOAT: {
            /* Not currently used, but included for completeness */
            float *data = *(float **)(base + def->storage_offset);
            if (data == NULL) Py_RETURN_NONE;  /* metadata_only mode */
            npy_intp dims[1] = {size};
            PyObject *arr = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT, data);
            if (arr) PyArray_ENABLEFLAGS((PyArrayObject *)arr, NPY_ARRAY_OWNDATA);
            return arr;
        }

        case PY_2D_FLOAT: {
            float *data = *(float **)(base + def->storage_offset);
            if (data == NULL) Py_RETURN_NONE;  /* metadata_only mode */
            return _init_2d_arr_float(size, def->elements_per_item, data);
        }

        case PY_STR_LIST: {
            char **data = *(char ***)(base + def->storage_offset);
            if (data == NULL) Py_RETURN_NONE;  /* metadata_only mode */
            return _c_arr_to_py_list(data, size);
        }

        default:
            return NULL;  /* PY_NONE or unknown */
    }
}

/**
 * @brief Convert mmCIF struct to Python dict.
 *
 * Creates NumPy arrays and Python objects from the parsed C data,
 * using the field registry to determine export types and names.
 * Returns NULL and sets Python exception on error.
 */
static PyObject *_c_to_py(mmCIF cif) {
    PyObject *dict = PyDict_New();
    if (dict == NULL) return NULL;

    const FieldDef *fields = _get_fields();

    /* Export special fields not in the registry */

    /* id: PDB identifier */
    PyObject *py_id = _c_str_to_py_str(cif.id);
    if (py_id == NULL) goto cleanup;
    if (PyDict_SetItemString(dict, "id", py_id) < 0) {
        Py_DECREF(py_id);
        goto cleanup;
    }
    Py_DECREF(py_id);  /* Dict owns the reference now */

    /* polymer_count: number of polymer atoms */
    PyObject *py_polymer = _c_int_to_py_int(cif.polymer);
    if (py_polymer == NULL) goto cleanup;
    if (PyDict_SetItemString(dict, "polymer_count", py_polymer) < 0) {
        Py_DECREF(py_polymer);
        goto cleanup;
    }
    Py_DECREF(py_polymer);

    /* atoms_per_chain: computed outside registry */
    PyObject *py_apc = _init_1d_arr_int(cif.chains, cif.atoms_per_chain);
    if (py_apc == NULL) goto cleanup;
    if (PyDict_SetItemString(dict, "atoms_per_chain", py_apc) < 0) {
        Py_DECREF(py_apc);
        goto cleanup;
    }
    Py_DECREF(py_apc);

    /* Export all registry fields with py_export != PY_NONE */
    for (int i = 0; i < FIELD_COUNT; i++) {
        const FieldDef *def = &fields[i];
        if (def->py_export == PY_NONE) continue;

        /* Get the key name (py_name if set, otherwise name) */
        const char *key = def->py_name ? def->py_name : def->name;

        PyObject *value = _export_field(&cif, def);
        if (value == NULL) goto cleanup;

        if (PyDict_SetItemString(dict, key, value) < 0) {
            Py_DECREF(value);
            goto cleanup;
        }
        Py_DECREF(value);  /* Dict owns the reference now */
    }

    return dict;

cleanup:
    Py_DECREF(dict);
    return NULL;
}


/* Block parsing functions are now in parser.c - see parser.h for declarations */

/* Forward declarations for parsing functions */
extern CifError _precompute_lines(mmBlock *block, CifErrorContext *ctx);
extern void _free_lines(mmBlock *block);
extern int _get_attr_index(mmBlock *block, const char *attr, CifErrorContext *ctx);
extern int _parse_int_inline(mmBlock *block, int line, int index);
extern char *_get_field_ptr(mmBlock *block, int row, int attr_idx, size_t *len);


/**
 * @brief Parse entity descriptions from _entity.pdbx_description.
 *
 * Maps descriptions from entity_id to per-chain via _struct_asym.entity_id.
 *
 * @param cif Output structure (must have chains already populated)
 * @param blocks Parsed blocks containing BLOCK_ENTITY and BLOCK_CHAIN
 * @param ctx Error context
 * @return CIF_OK on success, error code on failure
 */
static CifError _parse_descriptions(mmCIF *cif, mmBlockList *blocks, CifErrorContext *ctx) {
    /* Allocate descriptions array */
    cif->descriptions = calloc((size_t)cif->chains, sizeof(char *));
    if (!cif->descriptions) {
        CIF_SET_ERROR(ctx, CIF_ERR_ALLOC, "Failed to allocate descriptions");
        return CIF_ERR_ALLOC;
    }

    mmBlock *entity = &blocks->b[BLOCK_ENTITY];
    mmBlock *chain_block = &blocks->b[BLOCK_CHAIN];

    /* Check if entity block exists */
    if (entity->category == NULL) {
        LOG_DEBUG("No _entity block, descriptions will be empty");
        return CIF_OK;
    }

    /* Build entity_id -> description map (sized for number of chains + 1) */
    int entity_desc_size = cif->chains + 1;
    char **entity_desc = calloc((size_t)entity_desc_size, sizeof(char *));
    if (!entity_desc) {
        CIF_SET_ERROR(ctx, CIF_ERR_ALLOC, "Failed to allocate entity_desc");
        return CIF_ERR_ALLOC;
    }

    CifError err = _precompute_lines(entity, ctx);
    if (err != CIF_OK) { free(entity_desc); return err; }

    int e_id_idx = _get_attr_index(entity, "id", ctx);
    int e_desc_idx = _get_attr_index(entity, "pdbx_description", ctx);

    if (e_id_idx >= 0 && e_desc_idx >= 0) {
        for (int row = 0; row < entity->size; row++) {
            int entity_id = _parse_int_inline(entity, row, e_id_idx);
            if (entity_id < 0 || entity_id >= entity_desc_size) continue;

            size_t desc_len;
            const char *desc_ptr = _get_field_ptr(entity, row, e_desc_idx, &desc_len);
            if (desc_ptr && desc_len > 0) {
                /* Copy and null-terminate the description */
                char *desc = malloc(desc_len + 1);
                if (desc) {
                    memcpy(desc, desc_ptr, desc_len);
                    desc[desc_len] = '\0';
                    entity_desc[entity_id] = desc;
                }
            }
        }
    }
    _free_lines(entity);

    /* Map chains to descriptions via _struct_asym.entity_id */
    err = _precompute_lines(chain_block, ctx);
    if (err != CIF_OK) {
        /* Free allocated descriptions */
        for (int i = 0; i < entity_desc_size; i++) free(entity_desc[i]);
        free(entity_desc);
        return err;
    }

    int sa_entity_idx = _get_attr_index(chain_block, "entity_id", ctx);
    if (sa_entity_idx >= 0) {
        for (int row = 0; row < chain_block->size && row < cif->chains; row++) {
            int entity_id = _parse_int_inline(chain_block, row, sa_entity_idx);
            if (entity_id >= 0 && entity_id < entity_desc_size && entity_desc[entity_id]) {
                /* Duplicate for each chain (entity may be shared) */
                cif->descriptions[row] = strdup(entity_desc[entity_id]);
            }
        }
    }
    _free_lines(chain_block);

    /* Free temporary entity description map */
    for (int i = 0; i < entity_desc_size; i++) free(entity_desc[i]);
    free(entity_desc);

    return CIF_OK;
}


/**
 * @brief Load an mmCIF file and return parsed data as Python objects.
 *
 * Main entry point for the Python extension. Loads the file, parses
 * all blocks, extracts molecular data, and returns as a dict of
 * NumPy arrays and Python lists.
 *
 * @param self Module reference (unused)
 * @param args Python positional arguments (filename string)
 * @param kwargs Python keyword arguments:
 *        - load_descriptions (bool): If true, parse entity descriptions (default: false)
 * @return Dict of parsed data or NULL on error
 */
static PyObject *_load(PyObject *self, PyObject *args, PyObject *kwargs) {
    (void)self;

    __py_init();
    PROFILE_RESET();

    CifErrorContext ctx = CIF_ERROR_INIT;

    /* Parse arguments: filename (required) + optional keywords */
    static char *kwlist[] = {"filename", "load_descriptions", "metadata_only", NULL};
    const char *file = NULL;
    int load_descriptions = 0;  /* Default: false */
    int metadata_only = 0;      /* Default: false - load full data */

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s|pp", kwlist,
                                      &file, &load_descriptions, &metadata_only)) {
        return NULL;
    }

    /* Load the entire file into memory */
    PROFILE_START(file_load);
    char *buffer = NULL;
    CifError err = _load_file(file, &buffer, &ctx);
    PROFILE_END(file_load);
    if (err != CIF_OK) {
        return _set_py_error(&ctx, file);
    }
    char *cpy = buffer;  /* Keep original pointer for free */

    /* Initialize parse cursor with line tracking */
    ParseCursor cursor = {.ptr = buffer, .line = 1};

    mmCIF cif = {0};
    mmBlockList blocks = {0};

    /* Read and validate the PDB ID */
    cif.id = _get_id(&cursor, &ctx);
    if (cif.id == NULL) {
        free(cpy);
        return _set_py_error(&ctx, file);
    }
    _next_block(&cursor);

    /* Parse all blocks in the file */
    PROFILE_START(block_parse);
    while (*cursor.ptr != '\0') {
        mmBlock block = _read_block(&cursor, &ctx);
        if (block.category == NULL) {
            /* Block parsing failed */
            free(cif.id);
            _free_block_list(&blocks);
            free(cpy);
            return _set_py_error(&ctx, file);
        }
        _store_or_free_block(&block, &blocks);
    }
    PROFILE_END(block_parse);

    /* Extract molecular data from parsed blocks (includes line_precomp, metadata, batch_parse, residue_count) */
    err = _fill_cif(&cif, &blocks, metadata_only, &ctx);
    if (err != CIF_OK) {
        free(cif.id);
        _free_block_list(&blocks);
        free(cpy);
        return _set_py_error(&ctx, file);
    }

    /* Optionally parse descriptions (after _fill_cif so chains is populated) */
    /* Skip in metadata_only mode since we don't need descriptions for indexing */
    if (load_descriptions && !metadata_only) {
        err = _parse_descriptions(&cif, &blocks, &ctx);
        if (err != CIF_OK) {
            free(cif.id);
            _free_block_list(&blocks);
            free(cpy);
            return _set_py_error(&ctx, file);
        }
    }

    /* Free the file buffer and block metadata */
    free(cpy);
    _free_block_list(&blocks);

    /* Convert to Python objects */
    PROFILE_START(py_convert);
    PyObject *dict = _c_to_py(cif);
    if (dict == NULL) return NULL;

    /* Add descriptions to dict if loaded */
    if (load_descriptions && cif.descriptions) {
        PyObject *py_desc = _c_arr_to_py_list(cif.descriptions, cif.chains);
        if (py_desc == NULL) {
            Py_DECREF(dict);
            return NULL;
        }
        if (PyDict_SetItemString(dict, "descriptions", py_desc) < 0) {
            Py_DECREF(py_desc);
            Py_DECREF(dict);
            return NULL;
        }
        Py_DECREF(py_desc);

        /* Free descriptions array (strings were copied by _c_arr_to_py_list) */
        for (int i = 0; i < cif.chains; i++) {
            free(cif.descriptions[i]);
        }
        free(cif.descriptions);
    }

    PROFILE_END(py_convert);
    return dict;
}


#ifdef CIFFY_PROFILE
/**
 * @brief Get profiling data from the last _load() call.
 *
 * Returns a dict with timing breakdown for each parsing phase.
 * Only available when compiled with CIFFY_PROFILE defined.
 *
 * @return Dict with timing in seconds, or None if profiling disabled
 */
static PyObject *_get_profile(PyObject *self, PyObject *args) {
    (void)self;
    (void)args;

    PyObject *dict = PyDict_New();
    if (dict == NULL) return NULL;

    /* Helper macro to add a timing value to the dict */
    #define ADD_TIMING(name) do { \
        PyObject *val = PyFloat_FromDouble(g_profile.name); \
        if (val == NULL) { Py_DECREF(dict); return NULL; } \
        if (PyDict_SetItemString(dict, #name, val) < 0) { \
            Py_DECREF(val); Py_DECREF(dict); return NULL; \
        } \
        Py_DECREF(val); \
    } while(0)

    ADD_TIMING(file_load);
    ADD_TIMING(block_parse);
    ADD_TIMING(line_precomp);
    ADD_TIMING(metadata);
    ADD_TIMING(batch_parse);
    ADD_TIMING(residue_count);
    ADD_TIMING(py_convert);
    /* Sub-phases of batch_parse */
    ADD_TIMING(batch_coords);
    ADD_TIMING(batch_elements);
    ADD_TIMING(batch_types);

    #undef ADD_TIMING

    return dict;
}
#else
/**
 * @brief Stub when profiling is disabled - returns None.
 */
static PyObject *_get_profile(PyObject *self, PyObject *args) {
    (void)self;
    (void)args;
    Py_RETURN_NONE;
}
#endif


/**
 * @brief Save molecular structure data to an mmCIF file.
 *
 * Takes Python/NumPy data and writes it to a CIF file.
 *
 * @param self Module reference (unused)
 * @param args Python arguments tuple containing:
 *        - filename (str): Output file path
 *        - id (str): PDB identifier
 *        - coordinates (ndarray): (N, 3) float32 array
 *        - atoms (ndarray): (N,) int32 array of atom types
 *        - elements (ndarray): (N,) int32 array of element types
 *        - residues (ndarray): (R,) int32 array of residue types
 *        - atoms_per_res (ndarray): (R,) int32 array
 *        - atoms_per_chain (ndarray): (C,) int32 array
 *        - res_per_chain (ndarray): (C,) int32 array
 *        - chain_names (list): List of chain name strings
 *        - strand_names (list): List of strand ID strings
 *        - polymer_count (int): Number of polymer atoms
 *        - molecule_types (ndarray): (C,) int32 array of molecule types
 * @return None on success, NULL on error
 */
static PyObject *_save(PyObject *self, PyObject *args) {

    __py_init();

    CifErrorContext ctx = CIF_ERROR_INIT;
    PyObject *result = NULL;

    /* Parse arguments */
    const char *filename;
    const char *id;
    PyObject *py_coords, *py_atoms, *py_elements, *py_residues;
    PyObject *py_atoms_per_res, *py_atoms_per_chain, *py_res_per_chain;
    PyObject *py_chain_names, *py_strand_names, *py_molecule_types;
    int polymer_count;

    if (!PyArg_ParseTuple(args, "ssOOOOOOOOOiO",
            &filename, &id,
            &py_coords, &py_atoms, &py_elements, &py_residues,
            &py_atoms_per_res, &py_atoms_per_chain, &py_res_per_chain,
            &py_chain_names, &py_strand_names, &polymer_count, &py_molecule_types)) {
        return NULL;  /* PyArg_ParseTuple sets exception */
    }

    /* Build mmCIF structure from Python objects.
     * Note: Numpy arrays are borrowed references (no copy).
     * String arrays (names, strands) are copies that we own.
     */
    mmCIF cif = {0};
    int num_chains = 0;
    int num_strands = 0;

    cif.polymer = polymer_count;

    /* Copy ID string (we own this) */
    cif.id = strdup(id);
    if (cif.id == NULL) {
        PyErr_NoMemory();
        goto cleanup;
    }

    /* Extract numpy arrays (borrowed references - no allocation) */
    int coord_size;
    cif.coordinates = _numpy_to_float_arr(py_coords, &coord_size);
    if (cif.coordinates == NULL) goto cleanup;
    cif.atoms = coord_size / 3;

    cif.types = _numpy_to_int_arr(py_atoms, NULL);
    if (cif.types == NULL) goto cleanup;

    cif.elements = _numpy_to_int_arr(py_elements, NULL);
    if (cif.elements == NULL) goto cleanup;

    cif.sequence = _numpy_to_int_arr(py_residues, &cif.residues);
    if (cif.sequence == NULL) goto cleanup;

    cif.atoms_per_res = _numpy_to_int_arr(py_atoms_per_res, NULL);
    if (cif.atoms_per_res == NULL) goto cleanup;

    cif.atoms_per_chain = _numpy_to_int_arr(py_atoms_per_chain, &cif.chains);
    if (cif.atoms_per_chain == NULL) goto cleanup;

    cif.res_per_chain = _numpy_to_int_arr(py_res_per_chain, NULL);
    if (cif.res_per_chain == NULL) goto cleanup;

    cif.molecule_types = _numpy_to_int_arr(py_molecule_types, NULL);
    if (cif.molecule_types == NULL) goto cleanup;

    /* Extract string arrays (we own these copies) */
    cif.names = _py_list_to_c_arr(py_chain_names, &num_chains);
    if (cif.names == NULL) goto cleanup;

    cif.strands = _py_list_to_c_arr(py_strand_names, &num_strands);
    if (cif.strands == NULL) goto cleanup;

    /* Calculate non-polymer count and write */
    cif.nonpoly = cif.atoms - cif.polymer;

    CifError err = _write_cif(&cif, filename, &ctx);
    if (err != CIF_OK) {
        _set_py_error(&ctx, filename);
        goto cleanup;
    }

    /* Success */
    result = Py_None;
    Py_INCREF(result);

cleanup:
    /* Free only what we own: id string and string arrays */
    free(cif.id);
    if (cif.names) _free_c_str_arr(cif.names, num_chains);
    if (cif.strands) _free_c_str_arr(cif.strands, num_strands);

    return result;
}


/* Python module method table */
static PyMethodDef methods[] = {
    {"_load", (PyCFunction)_load, METH_VARARGS | METH_KEYWORDS,
     "Load an mmCIF file and return molecular structure data.\n\n"
     "Args:\n"
     "    filename (str): Path to the mmCIF file\n"
     "    load_descriptions (bool): If True, parse entity descriptions (default: False)\n\n"
     "Returns:\n"
     "    dict: {\n"
     "        'id': str,                    # PDB identifier\n"
     "        'coordinates': ndarray,       # (N, 3) float32\n"
     "        'atoms': ndarray,             # (N,) int32 atom types\n"
     "        'elements': ndarray,          # (N,) int32 element types\n"
     "        'residues': ndarray,          # (R,) int32 residue types\n"
     "        'atoms_per_res': ndarray,     # (R,) int32\n"
     "        'atoms_per_chain': ndarray,   # (C,) int32\n"
     "        'res_per_chain': ndarray,     # (C,) int32\n"
     "        'chain_names': list[str],     # chain names\n"
     "        'strand_names': list[str],    # strand names\n"
     "        'polymer_count': int,         # polymer atoms\n"
     "        'molecule_types': ndarray,    # (C,) int32\n"
     "        'descriptions': list[str],    # entity descriptions (if load_descriptions=True)\n"
     "    }\n\n"
     "Raises:\n"
     "    IOError: If file cannot be read\n"
     "    ValueError: If file format is invalid\n"
     "    KeyError: If required attributes are missing\n"
     "    MemoryError: If allocation fails\n"},
    {"_save", _save, METH_VARARGS,
     "Save molecular structure data to an mmCIF file.\n\n"
     "Args:\n"
     "    filename (str): Output file path\n"
     "    id (str): PDB identifier\n"
     "    coordinates (ndarray): (N, 3) float32 array of atom coordinates\n"
     "    atoms (ndarray): (N,) int32 array of atom type indices\n"
     "    elements (ndarray): (N,) int32 array of element indices\n"
     "    residues (ndarray): (R,) int32 array of residue type indices\n"
     "    atoms_per_res (ndarray): (R,) int32 array of atoms per residue\n"
     "    atoms_per_chain (ndarray): (C,) int32 array of atoms per chain\n"
     "    res_per_chain (ndarray): (C,) int32 array of residues per chain\n"
     "    chain_names (list): List of chain name strings\n"
     "    strand_names (list): List of strand ID strings\n"
     "    polymer_count (int): Number of polymer atoms\n"
     "    molecule_types (ndarray): (C,) int32 array of molecule types\n\n"
     "Raises:\n"
     "    IOError: If file cannot be written\n"
     "    TypeError: If arguments have wrong type\n"
     "    MemoryError: If allocation fails\n"},
    {"_get_profile", _get_profile, METH_NOARGS,
     "Get profiling data from the last _load() call.\n\n"
     "Returns:\n"
     "    dict or None: Timing breakdown if profiling enabled, else None.\n"
     "    Keys: file_load, block_parse, line_precomp, metadata,\n"
     "          batch_parse, residue_count, py_convert (all in seconds)\n"},
    {"_cartesian_to_internal", py_cartesian_to_internal, METH_VARARGS,
     "Convert Cartesian coordinates to internal coordinates.\n\n"
     "Args:\n"
     "    coords (ndarray): (N, 3) float64 Cartesian coordinates.\n"
     "    indices (ndarray): (M, 4) int64 Z-matrix indices.\n\n"
     "Returns:\n"
     "    tuple: (distances, angles, dihedrals), each (M,) float64.\n"},
    {"_build_bond_graph", py_build_bond_graph, METH_VARARGS,
     "Build molecular bond graph from polymer arrays.\n\n"
     "Args:\n"
     "    atoms (ndarray): (N,) int32 atom values.\n"
     "    sequence (ndarray): (R,) int32 residue type indices.\n"
     "    res_sizes (ndarray): (R,) int32 atoms per residue.\n"
     "    chain_lengths (ndarray): (C,) int32 residues per chain.\n\n"
     "Returns:\n"
     "    ndarray: (E, 2) int64 edge array [atom_i, atom_j].\n"},
    {"_edges_to_csr", py_edges_to_csr, METH_VARARGS,
     "Convert edge list to CSR format.\n\n"
     "Args:\n"
     "    edges (ndarray): (E, 2) int64 symmetric edge array.\n"
     "    n_atoms (int): Total number of atoms.\n\n"
     "Returns:\n"
     "    tuple: (offsets, neighbors) CSR arrays.\n"},
    {"_build_zmatrix_parallel", py_build_zmatrix_parallel, METH_VARARGS,
     "Build Z-matrix for all chains in parallel using OpenMP.\n\n"
     "Args:\n"
     "    offsets (ndarray): (n_atoms+1,) int64 CSR offsets.\n"
     "    neighbors (ndarray): (E,) int64 neighbor indices.\n"
     "    n_atoms (int): Total number of atoms.\n"
     "    chain_starts (ndarray): (n_chains,) int64 first atom per chain.\n"
     "    chain_sizes (ndarray): (n_chains,) int64 atoms per chain.\n"
     "    roots (ndarray): (n_chains,) int64 root atom per chain.\n\n"
     "Returns:\n"
     "    tuple: (zmatrix, counts) - Z-matrix entries and per-chain counts.\n"},
    {"_cartesian_to_internal_backward", py_cartesian_to_internal_backward, METH_VARARGS,
     "Backward pass for Cartesian to internal coordinate conversion.\n\n"
     "Args:\n"
     "    coords (ndarray): (N, 3) float32 Cartesian coordinates.\n"
     "    indices (ndarray): (M, 4) int64 Z-matrix indices.\n"
     "    distances (ndarray): (M,) float32 forward pass distances.\n"
     "    angles (ndarray): (M,) float32 forward pass angles.\n"
     "    grad_distances (ndarray): (M,) float32 upstream gradients.\n"
     "    grad_angles (ndarray): (M,) float32 upstream gradients.\n"
     "    grad_dihedrals (ndarray): (M,) float32 upstream gradients.\n\n"
     "Returns:\n"
     "    ndarray: (N, 3) float32 gradients for coordinates.\n"},
    {"_find_connected_components", py_find_connected_components, METH_VARARGS,
     "Find connected components in CSR graph.\n\n"
     "Args:\n"
     "    offsets (ndarray): (n_atoms+1,) int64 CSR offsets.\n"
     "    neighbors (ndarray): (E,) int64 neighbor indices.\n"
     "    n_atoms (int): Total number of atoms.\n\n"
     "Returns:\n"
     "    tuple: (roots, sizes, n_components).\n"},
    {"_nerf_reconstruct_leveled_anchored", py_nerf_reconstruct_leveled_anchored, METH_VARARGS,
     "Level-parallel NERF reconstruction with anchor coordinates.\n\n"
     "Places atoms in frame defined by anchor coordinates instead of canonical frame.\n"
     "Eliminates need for post-reconstruction Kabsch rotation.\n\n"
     "Args:\n"
     "    indices (ndarray): (M, 4) int64 Z-matrix indices (sorted by level).\n"
     "    distances (ndarray): (M,) float32 bond lengths.\n"
     "    angles (ndarray): (M,) float32 bond angles in radians.\n"
     "    dihedrals (ndarray): (M,) float32 dihedral angles in radians.\n"
     "    n_atoms (int): Total number of atoms.\n"
     "    level_offsets (ndarray): (n_levels+1,) int32 CSR-style offsets.\n"
     "    anchor_coords (ndarray): (n_components, 3, 3) float32 anchor positions.\n"
     "    component_ids (ndarray): (M,) int32 component ID per Z-matrix entry.\n\n"
     "Returns:\n"
     "    ndarray: (N, 3) float32 Cartesian coordinates.\n"},
    {"_nerf_reconstruct_backward_leveled_anchored", py_nerf_reconstruct_backward_leveled_anchored, METH_VARARGS,
     "Level-parallel backward pass for anchored NERF reconstruction.\n\n"
     "Args:\n"
     "    coords (ndarray): (N, 3) float32 reconstructed coordinates.\n"
     "    indices (ndarray): (M, 4) int64 Z-matrix indices (sorted by level).\n"
     "    distances (ndarray): (M,) float32 bond lengths.\n"
     "    angles (ndarray): (M,) float32 bond angles.\n"
     "    dihedrals (ndarray): (M,) float32 dihedral angles.\n"
     "    grad_coords (ndarray): (N, 3) float32 upstream gradients.\n"
     "    level_offsets (ndarray): (n_levels+1,) int32 CSR-style offsets.\n"
     "    anchor_coords (ndarray): (n_components, 3, 3) float32 anchor positions.\n"
     "    component_ids (ndarray): (M,) int32 component ID per Z-matrix entry.\n\n"
     "Returns:\n"
     "    tuple: (grad_distances, grad_angles, grad_dihedrals).\n"},
    {NULL, NULL, 0, NULL}
};

/* Python module definition */
static struct PyModuleDef _c = {
    PyModuleDef_HEAD_INIT,
    "_c",
    "Low-level C extension for parsing mmCIF files.",
    -1,
    methods
};

/* Module initialization function */
PyMODINIT_FUNC PyInit__c(void) {
    import_array();  /* Initialize NumPy C API */
    return PyModule_Create(&_c);
}
