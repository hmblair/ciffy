/**
 * @file module.c
 * @brief Python C extension entry point for ciffy.
 *
 * Provides the _load function that reads mmCIF files and returns
 * parsed molecular structure data as Python/NumPy objects.
 */

/* Define CIFFY_MAIN_MODULE before including headers so python.h knows to import numpy */
#define CIFFY_MAIN_MODULE
#include "module.h"
#include "log.h"


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
            return _init_1d_arr_int(size, data);
        }

        case PY_1D_FLOAT: {
            /* Not currently used, but included for completeness */
            float *data = *(float **)(base + def->storage_offset);
            npy_intp dims[1] = {size};
            PyObject *arr = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT, data);
            if (arr) PyArray_ENABLEFLAGS((PyArrayObject *)arr, NPY_ARRAY_OWNDATA);
            return arr;
        }

        case PY_2D_FLOAT: {
            float *data = *(float **)(base + def->storage_offset);
            return _init_2d_arr_float(size, def->elements_per_item, data);
        }

        case PY_STR_LIST: {
            char **data = *(char ***)(base + def->storage_offset);
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

    /* Build entity_id -> description map (max 100 entities) */
    char *entity_desc[100] = {NULL};

    CifError err = _precompute_lines(entity, ctx);
    if (err != CIF_OK) return err;

    int e_id_idx = _get_attr_index(entity, "id", ctx);
    int e_desc_idx = _get_attr_index(entity, "pdbx_description", ctx);

    if (e_id_idx >= 0 && e_desc_idx >= 0) {
        for (int row = 0; row < entity->size; row++) {
            int entity_id = _parse_int_inline(entity, row, e_id_idx);
            if (entity_id < 0 || entity_id >= 100) continue;

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
        for (int i = 0; i < 100; i++) free(entity_desc[i]);
        return err;
    }

    int sa_entity_idx = _get_attr_index(chain_block, "entity_id", ctx);
    if (sa_entity_idx >= 0) {
        for (int row = 0; row < chain_block->size && row < cif->chains; row++) {
            int entity_id = _parse_int_inline(chain_block, row, sa_entity_idx);
            if (entity_id >= 0 && entity_id < 100 && entity_desc[entity_id]) {
                /* Duplicate for each chain (entity may be shared) */
                cif->descriptions[row] = strdup(entity_desc[entity_id]);
            }
        }
    }
    _free_lines(chain_block);

    /* Free temporary entity description map */
    for (int i = 0; i < 100; i++) free(entity_desc[i]);

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

    CifErrorContext ctx = CIF_ERROR_INIT;

    /* Parse arguments: filename (required) + load_descriptions (optional keyword) */
    static char *kwlist[] = {"filename", "load_descriptions", NULL};
    const char *file = NULL;
    int load_descriptions = 0;  /* Default: false */

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s|p", kwlist,
                                      &file, &load_descriptions)) {
        return NULL;
    }

    /* Load the entire file into memory */
    char *buffer = NULL;
    CifError err = _load_file(file, &buffer, &ctx);
    if (err != CIF_OK) {
        return _set_py_error(&ctx, file);
    }
    char *cpy = buffer;  /* Keep original pointer for free */

    mmCIF cif = {0};
    mmBlockList blocks = {0};

    /* Read and validate the PDB ID */
    cif.id = _get_id(buffer, &ctx);
    if (cif.id == NULL) {
        free(cpy);
        return _set_py_error(&ctx, file);
    }
    _next_block(&buffer);

    /* Parse all blocks in the file */
    while (*buffer != '\0') {
        mmBlock block = _read_block(&buffer, &ctx);
        if (block.category == NULL) {
            /* Block parsing failed */
            free(cif.id);
            _free_block_list(&blocks);
            free(cpy);
            return _set_py_error(&ctx, file);
        }
        _store_or_free_block(&block, &blocks);
    }

    /* Extract molecular data from parsed blocks */
    err = _fill_cif(&cif, &blocks, &ctx);
    if (err != CIF_OK) {
        free(cif.id);
        _free_block_list(&blocks);
        free(cpy);
        return _set_py_error(&ctx, file);
    }

    /* Optionally parse descriptions (after _fill_cif so chains is populated) */
    if (load_descriptions) {
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

    return dict;
}


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
    return PyModule_Create(&_c);
}
