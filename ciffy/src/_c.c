/**
 * @file _c.c
 * @brief Python C extension entry point for ciffy.
 *
 * Provides the _load function that reads mmCIF files and returns
 * parsed molecular structure data as Python/NumPy objects.
 */

#include "_c.h"


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
 * @brief Create a 1D NumPy array from int data.
 */
static PyObject *_init_1d_arr_int(int size, int *data) {
    npy_intp dims[1] = {size};
    PyObject *arr = PyArray_SimpleNewFromData(1, dims, NPY_INT, data);
    if (arr == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Failed to create NumPy array");
    }
    return arr;
}


/**
 * @brief Create a 1D NumPy array from float data.
 */
static PyObject *_init_1d_arr_float(int size, float *data) {
    npy_intp dims[1] = {size};
    PyObject *arr = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT, data);
    if (arr == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Failed to create NumPy array");
    }
    return arr;
}


/**
 * @brief Create a 2D NumPy array from float data.
 */
static PyObject *_init_2d_arr_float(int size1, int size2, float *data) {
    npy_intp dims[2] = {size1, size2};
    PyObject *arr = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT, data);
    if (arr == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Failed to create NumPy array");
    }
    return arr;
}


/**
 * @brief Convert mmCIF struct to Python tuple.
 *
 * Creates NumPy arrays and Python objects from the parsed C data.
 * Returns NULL and sets Python exception on error.
 */
static PyObject *_c_to_py(mmCIF cif) {

    PyObject *py_id = _c_str_to_py_str(cif.id);
    if (py_id == NULL) return NULL;

    PyObject *chain_names_list = _c_arr_to_py_list(cif.names, cif.chains);
    if (chain_names_list == NULL) { Py_DECREF(py_id); return NULL; }

    PyObject *strand_names_list = _c_arr_to_py_list(cif.strands, cif.chains);
    if (strand_names_list == NULL) {
        Py_DECREF(py_id);
        Py_DECREF(chain_names_list);
        return NULL;
    }

    PyObject *coordinates = _init_2d_arr_float(cif.atoms, 3, cif.coordinates);
    if (coordinates == NULL) {
        Py_DECREF(py_id);
        Py_DECREF(chain_names_list);
        Py_DECREF(strand_names_list);
        return NULL;
    }

    PyObject *atoms_array = _init_1d_arr_int(cif.atoms, cif.types);
    if (atoms_array == NULL) {
        Py_DECREF(py_id);
        Py_DECREF(chain_names_list);
        Py_DECREF(strand_names_list);
        Py_DECREF(coordinates);
        return NULL;
    }

    PyObject *elements_array = _init_1d_arr_int(cif.atoms, cif.elements);
    if (elements_array == NULL) {
        Py_DECREF(py_id);
        Py_DECREF(chain_names_list);
        Py_DECREF(strand_names_list);
        Py_DECREF(coordinates);
        Py_DECREF(atoms_array);
        return NULL;
    }

    PyObject *residues_array = _init_1d_arr_int(cif.residues, cif.sequence);
    if (residues_array == NULL) {
        Py_DECREF(py_id);
        Py_DECREF(chain_names_list);
        Py_DECREF(strand_names_list);
        Py_DECREF(coordinates);
        Py_DECREF(atoms_array);
        Py_DECREF(elements_array);
        return NULL;
    }

    PyObject *atoms_per_res = _init_1d_arr_int(cif.residues, cif.atoms_per_res);
    if (atoms_per_res == NULL) {
        Py_DECREF(py_id);
        Py_DECREF(chain_names_list);
        Py_DECREF(strand_names_list);
        Py_DECREF(coordinates);
        Py_DECREF(atoms_array);
        Py_DECREF(elements_array);
        Py_DECREF(residues_array);
        return NULL;
    }

    PyObject *atoms_per_chain = _init_1d_arr_int(cif.chains, cif.atoms_per_chain);
    if (atoms_per_chain == NULL) {
        Py_DECREF(py_id);
        Py_DECREF(chain_names_list);
        Py_DECREF(strand_names_list);
        Py_DECREF(coordinates);
        Py_DECREF(atoms_array);
        Py_DECREF(elements_array);
        Py_DECREF(residues_array);
        Py_DECREF(atoms_per_res);
        return NULL;
    }

    PyObject *res_per_chain = _init_1d_arr_int(cif.chains, cif.res_per_chain);
    if (res_per_chain == NULL) {
        Py_DECREF(py_id);
        Py_DECREF(chain_names_list);
        Py_DECREF(strand_names_list);
        Py_DECREF(coordinates);
        Py_DECREF(atoms_array);
        Py_DECREF(elements_array);
        Py_DECREF(residues_array);
        Py_DECREF(atoms_per_res);
        Py_DECREF(atoms_per_chain);
        return NULL;
    }

    PyObject *polymer_count = _c_int_to_py_int(cif.polymer);
    if (polymer_count == NULL) {
        Py_DECREF(py_id);
        Py_DECREF(chain_names_list);
        Py_DECREF(strand_names_list);
        Py_DECREF(coordinates);
        Py_DECREF(atoms_array);
        Py_DECREF(elements_array);
        Py_DECREF(residues_array);
        Py_DECREF(atoms_per_res);
        Py_DECREF(atoms_per_chain);
        Py_DECREF(res_per_chain);
        return NULL;
    }

    return PyTuple_Pack(11,
        py_id, coordinates, atoms_array, elements_array, residues_array,
        atoms_per_res, atoms_per_chain, res_per_chain,
        chain_names_list, strand_names_list, polymer_count);
}


/**
 * @brief Skip past a multi-line attribute value.
 */
static void _skip_multiline_attr(char **buffer) {
    _advance_line(buffer);
    while (**buffer != ';') {
        _advance_line(buffer);
    }
    _advance_line(buffer);
}


/**
 * @brief Advance to the next block (skip to section end marker).
 */
static void _next_block(char **buffer) {
    while (!_is_section_end(*buffer)) {
        _advance_line(buffer);
    }
    _advance_line(buffer);
}


/**
 * @brief Parse a single mmCIF block.
 *
 * Reads block header, counts attributes, and for multi-entry blocks,
 * calculates line width and entry count.
 *
 * @param buffer Pointer to buffer pointer (modified in place)
 * @param ctx Error context for allocation failures
 * @return Parsed block structure (check category for NULL on error)
 */
static mmBlock _read_block(char **buffer, CifErrorContext *ctx) {

    mmBlock block = {0};

    /* Check if this is a single-entry block (no "loop_" prefix) */
    if (_eq(*buffer, "loop_")) {
        _advance_line(buffer);
    } else {
        block.single = true;
        block.size = 1;
    }

    block.head = *buffer;
    block.category = _get_category(block.head, ctx);
    if (block.category == NULL) {
        return block;  /* Error - ctx is already set */
    }

    /* Count attributes by scanning header lines */
    while (_eq(*buffer, block.category)) {
        block.attributes++;
        _advance_line(buffer);
        if (**buffer == ';') {
            _skip_multiline_attr(buffer);
        }
    }

    if (!block.single) {
        /* Multi-entry block: calculate offsets and line width */
        block.start = *buffer;
        block.offsets = _get_offsets(block.start, block.attributes, ctx);
        if (block.offsets == NULL) {
            free(block.category);
            block.category = NULL;
            return block;  /* Error - ctx is already set */
        }
        block.width = block.offsets[block.attributes] + 1;

        /* Count entries until section end */
        while (!_is_section_end(*buffer)) {
            *buffer += block.width;
            block.size++;

            /* If block is not homogeneous (different line widths), stop */
            if ((*buffer)[-1] != '\n') { break; }
        }
    }

    /* Skip past section end marker */
    _next_block(buffer);

    return block;
}


/**
 * @brief Free resources associated with a block.
 */
static void _free_block(mmBlock *block) {
    block->head = NULL;
    block->start = NULL;

    if (block->category != NULL) {
        free(block->category);
        block->category = NULL;
    }

    if (block->offsets != NULL) {
        free(block->offsets);
        block->offsets = NULL;
    }
}


/**
 * @brief Store a block if it's needed, otherwise free it.
 */
static void _store_or_free_block(mmBlock *block, mmBlockList *blocks) {

    if (_eq(block->category, "_atom_site.")) {
        blocks->atom = *block;
        return;
    }

    if (_eq(block->category, "_struct_asym.")) {
        blocks->chain = *block;
        return;
    }

    if (_eq(block->category, "_pdbx_poly_seq_scheme.")) {
        blocks->poly = *block;
        return;
    }

    if (_eq(block->category, "_pdbx_nonpoly_scheme.")) {
        blocks->nonpoly = *block;
        return;
    }

    if (_eq(block->category, "_struct_conn.")) {
        blocks->conn = *block;
        return;
    }

    _free_block(block);
}


/**
 * @brief Free all blocks in a block list.
 */
static void _free_block_list(mmBlockList *blocks) {
    _free_block(&blocks->atom);
    _free_block(&blocks->poly);
    _free_block(&blocks->nonpoly);
    _free_block(&blocks->conn);
    _free_block(&blocks->chain);
}


/**
 * @brief Load an mmCIF file and return parsed data as Python objects.
 *
 * Main entry point for the Python extension. Loads the file, parses
 * all blocks, extracts molecular data, and returns as a tuple of
 * NumPy arrays and Python lists.
 *
 * @param self Module reference (unused)
 * @param args Python arguments (filename string)
 * @return Tuple of (id, coordinates, atoms, elements, residues,
 *         atoms_per_res, atoms_per_chain, res_per_chain,
 *         chain_names, strand_names, nonpoly) or NULL on error
 */
static PyObject *_load(PyObject *self, PyObject *args) {

    __py_init();

    CifErrorContext ctx = CIF_ERROR_INIT;

    /* Get the filename from arguments */
    const char *file = _get_filename(args);
    if (file == NULL) { return NULL; }

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

    /* Free the file buffer and block metadata */
    free(cpy);
    _free_block_list(&blocks);

    /* Convert to Python objects */
    return _c_to_py(cif);
}


/* Python module method table */
static PyMethodDef methods[] = {
    {"_load", _load, METH_VARARGS,
     "Load an mmCIF file and return molecular structure data.\n\n"
     "Args:\n"
     "    filename (str): Path to the mmCIF file\n\n"
     "Returns:\n"
     "    tuple: (id, coordinates, atoms, elements, residues,\n"
     "            atoms_per_res, atoms_per_chain, res_per_chain,\n"
     "            chain_names, strand_names, polymer_count)\n\n"
     "Raises:\n"
     "    IOError: If file cannot be read\n"
     "    ValueError: If file format is invalid\n"
     "    KeyError: If required attributes are missing\n"
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
