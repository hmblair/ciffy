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
 * @brief Create a 1D NumPy array from int data.
 *
 * Sets NPY_ARRAY_OWNDATA so NumPy frees the memory when the array
 * is garbage collected.
 */
static PyObject *_init_1d_arr_int(int size, int *data) {
    npy_intp dims[1] = {size};
    PyObject *arr = PyArray_SimpleNewFromData(1, dims, NPY_INT, data);
    if (arr == NULL) {
        free(data);
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


/* Block parsing functions are now in parser.c - see parser.h for declarations */


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
 * @return None on success, NULL on error
 */
static PyObject *_save(PyObject *self, PyObject *args) {

    __py_init();

    CifErrorContext ctx = CIF_ERROR_INIT;

    /* Parse arguments */
    const char *filename;
    const char *id;
    PyObject *py_coords, *py_atoms, *py_elements, *py_residues;
    PyObject *py_atoms_per_res, *py_atoms_per_chain, *py_res_per_chain;
    PyObject *py_chain_names, *py_strand_names;
    int polymer_count;

    if (!PyArg_ParseTuple(args, "ssOOOOOOOOOi",
            &filename, &id,
            &py_coords, &py_atoms, &py_elements, &py_residues,
            &py_atoms_per_res, &py_atoms_per_chain, &py_res_per_chain,
            &py_chain_names, &py_strand_names, &polymer_count)) {
        return NULL;  /* PyArg_ParseTuple sets exception */
    }

    /* Build mmCIF structure from Python objects */
    mmCIF cif = {0};
    cif.polymer = polymer_count;

    /* Copy ID string (we need to own it for mmCIF struct) */
    cif.id = strdup(id);
    if (cif.id == NULL) {
        return PyErr_NoMemory();
    }

    /* Extract arrays (borrowed references - no copy needed) */
    int coord_size;
    cif.coordinates = _numpy_to_float_arr(py_coords, &coord_size);
    if (cif.coordinates == NULL) {
        free(cif.id);
        return NULL;  /* Exception already set */
    }
    cif.atoms = coord_size / 3;  /* Coordinates are N*3 */

    cif.types = _numpy_to_int_arr(py_atoms, NULL);
    if (cif.types == NULL) {
        free(cif.id);
        return NULL;
    }

    cif.elements = _numpy_to_int_arr(py_elements, NULL);
    if (cif.elements == NULL) {
        free(cif.id);
        return NULL;
    }

    cif.sequence = _numpy_to_int_arr(py_residues, &cif.residues);
    if (cif.sequence == NULL) {
        free(cif.id);
        return NULL;
    }

    cif.atoms_per_res = _numpy_to_int_arr(py_atoms_per_res, NULL);
    if (cif.atoms_per_res == NULL) {
        free(cif.id);
        return NULL;
    }

    cif.atoms_per_chain = _numpy_to_int_arr(py_atoms_per_chain, &cif.chains);
    if (cif.atoms_per_chain == NULL) {
        free(cif.id);
        return NULL;
    }

    cif.res_per_chain = _numpy_to_int_arr(py_res_per_chain, NULL);
    if (cif.res_per_chain == NULL) {
        free(cif.id);
        return NULL;
    }

    /* Extract chain name lists (need to copy because we need char** format) */
    int num_chains;
    cif.names = _py_list_to_c_arr(py_chain_names, &num_chains);
    if (cif.names == NULL) {
        free(cif.id);
        return NULL;
    }

    int num_strands;
    cif.strands = _py_list_to_c_arr(py_strand_names, &num_strands);
    if (cif.strands == NULL) {
        free(cif.id);
        _free_c_str_arr(cif.names, num_chains);
        return NULL;
    }

    /* Calculate non-polymer count */
    cif.nonpoly = cif.atoms - cif.polymer;

    /* Write to file */
    CifError err = _write_cif(&cif, filename, &ctx);

    /* Cleanup (only what we allocated - id and string arrays) */
    free(cif.id);
    _free_c_str_arr(cif.names, num_chains);
    _free_c_str_arr(cif.strands, num_strands);

    if (err != CIF_OK) {
        return _set_py_error(&ctx, filename);
    }

    Py_RETURN_NONE;
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
     "    polymer_count (int): Number of polymer atoms\n\n"
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
