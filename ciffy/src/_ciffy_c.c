#include "_ciffy_c.h"

static inline bool _eq(char *str1, char* str2) {

    return strncmp(str1, str2, strlen(str2)) == 0;

}

static inline bool _neq(char *str1, char* str2) {

    return strncmp(str1, str2, strlen(str2)) != 0;

}

static inline bool _is_section_end(char* line) {

    return *line == '#';

}

PyObject *_init_1d_arr_int(int size, int* data) {

    npy_intp dims[1] = {size};
    return PyArray_SimpleNewFromData(1, dims, NPY_INT, data);

}

PyObject *_init_1d_arr_float(int size, float* data) {

    npy_intp dims[1] = {size};
    return PyArray_SimpleNewFromData(1, dims, NPY_FLOAT, data);

}

PyObject *_init_2d_arr_float(int size1, int size2, float* data) {

    npy_intp dims[2] = {size1, size2};
    return PyArray_SimpleNewFromData(2, dims, NPY_FLOAT, data);

}


PyObject* _c_to_py(mmCIF cif) {

    PyObject *py_id = _c_str_to_py_str(cif.id);
    PyObject *chain_names_list = _c_arr_to_py_list(cif.names, cif.chains);

    PyObject *coordinates = _init_2d_arr_float(cif.atoms, 3, cif.coordinates);

    PyObject *atoms_array    = _init_1d_arr_int(cif.atoms, cif.types);
    PyObject *elements_array = _init_1d_arr_int(cif.atoms, cif.elements);
    PyObject *residues_array = _init_1d_arr_int(cif.residues, cif.sequence);

    PyObject *atoms_per_res = _init_1d_arr_int(cif.residues, cif.atoms_per_res);
    PyObject *atoms_per_chain = _init_1d_arr_int(cif.chains, cif.atoms_per_chain);
    PyObject *res_per_chain = _init_1d_arr_int(cif.chains, cif.res_per_chain);

    PyObject *nonpoly = _c_int_to_py_int(cif.nonpoly);

    return PyTuple_Pack(10, py_id, coordinates, atoms_array, elements_array, residues_array, atoms_per_res, atoms_per_chain, res_per_chain, chain_names_list, nonpoly);

}


void _skip_multiline_attr(char **buffer) {

    _advance_line(buffer);
    while (**buffer != ';') {
        _advance_line(buffer);
    }
    _advance_line(buffer);

}


void _next_block(char **buffer) {

    while (!_is_section_end(*buffer)) {
        _advance_line(buffer);
    }
    _advance_line(buffer);

}


mmBlock _read_block(char **buffer) {

    mmBlock block = {0};

    // Check if this is a single-entry block and handle appropriately
    // *buffer now points to the beginning of the header

    if (_eq(*buffer, "loop_")) {
        _advance_line(buffer);
    } else {
        block.single = true;
        block.size   = 1;
    }

    block.head     = *buffer;
    block.category = _get_category(block.head);

    // Count the number of attributes in the header
    // *buffer now points to the beginning of the data

    while (_eq(*buffer, block.category)) {

        block.attributes++;
        _advance_line(buffer);
        if (**buffer == ';') {
            _skip_multiline_attr(buffer);
        }

    }

    if (!block.single) {

        // Read the field offsets and line width
        // The latter includes the newline character

        block.start   = *buffer;
        block.offsets = _get_offsets(block.start, block.attributes);
        block.width   = block.offsets[block.attributes] + 1;

        // Count the number of entries in the block
        // *buffer now points to the end of the section

        while (!_is_section_end(*buffer)) {

            *buffer += block.width;
            block.size++;

            // If the block is not homogeneous, we will skip it

            if ((*buffer)[-1] != '\n') { break; }

        }

    }

    // Skip past the end of section marker
    // *buffer now points to the beginning of the next section

    _next_block(buffer);

    return block;

}


void _free_block(mmBlock *block) {

    block->head  = NULL;
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


void _store_or_free_block(mmBlock *block, mmBlockList* blocks) {

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


void _free_block_list(mmBlockList* blocks) {

    _free_block(&blocks->atom);
    _free_block(&blocks->poly);
    _free_block(&blocks->nonpoly);
    _free_block(&blocks->conn);

}


static PyObject* _load(PyObject* self, PyObject* args) {

    __py_init();

    // Get the filename

    const char *file = _get_filename(args);
    if (file == NULL) { return NULL; }

    // Load the file

    char *buffer = _load_file(file);
    char *cpy    = buffer;
    if (buffer == NULL) {
        return PyErr_Format(PyExc_IOError, "Failed to open file: %s", file);
    }

    mmCIF cif          = {0};
    mmBlockList blocks = {0};

    // Read the ID and validate the file

    cif.id = _get_id(buffer);
    if (cif.id == NULL) {
        free(cpy);
        return PyErr_Format(PyExc_IOError, "Invalid mmCIF file: %s", file);
    }
    _next_block(&buffer);

    // Load the relevant blocks

    while (*buffer != '\0') {
        mmBlock block = _read_block(&buffer);
        _store_or_free_block(&block, &blocks);
    }

    // Parse the blocks

    _fill_cif(&cif, &blocks);

    // Free

    free(cpy);
    _free_block_list(&blocks);

    // Convert to a PyObject pointer

    return _c_to_py(cif);

}

static PyMethodDef methods[] = {
    {"_load", _load, METH_VARARGS, "Load a CIF file and return coordinates, elements, and residues as NumPy arrays"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef _ciffy_c = {
    PyModuleDef_HEAD_INIT,
    "_ciffy_c",
    NULL,
    -1,
    methods
};

PyMODINIT_FUNC PyInit__ciffy_c(void) {
    return PyModule_Create(&_ciffy_c);
}
