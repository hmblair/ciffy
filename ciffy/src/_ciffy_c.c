#include "_ciffy_c.h"

const size_t COORDS = 3;

const char *ATTR_X = "Cartn_x";
const char *ATTR_Y = "Cartn_y";
const char *ATTR_Z = "Cartn_z";


bool _is_prefix(char *str, char* prefix) {

    return strncmp(str, prefix, strlen(prefix)) == 0;

}

bool _eq(char *str1, char* str2) {

    return strncmp(str1, str2, strlen(str2)) == 0;

}

bool _neq(char *str1, char* str2) {

    return strncmp(str1, str2, strlen(str2)) != 0;

}

char* _strip_quotes(char* str) {

    size_t len = strlen(str);
    if (str[len - 1] == '"') { str[len - 1] = '\0'; }
    if (str[0] == '"') { str++; } 
    return str;

}

void _allocate_and_increment_int(void **data, int val) {

    int *tmp = (int *)(*data);
    *(tmp++) = val;
    *data = (void *)tmp;

}

void _allocate_and_increment_float(void **data, float val) {

    float *tmp = (float *)(*data);
    *(tmp++) = val;
    *data = (void *)tmp;

}

void _increment_if_different(char *token, void **data, char** prev, int* total) {

    if (*prev == NULL) {
        *prev = token;
        (*total)++;
     }

    int *tmp = (int *)(*data);

    if (strcmp(token, *prev) != 0) {
        tmp++;
        *data = (void *)tmp;
        *prev = token;
        (*total)++;
    }

    (*tmp)++;

}

struct _LOOKUP *_parse_atom(char *token, int size) {

    token = _strip_quotes(token);
    return _lookup_atom(token, strlen(token));

}

void _parse_element(char *token, Field *field) {

    int val = -1;

    struct _LOOKUP *lookup = _lookup_element(token, strlen(token));
    if (lookup != NULL) { val = lookup->value; }
    // else { fprintf(stderr, "Failed to tokenize the element \"%s\".\n", token); }

    _allocate_and_increment_int(&field->data, val);

}


void _parse_model_num(char *token, Field *field) {

    int *tmp = (int *)(field->data);

    char *endptr;
    int num = strtol(token, &endptr, 10);
    // if (*endptr != '\0') {}

    tmp[num - 1]++;

}


void _parse_residue(char *token, Field *field) {

    int *tmp    = (int *)(field->data);

    if (token[0] == '.') {
        (*tmp)++;
        return;
    }

    char *endptr;
    int num = strtol(token, &endptr, 10);
    // if (*endptr != '\0') {}

    if (field->aux == NULL) {
        field->aux = calloc(1, sizeof(int));
        *(int *)field->aux = 1;
    }

    int prev = *(int *)field->aux;

    // If this is a new chain

    if (prev > num) {
        tmp += num;
        field->data = (void *)tmp;
    }

    // If this is a new residue

    else if (prev < num) {
        tmp += (num - prev);
        field->data = (void *)tmp;
    }

    (*tmp)++;
    *(int *)field->aux = num;

}

void _parse_sequence(char *token, Field *field) {

    int *tmp = (int *)(field->data);
    int val = -1;

    struct _LOOKUP *lookup = _lookup_residue(token, strlen(token));
    if (lookup != NULL) { val = lookup->value; }
    // else { fprintf(stderr, "Failed to tokenize the residue \"%s\".\n", token); }
    *tmp = val;
    tmp++;
    field->data = (void *)tmp;

}

void _parse_chain(char *token, Field *field) {

    int *tmp = (int *)(field->data);

    // If we advanced the pointer last time round

    if (field->aux == NULL) {

        field->aux = malloc(strlen(token) + 1);
        strcpy((char *)field->aux, token);

    }

    // If this is a new chain

    if (strcmp((char *)field->aux, token) != 0) {

        tmp++;
        field->data = (void *)tmp;

        free((char *)field->aux);
        field->aux = NULL;

    }

    (*tmp)++;

}

void _parse_chain_id(char *token, Field *field) {

    char **tmp = (char **)(field->data);

    *tmp = malloc(strlen(token) + 1);
    strcpy(*tmp, token);

    tmp++;
    field->data = (void *)tmp;

}

void _parse_coordinate(char *token, Field *field) {

    float val = strtof(token, NULL);
    _allocate_and_increment_float(&field->data, val);

}


int _get_offset(char *buffer, char delimiter, int n) {

    int offset  = 0;

    // Delimiters within single quotes are ignored
    // Single quotes within double quotes are ignored

    bool squotes = false;
    bool dquotes = false;

    for (int ix = 0; ix < n; ix++) {
        while (*buffer != delimiter || squotes) {
            if (*buffer == '\'' && !dquotes) { squotes = !squotes; }
            if (*buffer == '\"') { dquotes = !dquotes; }
            buffer++;
            offset++;
        }
        while (*buffer == delimiter) {
            buffer++;
            offset++;
        }
    }

    return offset;

}


int *_get_offsets(char *buffer, int fields) {

    int *offsets = calloc(fields + 1, sizeof(int));
    for (int ix = 0; ix <= fields; ix++) {
        offsets[ix] = _get_offset(buffer, ' ', ix);
    }

    return offsets;

}


int _get_width(char *buffer) {

    return _get_offset(buffer, '\n', 1);

}



char *_get_field(char *buffer) {

    // Skip whitespace

    while (*buffer == ' ') { buffer++; }

    // Read until we see whitespace again

    char *cpy = buffer;
    while (*buffer != ' ') { buffer++; }
    int length = buffer - cpy;

    char *field = malloc(length + 1);
    strncpy(field, cpy, length);
    field[length] = '\0';

    return field;

}



char *_get_next_field(char **buffer) {

    // Skip whitespace

    while (**buffer == ' ') { (*buffer)++; }

    // Read until we see whitespace again

    char *cpy = *buffer;
    while (**buffer != ' ') { (*buffer)++; }
    int length = *buffer - cpy;

    char *field = malloc(length + 1);
    strncpy(field, cpy, length);
    field[length] = '\0';

    return field;

}





bool _is_entry(char *line) {

    return strncmp(line, "_entry.id", 9) == 0;

}

bool _is_loop(char *line) {

    return strncmp(line, "loop_", 5) == 0;

}

bool _is_section_end(char* line) {

    return line[0] == '#';

}

PyObject *_init_1d_arr_int(int size, int* data) {

    npy_intp dims[1] = {size};
    return PyArray_SimpleNewFromData(1, dims, NPY_INT, data);

}

PyObject *_init_2d_arr_int(int size, int* data) {

    npy_intp dims[2] = {size, 2};
    return PyArray_SimpleNewFromData(2, dims, NPY_INT, data);

}

PyObject *_init_1d_arr_float(int size, float* data) {

    npy_intp dims[1] = {size};
    return PyArray_SimpleNewFromData(1, dims, NPY_FLOAT, data);

}

PyObject *_init_2d_arr_float(int size, float* data) {

    npy_intp dims[2] = {size, COORDS};
    return PyArray_SimpleNewFromData(2, dims, NPY_FLOAT, data);

}
















static inline char* _load_file(const char* name) {

    FILE *file = fopen(name, "r");

    if (file == NULL) {
        perror("Failed to open file");
        return NULL;
    }

    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    fseek(file, 0, SEEK_SET);

    char* buffer = malloc(size + 1);
    if (buffer == NULL) {
        perror("Failed to allocate memory");
        fclose(file);
        return NULL;
    }

    fread(buffer, 1, size, file);
    buffer[size] = '\0';
    fclose(file);

    return buffer;

}


void _advance_line(char **buffer) {

    while (**buffer != '\n' && **buffer != '\0') { (*buffer)++; }
    if (**buffer == '\n') { (*buffer)++; }

}

char *_get_category(char *buffer) {

    char *pos = strchr(buffer, '.');
    size_t length = pos - buffer + 1;

    char *result = malloc(length + 1);
    strncpy(result, buffer, length);
    result[length - 1] = '.';
    result[length] = '\0';

    return result;

}

char *_get_attr(char *buffer) {

    char *start = strchr(buffer, '.') + 1;
    char *end = strchr(start, ' ');
    size_t length = end - start;

    char *result = malloc(length + 1);
    strncpy(result, start, length);
    result[length] = '\0';

    return result;

}


void _fill_header_entry(Header* header, Field* field, int ix) {

    // The header tells us how large the field will be

    if (header->single) {
        field->size = 1;
    } else {
        field->size = header->entries;
    }

    // Allocate data if we haven't already

    if (field->data == NULL) {
        field->data = calloc(field->size, field->dsize);
    }

    // Keep track of the beginning of the data block

    field->base = field->data;

    // Insert the field into the correct position

    header->fields[ix] = field;

}

void _free_header(Header *header) {

    if (header->category != NULL) {
        free(header->category);
        header->category = NULL;
    }

    if (header->fields != NULL) {
        free(header->fields);
        header->fields = NULL;
    }

}

void _free_field(Field *field) {

    field->data = NULL;
    field->base = NULL;

    if (field->aux != NULL) {
        free(field->aux);
        field->aux = NULL;
    }

}


const char *_get_filename(PyObject* args) {

    const char *filename;
    if (!PyArg_ParseTuple(args, "s", &filename)) { return NULL; }
    return filename;

}


void _read_fields(Header header, char **buffer) {

    for (int line = 0; line < header.entries; line++) {

        for (int record = 0; record < header.size; record++) {

            // Read the next token on the line

            char *token = _get_next_field(buffer);

            // Get a pointer to the relevant field

            Field *field = header.fields[record];
            if (field == NULL) { continue; }

            // Parse the data

            field->parser(token, field);

        }

        _advance_line(buffer);

    }

}


PyObject *_c_str_to_str(char *str) {

    if (str == NULL) { str = ""; }
    return PyUnicode_FromString(str);

}


PyObject *_c_arr_to_list(char **arr, int size) {

    PyObject *list = PyList_New(size);

    for (int ix = 0; ix < size; ix++) {

        char *str = arr[ix];
        PyObject *pystr = _c_str_to_str(str);
        PyList_SetItem(list, ix, pystr);

    }

    return list;

}


PyObject* _c_to_py(mmCIF cif) {

    PyObject *py_id = _c_str_to_str(cif.id);
    PyObject *chain_names_list = _c_arr_to_list(cif.names, cif.chains);

    PyObject *coordinates = _init_2d_arr_float(cif.atoms, cif.coordinates);

    PyObject *atoms_array    = _init_1d_arr_int(cif.atoms, cif.types);
    PyObject *elements_array = _init_1d_arr_int(cif.atoms, cif.elements);
    PyObject *residues_array = _init_1d_arr_int(cif.residues, cif.sequence);

    PyObject *atoms_per_res = _init_1d_arr_int(cif.residues, cif.atoms_per_res);
    PyObject *atoms_per_chain = _init_1d_arr_int(cif.chains, cif.atoms_per_chain);
    PyObject *res_per_chain = _init_1d_arr_int(cif.chains, cif.res_per_chain);

    return PyTuple_Pack(9, py_id, coordinates, atoms_array, elements_array, residues_array, atoms_per_res, atoms_per_chain, res_per_chain, chain_names_list);

}


char *_get_id(char **buffer) {

    (void)_get_next_field(buffer);
    char *tmp = _get_next_field(buffer);

    int len = strlen(tmp);
    char *id = malloc(len + 1);
    strcpy(id, tmp);

    return id;

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


int _get_attr_index(mmBlock *block, const char *attr) {

    char *ptr = block->head;

    for (int ix = 0; ix < block->attributes; ix++) {

        char *curr = _get_attr(ptr);
        if (strncmp(curr, attr, strlen(attr)) == 0) { return ix; }
        _advance_line(&ptr);

    }

    return -1;

}


mmBlock _read_block(char **buffer) {

    mmBlock block = {0};

    // Check if this is a single-entry block and handle appropriately
    // *buffer now points to the beginning of the header

    if (_is_loop(*buffer)) {
        _advance_line(buffer);
    } else {
        block.single = true;
    }

    block.head     = *buffer;
    block.category = _get_category(block.head);

    // Count the number of attributes in the header
    // *buffer now points to the beginning of the data

    while (_is_prefix(*buffer, block.category)) {

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

    } else {

        block.size = 1;

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


char *_read_attribute(mmBlock* block, int line, int attr) {

    if (block->single) {

        char *ptr = block->head;
        for (int ix = 0; ix < attr; ix++) {
            _advance_line(&ptr);
        }

        (void)_get_next_field(&ptr);
        return _get_next_field(&ptr);

    } else {

        char *ptr = block->start + line * block->width + block->offsets[attr];
        return _get_field(ptr);

    }

}


bool _parse_data_entry(char **buffer) {

    if (_is_prefix(*buffer, "data_")) {
        _advance_line(buffer);
        _advance_line(buffer);
        return true;
    }

    return false;

}





int _str_to_int(const char *str) {

    int base = 10;
    char *endptr = NULL;

    int val = strtol(str, &endptr, base);
    if (*endptr != '\0') { val = -1; }

    return val;

}


int _lookup(HashTable func, const char *token) {

    int val = -1;
    token = _strip_quotes(token);
    struct _LOOKUP *lookup = func(token, strlen(token));
    if (lookup != NULL) {
        val = lookup->value;
    } else {
        // printf("Failed to parse %s.\n", token);
    }

    return val;

}


int *_parse_int(mmBlock *block, const char *attr) {

    int index  = _get_attr_index(block, attr);
    if (index == -1) { return 0; }

    int *array = calloc(block->size, sizeof(int));

    for (int line = 0; line < block->size; line++) {

        char *token = _read_attribute(block, line, index);
        array[line] = _str_to_int(token);

    }

    return array;

}


int *_parse_int_pair(mmBlock *block, const char *attr1, const char *attr2) {

    int index1  = _get_attr_index(block, attr1);
    if (index1 == -1) { return 0; }
    int index2  = _get_attr_index(block, attr2);
    if (index2 == -1) { return 0; }

    int *array = calloc(2 * block->size, sizeof(int));

    for (int line = 0; line < block->size; line++) {

        char *token1 = _read_attribute(block, line, index1);
        char *token2 = _read_attribute(block, line, index2);

        array[2 * line + 0] = _str_to_int(token1);
        array[2 * line + 1] = _str_to_int(token2);

    }

    return array;

}


char **_parse_str(mmBlock *block, const char *attr) {

    int index  = _get_attr_index(block, attr);
    if (index == -1) { return NULL; }

    char **array = calloc(block->size, sizeof(char *));

    for (int line = 0; line < block->size; line++) {

        array[line] = _read_attribute(block, line, index);

    }

    return array;

}




char **_get_unique(mmBlock *block, const char *attr, int *size) {

    int index = _get_attr_index(block, attr);
    if (index == -1) { return NULL; }

    char **str = calloc(*size > 0 ? *size : block->size, sizeof(char *));

    char *prev = NULL;
    int ix     = 0;

    for (int line = 0; line < block->size; line++) {

        char *token = _read_attribute(block, line, index);

        if (prev == NULL) {
            prev = token;
            str[ix] = token;
        }

        if(_neq(prev, token)) {
            prev = token; ix++;
            str[ix] = token;
        }

    }

    if (*size > 0) {
        return str;
    } else {
        *size = ix + 1;
        return realloc(str , *size * sizeof(int));
    }

}



float *_parse_coords(mmBlock *block) {

    int x_index = _get_attr_index(block, ATTR_X);
    int y_index = _get_attr_index(block, ATTR_Y);
    int z_index = _get_attr_index(block, ATTR_Z);

    int *indices = malloc(COORDS * sizeof(int));
    indices[0] = x_index;
    indices[1] = y_index;
    indices[2] = z_index;

    float *array = calloc(COORDS * block->size, sizeof(float));

    for (int line = 0; line < block->size; line++) {

        for (int ix = 0; ix < COORDS; ix++) {
            char *token = _read_attribute(block, line, indices[ix]);
            array[COORDS * line + ix] = strtof(token, NULL);
        }

    }

    free(indices);
    return array;

}


int *_parse_via_lookup(mmBlock *block, HashTable func, const char *attr) {

    int index  = _get_attr_index(block, attr);
    if (index == -1) { return NULL; }

    int *array = calloc(block->size, sizeof(int));

    for (int line = 0; line < block->size; line++) {

        char *token = _read_attribute(block, line, index);
        array[line] = _lookup(func, token);

    }

    return array;

}


int _count_first_unique(mmBlock *block, const char *attr) {

    int index  = _get_attr_index(block, attr);
    if (index == -1) { return 0; }

    int count = 0;
    char *prev = NULL;

    for (int line = 0; line < block->size; line++) {

        char *token = _read_attribute(block, line, index);

        if (prev == NULL) { prev = token; }
        if (_neq(prev, token)) { break; }

        count++;

    }

    return count;

}


int *_parse_sizes_relative(mmBlock *block, const char *attr, int *size) {

    int index  = _get_attr_index(block, attr);
    if (index == -1) { return NULL; }

    int *sizes = calloc(*size > 0 ? *size : block->size, sizeof(int));

    char *prev = NULL;
    int ix     = 0;

    for (int line = 0; line < block->size; line++) {

        char *token = _read_attribute(block, line, index);

        if (prev == NULL) { prev = token; }
        if (_neq(prev, token)) { prev = token; ix++; }

        sizes[ix]++;

    }

    if (*size > 0) {
        return sizes;
    } else {
        *size = ix + 1;
        return realloc(sizes, *size * sizeof(int));
    }

}


int *_parse_sizes_absolute(mmBlock *block, const char *attr, int *size) {

    int index  = _get_attr_index(block, attr);
    if (index == -1) { return NULL; }

    int *sizes = calloc(*size > 0 ? *size : block->size, sizeof(int));

    int prev = 1;
    int ix   = 0;

    for (int line = 0; line < block->size; line++) {

        char *token = _read_attribute(block, line, index);

        int num = _str_to_int(token);
        if (num < 0) { continue; }

        // If this is a new chain

        if (prev > num) { ix += num; }

        // If this is a new residue

        else if (prev < num) { ix += (num - prev); }

        sizes[ix]++;
        prev = num;

    }

    if (*size > 0) {
        return sizes;
    } else {
        *size = ix + 1;
        return realloc(sizes, *size * sizeof(int));
    }

}


const char *IS_HETERO = "group_PDB";
int *_is_hetero(mmBlock *block) {

    int index  = _get_attr_index(block, IS_HETERO);
    if (index == -1) { return NULL; }

    int *hetero = calloc(block->size, sizeof(int));

    for (int line = 0; line < block->size; line++) {

        char *token = _read_attribute(block, line, index);
        if (_is_prefix(token, "HETATM")) { hetero[line] = 1; }

    }

    return hetero;

}


typedef struct {

    mmBlock id;
    mmBlock atom;
    mmBlock poly;
    mmBlock nonpoly;
    mmBlock conn;

} mmBlockList;


void _store_or_free_block(mmBlock *block, mmBlockList* blocks) {

    if (_eq(block->category, "_entry.")) {
        blocks->id = *block;
        return;
    }

    if (_eq(block->category, "_atom_site.")) {
        blocks->atom = *block;
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

}


const char *ATTR_CHAIN_ID = "id";
const char *ATTR_RESIDUE = "mon_id";

const char *ATTR_CHAIN_ID_2 = "asym_id";

const char *ATTR_MODEL_NUM = "pdbx_PDB_model_num";
const char *ATTR_ELEMENT = "type_symbol";
const char *ATTR_ATOM = "label_atom_id";
const char *ATTR_RESIDUE_NUM = "label_seq_id";
const char *ATTR_CHAIN_ID_3 = "label_asym_id";


static PyObject* _load(PyObject* self, PyObject* args) {

    __py_init();
    const char *file = _get_filename(args);
    if (file == NULL) { return NULL; }

    char *buffer = _load_file(file);
    char *cpy    = buffer;
    if (buffer == NULL) {
        return PyErr_Format(PyExc_IOError, "Failed to open file: %s", file);
    }

    mmCIF cif = {0};
    mmBlockList blocks = {0};

    if (!_parse_data_entry(&buffer)) {
        return PyErr_Format(PyExc_IOError, "Invalid mmCIF file: %s", file);
    }

    while (*buffer != '\0') {

        mmBlock block = _read_block(&buffer);
        _store_or_free_block(&block, &blocks);

    }

    cif.id = *_parse_str(&blocks.id, "id");

    cif.atoms = _count_first_unique(&blocks.atom, ATTR_MODEL_NUM);
    blocks.atom.size = cif.atoms;

    cif.res_per_chain = _parse_sizes_relative(&blocks.poly, ATTR_CHAIN_ID_2, &cif.chains);
    cif.names = _get_unique(&blocks.poly, ATTR_CHAIN_ID_2, &cif.chains);

    cif.residues = blocks.poly.size;
    cif.sequence = _parse_via_lookup(&blocks.poly, _lookup_residue, ATTR_RESIDUE);

    // Read the atoms, elements, and coordinates

    cif.elements = _parse_via_lookup(&blocks.atom, _lookup_element, ATTR_ELEMENT);
    cif.types    = _parse_via_lookup(&blocks.atom, _lookup_atom, ATTR_ATOM);
    cif.coordinates = _parse_coords(&blocks.atom);

    // Compute the number of atoms in each residue and chain

    cif.atoms_per_res = _parse_sizes_absolute(&blocks.atom, ATTR_RESIDUE_NUM, &cif.residues);
    cif.atoms_per_chain = _parse_sizes_relative(&blocks.atom, ATTR_CHAIN_ID_3, &cif.chains);

    free(cpy);
    _free_block_list(&blocks);

    return _c_to_py(cif);

}

static PyMethodDef methods[] = {
    {"_load", _load, METH_VARARGS, "Load CIF file and return coordinates, elements, and residues as NumPy arrays"},
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
