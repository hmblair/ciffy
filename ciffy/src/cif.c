#include "cif.h"

// Include these here to avoid duplicate symbols

#include "hash/atom.c"
#include "hash/residue.c"
#include "hash/element.c"


const size_t COORDS = 3;
const char *ATTR_X = "Cartn_x";
const char *ATTR_Y = "Cartn_y";
const char *ATTR_Z = "Cartn_z";


static inline bool _eq(char *str1, char* str2) {

    return strncmp(str1, str2, strlen(str2)) == 0;

}


static inline bool _neq(char *str1, char* str2) {

    return strncmp(str1, str2, strlen(str2)) != 0;

}


char *_get_id(char *buffer) {

    char *prefix = "data_";
    if (_neq(buffer, prefix)) { return NULL; }
    buffer += 5;

    char *cpy = buffer;
    while (*buffer != '\n') { buffer++; }
    int length = buffer - cpy;

    char *id = malloc(length + 1);
    strncpy(id, cpy, length);
    id[length] = '\0';

    return id;

}


int *_parse_int(mmBlock *block, const char *attr) {

    int index  = _get_attr_index(block, attr);
    if (index == BAD_IX) { return NULL; }

    int *array = calloc(block->size, sizeof(int));

    for (int line = 0; line < block->size; line++) {

        char *token = _get_attr_by_line(block, line, index);
        array[line] = _str_to_int(token);

    }

    return array;

}


char **_parse_str(mmBlock *block, const char *attr) {

    int index  = _get_attr_index(block, attr);
    if (index == BAD_IX) { return NULL; }

    char **array = calloc(block->size, sizeof(char *));

    for (int line = 0; line < block->size; line++) {

        array[line] = _get_attr_by_line(block, line, index);

    }

    return array;

}


char **_get_unique(mmBlock *block, const char *attr, int *size) {

    int index = _get_attr_index(block, attr);
    if (index == BAD_IX) { return NULL; }

    char **str = calloc(*size > 0 ? *size : block->size, sizeof(char *));

    char *prev = NULL;
    int ix     = 0;

    for (int line = 0; line < block->size; line++) {

        char *token = _get_attr_by_line(block, line, index);

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
    if (x_index == BAD_IX) { return NULL; }

    int y_index = _get_attr_index(block, ATTR_Y);
    if (y_index == BAD_IX) { return NULL; }

    int z_index = _get_attr_index(block, ATTR_Z);
    if (z_index == BAD_IX) { return NULL; }

    int *indices = malloc(COORDS * sizeof(int));
    indices[0] = x_index;
    indices[1] = y_index;
    indices[2] = z_index;

    float *array = calloc(COORDS * block->size, sizeof(float));

    for (int line = 0; line < block->size; line++) {

        for (int ix = 0; ix < COORDS; ix++) {
            char *token = _get_attr_by_line(block, line, indices[ix]);
            array[COORDS * line + ix] = strtof(token, NULL);
        }

    }

    free(indices);
    return array;

}


int *_parse_via_lookup(mmBlock *block, HashTable func, const char *attr) {

    int index  = _get_attr_index(block, attr);
    if (index == BAD_IX) { return NULL; }

    int *array = calloc(block->size, sizeof(int));

    for (int line = 0; line < block->size; line++) {

        char *token = _get_attr_by_line(block, line, index);
        array[line] = _lookup(func, token);

    }

    return array;

}


int _unique(mmBlock *block, const char *attr) {

    int index = _get_attr_index(block, attr);
    if (index == BAD_IX) { return BAD_IX; }

    int count = 0;
    char *prev = NULL;

    for (int line = 0; line < block->size; line++) {

        char *token = _get_attr_by_line(block, line, index);

        if (prev == NULL || _neq(prev, token)) {
            prev = token;
            count++;
        }

    }

    return count;

}


int *_parse_sizes_relative(mmBlock *block, const char *attr, int *size) {

    int index  = _get_attr_index(block, attr);
    if (index == BAD_IX) { return NULL; }

    int *sizes = calloc(*size > 0 ? *size : block->size, sizeof(int));

    char *prev = NULL;
    int ix     = 0;

    for (int line = 0; line < block->size; line++) {

        char *token = _get_attr_by_line(block, line, index);

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


int *_parse_residue_sizes(
    mmBlock *block,
    const char *attr,
    int size,
    int *nonpoly,
    int *lengths
) {

    int index  = _get_attr_index(block, attr);
    if (index == -1) { return NULL; }

    int cindex  = _get_attr_index(block, "label_asym_id");
    if (cindex == -1) { return NULL; }

    int *sizes = calloc(size, sizeof(int));
    int offset = 0;

    char *pchain = NULL;

    for (int line = 0; line < block->size; line++) {

        char *ctoken = _get_attr_by_line(block, line, cindex);
        if (pchain == NULL) { pchain = ctoken; }
        if (_neq(pchain, ctoken)) {
            pchain = ctoken;
            offset += *lengths;
            lengths++;
        }

        char *token = _get_attr_by_line(block, line, index);

        int num = _str_to_int(token) - 1;
        if (num < 0) { (*nonpoly)++; continue; }

        sizes[offset + num]++;

    }

    return sizes;

}


const char *MODEL  = "pdbx_PDB_model_num";
const char *CHAIN  = "id";

const char *RES_PER_CHAIN = "asym_id";

const char *RESIDUE = "mon_id";
const char *ELEMENT = "type_symbol";
const char *ATOM    = "label_atom_id";

const char *ATTR_RESIDUE_NUM = "label_seq_id";
const char *ATTR_CHAIN_ID_3 = "label_asym_id";

void _fill_cif(mmCIF *cif, mmBlockList *blocks) {

    // Count the number of models, chains, residues, atoms

    cif->models   = _unique(&blocks->atom, MODEL);
    cif->chains   = blocks->chain.size;
    cif->residues = blocks->poly.size;

    blocks->atom.size /= cif->models;
    cif->atoms = blocks->atom.size;

    // Count the residues in each chain

    cif->res_per_chain = _parse_sizes_relative(&blocks->poly, RES_PER_CHAIN , &cif->chains);

    // Get chain names and the residue sequence

    cif->names    = _get_unique(&blocks->chain, CHAIN, &cif->chains);
    cif->sequence = _parse_via_lookup(&blocks->poly, _lookup_residue, RESIDUE);

    // Read the atoms, elements, and coordinates

    cif->types    = _parse_via_lookup(&blocks->atom, _lookup_atom, ATOM);
    cif->elements = _parse_via_lookup(&blocks->atom, _lookup_element, ELEMENT);
    cif->coordinates = _parse_coords(&blocks->atom);

    // Compute the number of atoms in each residue and chain

    cif->atoms_per_res = _parse_residue_sizes(&blocks->atom, ATTR_RESIDUE_NUM, cif->residues, &cif->nonpoly, cif->res_per_chain);
    cif->atoms_per_chain = _parse_sizes_relative(&blocks->atom, ATTR_CHAIN_ID_3, &cif->chains);

}
