#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>

#include "hash/atom.c"
#include "hash/residue.c"
#include "hash/element.c"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#define __py_init() if (PyArray_API == NULL) { import_array(); }

typedef struct Field Field;
typedef void (*Parser)(char *, Field *);
typedef void (*Init)(Field *);
typedef struct _LOOKUP *(*HashTable)(const char *, size_t);

struct Field {

    char *name;

    void *data;
    void *base;
    int size;
    int dsize;

    Parser parser;

    void *aux;

};

typedef struct {

    char *category;
    int size;
    int entries;
    bool single;
    Field **fields;

} Header;

typedef struct {

    char *id;
    char **names;

    int chains;
    int residues;
    int atoms;

    float *coordinates;
    int   *types;
    int   *elements;

    int *sequence;
    int *res_per_chain;
    int *atoms_per_chain;
    int *atoms_per_res;

} mmCIF;

typedef struct {

    char *category;
    int  attributes;
    int  size;
    int  width;

    bool single;
    bool var;

    char *head;
    char *start;
    int  *offsets;

} mmBlock;
