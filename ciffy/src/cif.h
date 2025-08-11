#ifndef _CIFFY_CIF_H
#define _CIFFY_CIF_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#include "io.h"

typedef struct {

    char *id;
    char **names;
    char **descriptions;

    int models;
    int chains;
    int residues;
    int atoms;

    int nonpoly;

    float *coordinates;
    int   *types;
    int   *elements;

    int *sequence;
    int *res_per_chain;
    int *atoms_per_chain;
    int *atoms_per_res;

} mmCIF;

typedef struct {

    mmBlock atom;
    mmBlock poly;
    mmBlock nonpoly;
    mmBlock conn;
    mmBlock chain;
    mmBlock entity;

} mmBlockList;

void _fill_cif(mmCIF *cif, mmBlockList *blocks);
char *_get_id(char *buffer);

#endif
