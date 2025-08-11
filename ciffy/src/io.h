#ifndef _CIFFY_IO_H
#define _CIFFY_IO_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#include "hash/lookup.h"

#define BAD_IX -1

typedef struct {

    // The block category

    char *category;

    // The number of attributes in the block

    int  attributes;

    // The number of entries in the block

    int  size;

    // The width of each line in the block (assuming it is constant)

    int  width;

    // Is this a single-entry block

    bool single;

    // Pointer to the start of the header

    char *head;

    // Pointer to the start of the data

    char *start;

    // Offset of each attribute

    int  *offsets;

} mmBlock;


char *_load_file(const char* name);
void  _advance_line(char **buffer);
int   _get_offset(char *buffer, char delimiter, int n);
int  *_get_offsets(char *buffer, int fields);
char *_get_field(char *buffer);
char *_get_field_and_advance(char **buffer);
char *_get_category(char *buffer);
char *_get_attr(char *buffer);
int  _get_attr_index(mmBlock *block, const char *attr);
char *_get_attr_by_line(mmBlock* block, int line, int index);
int  _str_to_int(const char *str);

typedef struct _LOOKUP *(*HashTable)(const char *, size_t);
int _lookup(HashTable func, char *token);

#endif
