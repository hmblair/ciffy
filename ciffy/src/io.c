#include "io.h"

static inline bool _eq(const char *str1, const char* str2) {

    return strncmp(str1, str2, strlen(str2)) == 0;

}

static inline bool _neq(const char *str1, const char* str2) {

    return strncmp(str1, str2, strlen(str2)) != 0;

}

static inline bool _is_section_end(const char* line) {

    return *line == '#';

}

char* _load_file(const char* name) {

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


int _get_offset(char *buffer, char delimiter, int n) {

    int offset = 0;

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


char *_get_field(char *buffer) {

    // Skip whitespace

    while (*buffer == ' ') { buffer++; }

    // Read until we see whitespace again
    // Ignore single quotes, unless inside double quotes

    bool squotes = false;
    bool dquotes = false;

    char *cpy = buffer;
    while (*buffer != ' ' || squotes) {
        if (*buffer == '\'' && !dquotes) { squotes = !squotes; }
        if (*buffer == '\"') { dquotes = !dquotes; }
        buffer++;
    }
    int length = buffer - cpy;

    char *field = malloc(length + 1);
    strncpy(field, cpy, length);
    field[length] = '\0';

    return field;

}


char *_get_field_and_advance(char **buffer) {

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


int _get_attr_index(mmBlock *block, const char *attr) {

    char *ptr = block->head;

    for (int ix = 0; ix < block->attributes; ix++) {

        char *curr = _get_attr(ptr);
        if (_eq(curr, attr)) { return ix; }
        _advance_line(&ptr);

    }

    return BAD_IX;

}


char *_get_attr_by_line(mmBlock* block, int line, int index) {

    if (block->single) {

        char *ptr = block->head;
        for (int ix = 0; ix < index; ix++) {
            _advance_line(&ptr);
        }

        (void)_get_field_and_advance(&ptr);
        return _get_field_and_advance(&ptr);

    } else {

        char *ptr = block->start + line * block->width + block->offsets[index];
        return _get_field(ptr);

    }

}


int _str_to_int(const char *str) {

    int base = 10;
    char *endptr = NULL;

    int val = strtol(str, &endptr, base);
    if (*endptr != '\0') { val = -1; }

    return val;

}


static inline char* _strip_quotes(char* str) {

    size_t len = strlen(str);
    if (str[len - 1] == '"') { str[len - 1] = '\0'; }
    if (str[0] == '"') { str++; } 
    return str;

}


int _lookup(HashTable func, char *token) {

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
