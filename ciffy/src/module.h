#ifndef _CIFFY_MODULE_H
#define _CIFFY_MODULE_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#include "io.h"
#include "python.h"
#include "parser.h"
#include "writer.h"
#include "registry.h"

#define __py_init() if (PyArray_API == NULL) { import_array(); }

#endif /* _CIFFY_MODULE_H */
