#ifndef _CIFFY_MODULE_H
#define _CIFFY_MODULE_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>

#include "cif/io.h"
#include "pyutils.h"
#include "cif/parser.h"
#include "cif/writer.h"
#include "cif/registry.h"

#define __py_init() if (PyArray_API == NULL) { import_array(); }

#endif /* _CIFFY_MODULE_H */
