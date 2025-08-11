#ifndef _CIFFY_H
#define _CIFFY_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#include "io.h"
#include "py.h"
#include "cif.h"

#define __py_init() if (PyArray_API == NULL) { import_array(); }

#endif
