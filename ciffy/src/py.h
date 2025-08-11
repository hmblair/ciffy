#ifndef _CIFFY_PY_H
#define _CIFFY_PY_H

#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

const char *_get_filename(PyObject* args);
PyObject *_c_str_to_py_str(char *str);
PyObject *_c_arr_to_py_list(char **arr, int size);
PyObject *_c_int_to_py_int(int value);

#endif
