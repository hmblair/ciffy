#ifndef _CIFFY_PY_H
#define _CIFFY_PY_H

/**
 * @file py.h
 * @brief Python C API helper functions.
 *
 * Provides utilities for converting between C types and Python objects.
 */

#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

/**
 * @brief Extract filename string from Python arguments.
 *
 * @param args Python argument tuple
 * @return Filename string (borrowed reference), or NULL on error
 */
const char *_get_filename(PyObject *args);

/**
 * @brief Convert C string to Python string object.
 *
 * @param str C string (NULL is converted to empty string)
 * @return New Python string object, or NULL on error
 */
PyObject *_c_str_to_py_str(char *str);

/**
 * @brief Convert C string array to Python list of strings.
 *
 * @param arr Array of C strings
 * @param size Number of elements in array
 * @return New Python list object, or NULL on error
 */
PyObject *_c_arr_to_py_list(char **arr, int size);

/**
 * @brief Convert C int to Python int object.
 *
 * @param value Integer value
 * @return New Python int object, or NULL on error
 */
PyObject *_c_int_to_py_int(int value);

#endif /* _CIFFY_PY_H */
