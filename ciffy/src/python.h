#ifndef _CIFFY_PYTHON_H
#define _CIFFY_PYTHON_H

/**
 * @file python.h
 * @brief Python C API helper functions.
 *
 * Provides utilities for converting between C types and Python objects.
 */

#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

/* NumPy multi-file setup: module.c defines the API, other files import it */
#ifndef CIFFY_MAIN_MODULE
#define NO_IMPORT_ARRAY
#endif
#define PY_ARRAY_UNIQUE_SYMBOL CIFFY_ARRAY_API

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


/* ============================================================================
 * Python-to-C conversion functions (for writing)
 * ============================================================================ */

/**
 * @brief Extract float pointer from NumPy array (borrowed reference).
 *
 * @param arr NumPy array object
 * @param size Output for array size (may be NULL)
 * @return Pointer to float data, or NULL on error
 */
float *_numpy_to_float_arr(PyObject *arr, int *size);

/**
 * @brief Extract int pointer from NumPy array (borrowed reference).
 *
 * @param arr NumPy array object
 * @param size Output for array size (may be NULL)
 * @return Pointer to int data, or NULL on error
 */
int *_numpy_to_int_arr(PyObject *arr, int *size);

/**
 * @brief Convert Python list of strings to C string array.
 *
 * Caller is responsible for freeing the returned array and each string.
 *
 * @param list Python list object
 * @param size Output for array size
 * @return Array of C strings, or NULL on error
 */
char **_py_list_to_c_arr(PyObject *list, int *size);

/**
 * @brief Free C string array allocated by _py_list_to_c_arr.
 *
 * @param arr Array of C strings
 * @param size Number of elements
 */
void _free_c_str_arr(char **arr, int size);

#endif /* _CIFFY_PYTHON_H */
