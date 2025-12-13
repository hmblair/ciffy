#ifndef _CIFFY_PYTHON_H
#define _CIFFY_PYTHON_H

/**
 * @file pyutils.h
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


/* ============================================================================
 * NumPy array validation helpers
 * ============================================================================ */

/**
 * @brief Validate a 2D float32 C-contiguous NumPy array.
 *
 * Sets Python exception on failure and returns NULL.
 *
 * @param obj Python object to validate
 * @param expected_cols Expected number of columns (second dimension)
 * @param name Parameter name for error messages
 * @return PyArrayObject pointer, or NULL on error
 */
static inline PyArrayObject *validate_float32_2d(
    PyObject *obj, npy_intp expected_cols, const char *name
) {
    if (!PyArray_Check(obj)) {
        PyErr_Format(PyExc_TypeError, "%s must be a NumPy array", name);
        return NULL;
    }
    PyArrayObject *arr = (PyArrayObject *)obj;
    if (PyArray_NDIM(arr) != 2 || PyArray_DIM(arr, 1) != expected_cols) {
        PyErr_Format(PyExc_ValueError, "%s must have shape (N, %ld)", name, (long)expected_cols);
        return NULL;
    }
    if (PyArray_TYPE(arr) != NPY_FLOAT32) {
        PyErr_Format(PyExc_TypeError, "%s must be float32", name);
        return NULL;
    }
    if (!PyArray_IS_C_CONTIGUOUS(arr)) {
        PyErr_Format(PyExc_ValueError, "%s must be C-contiguous", name);
        return NULL;
    }
    return arr;
}

/**
 * @brief Validate a 1D float32 C-contiguous NumPy array.
 *
 * Sets Python exception on failure and returns NULL.
 *
 * @param obj Python object to validate
 * @param name Parameter name for error messages
 * @return PyArrayObject pointer, or NULL on error
 */
static inline PyArrayObject *validate_float32_1d(PyObject *obj, const char *name) {
    if (!PyArray_Check(obj)) {
        PyErr_Format(PyExc_TypeError, "%s must be a NumPy array", name);
        return NULL;
    }
    PyArrayObject *arr = (PyArrayObject *)obj;
    if (PyArray_NDIM(arr) != 1) {
        PyErr_Format(PyExc_ValueError, "%s must be 1D", name);
        return NULL;
    }
    if (PyArray_TYPE(arr) != NPY_FLOAT32) {
        PyErr_Format(PyExc_TypeError, "%s must be float32", name);
        return NULL;
    }
    if (!PyArray_IS_C_CONTIGUOUS(arr)) {
        PyErr_Format(PyExc_ValueError, "%s must be C-contiguous", name);
        return NULL;
    }
    return arr;
}

/**
 * @brief Validate a 2D int64 C-contiguous NumPy array.
 *
 * Sets Python exception on failure and returns NULL.
 *
 * @param obj Python object to validate
 * @param expected_cols Expected number of columns (second dimension)
 * @param name Parameter name for error messages
 * @return PyArrayObject pointer, or NULL on error
 */
static inline PyArrayObject *validate_int64_2d(
    PyObject *obj, npy_intp expected_cols, const char *name
) {
    if (!PyArray_Check(obj)) {
        PyErr_Format(PyExc_TypeError, "%s must be a NumPy array", name);
        return NULL;
    }
    PyArrayObject *arr = (PyArrayObject *)obj;
    if (PyArray_NDIM(arr) != 2 || PyArray_DIM(arr, 1) != expected_cols) {
        PyErr_Format(PyExc_ValueError, "%s must have shape (N, %ld)", name, (long)expected_cols);
        return NULL;
    }
    if (PyArray_TYPE(arr) != NPY_INT64) {
        PyErr_Format(PyExc_TypeError, "%s must be int64", name);
        return NULL;
    }
    if (!PyArray_IS_C_CONTIGUOUS(arr)) {
        PyErr_Format(PyExc_ValueError, "%s must be C-contiguous", name);
        return NULL;
    }
    return arr;
}

#endif /* _CIFFY_PYTHON_H */
