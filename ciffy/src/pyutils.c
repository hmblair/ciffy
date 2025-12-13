/**
 * @file python.c
 * @brief Python C API helper functions.
 *
 * Provides utilities for converting between C types and Python objects.
 */

#include "pyutils.h"


const char *_get_filename(PyObject *args) {

    const char *filename;
    if (!PyArg_ParseTuple(args, "s", &filename)) {
        return NULL;  /* PyArg_ParseTuple sets exception */
    }
    return filename;
}


PyObject *_c_str_to_py_str(char *str) {

    if (str == NULL) {
        str = "";
    }

    PyObject *result = PyUnicode_FromString(str);
    if (result == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Failed to create Python string");
    }
    return result;
}


PyObject *_c_int_to_py_int(int value) {

    PyObject *result = PyLong_FromLong(value);
    if (result == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Failed to create Python int");
    }
    return result;
}


PyObject *_c_arr_to_py_list(char **arr, int size) {

    /* Handle NULL array */
    if (arr == NULL && size > 0) {
        PyErr_SetString(PyExc_ValueError,
            "Cannot convert NULL array with non-zero size to Python list");
        return NULL;
    }
    if (size <= 0) {
        return PyList_New(0);
    }

    PyObject *list = PyList_New(size);
    if (list == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Failed to create Python list");
        return NULL;
    }

    for (int ix = 0; ix < size; ix++) {

        char *str = arr[ix];
        PyObject *pystr = _c_str_to_py_str(str);
        if (pystr == NULL) {
            Py_DECREF(list);
            return NULL;  /* Exception already set */
        }

        /* PyList_SetItem steals reference, so no need to DECREF pystr */
        PyList_SetItem(list, ix, pystr);
    }

    return list;
}


/* ============================================================================
 * Python-to-C conversion functions (for writing)
 * ============================================================================ */

float *_numpy_to_float_arr(PyObject *arr, int *size) {

    if (!PyArray_Check(arr)) {
        PyErr_SetString(PyExc_TypeError, "Expected NumPy array");
        return NULL;
    }

    PyArrayObject *np_arr = (PyArrayObject *)arr;

    /* Must be contiguous for direct data access */
    if (!PyArray_IS_C_CONTIGUOUS(np_arr)) {
        PyErr_SetString(PyExc_ValueError, "Array must be C-contiguous");
        return NULL;
    }

    /* Check dtype is float32 */
    if (PyArray_TYPE(np_arr) != NPY_FLOAT32) {
        PyErr_SetString(PyExc_TypeError, "Array must be float32");
        return NULL;
    }

    if (size != NULL) {
        *size = (int)PyArray_SIZE(np_arr);
    }

    return (float *)PyArray_DATA(np_arr);
}


int *_numpy_to_int_arr(PyObject *arr, int *size) {

    if (!PyArray_Check(arr)) {
        PyErr_SetString(PyExc_TypeError, "Expected NumPy array");
        return NULL;
    }

    PyArrayObject *np_arr = (PyArrayObject *)arr;

    /* Must be contiguous for direct data access */
    if (!PyArray_IS_C_CONTIGUOUS(np_arr)) {
        PyErr_SetString(PyExc_ValueError, "Array must be C-contiguous");
        return NULL;
    }

    /* Check dtype is int32 */
    if (PyArray_TYPE(np_arr) != NPY_INT32) {
        PyErr_SetString(PyExc_TypeError, "Array must be int32");
        return NULL;
    }

    if (size != NULL) {
        *size = (int)PyArray_SIZE(np_arr);
    }

    return (int *)PyArray_DATA(np_arr);
}


char **_py_list_to_c_arr(PyObject *list, int *size) {

    if (!PyList_Check(list)) {
        PyErr_SetString(PyExc_TypeError, "Expected Python list");
        return NULL;
    }

    Py_ssize_t len = PyList_Size(list);
    if (size != NULL) {
        *size = (int)len;
    }

    char **arr = malloc((size_t)len * sizeof(char *));
    if (arr == NULL) {
        PyErr_NoMemory();
        return NULL;
    }

    for (Py_ssize_t i = 0; i < len; i++) {
        PyObject *item = PyList_GetItem(list, i);  /* Borrowed reference */
        if (!PyUnicode_Check(item)) {
            /* Free already allocated strings */
            for (Py_ssize_t j = 0; j < i; j++) {
                free(arr[j]);
            }
            free(arr);
            PyErr_SetString(PyExc_TypeError, "List must contain strings");
            return NULL;
        }

        Py_ssize_t str_len;
        const char *str = PyUnicode_AsUTF8AndSize(item, &str_len);
        if (str == NULL) {
            /* Free already allocated strings */
            for (Py_ssize_t j = 0; j < i; j++) {
                free(arr[j]);
            }
            free(arr);
            return NULL;  /* Exception already set */
        }

        /* Use known length to avoid strlen() call */
        arr[i] = malloc((size_t)str_len + 1);
        if (arr[i] == NULL) {
            /* Free already allocated strings */
            for (Py_ssize_t j = 0; j < i; j++) {
                free(arr[j]);
            }
            free(arr);
            PyErr_NoMemory();
            return NULL;
        }
        memcpy(arr[i], str, (size_t)str_len);
        arr[i][str_len] = '\0';
    }

    return arr;
}


void _free_c_str_arr(char **arr, int size) {
    if (arr == NULL) return;
    for (int i = 0; i < size; i++) {
        free(arr[i]);
    }
    free(arr);
}
