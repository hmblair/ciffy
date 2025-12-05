/**
 * @file py.c
 * @brief Python C API helper functions.
 *
 * Provides utilities for converting between C types and Python objects.
 */

#include "py.h"


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
