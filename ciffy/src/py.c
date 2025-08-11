#include "py.h"

const char *_get_filename(PyObject* args) {

    const char *filename;
    if (!PyArg_ParseTuple(args, "s", &filename)) { return NULL; }
    return filename;

}


PyObject *_c_str_to_py_str(char *str) {

    if (str == NULL) { str = ""; }
    return PyUnicode_FromString(str);

}


PyObject *_c_int_to_py_int(int value) {

    return PyLong_FromLong(value);

}


PyObject *_c_arr_to_py_list(char **arr, int size) {

    PyObject *list = PyList_New(size);

    for (int ix = 0; ix < size; ix++) {

        char *str = arr[ix];
        PyObject *pystr = _c_str_to_py_str(str);
        PyList_SetItem(list, ix, pystr);

    }

    return list;

}
