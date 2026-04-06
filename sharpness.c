#include <Python.h>
#include <math.h>
#include <stdint.h>

static PyObject* compute_sharpness(PyObject* self, PyObject* args) {
    Py_buffer view;
    int width, height;

    if (!PyArg_ParseTuple(args, "y*ii", &view, &width, &height))
        return NULL;

    uint8_t* data = (uint8_t*)view.buf;
    double mean = 0, var = 0;
    long count = 0;

    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int idx = (y * width + x) * 3;
            int top = ((y - 1) * width + x) * 3;
            int bot = ((y + 1) * width + x) * 3;
            int left = (y * width + (x - 1)) * 3;
            int right = (y * width + (x + 1)) * 3;

            float gray = data[idx] * 0.299f + data[idx + 1] * 0.587f +
                         data[idx + 2] * 0.114f;
            float tg = data[top] * 0.299f + data[top + 1] * 0.587f +
                       data[top + 2] * 0.114f;
            float bg = data[bot] * 0.299f + data[bot + 1] * 0.587f +
                       data[bot + 2] * 0.114f;
            float lg = data[left] * 0.299f + data[left + 1] * 0.587f +
                       data[left + 2] * 0.114f;
            float rg = data[right] * 0.299f + data[right + 1] * 0.587f +
                       data[right + 2] * 0.114f;

            float lap = 4 * gray - tg - bg - lg - rg;
            mean += lap;
            var += lap * lap;
            count++;
        }
    }

    mean /= count;
    var = var / count - mean * mean;

    PyBuffer_Release(&view);
    return PyFloat_FromDouble(var);
}

static PyMethodDef methods[] = {
    {"compute_sharpness", compute_sharpness, METH_VARARGS, "Compute sharpness"},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "sharpness_c", NULL,
                                    -1, methods};

PyMODINIT_FUNC PyInit_sharpness_c(void) { return PyModule_Create(&module); }