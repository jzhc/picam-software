#include <Python.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <turbojpeg.h>

/* ── SHARPNESS ─────────────────────────────────────────────────────────────
   Accepts a numpy buffer directly (zero copy), downsamples 2x in one pass,
   then computes Laplacian variance for sharpness score.
   Args: buffer, full_width, full_height
------------------------------------------------------------------------- */
static PyObject* compute_sharpness(PyObject* self, PyObject* args) {
    Py_buffer view;
    int fw, fh;

    if (!PyArg_ParseTuple(args, "y*ii", &view, &fw, &fh))
        return NULL;

    uint8_t* data = (uint8_t*)view.buf;

    int sw = fw / 2;
    int sh = fh / 2;

    // allocate downsampled grayscale buffer
    float* gray = (float*)malloc(sw * sh * sizeof(float));
    if (!gray) {
        PyBuffer_Release(&view);
        PyErr_NoMemory();
        return NULL;
    }

    // downsample and convert to grayscale in one pass
    for (int y = 0; y < sh; y++) {
        for (int x = 0; x < sw; x++) {
            int src = (y * 2 * fw + x * 2) * 3;
            gray[y * sw + x] = data[src] * 0.299f + data[src + 1] * 0.587f +
                               data[src + 2] * 0.114f;
        }
    }

    // Laplacian variance
    double mean = 0.0, var = 0.0;
    long count = 0;

    for (int y = 1; y < sh - 1; y++) {
        for (int x = 1; x < sw - 1; x++) {
            float g = gray[y * sw + x];
            float top = gray[(y - 1) * sw + x];
            float bot = gray[(y + 1) * sw + x];
            float lft = gray[y * sw + (x - 1)];
            float rgt = gray[y * sw + (x + 1)];
            float lap = 4 * g - top - bot - lft - rgt;
            mean += lap;
            var += lap * lap;
            count++;
        }
    }

    free(gray);
    PyBuffer_Release(&view);

    if (count == 0)
        return PyFloat_FromDouble(0.0);

    mean /= count;
    var = var / count - mean * mean;

    return PyFloat_FromDouble(var);
}

/* ── JPEG ENCODE ───────────────────────────────────────────────────────────
   Calls libturbojpeg C API directly — no Python wrapper overhead.
   Args: buffer, width, height, quality
   Returns: bytes object containing JPEG data
------------------------------------------------------------------------- */
static PyObject* encode_jpeg(PyObject* self, PyObject* args) {
    Py_buffer view;
    int width, height, quality;

    if (!PyArg_ParseTuple(args, "y*iii", &view, &width, &height, &quality))
        return NULL;

    tjhandle handle = tjInitCompress();
    if (!handle) {
        PyBuffer_Release(&view);
        PyErr_SetString(PyExc_RuntimeError, "Failed to init TurboJPEG");
        return NULL;
    }

    unsigned char* jpegBuf = NULL;
    unsigned long jpegSize = 0;

    int ret = tjCompress2(handle, (unsigned char*)view.buf, width, 0, height,
                          TJPF_RGB, &jpegBuf, &jpegSize, TJSAMP_420, quality,
                          TJFLAG_FASTDCT);

    tjDestroy(handle);
    PyBuffer_Release(&view);

    if (ret != 0) {
        PyErr_SetString(PyExc_RuntimeError, tjGetErrorStr());
        return NULL;
    }

    PyObject* result = PyBytes_FromStringAndSize((char*)jpegBuf, jpegSize);
    tjFree(jpegBuf);
    return result;
}

/* ── MODULE ────────────────────────────────────────────────────────────── */
static PyMethodDef methods[] = {{"compute_sharpness", compute_sharpness,
                                 METH_VARARGS,
                                 "Compute sharpness with 2x downsample"},
                                {"encode_jpeg", encode_jpeg, METH_VARARGS,
                                 "Encode RGB buffer to JPEG via libturbojpeg"},
                                {NULL, NULL, 0, NULL}};

static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "camera_utils", NULL,
                                    -1, methods};

PyMODINIT_FUNC PyInit_camera_utils(void) { return PyModule_Create(&module); }