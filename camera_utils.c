/*
 * camera_utils.c — Optimised Pi HQ Camera C helpers
 *
 * Optimisations vs. original:
 *   1. Fused 2x-downsample + Laplacian in a 3-row ring buffer — no
 *      full-frame malloc, memory-bandwidth cut by ~50 %.
 *   2. ARM NEON intrinsics for the Laplacian accumulation loop —
 *      4 floats per cycle on the Cortex-A53.
 *   3. GIL released during all CPU-bound work so the Python thread pool
 *      can overlap sharpness and JPEG encoding simultaneously.
 *   4. Persistent TurboJPEG handle + pre-allocated output buffer —
 *      no per-frame heap churn or handle init/destroy overhead.
 *   5. TJFLAG_NOREALLOC — turbojpeg never calls realloc once the buffer
 *      reaches its high-water mark.
 */

#include <Python.h>
#include <stdint.h>
#include <stdlib.h>
#include <turbojpeg.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
#define USE_NEON 1
#endif

/* ── STATIC RESOURCES ────────────────────────────────────────────────────
   Three ring-buffer rows (max width 1920) replace the per-call malloc.
   _jpeg_buf grows when needed but never shrinks.
   _tj_handle is initialised once and reused.                              */
#define MAX_PREVIEW_W 1920

static float _ring[3][MAX_PREVIEW_W];
static unsigned char* _jpeg_buf = NULL;
static unsigned long _jpeg_buf_size = 0;
static tjhandle _tj_handle = NULL;

/* ── HELPERS ─────────────────────────────────────────────────────────────*/

/*
 * Downsample row ds_y (2×) from the full-res RGB image and convert to grey.
 * Source row = ds_y*2; horizontal stride = 6 bytes (skip one RGB pixel).
 */
static inline void downsample_row(const uint8_t* src, int row_stride, int ds_y,
                                  float* dst, int sw) {
    const uint8_t* row = src + (size_t)ds_y * 2 * row_stride;
    for (int x = 0; x < sw; ++x) {
        const uint8_t* p = row + x * 6;
        dst[x] = p[0] * 0.299f + p[1] * 0.587f + p[2] * 0.114f;
    }
}

/*
 * Accumulate Laplacian stats for one interior row of the downsampled image.
 *
 * NEON path  — 4 floats/cycle; lap = 4·cur − prev − next − left − right.
 * Scalar tail — handles remainder + non-NEON fallback.
 */
static void laplacian_row(const float* restrict prev, const float* restrict cur,
                          const float* restrict next, int sw, double* sum,
                          double* sum_sq, long* count) {
    int x = 1;
    int end = sw - 1;

#ifdef USE_NEON
    float32x4_t vacc = vdupq_n_f32(0.f);
    float32x4_t vacc2 = vdupq_n_f32(0.f);

    /* Each NEON pass covers columns x … x+3.
       Left  = vld1q(cur+x-1) → [cur[x-1], cur[x], cur[x+1], cur[x+2]]
       Right = vld1q(cur+x+1) → [cur[x+1], cur[x+2], cur[x+3], cur[x+4]]
       Bounds: x ≥ 1 and x+4 ≤ end ≤ sw-1, so cur[x+4] ≤ cur[sw-1]. ✓  */
    for (; x + 4 <= end; x += 4) {
        float32x4_t c = vld1q_f32(cur + x);
        float32x4_t top = vld1q_f32(prev + x);
        float32x4_t bot = vld1q_f32(next + x);
        float32x4_t lft = vld1q_f32(cur + x - 1);
        float32x4_t rgt = vld1q_f32(cur + x + 1);

        float32x4_t lap = vsubq_f32(
            vsubq_f32(vsubq_f32(vsubq_f32(vmulq_n_f32(c, 4.f), top), bot), lft),
            rgt);

        vacc = vaddq_f32(vacc, lap);
        vacc2 = vmlaq_f32(vacc2, lap, lap);
    }
    /* Elements processed = x − 1  (x started at 1, steps of 4). */
    *count += (long)(x - 1);

    /* Horizontal reduction. */
    float32x2_t s2 = vpadd_f32(vget_low_f32(vacc), vget_high_f32(vacc));
    float32x2_t sq2 = vpadd_f32(vget_low_f32(vacc2), vget_high_f32(vacc2));
    s2 = vpadd_f32(s2, s2);
    sq2 = vpadd_f32(sq2, sq2);
    *sum += (double)vget_lane_f32(s2, 0);
    *sum_sq += (double)vget_lane_f32(sq2, 0);
#endif

    for (; x < end; ++x) {
        float lap = 4.f * cur[x] - prev[x] - next[x] - cur[x - 1] - cur[x + 1];
        *sum += (double)lap;
        *sum_sq += (double)lap * lap;
        ++(*count);
    }
}

/* ── SHARPNESS ───────────────────────────────────────────────────────────
 * compute_sharpness(buffer, full_width, full_height) -> float
 *
 * Accepts any buffer-protocol object (numpy array, memoryview, bytes).
 * GIL is released for the entire computation so Python's thread pool can
 * run encode_jpeg concurrently on another core.
 *
 * Returns the Laplacian variance (higher = sharper image).               */
static PyObject* compute_sharpness(PyObject* self, PyObject* args) {
    Py_buffer view;
    int fw, fh;

    if (!PyArg_ParseTuple(args, "y*ii", &view, &fw, &fh))
        return NULL;

    int sw = fw / 2;
    int sh = fh / 2;

    if (sw > MAX_PREVIEW_W || sw < 3 || sh < 3) {
        PyBuffer_Release(&view);
        PyErr_SetString(PyExc_ValueError,
                        "Image dimensions out of supported range "
                        "(max width 1920, min 6x6)");
        return NULL;
    }

    const uint8_t* data = (const uint8_t*)view.buf;
    const int stride = fw * 3;
    double dsum = 0.0, dsum_sq = 0.0;
    long count = 0;

    Py_BEGIN_ALLOW_THREADS /* ── GIL released ── */

        /* Prime ring rows 0 and 1 before entering the sliding window loop. */
        downsample_row(data, stride, 0, _ring[0], sw);
    downsample_row(data, stride, 1, _ring[1], sw);

    /*
     * Slide a 3-row window.  For y = 1 … sh-2:
     *   a) Compute downsampled row y+1 into the free ring slot.
     *   b) Accumulate Laplacian stats for row y (has prev, cur, next).
     */
    for (int y = 1; y < sh - 1; ++y) {
        int slot_next = (y + 1) % 3;
        downsample_row(data, stride, y + 1, _ring[slot_next], sw);
        laplacian_row(_ring[(y - 1) % 3], _ring[y % 3], _ring[slot_next], sw,
                      &dsum, &dsum_sq, &count);
    }

    Py_END_ALLOW_THREADS /* ── GIL reacquired ── */

        PyBuffer_Release(&view);

    if (count == 0)
        return PyFloat_FromDouble(0.0);

    double mean = dsum / (double)count;
    double var = dsum_sq / (double)count - mean * mean;
    return PyFloat_FromDouble(var < 0.0 ? 0.0 : var);
}

/* ── JPEG ENCODE ─────────────────────────────────────────────────────────
 * encode_jpeg(buffer, width, height, quality) -> bytes
 *
 * Persistent handle + pre-allocated buffer + TJFLAG_NOREALLOC means zero
 * heap activity after the first frame at each resolution.
 * GIL released during the actual JPEG compression loop.                  */
static PyObject* encode_jpeg(PyObject* self, PyObject* args) {
    Py_buffer view;
    int width, height, quality;

    if (!PyArg_ParseTuple(args, "y*iii", &view, &width, &height, &quality))
        return NULL;

    /* Grow the output buffer only when necessary. */
    unsigned long needed = (unsigned long)width * height * 3 + 65536UL;
    if (!_jpeg_buf || _jpeg_buf_size < needed) {
        if (_jpeg_buf)
            tjFree(_jpeg_buf);
        _jpeg_buf = tjAlloc((int)needed);
        if (!_jpeg_buf) {
            _jpeg_buf_size = 0;
            PyBuffer_Release(&view);
            PyErr_NoMemory();
            return NULL;
        }
        _jpeg_buf_size = needed;
    }

    /* Lazy-init persistent compress handle. */
    if (!_tj_handle) {
        _tj_handle = tjInitCompress();
        if (!_tj_handle) {
            PyBuffer_Release(&view);
            PyErr_SetString(PyExc_RuntimeError,
                            "Failed to initialise TurboJPEG compressor");
            return NULL;
        }
    }

    unsigned long jpeg_size = _jpeg_buf_size;
    int ret;

    Py_BEGIN_ALLOW_THREADS /* ── GIL released ── */

        ret = tjCompress2(_tj_handle, (const unsigned char*)view.buf, width, 0,
                          height, TJPF_RGB, &_jpeg_buf, &jpeg_size, TJSAMP_420,
                          quality, TJFLAG_FASTDCT | TJFLAG_NOREALLOC);

    Py_END_ALLOW_THREADS /* ── GIL reacquired ── */

        PyBuffer_Release(&view);

    if (ret != 0) {
        PyErr_SetString(PyExc_RuntimeError, tjGetErrorStr());
        return NULL;
    }

    return PyBytes_FromStringAndSize((char*)_jpeg_buf, (Py_ssize_t)jpeg_size);
}

/* ── MODULE CLEANUP ──────────────────────────────────────────────────────*/
static void module_free(void* m) {
    (void)m;
    if (_tj_handle) {
        tjDestroy(_tj_handle);
        _tj_handle = NULL;
    }
    if (_jpeg_buf) {
        tjFree(_jpeg_buf);
        _jpeg_buf = NULL;
    }
    _jpeg_buf_size = 0;
}

/* ── MODULE DEFINITION ───────────────────────────────────────────────────*/
static PyMethodDef methods[] = {
    {"compute_sharpness", compute_sharpness, METH_VARARGS,
     "compute_sharpness(buf, w, h) -> float\n"
     "Laplacian variance with 2x downsample + 3-row ring-buffer fusion.\n"
     "Accepts numpy arrays, memoryviews, or bytes. GIL released."},
    {"encode_jpeg", encode_jpeg, METH_VARARGS,
     "encode_jpeg(buf, w, h, quality) -> bytes\n"
     "Encode RGB buffer to JPEG via libturbojpeg.\n"
     "Persistent handle + pre-alloc buffer. GIL released."},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    "camera_utils",
    "Optimised Pi HQ Camera helpers — sharpness + JPEG encode.",
    -1,
    methods,
    NULL,
    NULL,
    NULL,
    module_free};

PyMODINIT_FUNC PyInit_camera_utils(void) {
    return PyModule_Create(&module_def);
}