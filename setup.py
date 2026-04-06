from setuptools import setup, Extension

setup(
    ext_modules=[Extension("sharpness_c", ["sharpness.c"], extra_compile_args=["-O3"])]
)