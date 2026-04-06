from setuptools import setup, Extension

setup(
    ext_modules=[
        Extension(
            "camera_utils",
            ["camera_utils.c"],
            libraries=["turbojpeg"],
            extra_compile_args=["-O3"]
        )
    ]
)