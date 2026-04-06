"""
setup.py — Build camera_utils C extension for Pi HQ Camera.

Build:
    pip install --break-system-packages .
  or (editable / in-place):
    pip install --break-system-packages -e .

Architecture flags:
  aarch64 / arm64  — Pi Zero 2 W running 64-bit OS.  NEON is always present
                     on ARMv8; -march=native enables the compiler to use it.
  armv7l / armhf   — 32-bit Pi OS.  We explicitly request NEON + hard-float.
  Other            — Compiles cleanly without NEON (scalar fallback in C).
"""

import platform
from setuptools import Extension, setup

machine = platform.machine().lower()

compile_args = [
    "-O3",
    "-ffast-math",       # allows reassociation; safe for our use-case
    "-funroll-loops",    # unroll the inner Laplacian / downsample loops
]
define_macros = []

if machine in ("aarch64", "arm64"):
    # 64-bit ARMv8: NEON is mandatory, just tell the compiler to use it.
    compile_args += ["-march=native"]

elif "arm" in machine:
    # 32-bit ARMv7 (armhf / armv7l):
    compile_args += [
        "-mfpu=neon-fp-armv8",
        "-mfloat-abi=hard",
        "-march=armv8-a",
    ]

# On x86/x86_64 (dev machines) the C code compiles fine; NEON blocks are
# guarded by #ifdef __ARM_NEON so they are simply skipped.

ext = Extension(
    "camera_utils",
    sources=["camera_utils.c"],
    libraries=["turbojpeg"],
    extra_compile_args=compile_args,
    define_macros=define_macros,
)

setup(
    name="pi-camera-utils",
    version="2.0.0",
    description="Optimised C helpers for Pi HQ Camera (sharpness + JPEG encode)",
    ext_modules=[ext],
)