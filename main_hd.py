#!/usr/bin/env python3
"""
Pi HQ Camera — Native DRM preview with terminal controls.
Optimized for Raspberry Pi OS Lite (Command Line Only).

Controls (single keypress in the terminal, no Enter needed):
  1  →  low    preset  (320×240  / 30 fps)
  2  →  medium preset  (640×480  / 20 fps)
  3  →  high   preset  (1280×960 / 10 fps)
  q  →  quit
"""

import logging
import sys
import termios
import threading
import time
import tty
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from picamera2 import Picamera2, Preview

import camera_utils

# ── CONFIG ───────────────────────────────────────────────────────────────────
LOG_LEVEL      = logging.INFO

# Fraction of each axis used for the centre-crop sharpness window.
# 0.5 → centre 50 % × 50 % = 25 % of pixels, ~4× less work.
SHARPNESS_CROP = 0.5

PRESETS = {
    "low":    {"size": (320,  240), "fps": 90, "label": "320×240  / 90 fps"},
    "medium": {"size": (640,  480), "fps": 60, "label": "640×480  / 60 fps"},
    "high":   {"size": (1280, 960), "fps": 40, "label": "1280×960 / 40 fps"},
}
DEFAULT_PRESET = "medium"

# ── LOGGING ──────────────────────────────────────────────────────────────────
# Force logging to use \r\n so it doesn't staircase when terminal is in raw mode
logging.basicConfig(level=LOG_LEVEL, format="[%(levelname)s] %(message)s")
logging.StreamHandler.terminator = "\r\n"
log = logging.getLogger(__name__)

# ── SHARED STATE ─────────────────────────────────────────────────────────────
camera         = None

sharpness_val  = 0.0
sharpness_pct  = 0.0
sharp_lock     = threading.Lock()

current_preset = DEFAULT_PRESET
target_fps     = PRESETS[DEFAULT_PRESET]["fps"]
_preset_lock   = threading.Lock()
_pending_preset: str | None = None   # written by input thread, read by frame loop

_quit          = threading.Event()   # set by either thread to trigger clean exit

_sharp_history = deque(maxlen=300)
_executor      = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sharp")

# ── SHARPNESS HELPERS ────────────────────────────────────────────────────────

def _sharpness_to_pct(value: float) -> float:
    """Normalise raw Laplacian variance to 0–100 % relative to recent peak."""
    _sharp_history.append(value)
    if len(_sharp_history) < 10:
        return 0.0
    n_top     = max(1, len(_sharp_history) // 10)
    reference = sum(sorted(_sharp_history, reverse=True)[:n_top]) / n_top
    return min(100.0, (value / reference) * 100.0) if reference else 0.0


def _sharpness_worker(raw: bytes, w: int, h: int):
    """Runs in thread pool; C function releases the GIL internally."""
    val = camera_utils.compute_sharpness(raw, w, h)
    pct = _sharpness_to_pct(val)
    return val, pct


# ── PRESET SWITCHING ─────────────────────────────────────────────────────────

def _apply_preset(name: str) -> None:
    """Reconfigure the camera pipeline. Must be called from the frame-loop thread."""
    global current_preset, target_fps

    if name not in PRESETS or name == current_preset:
        return

    p = PRESETS[name]
    # Use explicit terminal carriage returns
    sys.stdout.write(f"\r\n[INFO] Switching preset → {p['label']}\r\n")
    sys.stdout.flush()

    camera.stop()
    
    config = camera.create_preview_configuration(
        main={"size": p["size"], "format": "RGB888"},
        controls={
            "FrameRate": float(p["fps"]),
            "AeEnable":  True,
            "AwbEnable": True,
        },
        buffer_count=4,   # DRM holds 1, capture holds 1, 2 spare — no starvation
    )
    
    camera.configure(config)
    # We DO NOT call start_preview() here. The DRM preview is already hooked up.
    # picamera2 will automatically route the new stream to the existing preview.
    camera.start()

    with _preset_lock:
        current_preset = name
        target_fps     = p["fps"]


# ── TERMINAL INPUT ────────────────────────────────────────────────────────────

def _print_controls() -> None:
    """Print the key legend. Uses \\r\\n because the terminal is in raw mode."""
    lines = [
        "",
        "┌─────────────────────────────────┐",
        "│     Pi HQ Camera — Controls     │",
        "├─────────────────────────────────┤",
        "│  1  →  low    (320×240  / 30fps)│",
        "│  2  →  medium (640×480  / 20fps)│",
        "│  3  →  high   (1280×960 / 10fps)│",
        "│  q  →  quit                     │",
        "└─────────────────────────────────┘",
        "",
    ]
    sys.stdout.write("\r\n".join(lines) + "\r\n")
    sys.stdout.flush()


def terminal_input_loop() -> None:
    """
    Reads single keypresses without requiring Enter, using raw terminal mode.
    Runs in a daemon thread so the frame loop can remain on the main thread.
    """
    global _pending_preset

    fd  = sys.stdin.fileno()
    old = termios.tcgetattr(fd)

    key_map = {"1": "low", "2": "medium", "3": "high"}

    try:
        tty.setraw(fd)
        _print_controls()

        while not _quit.is_set():
            ch = sys.stdin.read(1)

            if ch == "q":
                _quit.set()
                break

            if ch in key_map:
                with _preset_lock:
                    _pending_preset = key_map[ch]

    finally:
        # Always restore the terminal, even if an exception is raised.
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


# ── FRAME LOOP ────────────────────────────────────────────────────────────────

def frame_loop() -> None:
    """
    Main loop: zero-copy capture → sharpness (async) → terminal stats.

    Uses capture_request() instead of capture_array() so the raw frame data
    is accessed directly from the camera DMA buffer with no full-frame copy.
    Only the small centre-crop is copied (via tobytes()) to hand off to the
    sharpness worker. The request is released immediately after the crop is
    taken, returning the DMA buffer to the pipeline as fast as possible.
    """
    global _pending_preset, sharpness_val, sharpness_pct

    _sharp_future = None
    frame_count   = 0

    fps_counter   = 0
    current_fps   = 0.0
    last_time     = time.monotonic()

    sys.stdout.write("\r\n[INFO] Warming up camera (2s)...\r\n")
    sys.stdout.flush()
    time.sleep(2)

    while not _quit.is_set():
        # ── Apply any pending preset switch ───────────────────────────────
        with _preset_lock:
            pending         = _pending_preset
            _pending_preset = None

        if pending:
            _apply_preset(pending)

        try:
            # Zero-copy acquire: returns a handle to the DMA buffer directly.
            # No numpy array is allocated; the frame stays in camera memory.
            request = camera.capture_request()
            buf     = request.make_buffer("main")   # memoryview, no copy

            # Wrap in numpy without copying so we can slice for the crop.
            with _preset_lock:
                w_f, h_f = PRESETS[current_preset]["size"]
            frame = np.frombuffer(buf, dtype=np.uint8).reshape(h_f, w_f, 3)

            # ── Collect finished sharpness result ─────────────────────────
            if _sharp_future is not None and _sharp_future.done():
                try:
                    sv, sp = _sharp_future.result()
                    with sharp_lock:
                        sharpness_val = sv
                        sharpness_pct = sp
                except Exception:
                    pass
                _sharp_future = None

            # ── Submit centre-crop sharpness job every 3rd frame ──────────
            # tobytes() copies only the crop (~25 % of frame pixels).
            # request.release() then happens immediately, returning the DMA
            # buffer to the camera pipeline without waiting on the worker.
            if frame_count % 3 == 0 and _sharp_future is None:
                ch   = int(h_f * SHARPNESS_CROP) & ~1
                cw   = int(w_f * SHARPNESS_CROP) & ~1
                y0   = (h_f - ch) // 2
                x0   = (w_f - cw) // 2
                crop = np.ascontiguousarray(frame[y0:y0 + ch, x0:x0 + cw])
                _sharp_future = _executor.submit(
                    _sharpness_worker, crop.tobytes(), cw, ch)

            # Release DMA buffer as early as possible so the camera pipeline
            # can reuse it for the next frame without waiting on us.
            request.release()

            frame_count += 1
            fps_counter += 1

            # ── Calculate FPS (once per second) ───────────────────────────
            now = time.monotonic()
            if now - last_time >= 1.0:
                current_fps = fps_counter / (now - last_time)
                fps_counter = 0
                last_time   = now

            # ── Print terminal readout every 10th frame only ───────────────
            # A flush syscall every frame is measurable overhead at high FPS.
            # Updating ~6x per second is plenty for a human to read.
            if frame_count % 10 == 0:
                with sharp_lock:
                    sv = sharpness_val
                    sp = sharpness_pct
                with _preset_lock:
                    label = PRESETS[current_preset]["label"]

                status = (
                    f"\r  [ Preset: {label:<16} ]"
                    f"   [ Sharpness: {sv:5.0f} ({sp:5.1f}%) ]"
                    f"   [ FPS: {current_fps:4.1f} ]   "
                )
                sys.stdout.write(status)
                sys.stdout.flush()

        except Exception as exc:
            sys.stdout.write(f"\r\n[WARNING] Frame error: {exc}\r\n")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main() -> None:
    global camera

    log.info("Starting camera…")
    camera = Picamera2()

    p      = PRESETS[DEFAULT_PRESET]
    config = camera.create_preview_configuration(
        main={"size": p["size"], "format": "RGB888"},
        controls={
            "FrameRate": float(p["fps"]),
            "AeEnable":  True,
            "AwbEnable": True,
        },
        buffer_count=4,   # DRM holds 1, capture holds 1, 2 spare — no starvation
    )
    camera.configure(config)
    
    # Attach DRM preview ONCE here.
    camera.start_preview(Preview.DRM)
    camera.start()
    
    log.info("Camera started — preset: %s", p["label"])

    # Input thread is a daemon so it exits automatically if the main thread exits.
    threading.Thread(
        target=terminal_input_loop, daemon=True, name="term-input"
    ).start()

    try:
        frame_loop()   # blocks until _quit is set
    except KeyboardInterrupt:
        _quit.set()
    finally:
        sys.stdout.write("\r\n[INFO] Shutting down...\r\n")
        camera.stop()
        _executor.shutdown(wait=False)
        sys.stdout.write("[INFO] Shutdown complete.\r\n")


if __name__ == "__main__":
    main()