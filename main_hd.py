#!/usr/bin/env python3
"""
Pi HQ Camera — HDMI preview via pygame with terminal controls.

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
import pygame
from picamera2 import Picamera2
from picamera2.display import DrmPreview

import camera_utils

# ── CONFIG ───────────────────────────────────────────────────────────────────
LOG_LEVEL      = logging.INFO

# Fraction of each axis used for the centre-crop sharpness window.
# 0.5 → centre 50 % × 50 % = 25 % of pixels, ~4× less work.
SHARPNESS_CROP = 0.5

PRESETS = {
    "low":    {"size": (320,  240), "fps": 30, "label": "320×240  / 30 fps"},
    "medium": {"size": (640,  480), "fps": 20, "label": "640×480  / 20 fps"},
    "high":   {"size": (1280, 960), "fps": 10, "label": "1280×960 / 10 fps"},
}
DEFAULT_PRESET = "medium"

# ── LOGGING ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=LOG_LEVEL, format="[%(levelname)s] %(message)s")
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
    log.info("Switching preset → %s", p["label"])

    camera.stop()
    config = camera.create_preview_configuration(
        main={"size": p["size"], "format": "RGB888"},
        controls={
            "FrameRate": float(p["fps"]),
            "AeEnable":  True,
            "AwbEnable": True,
        },
        buffer_count=2,
    )
    camera.configure(config)
    camera.start_preview(DrmPreview())
    camera.start()

    with _preset_lock:
        current_preset = name
        target_fps     = p["fps"]

    log.info("Preset active: %s", p["label"])


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
    Restores normal terminal state on exit regardless of how the loop ends.
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
                log.info("Quit requested via terminal.")
                _quit.set()
                break

            if ch in key_map:
                with _preset_lock:
                    _pending_preset = key_map[ch]
                sys.stdout.write(f"\r  → switching to {key_map[ch]}…\r\n")
                sys.stdout.flush()

    finally:
        # Always restore the terminal, even if an exception is raised.
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


# ── OVERLAY HELPERS ───────────────────────────────────────────────────────────

def _focus_colour(pct: float) -> tuple[int, int, int]:
    if pct >= 75:
        return (74,  222, 154)   # green
    if pct >= 45:
        return (240, 160,  80)   # amber
    return     (240,  96,  96)   # red


def _draw_overlay(screen: pygame.Surface, font: pygame.font.Font,
                  sv: float, sp: float, label: str) -> None:
    """Render sharpness + preset info into the top-left corner of the surface."""
    col   = _focus_colour(sp)
    lines = [
        f"Preset : {label}",
        f"Sharp  : {sv:.0f}   ({sp:.1f} %)",
    ]
    y = 8
    for line in lines:
        # Dark shadow pass for legibility over any background colour.
        screen.blit(font.render(line, True, (0, 0, 0)), (9, y + 1))
        screen.blit(font.render(line, True, col),       (8, y))
        y += 22


# ── FRAME LOOP ────────────────────────────────────────────────────────────────

def frame_loop() -> None:
    """
    Main loop: capture → sharpness (async) → pygame blit → overlay.
    Must run on the main thread (pygame requires this on most platforms).
    """
    global _pending_preset, sharpness_val, sharpness_pct

    pygame.init()
    p      = PRESETS[current_preset]
    screen = pygame.display.set_mode(p["size"], pygame.RESIZABLE | pygame.NOFRAME)
    pygame.display.set_caption("Pi HQ Camera")
    font  = pygame.font.SysFont("monospace", 16, bold=True)
    clock = pygame.time.Clock()

    _sharp_future = None
    frame_count   = 0

    log.info("Frame loop: warming up (2 s)…")
    time.sleep(2)
    log.info("Frame loop: running.")

    while not _quit.is_set():
        # ── Pygame window events (e.g. display manager close) ─────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                _quit.set()

        # ── Apply any pending preset switch ───────────────────────────────
        with _preset_lock:
            pending         = _pending_preset
            _pending_preset = None

        if pending:
            _apply_preset(pending)
            new_size = PRESETS[current_preset]["size"]
            screen = pygame.display.set_mode(
                new_size, pygame.RESIZABLE | pygame.NOFRAME)

        try:
            frame    = camera.capture_array()         # numpy RGB888
            h_f, w_f = frame.shape[:2]

            # ── Collect finished sharpness result ─────────────────────────
            if _sharp_future is not None and _sharp_future.done():
                try:
                    sv, sp = _sharp_future.result()
                    with sharp_lock:
                        sharpness_val = sv
                        sharpness_pct = sp
                except Exception as exc:
                    log.debug("Sharpness result error: %s", exc)
                _sharp_future = None

            # ── Submit centre-crop sharpness job every 3rd frame ──────────
            if frame_count % 3 == 0 and _sharp_future is None:
                ch   = int(h_f * SHARPNESS_CROP) & ~1   # keep even for 2× downsample
                cw   = int(w_f * SHARPNESS_CROP) & ~1
                y0   = (h_f - ch) // 2
                x0   = (w_f - cw) // 2
                crop = np.ascontiguousarray(frame[y0:y0 + ch, x0:x0 + cw])
                _sharp_future = _executor.submit(
                    _sharpness_worker, crop.tobytes(), cw, ch)

            frame_count += 1

            # ── Blit frame to display ─────────────────────────────────────
            # swapaxes: numpy is (H, W, 3); pygame surfarray expects (W, H, 3).
            surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
            screen.blit(surface, (0, 0))

            # ── Draw HUD overlay ──────────────────────────────────────────
            with sharp_lock:
                sv = sharpness_val
                sp = sharpness_pct
            with _preset_lock:
                label = PRESETS[current_preset]["label"]

            _draw_overlay(screen, font, sv, sp, label)
            pygame.display.flip()

        except Exception as exc:
            log.warning("Frame error: %s", exc)

        clock.tick(target_fps)   # caps loop to preset FPS, absorbs any surplus


    pygame.quit()


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
        buffer_count=2,
    )
    camera.configure(config)
    camera.start()
    log.info("Camera started — preset: %s", p["label"])

    # Input thread is a daemon so it exits automatically if the main thread exits.
    threading.Thread(
        target=terminal_input_loop, daemon=True, name="term-input"
    ).start()

    try:
        frame_loop()   # blocks until _quit is set
    finally:
        camera.stop()
        _executor.shutdown(wait=False)
        log.info("Shutdown complete.")


if __name__ == "__main__":
    main()