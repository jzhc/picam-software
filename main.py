#!/usr/bin/env python3
"""
Pi HQ Camera — Live Preview with Dynamic Resolution
Stream: http://<pi-ip>:8000

Optimisations vs. original:
  • ThreadPoolExecutor overlaps sharpness (C/NEON, GIL released) with
    JPEG encode (also GIL released) so both run on separate cores.
  • memoryview passed directly to C — zero Python-side copy for encode.
  • frame.tobytes() copy only for the async sharpness path (every 3rd frame).
  • Deadline-based adaptive sleep removes accumulated drift.
  • Dynamic resolution/FPS presets switchable via HTTP or UI buttons.
  • Double-buffered JPEG store — HTTP handler always gets the last complete
    frame without blocking the capture loop.
"""

import json
import logging
import socket
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
from picamera2 import Picamera2

import camera_utils

# ── CONFIG ─────────────────────────────────────────────────────────────────
JPEG_QUALITY      = 70
SERVER_PORT       = 8000
LOG_LEVEL         = logging.INFO

# Fraction of each axis to use for the centre-crop sharpness window.
# 0.5 → centre 50 % of width × 50 % of height = 25 % of total pixels.
# Reducing this cuts sharpness compute time proportionally with no loss
# of focus accuracy for a centred subject.
SHARPNESS_CROP    = 0.3

# Resolution / framerate presets.
# "fps" controls the camera pipeline rate AND our processing deadline.
PRESETS = {
    "low":    {"size": (320,  240),  "fps": 30, "label": "320×240 / 30 fps"},
    "medium": {"size": (640,  480),  "fps": 20, "label": "640×480 / 20 fps"},
    "high":   {"size": (1280, 960),  "fps": 10, "label": "1280×960 / 10 fps"},
}
DEFAULT_PRESET = "medium"

# ── LOGGING ─────────────────────────────────────────────────────────────────
logging.basicConfig(level=LOG_LEVEL, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── GLOBAL STATE ─────────────────────────────────────────────────────────────
camera         = None

# Double-buffer: writer fills _bufs[_write_idx ^ 1]; reader reads _bufs[_read_idx].
_bufs          = [None, None]
_read_idx      = 0
_buf_lock      = threading.Lock()

sharpness_val  = 0.0
sharpness_pct  = 0.0
sharp_lock     = threading.Lock()

current_preset = DEFAULT_PRESET
target_fps     = PRESETS[DEFAULT_PRESET]["fps"]
_preset_lock   = threading.Lock()
_pending_preset: str | None = None  # written by HTTP thread, read by frame loop

# Rolling history for sharpness-to-percent conversion (last 300 readings).
_sharp_history  = deque(maxlen=300)

# Thread pool: one worker for async sharpness (GIL released in C → true concurrency).
_executor      = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sharp")
_sharp_future  = None

# ── SHARPNESS HELPERS ─────────────────────────────────────────────────────

def _sharpness_to_pct(value: float) -> float:
    """Normalise raw Laplacian variance to 0-100 % relative to recent peak."""
    _sharp_history.append(value)
    if len(_sharp_history) < 10:
        return 0.0
    n_top    = max(1, len(_sharp_history) // 10)
    reference = sum(sorted(_sharp_history, reverse=True)[:n_top]) / n_top
    return min(100.0, (value / reference) * 100.0) if reference else 0.0


def _sharpness_worker(raw: bytes, w: int, h: int):
    """Run in the thread pool. C function releases the GIL internally."""
    val = camera_utils.compute_sharpness(raw, w, h)
    pct = _sharpness_to_pct(val)
    return val, pct


# ── PRESET SWITCHING ───────────────────────────────────────────────────────

def _apply_preset(name: str) -> None:
    """
    Reconfigure the camera for a new preset.
    Must be called from the frame loop thread (the only camera user).
    """
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
    camera.start()

    with _preset_lock:
        current_preset = name
        target_fps     = p["fps"]

    log.info("Preset active: %s", p["label"])


# ── FRAME LOOP ─────────────────────────────────────────────────────────────

def frame_loop() -> None:
    global _bufs, _read_idx, sharpness_val, sharpness_pct
    global _sharp_future, _pending_preset

    log.info("Frame loop: warming up (2 s)...")
    time.sleep(2)
    log.info("Frame loop: running.")

    frame_count = 0

    while True:
        # Deadline-based timing removes accumulated drift.
        deadline = time.monotonic() + 1.0 / max(1, target_fps)

        # ── Check for pending preset switch ──────────────────────────────
        with _preset_lock:
            pending = _pending_preset
            if pending:
                _pending_preset = None
        if pending:
            _apply_preset(pending)

        try:
            frame = camera.capture_array()    # numpy array, RGB888
            h, w  = frame.shape[:2]

            # ── Collect sharpness result if the worker finished ───────────
            if _sharp_future is not None and _sharp_future.done():
                try:
                    sv, sp = _sharp_future.result()
                    with sharp_lock:
                        sharpness_val = sv
                        sharpness_pct = sp
                except Exception as exc:
                    log.debug("Sharpness result error: %s", exc)
                _sharp_future = None

            # ── Submit sharpness job every 3rd frame ──────────────────────
            # Analyse only the centre crop (SHARPNESS_CROP fraction of each
            # axis) — proportionally fewer pixels, same focus accuracy for a
            # centred subject.  np.ascontiguousarray ensures the slice is a
            # packed RGB buffer before tobytes() copies it for the worker.
            if frame_count % 3 == 0 and _sharp_future is None:
                ch = int(h * SHARPNESS_CROP) & ~1   # keep even (downsample needs it)
                cw = int(w * SHARPNESS_CROP) & ~1
                y0, x0 = (h - ch) // 2, (w - cw) // 2
                crop = np.ascontiguousarray(frame[y0:y0 + ch, x0:x0 + cw])
                _sharp_future = _executor.submit(
                    _sharpness_worker, crop.tobytes(), cw, ch)

            frame_count += 1

            # ── Encode JPEG (zero-copy memoryview → C, GIL released) ──────
            frame_c = np.ascontiguousarray(frame)   # no-op if already C-order
            buf = camera_utils.encode_jpeg(memoryview(frame_c), w, h, JPEG_QUALITY)

            # ── Double-buffer swap ─────────────────────────────────────────
            # Write into the slot NOT currently being read, then flip the
            # read index.  The HTTP handler only grabs _bufs[_read_idx].
            write_slot = _read_idx ^ 1
            _bufs[write_slot] = buf
            with _buf_lock:
                _read_idx = write_slot

        except Exception as exc:
            log.warning("Frame error: %s", exc)

        # Adaptive sleep: honour the deadline regardless of how long the
        # work above took.
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)


# ── HTML ────────────────────────────────────────────────────────────────────
# Intentionally kept as a single string to avoid any file-serving complexity
# on the Pi.  Fonts load from Google Fonts if the browser has internet access,
# otherwise falls back gracefully to system monospace.
# Note: String is now defined normally (without b"")

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Pi HQ Camera</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Azeret+Mono:wght@400;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg:      #0e0e10;
      --panel:   #161618;
      --border:  #2a2a2e;
      --green:   #4ade9a;
      --amber:   #f0a050;
      --red:     #f06060;
      --label:   #888896;
      --text:    #c8c8d4;
      --bright:  #eeeef4;
    }
    *, *::before, *::after { margin:0; padding:0; box-sizing:border-box; }

    body {
      background: var(--bg);
      color: var(--text);
      font-family: 'Azeret Mono', 'Courier New', monospace;
      min-height: 100vh;
      display: grid;
      grid-template-rows: 1fr;
      overflow: hidden;
    }

    main {
      display: grid;
      grid-template-columns: 1fr 256px;
      overflow: hidden;
    }

    /* ── VIDEO PANEL ─────────────────────────────────────────────────── */
    .video-panel {
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 20px;
      border-right: 1px solid var(--border);
      overflow: hidden;
    }
    .frame-wrap { display: inline-flex; }
    .frame-wrap img {
      display: block;
      max-width: 100%;
      max-height: calc(100vh - 40px);
      object-fit: contain;
      image-rendering: crisp-edges;
    }

    /* ── SIDEBAR ─────────────────────────────────────────────────────── */
    .sidebar {
      display: flex;
      flex-direction: column;
      overflow-y: auto;
      scrollbar-width: none;
    }
    .sidebar::-webkit-scrollbar { display: none; }

    .block {
      padding: 16px;
      border-bottom: 1px solid var(--border);
    }
    .block-title {
      font-size: 9px;
      letter-spacing: 2px;
      text-transform: uppercase;
      color: var(--label);
      margin-bottom: 12px;
    }

    /* ── STAT TILES ──────────────────────────────────────────────────── */
    .stat-row { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    .stat-tile {
      background: var(--panel);
      border: 1px solid var(--border);
      padding: 10px 12px;
    }
    .stat-label {
      font-size: 8px;
      letter-spacing: 1.5px;
      text-transform: uppercase;
      color: var(--label);
      margin-bottom: 6px;
    }
    .stat-val {
      font-size: 22px;
      font-weight: 600;
      color: var(--green);
      line-height: 1;
      transition: color .3s;
    }

    /* ── FOCUS BAR ───────────────────────────────────────────────────── */
    .focus-pct {
      font-size: 28px;
      font-weight: 600;
      color: var(--green);
      line-height: 1;
      margin-bottom: 10px;
      transition: color .3s;
    }
    .bar-track {
      height: 4px;
      background: var(--border);
      border-radius: 2px;
    }
    .bar-fill {
      height: 100%;
      background: var(--green);
      width: 0%;
      border-radius: 2px;
      transition: width .2s ease, background-color .3s;
    }

    /* ── SPARKLINE ───────────────────────────────────────────────────── */
    #sparkline { width: 100%; height: 52px; display: block; }

    /* ── PRESET BUTTONS ──────────────────────────────────────────────── */
    .preset-list { display: flex; flex-direction: column; gap: 6px; }
    .preset-btn {
      background: transparent;
      border: 1px solid var(--border);
      color: var(--text);
      font-family: inherit;
      font-size: 11px;
      padding: 8px 10px;
      cursor: pointer;
      text-align: left;
      border-radius: 2px;
      transition: border-color .15s, color .15s, background .15s;
    }
    .preset-btn:hover { border-color: var(--text); color: var(--bright); }
    .preset-btn.active {
      border-color: var(--green);
      color: var(--green);
    }
  </style>
</head>
<body>
  <main>
    <div class="video-panel">
      <div class="frame-wrap">
        <img id="feed" src="/frame.jpg" alt="Live stream">
      </div>
    </div>

    <div class="sidebar">

      <div class="block">
        <div class="block-title">Metrics</div>
        <div class="stat-row">
          <div class="stat-tile">
            <div class="stat-label">Sharpness</div>
            <div class="stat-val" id="sharp-val">&#8212;</div>
          </div>
          <div class="stat-tile">
            <div class="stat-label">FPS</div>
            <div class="stat-val" id="fps-val">&#8212;</div>
          </div>
        </div>
      </div>

      <div class="block">
        <div class="block-title">Relative Focus</div>
        <div class="focus-pct" id="focus-pct">0.0%</div>
        <div class="bar-track"><div class="bar-fill" id="bar"></div></div>
      </div>

      <div class="block">
        <div class="block-title">Sharpness History</div>
        <canvas id="sparkline" width="224" height="52"></canvas>
      </div>

      <div class="block">
        <div class="block-title">Resolution / FPS</div>
        <div class="preset-list">
          <button class="preset-btn" data-preset="low"
                  onclick="setPreset(this)">320 &times; 240 &middot; 30 fps</button>
          <button class="preset-btn active" data-preset="medium"
                  onclick="setPreset(this)">640 &times; 480 &middot; 20 fps</button>
          <button class="preset-btn" data-preset="high"
                  onclick="setPreset(this)">1280 &times; 960 &middot; 10 fps</button>
        </div>
      </div>

    </div>
  </main>

  <script>
    const hist  = [];
    const MAX_H = 150;
    let fc      = 0;
    let lastT   = Date.now();

    /* ── Sparkline ───────────────────────────────────────────────────── */
    function drawSparkline(data) {
      const canvas = document.getElementById('sparkline');
      const ctx    = canvas.getContext('2d');
      const W = canvas.width, H = canvas.height;
      ctx.clearRect(0, 0, W, H);
      if (data.length < 2) return;

      const min   = Math.min(...data);
      const max   = Math.max(...data);
      const range = (max - min) || 1;
      const pts   = data.map((v, i) => ({
        x: (i / (data.length - 1)) * W,
        y: H - 2 - ((v - min) / range) * (H - 8)
      }));

      ctx.strokeStyle = '#4ade9a';
      ctx.lineWidth   = 1.5;
      ctx.beginPath();
      pts.forEach((p, i) => i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y));
      ctx.stroke();

      ctx.beginPath();
      pts.forEach((p, i) => i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y));
      ctx.lineTo(W, H); ctx.lineTo(0, H); ctx.closePath();
      ctx.fillStyle = 'rgba(74,222,154,.08)';
      ctx.fill();
    }

    /* ── Focus colour ────────────────────────────────────────────────── */
    function focusColor(pct) {
      if (pct >= 75) return 'var(--green)';
      if (pct >= 45) return 'var(--amber)';
      return 'var(--red)';
    }

    /* ── Frame fetching ──────────────────────────────────────────────── */
    const visibleFeed = document.getElementById('feed');
    const bufferImg   = new Image();

    bufferImg.onload = () => {
      visibleFeed.src = bufferImg.src;
      fc++;
      requestNextFrame();
    };
    bufferImg.onerror = () => setTimeout(requestNextFrame, 500);

    function requestNextFrame() {
      bufferImg.src = '/frame.jpg?' + Date.now();
    }

    /* ── Status polling ──────────────────────────────────────────────── */
    function fetchStatus() {
      fetch('/status')
        .then(r => r.json())
        .then(d => {
          const sharp  = parseFloat(d.sharpness);
          const pct    = parseFloat(d.percent);
          const preset = d.preset;

          document.getElementById('sharp-val').textContent =
            isNaN(sharp) ? '—' : sharp.toFixed(0);

          const col  = focusColor(pct);
          const fpEl = document.getElementById('focus-pct');
          fpEl.textContent  = pct.toFixed(1) + '%';
          fpEl.style.color  = col;
          const bar = document.getElementById('bar');
          bar.style.width           = Math.min(100, pct) + '%';
          bar.style.backgroundColor = col;

          document.querySelectorAll('.preset-btn').forEach(b => {
            b.classList.toggle('active', b.dataset.preset === preset);
          });

          if (!isNaN(sharp)) {
            hist.push(sharp);
            if (hist.length > MAX_H) hist.shift();
            drawSparkline(hist);
          }
        })
        .catch(() => {})
        .finally(() => setTimeout(fetchStatus, 250));
    }

    /* ── FPS counter ─────────────────────────────────────────────────── */
    setInterval(() => {
      const now     = Date.now();
      const elapsed = (now - lastT) / 1000;
      if (elapsed >= 1.0) {
        document.getElementById('fps-val').textContent = Math.round(fc / elapsed);
        fc = 0;
        lastT = now;
      }
    }, 100);

    /* ── Preset control ──────────────────────────────────────────────── */
    function setPreset(btn) {
      fetch('/config?preset=' + btn.dataset.preset).catch(() => {});
    }

    requestNextFrame();
    fetchStatus();
  </script>
</body>
</html>"""


# ── HTTP SERVER ─────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        path   = parsed.path

        if   path == "/":          self._serve_html()
        elif path == "/frame.jpg": self._serve_frame()
        elif path == "/status":    self._serve_status()
        elif path == "/config":    self._handle_config(parsed.query)
        else:                      self.send_error(404)

    # ── Endpoints ──────────────────────────────────────────────────────────

    def _serve_html(self):
        # Encode the standard Python string to bytes before sending
        self._write(200, "text/html; charset=utf-8", HTML.encode('utf-8'))

    def _serve_frame(self):
        with _buf_lock:
            frame = _bufs[_read_idx]
        if frame:
            self._write(200, "image/jpeg", frame)
        else:
            self.send_error(503)

    def _serve_status(self):
        with sharp_lock:
            sv = sharpness_val
            sp = sharpness_pct
        with _preset_lock:
            preset = current_preset
            tfps   = target_fps
        res = "×".join(str(d) for d in PRESETS[preset]["size"])
        payload = json.dumps({
            "sharpness":  f"{sv:.1f}",
            "percent":    f"{sp:.1f}",
            "preset":     preset,
            "resolution": res,
            "target_fps": tfps,
        }).encode()
        self._write(200, "application/json", payload)

    def _handle_config(self, query: str):
        global _pending_preset
        params = parse_qs(query)

        if "preset" in params:
            name = params["preset"][0]
            if name in PRESETS:
                with _preset_lock:
                    _pending_preset = name

        self._write(200, "application/json", b'{"ok":true}')

    # ── Helpers ────────────────────────────────────────────────────────────

    def _write(self, code: int, ctype: str, data: bytes):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", len(data))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *_):
        pass   # suppress per-request logs

    def handle_error(self, request, client_address):
        pass


# ── UTILITIES ───────────────────────────────────────────────────────────────

def _get_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "localhost"


# ── MAIN ────────────────────────────────────────────────────────────────────

def main():
    global camera

    log.info("Starting camera...")
    camera = Picamera2()

    p = PRESETS[DEFAULT_PRESET]
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

    threading.Thread(target=frame_loop, daemon=True, name="frame-loop").start()

    ip = _get_ip()
    print()
    print("=" * 44)
    print(f"  Stream : http://{ip}:{SERVER_PORT}")
    print(f"  Preset : {p['label']}")
    print("=" * 44)
    print()

    server = ThreadingHTTPServer(("0.0.0.0", SERVER_PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down...")
    finally:
        # Use server_close() instead of shutdown() to prevent deadlock on Ctrl+C
        server.server_close()
        
        # Stop the camera and kill the thread pool
        camera.stop()
        _executor.shutdown(wait=False)


if __name__ == "__main__":
    main()