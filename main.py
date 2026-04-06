#!/usr/bin/env python3
"""
Pi HQ Camera — Live Preview with Dynamic Resolution & Auto-FPS
Stream: http://<pi-ip>:8000

Optimisations vs. original:
  • ThreadPoolExecutor overlaps sharpness (C/NEON, GIL released) with
    JPEG encode (also GIL released) so both run on separate cores.
  • memoryview passed directly to C — zero Python-side copy for encode.
  • frame.tobytes() copy only for the async sharpness path (every 3rd frame).
  • Deadline-based adaptive sleep removes accumulated drift.
  • Dynamic resolution/FPS presets switchable via HTTP or UI buttons.
  • Auto-FPS: scene-activity detector adjusts processing rate without
    touching the camera pipeline (no reconfigure needed).
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
JPEG_QUALITY   = 70
SERVER_PORT    = 8000
LOG_LEVEL      = logging.INFO

# Resolution / framerate presets.
# "fps" controls the camera pipeline rate AND our processing deadline.
PRESETS = {
    "low":    {"size": (320,  240),  "fps": 30, "label": "320×240 / 30 fps"},
    "medium": {"size": (640,  480),  "fps": 20, "label": "640×480 / 20 fps"},
    "high":   {"size": (1280, 960),  "fps": 10, "label": "1280×960 / 10 fps"},
}
DEFAULT_PRESET = "medium"

# Auto-FPS: variance of the last N sharpness readings is used as a proxy
# for scene activity.  Below LOW → idle → halve the processing rate.
# Above HIGH → motion → restore full preset rate.
AUTO_FPS_WINDOW           = 30    # readings
AUTO_FPS_VARIANCE_LOW     = 300.0
AUTO_FPS_VARIANCE_HIGH    = 3000.0

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

auto_fps_enabled = True
_auto_fps_lock   = threading.Lock()

# Rolling history for sharpness-to-percent conversion (last 300 readings).
_sharp_history  = deque(maxlen=300)
# Shorter window for scene-activity / auto-FPS detection.
_scene_window   = deque(maxlen=AUTO_FPS_WINDOW)

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


# ── AUTO-FPS ──────────────────────────────────────────────────────────────

def _update_auto_fps(sharpness: float) -> None:
    """
    Adjust `target_fps` based on scene activity (variance of sharpness).
    Operates within the bounds of the current preset's configured FPS.
    Only adjusts our *processing* sleep — the camera pipeline is unchanged.
    """
    global target_fps, auto_fps_enabled

    with _auto_fps_lock:
        if not auto_fps_enabled:
            return

    _scene_window.append(sharpness)
    if len(_scene_window) < AUTO_FPS_WINDOW // 2:
        return

    vals = list(_scene_window)
    mean = sum(vals) / len(vals)
    var  = sum((v - mean) ** 2 for v in vals) / len(vals)

    with _preset_lock:
        max_fps = PRESETS[current_preset]["fps"]
    min_fps = max(5, max_fps // 3)

    if var < AUTO_FPS_VARIANCE_LOW:
        new_fps = min_fps
    elif var > AUTO_FPS_VARIANCE_HIGH:
        new_fps = max_fps
    else:
        t = (var - AUTO_FPS_VARIANCE_LOW) / (
            AUTO_FPS_VARIANCE_HIGH - AUTO_FPS_VARIANCE_LOW)
        new_fps = int(min_fps + t * (max_fps - min_fps))

    target_fps = max(min_fps, min(max_fps, new_fps))


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
                    _update_auto_fps(sv)
                except Exception as exc:
                    log.debug("Sharpness result error: %s", exc)
                _sharp_future = None

            # ── Submit sharpness job every 3rd frame ──────────────────────
            # frame.tobytes() copies the data so the camera buffer can be
            # safely reused before the worker finishes.
            if frame_count % 3 == 0 and _sharp_future is None:
                _sharp_future = _executor.submit(
                    _sharpness_worker, frame.tobytes(), w, h)

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
  <link href="https://fonts.googleapis.com/css2?family=Azeret+Mono:wght@300;400;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg:      #070708;
      --panel:   #0d0d0f;
      --border:  #1a1a1f;
      --green:   #3dffa0;
      --amber:   #ffb347;
      --red:     #ff5f5f;
      --dim:     #383840;
      --text:    #909098;
      --bright:  #d8d8e0;
    }
    *, *::before, *::after { margin:0; padding:0; box-sizing:border-box; }

    body {
      background: var(--bg);
      color: var(--text);
      font-family: 'Azeret Mono', 'Courier New', monospace;
      min-height: 100vh;
      display: grid;
      /* Changed from 3 rows to a single row taking up all the space */
      grid-template-rows: 1fr; 
      overflow: hidden;
    }

    /* ── MAIN GRID ───────────────────────────────────────────────────── */
    main {
      display: grid;
      grid-template-columns: 1fr 260px;
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
    .frame-wrap {
      position: relative;
      display: inline-flex;
    }
    .frame-wrap img {
      display: block;
      max-width: 100%;
      /* Increased max-height since the top/bottom bars are gone */
      max-height: calc(100vh - 40px);
      object-fit: contain;
      image-rendering: crisp-edges;
    }

    /* ── SIDEBAR ─────────────────────────────────────────────────────── */
    .sidebar {
      display: flex;
      flex-direction: column;
      gap: 0;
      overflow-y: auto;
      scrollbar-width: none;
    }
    .sidebar::-webkit-scrollbar { display: none; }

    .block {
      padding: 14px 16px;
      border-bottom: 1px solid var(--border);
    }
    .block-title {
      font-size: 8px;
      letter-spacing: 3px;
      text-transform: uppercase;
      color: var(--dim);
      margin-bottom: 10px;
    }

    /* Stat tiles */
    .stat-row { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    .stat-tile {
      background: var(--panel);
      border: 1px solid var(--border);
      padding: 10px;
    }
    .stat-label {
      font-size: 7px;
      letter-spacing: 2px;
      text-transform: uppercase;
      color: var(--dim);
      margin-bottom: 5px;
    }
    .stat-val {
      font-size: 20px;
      font-weight: 700;
      color: var(--green);
      line-height: 1;
      transition: color .3s;
    }
    .stat-val.amber { color: var(--amber); }
    .stat-val.red   { color: var(--red);   }

    /* Focus bar */
    .focus-pct {
      font-size: 26px;
      font-weight: 700;
      color: var(--green);
      line-height: 1;
      margin-bottom: 8px;
      transition: color .3s;
    }
    .bar-track {
      height: 3px;
      background: var(--border);
    }
    .bar-fill {
      height: 100%;
      background: var(--green);
      width: 0%;
      transition: width .2s ease, background-color .3s;
    }

    /* Sparkline */
    #sparkline { width: 100%; height: 52px; display: block; }

    /* Preset buttons */
    .preset-list { display: flex; flex-direction: column; gap: 5px; }
    .preset-btn {
      background: transparent;
      border: 1px solid var(--border);
      color: var(--text);
      font-family: inherit;
      font-size: 10px;
      letter-spacing: 1px;
      padding: 7px 10px;
      cursor: pointer;
      text-align: left;
      transition: border-color .15s, color .15s, background .15s;
    }
    .preset-btn:hover  { border-color: var(--green); color: var(--bright); }
    .preset-btn.active {
      border-color: var(--green);
      color: var(--green);
      background: rgba(61,255,160,.05);
    }

    /* Auto-FPS toggle */
    .toggle-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .toggle-label { font-size: 9px; letter-spacing: 2px; text-transform: uppercase; }
    .pill {
      width: 34px; height: 18px;
      background: var(--border);
      border-radius: 9px;
      position: relative;
      cursor: pointer;
      transition: background .2s;
      flex-shrink: 0;
    }
    .pill.on { background: var(--green); }
    .pill::after {
      content: '';
      position: absolute;
      width: 14px; height: 14px;
      background: var(--bg);
      border-radius: 50%;
      top: 2px; left: 2px;
      transition: transform .2s;
    }
    .pill.on::after { transform: translateX(16px); }

  </style>
</head>
<body>
  <main>
    <div class="video-panel">
      <div class="frame-wrap" id="frame-wrap">
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
            <div class="stat-val amber" id="fps-val">&#8212;</div>
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
        <canvas id="sparkline" width="228" height="52"></canvas>
      </div>

      <div class="block">
        <div class="block-title">Resolution / FPS</div>
        <div class="preset-list">
          <button class="preset-btn" data-preset="low"
                  onclick="setPreset(this)">&#9656; 320&times;240 &middot; 30 fps</button>
          <button class="preset-btn active" data-preset="medium"
                  onclick="setPreset(this)">&#9656; 640&times;480 &middot; 20 fps</button>
          <button class="preset-btn" data-preset="high"
                  onclick="setPreset(this)">&#9656; 1280&times;960 &middot; 10 fps</button>
        </div>
      </div>

      <div class="block">
        <div class="block-title">Auto-FPS</div>
        <div class="toggle-row">
          <span class="toggle-label">Scene Adaptive</span>
          <div class="pill on" id="afps-pill" onclick="toggleAutoFps(this)"></div>
        </div>
      </div>

    </div></main>

  <script>
    /* ── State ─────────────────────────────────────────────────────── */
    const hist    = [];
    const MAX_H   = 150;
    let fc        = 0;
    let lastT     = Date.now();
    let measFps   = 0;

    /* ── Sparkline ──────────────────────────────────────────────────── */
    function drawSparkline(data) {
      const canvas = document.getElementById('sparkline');
      const ctx    = canvas.getContext('2d');
      const W = canvas.width, H = canvas.height;
      ctx.clearRect(0, 0, W, H);
      if (data.length < 2) return;

      const min = Math.min(...data);
      const max = Math.max(...data);
      const range = (max - min) || 1;

      const pts = data.map((v, i) => ({
        x: (i / (data.length - 1)) * W,
        y: H - 2 - ((v - min) / range) * (H - 8)
      }));

      /* Glow line */
      ctx.save();
      ctx.strokeStyle = '#3dffa0';
      ctx.lineWidth   = 1.5;
      ctx.shadowColor = '#3dffa0';
      ctx.shadowBlur  = 6;
      ctx.beginPath();
      pts.forEach((p, i) => i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y));
      ctx.stroke();
      ctx.restore();

      /* Fill */
      ctx.beginPath();
      pts.forEach((p, i) => i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y));
      ctx.lineTo(W, H); ctx.lineTo(0, H); ctx.closePath();
      ctx.fillStyle = 'rgba(61,255,160,.06)';
      ctx.fill();
    }

    /* ── Focus bar colour ────────────────────────────────────────────── */
    function focusColor(pct) {
      if (pct >= 75) return 'var(--green)';
      if (pct >= 45) return 'var(--amber)';
      return 'var(--red)';
    }

    /* ── Frame Fetching (Event Driven, No Flickering) ────────────────── */
    const visibleFeed = document.getElementById('feed');
    const bufferImg = new Image();

    bufferImg.onload = () => {
      // Only swap the source once the image is fully downloaded
      visibleFeed.src = bufferImg.src;
      fc++;
      // Immediately fetch the next frame
      requestNextFrame();
    };

    bufferImg.onerror = () => {
      // If a frame drops/fails, wait half a second before trying again
      setTimeout(requestNextFrame, 500);
    };

    function requestNextFrame() {
      bufferImg.src = '/frame.jpg?' + Date.now();
    }

    /* ── Status Fetching (Promise Chaining) ──────────────────────────── */
    function fetchStatus() {
      fetch('/status')
        .then(r => r.json())
        .then(d => {
          const sharp  = parseFloat(d.sharpness);
          const pct    = parseFloat(d.percent);
          const preset = d.preset;

          /* Sharpness tile */
          document.getElementById('sharp-val').textContent = isNaN(sharp) ? '—' : sharp.toFixed(0);

          /* Focus bar */
          const col = focusColor(pct);
          const fpEl = document.getElementById('focus-pct');
          fpEl.textContent = pct.toFixed(1) + '%';
          fpEl.style.color = col;
          const bar = document.getElementById('bar');
          bar.style.width           = Math.min(100, pct) + '%';
          bar.style.backgroundColor = col;

          /* Preset buttons */
          document.querySelectorAll('.preset-btn').forEach(b => {
            b.classList.toggle('active', b.dataset.preset === preset);
          });

          /* Sparkline */
          if (!isNaN(sharp)) {
            hist.push(sharp);
            if (hist.length > MAX_H) hist.shift();
            drawSparkline(hist);
          }
        })
        .catch(() => {})
        .finally(() => {
          // Wait 250ms after the last request finished before polling again
          setTimeout(fetchStatus, 250); 
        });
    }

    /* ── Lightweight UI Loop (FPS Calculation only) ──────────── */
    function updateUI() {
      const now = Date.now();
      const elapsed = (now - lastT) / 1000;
      if (elapsed >= 1.0) {
        measFps = Math.round(fc / elapsed);
        fc = 0; 
        lastT = now;
        document.getElementById('fps-val').textContent = measFps;
      }
    }

    /* ── Controls ────────────────────────────────────────────────────── */
    function setPreset(btn) {
      fetch('/config?preset=' + btn.dataset.preset).catch(() => {});
    }

    function toggleAutoFps(pill) {
      pill.classList.toggle('on');
      const on = pill.classList.contains('on');
      fetch('/config?auto_fps=' + (on ? '1' : '0')).catch(() => {});
    }

    // Bootstrap the loops
    setInterval(updateUI, 100); 
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
        global auto_fps_enabled
        with sharp_lock:
            sv = sharpness_val
            sp = sharpness_pct
        with _preset_lock:
            preset = current_preset
            tfps   = target_fps
        with _auto_fps_lock:
            afps = auto_fps_enabled
        res = "×".join(str(d) for d in PRESETS[preset]["size"])
        payload = json.dumps({
            "sharpness":  f"{sv:.1f}",
            "percent":    f"{sp:.1f}",
            "preset":     preset,
            "resolution": res,
            "target_fps": tfps,
            "auto_fps":   afps,
        }).encode()
        self._write(200, "application/json", payload)

    def _handle_config(self, query: str):
        global _pending_preset, auto_fps_enabled
        params = parse_qs(query)

        if "preset" in params:
            name = params["preset"][0]
            if name in PRESETS:
                with _preset_lock:
                    _pending_preset = name

        if "auto_fps" in params:
            val = params["auto_fps"][0]
            with _auto_fps_lock:
                auto_fps_enabled = val == "1"

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