#!/usr/bin/env python3
"""
Pi HQ Camera - Preview + live sharpness (Polling method)
----------------------------------------------
Stream: http://<pi-ip>:8000
No GPIO. Preview only.
"""

from picamera2 import Picamera2
from PIL import Image
from collections import deque
import sharpness_c
import numpy as np
import time, threading, socket, io, json, logging
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

# ── CONFIG ─────────────────────────────────────
PREVIEW_RESOLUTION = (640, 480)
SERVER_PORT        = 8000
LOG_LEVEL          = logging.INFO
# ───────────────────────────────────────────────

logging.basicConfig(level=LOG_LEVEL, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── GLOBALS ────────────────────────────────────
camera         = None
jpeg_frame     = None
frame_lock     = threading.Lock()
sharpness_val  = 0.0
sharpness_pct  = 0.0
sharp_lock     = threading.Lock()

# rolling buffer of last 300 readings (~30 seconds at 10fps)
_sharp_history = deque(maxlen=300)

# ── SHARPNESS ──────────────────────────────────
def compute_sharpness(frame):
    rgb = frame[:, :, :3].astype('uint8')
    h, w = rgb.shape[:2]
    return sharpness_c.compute_sharpness(bytes(rgb), w, h)

def sharpness_to_pct(current):
    _sharp_history.append(current)
    if len(_sharp_history) < 10:
        return 0.0
    # average of top 10% of readings = "best focus" reference
    sorted_vals = sorted(_sharp_history, reverse=True)
    top_n = max(1, len(sorted_vals) // 10)
    reference = sum(sorted_vals[:top_n]) / top_n
    if reference == 0:
        return 0.0
    return min(100.0, (current / reference) * 100)

# ── FRAME LOOP ─────────────────────────────────
def frame_loop():
    global jpeg_frame, sharpness_val, sharpness_pct

    log.info("Frame loop: warming up...")
    time.sleep(2)
    log.info("Frame loop: running.")

    frame_count = 0

    while True:
        try:
            frame = camera.capture_array()

            if frame_count % 3 == 0:
                sharpness = compute_sharpness(frame)
                pct = sharpness_to_pct(sharpness)
                with sharp_lock:
                    sharpness_val = sharpness
                    sharpness_pct = pct

            frame_count += 1

            img = Image.fromarray(frame[:, :, :3].astype('uint8'))
            buf = io.BytesIO()
            img.save(buf, format='JPEG', quality=70)
            with frame_lock:
                jpeg_frame = buf.getvalue()

        except Exception as e:
            log.warning(f"Frame error: {e}")
            time.sleep(0.1)

        time.sleep(0.1)   # ~10 FPS

# ── HTML ───────────────────────────────────────
HTML = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Pi Camera</title>
    <style>
        * { margin:0; padding:0; box-sizing:border-box; }
        body {
            background:#0d0d0d; color:#ccc;
            font-family:'Courier New',monospace;
            display:flex; flex-direction:column;
            align-items:center; justify-content:center;
            min-height:100vh; gap:12px; padding:20px;
        }
        h1 { font-size:11px; letter-spacing:5px; color:#444; text-transform:uppercase; }
        img { width:640px; max-width:100%; border:1px solid #1a1a1a; background:#111; }
        .stats {
            display:flex; gap:40px; font-size:13px; color:#555;
        }
        .stat { display:flex; flex-direction:column; align-items:center; gap:4px; }
        .label { font-size:10px; letter-spacing:2px; text-transform:uppercase; color:#444; }
        .val { color:#7ec87e; font-weight:bold; font-size:20px; }
    </style>
</head>
<body>
    <h1>&#9632; Pi HQ Camera &mdash; Live Preview</h1>
    <img src="/frame.jpg" alt="stream">
    <div class="stats">
        <div class="stat">
            <span class="label">Sharpness</span>
            <span class="val" id="sharp">loading...</span>
        </div>
        <div class="stat">
            <span class="label">Focus %</span>
            <span class="val" id="pct">loading...</span>
        </div>
    </div>
    <script>
        setInterval(function() {
            document.querySelector('img').src = '/frame.jpg?' + Date.now();
            fetch('/status')
                .then(r => r.json())
                .then(d => {
                    document.getElementById('sharp').textContent = d.sharpness;
                    document.getElementById('pct').textContent   = d.percent + '%';
                })
                .catch(() => {});
        }, 100);
    </script>
</body>
</html>"""

# ── HTTP SERVER ────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split('?')[0]

        if   path == "/":          self._html()
        elif path == "/frame.jpg": self._frame()
        elif path == "/status":    self._status()
        else:                      self.send_error(404)

    def _html(self):
        data = HTML.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", len(data))
        self.end_headers()
        self.wfile.write(data)

    def _frame(self):
        with frame_lock:
            frame = jpeg_frame
        if frame:
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", len(frame))
            self.end_headers()
            self.wfile.write(frame)
        else:
            self.send_error(503)

    def _status(self):
        with sharp_lock:
            s = sharpness_val
            p = sharpness_pct
        data = json.dumps({"sharpness": f"{s:.1f}", "percent": f"{p:.1f}"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(data))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass

# ── UTILS ──────────────────────────────────────
def get_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "localhost"

# ── MAIN ───────────────────────────────────────
def main():
    global camera
    log.info("Starting camera...")
    camera = Picamera2()

    config = camera.create_preview_configuration(
        main={"size": PREVIEW_RESOLUTION},
        controls={
            "AeEnable":  True,
            "AwbEnable": True,
        }
    )
    camera.configure(config)
    camera.start()
    log.info("Camera started.")

    threading.Thread(target=frame_loop, daemon=True).start()

    ip = get_ip()
    print()
    print("=" * 40)
    print(f"  Stream: http://{ip}:{SERVER_PORT}")
    print("=" * 40)
    print()

    try:
        ThreadingHTTPServer(("0.0.0.0", SERVER_PORT), Handler).serve_forever()
    except KeyboardInterrupt:
        log.info("Stopping...")
    finally:
        camera.stop()

if __name__ == "__main__":
    main()