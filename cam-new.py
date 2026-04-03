#!/usr/bin/env python3
"""
Pi HQ Camera - Preview + live sharpness (Polling method)
----------------------------------------------
Stream: http://<pi-ip>:8000
No GPIO. Preview only.
"""

from picamera2 import Picamera2
from scipy import ndimage
from PIL import Image
import numpy as np
import time, threading, socket, io, json, logging
from http.server import HTTPServer, BaseHTTPRequestHandler

# ── CONFIG ─────────────────────────────────────
PREVIEW_RESOLUTION = (640, 480)
SERVER_PORT        = 8000
LOG_LEVEL          = logging.INFO
# ───────────────────────────────────────────────

logging.basicConfig(level=LOG_LEVEL, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── GLOBALS ────────────────────────────────────
camera        = None
jpeg_frame    = None
frame_lock    = threading.Lock()
sharpness_val = 0.0
sharp_lock    = threading.Lock()

# ── SHARPNESS ──────────────────────────────────
_lap = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]], dtype=np.float32)

def compute_sharpness(frame):
    if len(frame.shape) == 3:
        gray = np.mean(frame[:, :, :3], axis=2)
    else:
        gray = frame
    lap = ndimage.convolve(gray.astype(np.float32), _lap)
    return float(np.var(lap))

# ── FRAME LOOP ─────────────────────────────────
def frame_loop():
    global jpeg_frame, sharpness_val

    log.info("Frame loop: warming up...")
    time.sleep(2)
    log.info("Frame loop: running.")

    while True:
        try:
            frame = camera.capture_array()

            sharpness = compute_sharpness(frame)
            with sharp_lock:
                sharpness_val = sharpness

            img = Image.fromarray(frame[:, :, :3].astype('uint8'))
            buf = io.BytesIO()
            img.save(buf, format='JPEG', quality=85)
            with frame_lock:
                jpeg_frame = buf.getvalue()

        except Exception as e:
            log.warning(f"Frame error: {e}")
            time.sleep(0.1)

        time.sleep(0.033)   # ~30 FPS

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
        .stats { font-size:13px; color:#555; }
        .val { color:#7ec87e; font-weight:bold; font-size:20px; }
    </style>
</head>
<body>
    <h1>&#9632; Pi HQ Camera &mdash; Live Preview</h1>
    <img src="/frame.jpg" alt="stream">
    <div class="stats">Sharpness: <span class="val" id="sharp">loading...</span></div>
    <script>
        setInterval(function() {
            document.querySelector('img').src = '/frame.jpg?' + Date.now();
            fetch('/status')
                .then(r => r.json())
                .then(d => { document.getElementById('sharp').textContent = d.sharpness; })
                .catch(() => {});
        }, 100);
    </script>
</body>
</html>"""

# ── HTTP SERVER ────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        # Strip timestamp query parameters used for cache busting
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
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        with frame_lock:
            frame = jpeg_frame
            
        if frame:
            self.send_header("Content-Length", len(frame))
            self.end_headers()
            self.wfile.write(frame)
        else:
            self.end_headers()

    def _status(self):
        with sharp_lock:
            s = sharpness_val
        data = json.dumps({"sharpness": f"{s:.1f}"}).encode()
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
        HTTPServer(("0.0.0.0", SERVER_PORT), Handler).serve_forever()
    except KeyboardInterrupt:
        log.info("Stopping...")
    finally:
        camera.stop()

if __name__ == "__main__":
    main()