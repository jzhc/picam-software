#!/usr/bin/env python3
"""
Simple Picamera2 HTTP streaming server with live sharpness display.
Stream to: http://10.0.0.234:8000 in your Mac browser
"""

from picamera2 import Picamera2
from scipy import ndimage
import numpy as np
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import io
from PIL import Image

class SharpnessMonitor:
    def __init__(self):
        self.laplacian = np.array([[0, -1, 0],
                                    [-1, 4, -1],
                                    [0, -1, 0]], dtype=np.float32)
        self.last_sharpness = 0.0
        self.lock = threading.Lock()
    
    def compute_sharpness(self, frame):
        if len(frame.shape) == 3:
            gray = np.mean(frame[:, :, :3], axis=2)
        else:
            gray = frame
        
        lap = ndimage.convolve(gray, self.laplacian)
        return np.var(lap)
    
    def update(self, frame):
        sharpness = self.compute_sharpness(frame)
        with self.lock:
            self.last_sharpness = sharpness
    
    def get_sharpness(self):
        with self.lock:
            return self.last_sharpness

# Global camera, monitor, and frame buffer
camera = None
monitor = None
latest_frame = None
latest_jpeg = None
frame_lock = threading.Lock()

class StreamingHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            # Serve HTML page
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            
            html = """
            <html>
            <head>
                <title>Picamera2 Stream</title>
                <style>
                    body { 
                        font-family: 'Segoe UI', Arial, sans-serif; 
                        margin: 0; 
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        min-height: 100vh;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                    }
                    .container { 
                        background: white; 
                        padding: 30px; 
                        border-radius: 12px;
                        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
                        max-width: 700px;
                        text-align: center;
                    }
                    h1 { 
                        color: #333; 
                        margin: 0 0 20px 0;
                    }
                    img { 
                        width: 100%; 
                        border-radius: 8px;
                        border: 2px solid #ddd;
                        margin-bottom: 20px;
                        max-width: 640px;
                    }
                    .stats { 
                        background: #f5f5f5;
                        padding: 20px;
                        border-radius: 8px;
                        margin-top: 20px;
                    }
                    .stat-label {
                        font-size: 14px;
                        color: #666;
                        margin-bottom: 5px;
                    }
                    .sharpness { 
                        color: #667eea; 
                        font-size: 32px;
                        font-weight: bold;
                    }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>📷 Camera Preview</h1>
                    <img src="/frame.jpg" alt="Camera stream">
                    <div class="stats">
                        <div class="stat-label">Sharpness Value:</div>
                        <div class="sharpness" id="sharp">--</div>
                    </div>
                </div>
                <script>
                    // Update image and sharpness every 100ms
                    setInterval(function() {
                        document.querySelector('img').src = '/frame.jpg?' + Date.now();
                        fetch('/sharpness')
                            .then(r => r.text())
                            .then(data => {
                                document.getElementById('sharp').textContent = data;
                            })
                            .catch(() => {});
                    }, 100);
                </script>
            </body>
            </html>
            """
            self.wfile.write(html.encode('utf-8'))
        
        elif self.path == '/sharpness':
            # Return current sharpness value
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            sharpness = monitor.get_sharpness()
            self.wfile.write(f"{sharpness:.1f}".encode())
        
        elif self.path == '/frame.jpg':
            # Return current JPEG frame
            global latest_jpeg
            self.send_response(200)
            self.send_header('Content-Type', 'image/jpeg')
            
            with frame_lock:
                if latest_jpeg:
                    self.send_header('Content-Length', len(latest_jpeg))
                    self.end_headers()
                    self.wfile.write(latest_jpeg)
                else:
                    self.end_headers()
    
    def log_message(self, format, *args):
        # Suppress request logging
        pass

def capture_and_serve():
    """Continuously capture frames and convert to JPEG"""
    global latest_frame, latest_jpeg
    
    print("[*] Waiting for camera to warm up...")
    time.sleep(2)
    
    print("[*] Starting frame capture...")
    
    try:
        while True:
            # Capture frame
            frame = camera.capture_array()
            
            # Update sharpness
            monitor.update(frame)
            
            # Convert to JPEG
            img = Image.fromarray(frame[:, :, :3].astype('uint8'))
            jpeg_buffer = io.BytesIO()
            img.save(jpeg_buffer, format='JPEG', quality=85)
            jpeg_data = jpeg_buffer.getvalue()
            
            # Update buffer
            with frame_lock:
                latest_frame = frame
                latest_jpeg = jpeg_data
            
            time.sleep(0.033)  # ~30 FPS
    
    except KeyboardInterrupt:
        pass

def main():
    global camera, monitor
    
    print("=" * 60)
    print("Picamera2 HTTP Streaming Server")
    print("=" * 60)
    
    # Initialize camera
    print("[*] Initializing camera...")
    camera = Picamera2()
    monitor = SharpnessMonitor()
    
    config = camera.create_preview_configuration(
          main={"size": (640, 480)},
          controls={
               "AeEnable": True,   # Enable auto-exposure
               "AwbEnable": True,  # Enable auto-white balance
               "ExposureTime": 20000,
               "AnalogueGain": 2.0
     }
    )
    camera.configure(config)
    camera.start()
    
    # Start capture thread
    print("[*] Starting capture thread...")
    capture_thread = threading.Thread(target=capture_and_serve, daemon=True)
    capture_thread.start()
    
    # Start HTTP server
    print()
    print("=" * 60)
    print("🌐 STREAM AVAILABLE AT:")
    print("   http://10.0.0.234:8000")
    print()
    print("Open this URL in your Mac browser")
    print("=" * 60)
    print()
    
    try:
        server = HTTPServer(('0.0.0.0', 8000), StreamingHandler)
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Shutting down...")
    finally:
        try:
            camera.stop()
        except:
            pass

if __name__ == "__main__":
    main()