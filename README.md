# Pi HQ Camera — Live Preview + Focus Assistant

## Dependencies
```bash
sudo apt update && sudo apt install -y \
    python3-picamera2 \
    python3-numpy \
    python3-setuptools \
    python3-dev \
    python3-pip \
    libturbojpeg0-dev
```
```bash
python3 -m pip install PyTurboJPEG --break-system-packages
```

## ⚠️ Build First
```bash
sudo python3 setup.py build_ext --inplace
```
Only needs to be run once. Compiles the C extension (`camera_utils`) which handles sharpness computation and JPEG encoding. The main script will crash if this step is skipped.

## Running
```bash
python3 main.py
```
Open the URL printed in the terminal in any browser on the same WiFi network. Use `http://` not `https://`.

## Files
| File | Description |
|---|---|
| `main.py` | Main application — camera, web server, sharpness logic |
| `main_hd.py` | CLI application — Native HDMI preview (DRM) and terminal statistics |
| `camera_utils.c` | C extension — sharpness computation, downsampling, and JPEG encoding via libturbojpeg |
| `setup.py` | Builds the C extension |

## Interface Values

- **Sharpness** — raw Laplacian variance value
- **Focus %** — relative focus percentage based on a rolling best-focus reference

Focus % self-calibrates over ~30 seconds. The sharpest point seen becomes the 100% reference and everything else is scored relative to it.
