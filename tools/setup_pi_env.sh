#!/usr/bin/env bash
set -euo pipefail

# setup_pi_env.sh
# Prepare a Raspberry Pi for running the YOLO camera app in a Python venv.
# This installs system packages, creates a venv, and attempts to install
# Python dependencies. Manual step is required to install the correct
# PyTorch wheel for your Pi model / OS.

echo "--- Updating apt and installing system packages (may require sudo) ---"
sudo apt update
sudo apt install -y python3-venv python3-pip build-essential cmake git pkg-config \
    libjpeg-dev libpng-dev libtiff-dev libavcodec-dev libavformat-dev libswscale-dev \
    libv4l-dev libxvidcore-dev libx264-dev libopenblas-dev libatlas-base-dev \
    libatlas3-base libopenjp2-7-dev libgtk2.0-dev libcanberra-gtk-module \
    python3-picamera2 libcamera-apps

echo "--- Create Python virtualenv in .venv ---"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

echo "--- Install OpenCV (headless recommended on Pi) ---"
# Use headless OpenCV to avoid GUI issues on most Pi setups
pip install opencv-python-headless

echo "--- Install ultralytics (will require torch) ---"
# Ultralytics requires torch. On Raspberry Pi, installing torch often requires
# a pre-built wheel appropriate for your CPU and OS. There are community wheels
# and sometimes official wheels for specific Pi OS releases. Installing torch
# via pip without an appropriate wheel may fail.

echo "Important: installing torch on Raspberry Pi often requires a prebuilt wheel."
echo "If you know a wheel URL for your Pi/OS, install it now with:"
echo "  pip install <path-or-url-to-torch-wheel>"
echo "If you don't have a wheel, try pip directly (may fail):"
echo "  pip install torch"

echo "After torch is installed successfully, install ultralytics:" 
echo "  pip install ultralytics"

echo "--- Install other Python deps from repo requirements.txt ---"
# Install requirements besides torch which we treated separately
pip install -r ../requirements.txt || true

echo "--- Setup complete (manual torch wheel step may be required) ---"
echo "To run the YOLO camera demo (headless quick test):"
echo "  source .venv/bin/activate"
echo "  python3 ../app_yolo_camera.py --no-yolo --no-gui --frames 1"

echo "To run with YOLO once torch & ultralytics are installed:" 
echo "  python3 ../app_yolo_camera.py --yolo-weights yolov8n.pt"

exit 0
