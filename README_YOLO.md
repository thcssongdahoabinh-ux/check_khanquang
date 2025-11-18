# Running the YOLO camera app

This project includes `app_yolo_camera.py`, a sample script that captures camera frames
and runs YOLOv8 inference when `ultralytics` is available. Because `ultralytics` and
`torch` have platform-specific installation requirements (especially on Raspberry Pi),
follow these steps to set up a safe Python environment.

1) Create and activate a virtual environment

```bash
cd /home/admin/Apps/check_khanquang
python3 -m venv .venv
source .venv/bin/activate
```

2) Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Note: On Raspberry Pi, install system packages first:

```bash
sudo apt update
sudo apt install -y python3-picamera2 libcamera-apps
```

3) (Optional) If you can't install `opencv-contrib-python` with GUI support, use headless OpenCV:

```bash
pip uninstall opencv-contrib-python
pip install opencv-python-headless
```

4) Run the app

Headless quick smoke test (no YOLO):

```bash
python3 app_yolo_camera.py --no-yolo --no-gui --frames 1
```

Run with YOLO (requires `ultralytics` + `torch`):

```bash
python3 app_yolo_camera.py --yolo-weights yolov8n.pt
```

The script will attempt to download `yolov8n.pt` automatically if it's not present.

Troubleshooting:
- If camera initialization fails, check `tools/env_check.py` and ensure `/dev/video0` or
  libcamera utilities are present.
- For Raspberry Pi, refer to `DEPLOYMENT_RASPBERRY_PI.md` in this repo for more details.
