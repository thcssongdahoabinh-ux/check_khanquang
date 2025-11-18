"""Environment check helper for camera + YOLO app.

Reports availability of key runtime components: Python modules, system devices,
and helpful hints for installing missing pieces.
"""
import importlib
import shutil
import sys
import os

def check_module(name):
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False

def which(cmd):
    return shutil.which(cmd) is not None

def main():
    print('Environment check for check_khanquang camera app')
    print('-----------------------------------------------')
    modules = ['cv2', 'ultralytics', 'torch', 'picamera2']
    for m in modules:
        print(f"Module {m}: {'FOUND' if check_module(m) else 'MISSING'}")

    # Check for libcamera utilities on system
    print(f"libcamera-hello: {'FOUND' if which('libcamera-hello') else 'MISSING'}")
    print(f"vcgencmd: {'FOUND' if which('vcgencmd') else 'MISSING'}")

    # Check for camera devices
    dev_exists = any(os.path.exists(p) for p in ['/dev/video0', '/dev/media0'])
    print(f"Camera device present (/dev/video0 or /dev/media0): {'YES' if dev_exists else 'NO'}")

    print('\nNotes:')
    print('- If modules are missing, create a Python venv and run pip install -r requirements.txt')
    print('- On Raspberry Pi, install system packages: apt install python3-picamera2 libcamera-apps')

if __name__ == '__main__':
    main()
