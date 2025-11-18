"""Ad-hoc test runner for camera_backend.py when pytest isn't available.

This script replicates the behavior of `tests/test_camera_backend.py` using
simple assertions and module import tricks so it can run under plain Python.

Exit code 0 on success, 1 on failure.
"""
import importlib
import sys
import os
from unittest import mock

# Ensure project root is on sys.path so imports like `camera_backend` work
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

failures = 0

print('Running ad-hoc camera_backend tests...')

# Test 1: has_picamera2 when available
try:
    fake_mod = mock.MagicMock()
    sys.modules['picamera2'] = fake_mod
    cb = importlib.reload(importlib.import_module('camera_backend'))
    assert cb.has_picamera2() is True
    print('PASS: has_picamera2 when available')
except Exception as e:
    print('FAIL: has_picamera2 when available ->', e)
    failures += 1
finally:
    # cleanup
    if 'picamera2' in sys.modules:
        del sys.modules['picamera2']

# Test 2: has_picamera2 when not available
try:
    # Force imports of 'picamera2' to fail by patching builtins.__import__
    import builtins

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == 'picamera2' or (isinstance(name, str) and name.startswith('picamera2')):
            raise ImportError('simulated missing picamera2')
        return real_import(name, globals, locals, fromlist, level)

    builtins.__import__ = fake_import
    try:
        if 'picamera2' in sys.modules:
            del sys.modules['picamera2']
        cb = importlib.reload(importlib.import_module('camera_backend'))
        assert cb.has_picamera2() is False
        print('PASS: has_picamera2 when not available')
    finally:
        builtins.__import__ = real_import
except Exception as e:
    print('FAIL: has_picamera2 when not available ->', e)
    failures += 1

# Test 3: init_videocapture fallback (mock cv2.VideoCapture)
try:
    fake_cap = mock.MagicMock()
    fake_cap.isOpened.return_value = True

    fake_cv2 = mock.MagicMock()
    fake_cv2.VideoCapture.return_value = fake_cap

    sys.modules['cv2'] = fake_cv2
    cb = importlib.reload(importlib.import_module('camera_backend'))
    cap = cb.init_videocapture(index=0, try_alternate=False)
    assert cap is not None
    print('PASS: init_videocapture fallback with mocked cv2')
except Exception as e:
    print('FAIL: init_videocapture fallback ->', e)
    failures += 1
finally:
    if 'cv2' in sys.modules:
        del sys.modules['cv2']

if failures:
    print(f"{failures} test(s) failed")
    sys.exit(1)
else:
    print('All tests passed')
    sys.exit(0)
