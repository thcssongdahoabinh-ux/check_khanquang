"""Unit tests for camera_backend.py.

These tests mock external dependencies (picamera2 and cv2.VideoCapture) so they
can run on CI or dev machines without actual cameras.
"""
import importlib
import sys
from unittest import mock


def test_has_picamera2_when_available(monkeypatch):
    # Simulate picamera2 available
    fake_mod = mock.MagicMock()
    monkeypatch.setitem(sys.modules, 'picamera2', fake_mod)
    # reload module under test to pick up monkeypatched import
    cb = importlib.reload(importlib.import_module('camera_backend'))
    assert cb.has_picamera2() is True


def test_has_picamera2_when_not_available(monkeypatch):
    # Ensure picamera2 not in modules
    if 'picamera2' in sys.modules:
        del sys.modules['picamera2']
    cb = importlib.reload(importlib.import_module('camera_backend'))
    assert cb.has_picamera2() is False


def test_init_videocapture_fallback(monkeypatch):
    # Mock cv2.VideoCapture behavior
    fake_cap = mock.MagicMock()
    fake_cap.isOpened.return_value = True

    fake_cv2 = mock.MagicMock()
    fake_cv2.VideoCapture.return_value = fake_cap

    monkeypatch.setitem(sys.modules, 'cv2', fake_cv2)
    cb = importlib.reload(importlib.import_module('camera_backend'))
    cap = cb.init_videocapture(index=0, try_alternate=False)
    assert cap is not None
