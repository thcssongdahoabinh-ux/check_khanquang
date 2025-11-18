"""camera_backend.py

Small portable backend helpers to initialize Picamera2 or OpenCV VideoCapture.

These helpers are intentionally small and hardware-aware; unit tests mock external
libraries to validate behavior without needing a camera attached.
"""
from typing import Optional, Tuple
import time

# Try to import Picamera2. Real import may fail on non-Pi systems; callers should
# handle the absence gracefully.
try:
    from picamera2 import Picamera2  # type: ignore
    _HAS_PICAMERA2 = True
except Exception:
    Picamera2 = None
    _HAS_PICAMERA2 = False

# Try to import rpicam (alternative Raspberry Pi camera library requested)
try:
    import rpicam  # type: ignore
    _HAS_RPICAM = True
except Exception:
    rpicam = None
    _HAS_RPICAM = False
    # If there's no python package, check for system rpicam utilities (cli)
    import shutil
    if shutil.which('rpicam-hello') or shutil.which('rpicam-capture'):
        _HAS_RPICAM = True

try:
    import cv2
    _HAS_CV2 = True
except Exception:
    cv2 = None
    _HAS_CV2 = False


def has_picamera2() -> bool:
    """Return True if Picamera2 appears importable on this system."""
    return _HAS_PICAMERA2


def has_rpicam() -> bool:
    """Return True if rpicam appears importable on this system."""
    return _HAS_RPICAM



def init_picamera2(preview_size: Tuple[int, int] = (640, 480), warmup: float = 1.0):
    """Try to configure and start a Picamera2 instance.

    Returns the Picamera2 instance on success, or None on failure.
    """
    if not _HAS_PICAMERA2:
        return None
    try:
        picam2 = Picamera2()
        # Try a few common configuration helpers to maximize compatibility
        tried = []
        try:
            cfg = picam2.create_preview_configuration(main={"size": preview_size})
            picam2.configure(cfg)
            picam2.start()
            time.sleep(warmup)
            return picam2
        except Exception as e1:
            tried.append(('preview', str(e1)))
        try:
            cfg = picam2.create_video_configuration(main={"size": preview_size})
            picam2.configure(cfg)
            picam2.start()
            time.sleep(warmup)
            return picam2
        except Exception as e2:
            tried.append(('video', str(e2)))
        try:
            cfg = picam2.create_still_configuration(main={"size": preview_size})
            picam2.configure(cfg)
            picam2.start()
            time.sleep(warmup)
            return picam2
        except Exception as e3:
            tried.append(('still', str(e3)))
        # If none of the above succeeded, log attempts for diagnostics and return None
        # Avoid raising to keep callers simple; they should handle None.
        # Print to stderr to surface info in logs/debug runs.
        try:
            import sys as _sys
            _sys.stderr.write(f"init_picamera2: failed configurations: {tried}\n")
        except Exception:
            pass
        return None
    except Exception:
        return None


def init_rpicam(preview_size: Tuple[int, int] = (640, 480), warmup: float = 1.0):
    """Try to initialize an rpicam camera and return an adapter with
    a `capture_array()` method (similar to Picamera2). Returns None on failure.
    """
    if not _HAS_RPICAM:
        return None

    # If a python package exists, attempt to use it; otherwise fall back to
    # invoking system CLI utilities like `rpicam-hello` or `rpicam-capture`.
    try:
        if rpicam is not None:
            cam = None
            if hasattr(rpicam, 'RPiCamera'):
                cam = rpicam.RPiCamera()
            elif hasattr(rpicam, 'PiCamera'):
                cam = rpicam.PiCamera()
            elif hasattr(rpicam, 'Camera'):
                cam = rpicam.Camera()
            elif hasattr(rpicam, 'open'):
                cam = rpicam.open()

            if cam is not None:
                class _RpiCamAdapterPy:
                    def __init__(self, cam):
                        self._cam = cam

                    def capture_array(self):
                        if hasattr(self._cam, 'capture_array'):
                            return self._cam.capture_array()
                        if hasattr(self._cam, 'read'):
                            ret, frame = self._cam.read()
                            if ret:
                                return frame
                            raise RuntimeError('rpicam read() returned false')
                        if hasattr(self._cam, 'capture'):
                            import io
                            import numpy as _np
                            import cv2 as _cv2
                            buf = io.BytesIO()
                            try:
                                self._cam.capture(buf, format='jpeg')
                            except Exception:
                                try:
                                    self._cam.capture(format='jpeg', output=buf)
                                except Exception:
                                    raise
                            buf.seek(0)
                            data = _np.frombuffer(buf.read(), dtype=_np.uint8)
                            frame = _cv2.imdecode(data, _cv2.IMREAD_COLOR)
                            return frame
                        raise RuntimeError('rpicam has no known capture method')

                    def stop(self):
                        for fn in ('stop', 'close', 'release'):
                            if hasattr(self._cam, fn):
                                try:
                                    getattr(self._cam, fn)()
                                except Exception:
                                    pass

                try:
                    time.sleep(warmup)
                except Exception:
                    pass
                return _RpiCamAdapterPy(cam)

    except Exception:
        # Fall back to CLI path
        pass

    # CLI-based adapter: run rpicam-hello or rpicam-capture to produce a JPEG,
    # then load it with OpenCV. This is slower but maximally compatible.
    try:
        import shutil, subprocess, tempfile, cv2 as _cv2

        cli_cmd = None
        # Prefer a CLI that can directly capture a JPEG
        if shutil.which('rpicam-capture'):
            cli_cmd = 'rpicam-capture'
        elif shutil.which('libcamera-jpeg'):
            cli_cmd = 'libcamera-jpeg'
        elif shutil.which('libcamera-still'):
            cli_cmd = 'libcamera-still'
        elif shutil.which('rpicam-hello'):
            cli_cmd = 'rpicam-hello'
        else:
            # No CLI available
            return None

        class _RpiCamAdapterCLI:
            def __init__(self, cmd):
                self._cmd = cmd

            def capture_array(self):
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tf:
                    fname = tf.name
                try:
                    if self._cmd == 'rpicam-capture':
                        subprocess.check_call([self._cmd, '-o', fname])
                    elif self._cmd == 'libcamera-jpeg':
                        subprocess.check_call(['libcamera-jpeg', '-o', fname])
                    elif self._cmd == 'libcamera-still':
                        subprocess.check_call(['libcamera-still', '-o', fname])
                    elif self._cmd == 'rpicam-hello':
                        # rpicam-hello doesn't support direct capture; try libcamera-jpeg as fallback
                        if shutil.which('libcamera-jpeg'):
                            subprocess.check_call(['libcamera-jpeg', '-o', fname])
                        elif shutil.which('libcamera-still'):
                            subprocess.check_call(['libcamera-still', '-o', fname])
                        else:
                            # As a last resort, attempt to run rpicam-hello (may not save image)
                            subprocess.check_call([self._cmd])
                    img = _cv2.imread(fname)
                    return img
                finally:
                    try:
                        os.remove(fname)
                    except Exception:
                        pass

            def stop(self):
                return

        return _RpiCamAdapterCLI(cli_cmd)
    except Exception:
        return None


def init_videocapture(index: int = 0, try_alternate: bool = False, max_retries: int = 3, retry_delay: float = 0.2):
    """Initialize OpenCV VideoCapture.

    If try_alternate is True and the requested index fails, this will try indices
    0..3. Returns the opened cv2.VideoCapture instance or None.
    """
    if not _HAS_CV2:
        return None

    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        return cap

    if try_alternate:
        for i in range(0, 4):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                return cap
    # Try a few read retries to allow the camera to warm up
    attempts = 0
    while attempts < max_retries:
        time.sleep(retry_delay)
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            return cap
        attempts += 1

    return None
