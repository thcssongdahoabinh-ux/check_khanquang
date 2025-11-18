#!/usr/bin/env python3
"""Simple CLI runner to validate camera backends on the current machine.

Usage examples:
  python3 scripts/check_camera_runner.py         # tries Picamera2 then VideoCapture
  python3 scripts/check_camera_runner.py --no-picam  # skip Picamera2 test
"""
import argparse
import sys
from camera_backend import has_picamera2, init_picamera2, init_videocapture


def main():
    parser = argparse.ArgumentParser(description="Check Picamera2 and VideoCapture backends")
    parser.add_argument("--no-picam", action="store_true", help="Skip testing Picamera2")
    parser.add_argument("--camera-index", type=int, default=0, help="VideoCapture index to try")
    parser.add_argument("--try-alternate-indices", action="store_true", help="Try indices 0..3 if initial fails")
    args = parser.parse_args()

    ok = True

    if not args.no_picam:
        print("Checking Picamera2...", end=" ")
        if has_picamera2():
            pic = init_picamera2()
            if pic is not None:
                print("OK (Picamera2 started)")
                try:
                    pic.stop()
                except Exception:
                    pass
            else:
                print("FAILED to start Picamera2")
                ok = False
        else:
            print("NOT AVAILABLE")

    print(f"Checking OpenCV VideoCapture (index {args.camera_index})...", end=" ")
    cap = init_videocapture(index=args.camera_index, try_alternate=args.try_alternate_indices)
    if cap is not None:
        print("OK (VideoCapture opened)")
        try:
            cap.release()
        except Exception:
            pass
    else:
        print("FAILED to open VideoCapture")
        ok = False

    if ok:
        print("All checks passed")
        return 0
    else:
        print("Some checks failed")
        return 2


if __name__ == "__main__":
    sys.exit(main())
