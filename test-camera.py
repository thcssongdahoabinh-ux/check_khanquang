import os
import sys
import time
import cv2
import argparse

# Try to import Picamera2; if it's not available, fall back to OpenCV VideoCapture
try:
	from picamera2 import Picamera2  # type: ignore
	_PICAMERA2_AVAILABLE = True
except Exception:
	Picamera2 = None
	_PICAMERA2_AVAILABLE = False

# Optional: try to load a YOLO model if ultralytics is installed and a weights file is present
model = None

"""
test-camera.py

Picamera2 + OpenCV preview script with optional YOLO inference.

Features:
- Uses Picamera2 when available; falls back to OpenCV VideoCapture.
- Optional YOLO inference (ultralytics) unless --no-yolo is specified.
- Headless-friendly: can run without GUI (use --no-gui or will auto-fallback if OpenCV lacks GUI).
- Options to try alternate camera indices and save frames to disk.
"""

import os
import sys
import time
import cv2
import argparse

# Try to import Picamera2; if it's not available, fall back to OpenCV VideoCapture
try:
	from picamera2 import Picamera2  # type: ignore
	_PICAMERA2_AVAILABLE = True
except Exception:
	Picamera2 = None
	_PICAMERA2_AVAILABLE = False

parser = argparse.ArgumentParser(description="Picamera2/OpenCV preview with optional YOLO")
parser.add_argument("--no-yolo", action="store_true", help="Disable YOLO model loading and inference")
parser.add_argument("--no-gui", action="store_true", help="Run in headless mode (do not open GUI windows)")
parser.add_argument("--camera-index", type=int, default=0, help="Fallback camera index for OpenCV VideoCapture (default: 0)")
parser.add_argument("--try-alternate-indices", action="store_true", help="If VideoCapture fails, try indices 0..3")
parser.add_argument("--save-frames", type=str, default=None, help="Directory to save captured frames (optional)")
parser.add_argument("--yolo-weights", type=str, default="yolov8n.pt", help="YOLO weights path (default: yolov8n.pt)")
args = parser.parse_args()

# Optional: try to load a YOLO model if ultralytics is installed and a weights file is present
model = None
if not args.no_yolo:
	try:
		from ultralytics import YOLO  # type: ignore
		model = YOLO(args.yolo_weights)
		print(f"YOLO model loaded: {args.yolo_weights}")
	except Exception:
		model = None
		print("YOLO not available or failed to load; continuing without YOLO.")

use_picam = _PICAMERA2_AVAILABLE
gui_forced_off = args.no_gui

if use_picam:
	picam2 = Picamera2()
	picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
	picam2.start()
	time.sleep(1)   # Allow camera to warm up
	print("Starting Picamera2 preview. Press 'q' to quit.")
else:
	cam_index = args.camera_index if args.camera_index is not None else 0
	cap = cv2.VideoCapture(cam_index)
	if not cap.isOpened():
		if args.try_alternate_indices:
			print(f"VideoCapture index {cam_index} failed; trying alternate indices 0..3")
			for i in range(0, 4):
				cap = cv2.VideoCapture(i)
				if cap.isOpened():
					print(f"Opened camera index {i}")
					break
			else:
				print("Error: cannot open any camera indices 0..3 (VideoCapture).")
				sys.exit(1)
		else:
			print(f"Error: cannot open camera (VideoCapture({cam_index})).")
			sys.exit(1)
	print("Starting OpenCV VideoCapture preview. Press 'q' to quit.")

# Frame saving setup
save_dir = None
if args.save_frames:
	save_dir = os.path.abspath(args.save_frames)
	os.makedirs(save_dir, exist_ok=True)
	print(f"Saving frames to: {save_dir}")

try:
	frame_counter = 0
	while True:
		# Capture frame depending on available backend
		if use_picam:
			frame = picam2.capture_array()
		else:
			ret, frame = cap.read()
			if not ret:
				# Try a few times before giving up (camera may be warming up)
				print("Warning: failed to read frame from VideoCapture, retrying...")
				retry = 0
				max_retries = 5
				while retry < max_retries and not ret:
					time.sleep(0.2)
					ret, frame = cap.read()
					retry += 1
				if not ret:
					print("Error: unable to read frames from VideoCapture after retries. Exiting.")
					break

		# If a YOLO model is available, run inference and use the annotated frame if possible
		if model is not None:
			try:
				results = model(frame)
				try:
					annotated = results[0].plot()
					if annotated is not None:
						frame = annotated
				except Exception:
					pass
			except Exception:
				pass

		# Optionally save frames
		if save_dir:
			try:
				fname = os.path.join(save_dir, f"frame_{frame_counter:06d}.jpg")
				cv2.imwrite(fname, frame)
			except Exception:
				pass

		# Display or headless output
		if gui_forced_off:
			if frame_counter % 30 == 0:
				try:
					h, w = frame.shape[:2]
					print(f"Captured frame (headless mode): {w}x{h}")
				except Exception:
					print("Captured frame (headless mode)")
			time.sleep(0.03)
		else:
			try:
				cv2.imshow("Camera", frame)
				if cv2.waitKey(1) & 0xFF == ord("q"):
					break
			except cv2.error:
				if frame_counter % 30 == 0:
					try:
						h, w = frame.shape[:2]
						print(f"Captured frame (headless): {w}x{h}")
					except Exception:
						print("Captured frame (headless)")
				time.sleep(0.03)

		frame_counter += 1
finally:
	if use_picam:
		try:
			picam2.stop()
		except Exception:
			pass
	else:
		try:
			cap.release()
		except Exception:
			pass

	try:
		cv2.destroyAllWindows()
	except Exception:
		pass
