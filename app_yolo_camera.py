from picamera2 import Picamera2
from ultralytics import YOLO
import cv2
import time

def main():
    model = YOLO("yolov8n.pt")
    print("YOLO model loaded")

    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"size": (640, 480)},
        buffer_count=3
    )
    picam2.configure(config)
    picam2.start()
    print("Camera started")

    time.sleep(1)

    while True:
        frame = picam2.capture_array()   # RGB from Picamera2

        # Ensure no alpha channel
        if frame.shape[2] == 4:
            frame = frame[:, :, :3]

        # FIX COLOR: convert RGB -> BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Run YOLO on BGR frame
        results = model(frame_bgr, verbose=False)[0]

        # Render YOLO bounding boxes
        annotated = results.plot()

        # Display correctly colored frame
        cv2.imshow("YOLOv8 Detection", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Print detections
        for box in results.boxes:
            cls = model.names[int(box.cls)]
            conf = float(box.conf)
            print(f"Detected: {cls} ({conf:.2f})")

    picam2.stop()
    cv2.destroyAllWindows()
    print("Camera stopped")


if __name__ == "__main__":
    main()
