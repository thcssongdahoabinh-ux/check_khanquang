## Attendance-Enabled Red Scarf Monitor

This project pairs YOLOv8-based scarf detection with a lightweight attendance
workflow so schools can supervise uniform compliance and presence from the same
camera feed.

### Key Features

- **Live monitoring** – `app.py` continuously detects students and
  highlights missing red scarves while streaming frames to the web dashboard.
- **One-click capture control** – Start or stop violation snapshot capture
  directly from the Red Scarf Monitor page without restarting the service.
- **Student management** – The new `admin.html` screen lets administrators add
  students, capture reference face samples straight from the running camera, and
  review stored samples.
- **Face recognition & attendance** – Captured samples train an on-device
  LBPH recogniser. Recognised students are overlaid on the stream and logged as
  alternating check-in / check-out events in SQLite (`attendance_logs` table).
- **REST API** – `/api/students`, `/api/students/<id>/capture`, and
  `/api/attendance/logs` back the admin UI and can be integrated with external
  systems if needed.

### How It Works

1. A single camera feed is processed by YOLOv8 to spot students and detect
   whether the required red scarf is present.
2. In parallel, face crops from the same frame are passed to an LBPH-based
   recogniser that matches them against stored student samples.
3. The web dashboard overlays each detection with both scarf status and
   recognised student identity, then:
   - saves any scarf violation frames and registers them under the `violations`
     table;
   - alternates between check-in and check-out attendance events per recognised
     student, enforcing configured cooldowns to avoid duplicate entries.

### Running the Monitor

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Launch the monitor in web mode to expose the dashboard and admin console:

   ```bash
   python app.py --mode web
   ```

3. Open `http://localhost:8000/webcam.html` for the live monitor or
   `http://localhost:8000/admin.html` to manage students and attendance.

### Configuration

Editable settings live in `config/config.yaml`. New options include:

- `attendance_enabled` – master switch for the attendance workflow.
- `red_min_pixels` – minimum absolute red pixel count (after masking) to treat a scarf as present, helping close-range detections.
- `student_samples_dir` – storage root for captured face samples.
- `attendance_confidence_threshold` – minimum confidence (0–1) required to log
  attendance after recognition.
- `attendance_cooldown_seconds` – cooldown before a student can be logged again.
- `face_distance_threshold` – LBPH distance upper bound before a prediction is
  rejected.

Use the Settings page (`settings.html`) to update values via the browser; changes
persist to disk and apply after the monitor restarts.

### Data Storage

The SQLite database defined by `db_path` now contains:

- `students` – enrolled students plus `last_seen_at`.
- `student_samples` – metadata for captured sample images.
- `attendance_logs` – timestamped check-in/check-out entries.
- `violations` – existing red scarf violations table.

Sample images are saved under `images/students/` (configurable) and served through
`/student_samples/<path>`.

### Notes & Next Steps

- Accurate recognition depends on high-quality, well-lit face samples. Gather at
  least a couple of captures per student for best results.
- The recogniser retrains automatically after each new sample. For large cohorts
  consider moving training off the main thread or caching encodings.
- Extend `/api/attendance/logs` or add export endpoints if the attendance data
  needs to integrate with external systems.


