"""Standalone red scarf monitoring script.

This script combines YOLOv8 person detection with a manual red-color analysis
around the neck region to identify students not wearing a red scarf. It handles
tracking, cooldowns, alerting (sound + voice), snapshot saving, SQLite logging,
CSV export, and can stream annotated frames to a lightweight Flask backend for
browser viewing.

Run with:

    python red_scarf_monitor.py                 # OpenCV window
    python red_scarf_monitor.py --mode web      # MJPEG web stream

Press 'q' to quit, 'e' to export violations to CSV on demand (GUI mode).
"""

from __future__ import annotations

import argparse
import datetime as dt
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

import yaml

from flask import Flask, Response, abort, jsonify, redirect, request, send_from_directory

try:
    import mediapipe as mp  # type: ignore

    MP_POSE = mp.solutions.pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
except Exception:  # pragma: no cover - optional dependency
    MP_POSE = None

try:
    from playsound import playsound  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    playsound = None

try:
    import pyttsx3  # type: ignore

    TTS_ENGINE = pyttsx3.init()
except Exception:  # pragma: no cover - optional dependency
    TTS_ENGINE = None


_LATEST_FRAME: Optional[np.ndarray] = None
_FRAME_LOCK = threading.Lock()
_STATS_LOCK = threading.Lock()
_LATEST_STATS: Dict[str, Any] = {
    "timestamp": None,
    "total": 0,
    "with_scarf": 0,
    "missing": 0,
    "fps": 0.0,
    "monitoring": False,
    "message": "Initializing",
}
_GLOBAL_SETTINGS: Optional["Settings"] = None
_VIOLATION_STORE: Optional["ViolationStore"] = None
_CONFIG_PATH: Optional[Path] = None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class Settings:
    model_path: Path
    camera_index: int
    imgsz: int
    confidence: float
    red_ratio_threshold: float
    cooldown_seconds: float
    tracker_distance: float
    tracker_expiry: float
    school_start_hour: int
    school_end_hour: int
    enable_sound_alert: bool
    enable_voice_alert: bool
    save_violation_images: bool
    snapshot_dir: Path
    db_path: Path
    csv_path: Path
    alarm_file: Optional[Path]
    time_window_enabled: bool


_CONFIG_FIELD_TYPES: Dict[str, type] = {
    "model_path": str,
    "camera_index": int,
    "img_size": int,
    "confidence_threshold": float,
    "red_ratio_threshold": float,
    "cooldown_seconds": float,
    "cooldown_expiry_seconds": float,
    "tracker_distance": float,
    "school_start_hour": int,
    "school_end_hour": int,
    "time_window_enabled": bool,
    "enable_voice_alert": bool,
    "enable_sound_alert": bool,
    "save_violation_images": bool,
    "alarm_file": str,
    "snapshot_dir": str,
    "db_path": str,
    "log_csv_path": str,
}


def load_settings(config_path: Path) -> Settings:
    global _CONFIG_PATH
    _CONFIG_PATH = config_path.resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    project_root = config_path.parent.parent

    def resolve_path(relative: str | None, default: str) -> Path:
        target = Path(relative or default)
        if not target.is_absolute():
            target = (project_root / target).resolve()
        return target

    alarm_file = data.get("alarm_file")
    alarm_path = resolve_path(alarm_file, "config/alarm.mp3") if alarm_file else None

    return Settings(
        model_path=resolve_path(data.get("model_path"), "model/best.pt"),
        camera_index=int(data.get("camera_index", 0)),
        imgsz=int(data.get("img_size", 640)),
        confidence=float(data.get("confidence_threshold", 0.6)),
        red_ratio_threshold=float(data.get("red_ratio_threshold", 0.08)),
        cooldown_seconds=float(data.get("cooldown_seconds", 10)),
        tracker_distance=float(data.get("tracker_distance", 80)),
        tracker_expiry=float(data.get("cooldown_expiry_seconds", 15)),
        school_start_hour=int(data.get("school_start_hour", 6)),
        school_end_hour=int(data.get("school_end_hour", 12)),
        enable_sound_alert=bool(data.get("enable_sound_alert", True)),
        enable_voice_alert=bool(data.get("enable_voice_alert", True)),
        save_violation_images=bool(data.get("save_violation_images", True)),
        snapshot_dir=resolve_path(data.get("snapshot_dir"), "images/violations"),
        db_path=resolve_path(data.get("db_path"), "database/violations.db"),
        csv_path=resolve_path(data.get("log_csv_path"), "logs/violations.csv"),
        alarm_file=alarm_path,
        time_window_enabled=bool(data.get("time_window_enabled", True)),
    )


def _assert_config_path() -> Path:
    if _CONFIG_PATH is None:
        raise RuntimeError("Configuration path is not set")
    return _CONFIG_PATH


def _load_config_dict() -> Dict[str, Any]:
    path = _assert_config_path()
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a mapping")
    return dict(data)


def _write_config_dict(data: Dict[str, Any]) -> None:
    path = _assert_config_path()
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False, allow_unicode=False)


def _coerce_config_value(key: str, value: Any) -> Any:
    expected = _CONFIG_FIELD_TYPES.get(key)
    if expected is None:
        raise KeyError(key)

    if expected is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "on", "y"}:
                return True
            if lowered in {"0", "false", "no", "off", "n"}:
                return False
        raise ValueError(f"Invalid boolean for '{key}'")

    if expected is int:
        if isinstance(value, bool):
            raise ValueError(f"Invalid integer for '{key}'")
        return int(value)

    if expected is float:
        if isinstance(value, bool):
            raise ValueError(f"Invalid float for '{key}'")
        return float(value)

    if expected is str:
        if value is None:
            return ""
        return str(value)

    return value


def _update_config(values: Dict[str, Any]) -> Dict[str, Any]:
    config = _load_config_dict()
    updated: Dict[str, Any] = {}
    for key, raw_value in values.items():
        if key not in _CONFIG_FIELD_TYPES:
            continue
        try:
            coerced = _coerce_config_value(key, raw_value)
        except (ValueError, KeyError) as exc:
            raise ValueError(f"Invalid value for '{key}': {exc}") from exc
        updated[key] = coerced

    if not updated:
        return config

    for key in _CONFIG_FIELD_TYPES:
        if key in updated:
            config[key] = updated[key]

    _write_config_dict(config)
    return config


def _public_config_view() -> Dict[str, Any]:
    config = _load_config_dict()
    view: Dict[str, Any] = {}
    for key in _CONFIG_FIELD_TYPES:
        if key in config:
            view[key] = config[key]
    return view


# ---------------------------------------------------------------------------
# Tracking & cooldown helpers
# ---------------------------------------------------------------------------


@dataclass
class TrackedObject:
    center: Tuple[float, float]
    last_seen: float


class SimpleTracker:
    def __init__(self, max_distance: float, expiry_seconds: float) -> None:
        self._max_distance = max_distance
        self._expiry = expiry_seconds
        self._next_id = 1
        self._objects: Dict[int, TrackedObject] = {}

    def update(self, centers: Iterable[Tuple[float, float]]) -> List[int]:
        centers = list(centers)
        now = time.time()
        assignments: List[int] = []
        used_ids: set[int] = set()

        for cx, cy in centers:
            best_id = None
            best_dist = None
            for object_id, tracked in self._objects.items():
                if object_id in used_ids:
                    continue
                dist = np.hypot(cx - tracked.center[0], cy - tracked.center[1])
                if dist <= self._max_distance and (best_dist is None or dist < best_dist):
                    best_dist = dist
                    best_id = object_id

            if best_id is not None:
                self._objects[best_id] = TrackedObject(center=(cx, cy), last_seen=now)
                used_ids.add(best_id)
                assignments.append(best_id)
            else:
                object_id = self._next_id
                self._next_id += 1
                self._objects[object_id] = TrackedObject(center=(cx, cy), last_seen=now)
                used_ids.add(object_id)
                assignments.append(object_id)

        expired = [oid for oid, tracked in self._objects.items() if now - tracked.last_seen > self._expiry]
        for oid in expired:
            self._objects.pop(oid, None)

        return assignments


class CooldownTracker:
    def __init__(self, cooldown_seconds: float) -> None:
        self._cooldown = cooldown_seconds
        self._last_alert: Dict[int, float] = {}

    def ready(self, track_id: int) -> bool:
        now = time.time()
        last = self._last_alert.get(track_id, 0.0)
        return now - last > self._cooldown

    def mark(self, track_id: int) -> None:
        self._last_alert[track_id] = time.time()


# ---------------------------------------------------------------------------
# Alerts and logging
# ---------------------------------------------------------------------------


class AlertManager:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._voice_lock = threading.Lock()

    def sound(self) -> None:
        if not self._settings.enable_sound_alert or playsound is None:
            return
        alarm = self._settings.alarm_file
        if alarm is None or not alarm.exists():
            return
        threading.Thread(target=playsound, args=(str(alarm),), daemon=True).start()

    def voice(self, text: str) -> None:
        if not self._settings.enable_voice_alert or TTS_ENGINE is None:
            return

        def _speak() -> None:
            with self._voice_lock:
                try:
                    TTS_ENGINE.say(text)
                    TTS_ENGINE.runAndWait()
                except Exception:
                    pass

        threading.Thread(target=_speak, daemon=True).start()


class ViolationStore:
    def __init__(self, settings: Settings) -> None:
        self._db_path = settings.db_path
        self._csv_path = settings.csv_path
        self._snapshot_dir = settings.snapshot_dir
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS violations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    track_id INTEGER,
                    violation TEXT,
                    confidence REAL,
                    image_path TEXT
                )
                """
            )

    def log(self, track_id: int, confidence: float, image_path: Optional[Path]) -> None:
        timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
        stored_path: Optional[str] = None
        if image_path is not None:
            try:
                stored_path = str(image_path.relative_to(self._snapshot_dir))
            except ValueError:
                stored_path = str(image_path)
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "INSERT INTO violations (timestamp, track_id, violation, confidence, image_path) VALUES (?,?,?,?,?)",
                (timestamp, track_id, "NO_SCARF", float(confidence), stored_path),
            )

    def export_csv(self) -> Path:
        rows: List[Tuple[str, int, str, float, Optional[str]]] = []
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute(
                "SELECT timestamp, track_id, violation, confidence, image_path FROM violations ORDER BY id"
            )
            rows = cursor.fetchall()

        import csv

        with self._csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["timestamp", "track_id", "violation", "confidence", "image_path"])
            writer.writerows(rows)

        return self._csv_path

    def fetch_events(
        self,
        *,
        year: Optional[int] = None,
        month: Optional[int] = None,
        day: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, List[int]], int]:
        items: List[Dict[str, Any]] = []
        limit = max(int(limit or 1), 1)
        offset = max(int(offset or 0), 0)

        where_clauses: List[str] = []
        params: List[Any] = []

        start_dt = dt.date.fromisoformat(start_date) if start_date else None
        end_dt = dt.date.fromisoformat(end_date) if end_date else None

        if year is not None:
            where_clauses.append("substr(timestamp,1,4) = ?")
            params.append(f"{int(year):04d}")
        if month is not None:
            where_clauses.append("substr(timestamp,6,2) = ?")
            params.append(f"{int(month):02d}")
        if day is not None:
            where_clauses.append("substr(timestamp,9,2) = ?")
            params.append(f"{int(day):02d}")
        if start_dt is not None:
            where_clauses.append("substr(timestamp,1,10) >= ?")
            params.append(start_dt.isoformat())
        if end_dt is not None:
            where_clauses.append("substr(timestamp,1,10) <= ?")
            params.append(end_dt.isoformat())

        where_sql = f" WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        with sqlite3.connect(self._db_path) as conn:
            total = conn.execute(
                f"SELECT COUNT(*) FROM violations{where_sql}",
                params,
            ).fetchone()[0]

            cursor = conn.execute(
                f"SELECT timestamp, track_id, violation, confidence, image_path FROM violations{where_sql} ORDER BY id DESC LIMIT ? OFFSET ?",
                params + [limit, offset],
            )
            rows = cursor.fetchall()

            year_rows = conn.execute(
                "SELECT DISTINCT substr(timestamp,1,4) FROM violations WHERE timestamp IS NOT NULL ORDER BY 1 DESC"
            ).fetchall()
            years = [int(y[0]) for y in year_rows if y[0] and y[0].isdigit()]

            month_rows = conn.execute(
                "SELECT DISTINCT substr(timestamp,6,2) FROM violations WHERE timestamp IS NOT NULL ORDER BY 1"
            ).fetchall()
            months = [int(m[0]) for m in month_rows if m[0] and m[0].isdigit()]

            day_rows = conn.execute(
                "SELECT DISTINCT substr(timestamp,9,2) FROM violations WHERE timestamp IS NOT NULL ORDER BY 1"
            ).fetchall()
            days = [int(d[0]) for d in day_rows if d[0] and d[0].isdigit()]

        snapshot_root = self._snapshot_dir.resolve()

        for stamp, track_id, violation, confidence, image_path in rows:
            try:
                ts = dt.datetime.fromisoformat(stamp)
            except Exception:
                continue

            if year is not None and ts.year != year:
                continue
            if month is not None and ts.month != month:
                continue
            if day is not None and ts.day != day:
                continue

            if start_dt is not None and ts.date() < start_dt:
                continue
            if end_dt is not None and ts.date() > end_dt:
                continue

            local_ts = ts.astimezone()
            rel_path: Optional[Path] = None
            if image_path:
                candidate = Path(image_path)
                if not candidate.is_absolute():
                    rel_path = candidate
                else:
                    try:
                        rel_path = candidate.relative_to(self._snapshot_dir)
                    except ValueError:
                        rel_path = None

            image_url: Optional[str] = None
            if rel_path is not None:
                full_path = (self._snapshot_dir / rel_path).resolve()
                if full_path.exists() and str(full_path).startswith(str(snapshot_root)):
                    image_url = f"/snapshots/{rel_path.as_posix()}"

            items.append(
                {
                    "timestamp": ts.isoformat(),
                    "local_time": local_ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "track_id": int(track_id),
                    "violation": violation,
                    "confidence": float(confidence),
                    "image_url": image_url,
                }
            )

        filters = {
            "years": years,
            "months": months or list(range(1, 13)),
            "days": days or list(range(1, 32)),
        }

        return items, filters, int(total)


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------


def within_time_window(settings: Settings) -> bool:
    if not settings.time_window_enabled:
        return True

    now = dt.datetime.now().time()
    start = dt.time(hour=settings.school_start_hour)
    end = dt.time(hour=settings.school_end_hour)
    if start <= end:
        return start <= now <= end
    return now >= start or now <= end


def neck_box_from_bbox(bbox: Tuple[int, int, int, int], frame_shape: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    h, w = frame_shape[:2]
    bw = x2 - x1
    bh = y2 - y1
    cx = int(x1 + bw * 0.5)
    cy = int(y1 + bh * 0.18)
    nw = max(int(bw * 0.5), 120)
    nh = max(int(bh * 0.18), 80)
    nx1 = max(cx - nw // 2, 0)
    ny1 = max(cy - nh // 2, 0)
    nx2 = min(cx + nw // 2, w)
    ny2 = min(cy + nh // 2, h)
    return nx1, ny1, nx2 - nx1, ny2 - ny1


def neck_box_from_pose(frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    if MP_POSE is None:
        return None

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = MP_POSE.process(rgb)
    if not results.pose_landmarks:
        return None

    landmarks = results.pose_landmarks.landmark
    h, w = frame.shape[:2]
    try:
        left_sh = landmarks[11]
        right_sh = landmarks[12]
        nose = landmarks[0]
        lx, ly = int(left_sh.x * w), int(left_sh.y * h)
        rx, ry = int(right_sh.x * w), int(right_sh.y * h)
        nx, ny = int(nose.x * w), int(nose.y * h)
        cx = (lx + rx) // 2
        cy = (ly + ry) // 2
        if lx == 0 and rx == 0:
            cx, cy = nx, ny + 40
        nw = 120
        nh = 80
        x1 = max(cx - nw // 2, 0)
        y1 = max(cy - nh // 2, 0)
        x2 = min(cx + nw // 2, w)
        y2 = min(cy + nh // 2, h)
        return x1, y1, x2 - x1, y2 - y1
    except Exception:
        return None


def red_ratio(crop: np.ndarray) -> Tuple[float, Optional[Tuple[int, int, int, int]]]:
    if crop is None or crop.size == 0:
        return 0.0, None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 70, 50], dtype=np.uint8)
    upper_red1 = np.array([10, 255, 255], dtype=np.uint8)
    lower_red2 = np.array([160, 70, 50], dtype=np.uint8)
    upper_red2 = np.array([180, 255, 255], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    total_pixels = mask.size
    if total_pixels == 0:
        return 0.0, None

    red_pixels = cv2.countNonZero(mask)
    ratio = red_pixels / float(total_pixels)

    if red_pixels == 0:
        return ratio, None

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return ratio, None

    largest = max(contours, key=cv2.contourArea)
    rx, ry, rw, rh = cv2.boundingRect(largest)
    return ratio, (rx, ry, rw, rh)


def save_snapshot(settings: Settings, frame: np.ndarray, track_id: int) -> Path:
    settings.snapshot_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = settings.snapshot_dir / f"violation_{timestamp}_id{track_id}.jpg"
    cv2.imwrite(str(filename), frame)
    return filename


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def run_monitor(
    settings: Settings,
    frame_callback: Optional[Callable[[np.ndarray], None]] = None,
    display: bool = True,
    stop_event: Optional[threading.Event] = None,
) -> None:
    if settings.model_path.exists():
        model_source = str(settings.model_path)
        print(f"[INFO] Loading YOLO model from '{model_source}'")
    else:
        model_source = "yolov8n.pt"
        print(
            f"[INFO] Local model not found at {settings.model_path}. Falling back to '{model_source}'"
        )
        print("[INFO] Downloading YOLO weights – first run can take a minute. Please wait...")

    model = YOLO(model_source)
    print("[INFO] YOLO model ready.")
    tracker = SimpleTracker(settings.tracker_distance, settings.tracker_expiry)
    cooldown = CooldownTracker(settings.cooldown_seconds)
    alerts = AlertManager(settings)
    store = ViolationStore(settings)

    global _GLOBAL_SETTINGS, _VIOLATION_STORE
    _GLOBAL_SETTINGS = settings
    _VIOLATION_STORE = store

    cap = cv2.VideoCapture(settings.camera_index)
    if not cap.isOpened():
        raise RuntimeError("Unable to open camera")

    print("Running...")
    if display:
        print("PRESS 'q' to quit, 'e' to export CSV, 's' to save manual snapshot")
    elif frame_callback:
        print("Streaming frames to web clients. Press Ctrl+C to stop.")

    last_violation_text = ""
    fps = 0.0
    last_time = time.time()

    _set_latest_stats(total=0, with_scarf=0, missing=0, fps=0.0, monitoring=False, message="Starting camera...")

    while True:
        if stop_event and stop_event.is_set():
            break

        success, frame = cap.read()
        if not success:
            print("[WARN] Camera read failed. Reconnecting...")
            cap.release()
            time.sleep(1)
            cap = cv2.VideoCapture(settings.camera_index)
            continue

        if not within_time_window(settings):
            cv2.putText(frame, "Outside monitoring window", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            _set_latest_stats(
                total=0,
                with_scarf=0,
                missing=0,
                fps=0.0,
                monitoring=False,
                message="Outside monitoring window",
            )
            if frame_callback:
                frame_callback(frame)
            if display:
                cv2.imshow("Red Scarf Monitor", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                time.sleep(0.05)
            continue

        results = model.predict(frame, imgsz=settings.imgsz, conf=settings.confidence, verbose=False)

        person_boxes: List[Tuple[int, int, int, int]] = []
        centers: List[Tuple[int, int]] = []
        confidences: List[float] = []

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls != 0:  # person class in COCO
                    continue
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                person_boxes.append((x1, y1, x2, y2))
                centers.append(((x1 + x2) // 2, (y1 + y2) // 2))
                confidences.append(conf)

        assigned_ids = tracker.update(centers)

        scarf_count = 0
        violations = 0

        for idx, bbox in enumerate(person_boxes):
            x1, y1, x2, y2 = bbox
            track_id = assigned_ids[idx] if idx < len(assigned_ids) else -1
            confidence = confidences[idx]

            neck = neck_box_from_pose(frame)
            if neck is None:
                neck = neck_box_from_bbox(bbox, frame.shape)

            nx, ny, nw, nh = neck
            crop = frame[ny : ny + nh, nx : nx + nw]
            ratio, red_bbox = red_ratio(crop)
            has_scarf = ratio >= settings.red_ratio_threshold

            person_color = (0, 255, 0)
            scarf_color = (0, 0, 255)
            no_scarf_color = (0, 165, 255)

            status = f"Scarf: {'YES' if has_scarf else 'NO'} {ratio * 100:.1f}% ID:{track_id}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), person_color, 2)

            if has_scarf:
                if red_bbox is not None:
                    rx, ry, rw, rh = red_bbox
                    cv2.rectangle(frame, (nx + rx, ny + ry), (nx + rx + rw, ny + ry + rh), scarf_color, 2)
                else:
                    cv2.rectangle(frame, (nx, ny), (nx + nw, ny + nh), scarf_color, 2)
                status_color = scarf_color
            else:
                cv2.rectangle(frame, (nx, ny), (nx + nw, ny + nh), no_scarf_color, 2)
                status_color = no_scarf_color

            cv2.putText(frame, status, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)

            if has_scarf:
                scarf_count += 1
            else:
                violations += 1
                if track_id >= 0 and cooldown.ready(track_id):
                    cooldown.mark(track_id)
                    snapshot_path = save_snapshot(settings, frame, track_id) if settings.save_violation_images else None
                    store.log(track_id, confidence, snapshot_path)
                    alerts.sound()
                    alerts.voice("Warning, student without red scarf")
                    last_violation_text = f"Violation ID {track_id} @ {dt.datetime.now().strftime('%H:%M:%S')}"

        now = time.time()
        elapsed = now - last_time
        if elapsed > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / elapsed) if fps > 0 else 1.0 / elapsed
        last_time = now

        hud = f"Total: {len(person_boxes)}  With Scarf: {scarf_count}  Missing: {violations}  FPS: {fps:.1f}"
        cv2.putText(frame, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2)
        if last_violation_text:
            cv2.putText(frame, last_violation_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 2)

        _set_latest_stats(
            total=len(person_boxes),
            with_scarf=scarf_count,
            missing=violations,
            fps=fps,
            monitoring=True,
            message=last_violation_text or "Monitoring...",
        )

        if frame_callback:
            frame_callback(frame)

        if display:
            cv2.imshow("Red Scarf Monitor", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("e"):
                csv_path = store.export_csv()
                print(f"[INFO] Exported CSV to {csv_path}")
            if key == ord("s"):
                manual = save_snapshot(settings, frame, track_id=-1)
                print(f"[INFO] Manual snapshot saved to {manual}")
        else:
            time.sleep(0.001)

    cap.release()
    if display:
        cv2.destroyAllWindows()


def _update_latest_frame(frame: np.ndarray) -> None:
    global _LATEST_FRAME
    with _FRAME_LOCK:
        _LATEST_FRAME = frame.copy()


def _set_latest_stats(
    *,
    total: int,
    with_scarf: int,
    missing: int,
    fps: float,
    monitoring: bool,
    message: str,
) -> None:
    with _STATS_LOCK:
        _LATEST_STATS.update(
            {
                "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
                "total": int(total),
                "with_scarf": int(with_scarf),
                "missing": int(missing),
                "fps": round(float(fps), 2),
                "monitoring": monitoring,
                "message": message,
            }
        )


def _generate_mjpeg_frames() -> Iterable[bytes]:
    jpeg_params = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
    while True:
        with _FRAME_LOCK:
            frame = None if _LATEST_FRAME is None else _LATEST_FRAME.copy()
        if frame is None:
            time.sleep(0.05)
            continue
        success, buffer = cv2.imencode(".jpg", frame, jpeg_params)
        if not success:
            time.sleep(0.01)
            continue
        chunk = buffer.tobytes()
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + chunk + b"\r\n"
        time.sleep(0.03)


def run_web_monitor(settings: Settings, host: str, port: int) -> None:
    stop_event = threading.Event()

    monitor_thread = threading.Thread(
        target=run_monitor,
        args=(settings,),
        kwargs={"frame_callback": _update_latest_frame, "display": False, "stop_event": stop_event},
        daemon=True,
    )
    monitor_thread.start()

    app = Flask(__name__, static_folder=str(Path(__file__).parent), static_url_path="")

    @app.route("/")
    def root() -> Response:
        return redirect("/webcam.html")

    @app.route("/webcam.html")
    def webcam_page() -> Response:
        return send_from_directory(app.static_folder, "webcam.html")

    @app.route("/captures.html")
    def captures_page() -> Response:
        return send_from_directory(app.static_folder, "captures.html")

    @app.route("/settings.html")
    def settings_page() -> Response:
        return send_from_directory(app.static_folder, "settings.html")

    @app.route("/video_feed")
    def video_feed() -> Response:
        return Response(_generate_mjpeg_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/api/status")
    def api_status() -> Response:
        with _STATS_LOCK:
            payload = dict(_LATEST_STATS)
        payload.setdefault("timestamp", dt.datetime.now(dt.timezone.utc).isoformat())
        return jsonify(payload)

    @app.route("/api/violations")
    def api_violations() -> Response:
        if _VIOLATION_STORE is None:
            return jsonify({"items": [], "filters": {"years": [], "months": list(range(1, 13)), "days": list(range(1, 32))}})

        def parse_param(name: str) -> Optional[int]:
            raw = request.args.get(name)
            if raw is None or raw.lower() == "all":
                return None
            try:
                value = int(raw)
            except ValueError:  # pragma: no cover - defensive
                abort(400, description=f"Invalid {name} parameter")
            return value

        year = parse_param("year")
        month = parse_param("month")
        day = parse_param("day")

        def parse_date(name: str) -> Optional[str]:
            raw = request.args.get(name)
            if not raw:
                return None
            try:
                dt.date.fromisoformat(raw)
            except ValueError:
                abort(400, description=f"Invalid {name} parameter")
            return raw

        start_date = parse_date("start")
        end_date = parse_date("end")

        per_page_param = request.args.get("per_page") or request.args.get("limit")
        try:
            per_page = int(per_page_param) if per_page_param else 50
        except ValueError:
            abort(400, description="Invalid per_page parameter")
        per_page = max(1, min(per_page, 500))

        page = request.args.get("page", default=1, type=int)
        if page is None or page < 1:
            page = 1

        offset = (page - 1) * per_page

        items, filters, total = _VIOLATION_STORE.fetch_events(
            year=year,
            month=month,
            day=day,
            start_date=start_date,
            end_date=end_date,
            limit=per_page,
            offset=offset,
        )

        total_pages = (total + per_page - 1) // per_page if total else 0

        pagination = {
            "page": page,
            "per_page": per_page,
            "total": total,
            "total_pages": total_pages,
            "has_prev": page > 1 and total > 0,
            "has_next": page < total_pages,
        }

        return jsonify({"items": items, "filters": filters, "pagination": pagination})

    @app.route("/api/settings", methods=["GET", "POST"])
    def api_settings() -> Response:
        try:
            view = _public_config_view()
        except Exception as exc:  # pragma: no cover - defensive
            abort(500, description=str(exc))

        if request.method == "GET":
            return jsonify({"config": view})

        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            abort(400, description="Expected JSON object")

        try:
            updated = _update_config(payload)
        except ValueError as exc:
            abort(400, description=str(exc))
        except Exception as exc:  # pragma: no cover - defensive
            abort(500, description=str(exc))

        return jsonify(
            {
                "config": {k: updated.get(k) for k in _CONFIG_FIELD_TYPES},
                "message": "Settings saved. Restart the monitor to apply changes.",
            }
        )

    @app.route("/snapshots/<path:filename>")
    def snapshots(filename: str) -> Response:
        if _GLOBAL_SETTINGS is None:
            abort(404)
        snapshot_dir = _GLOBAL_SETTINGS.snapshot_dir.resolve()
        target = (snapshot_dir / Path(filename)).resolve()
        if not str(target).startswith(str(snapshot_dir)) or not target.exists():
            abort(404)
        relative = target.relative_to(snapshot_dir)
        return send_from_directory(str(snapshot_dir), relative.as_posix())

    try:
        app.run(host=host, port=port, debug=False, threaded=True, use_reloader=False)
    finally:
        stop_event.set()
        monitor_thread.join(timeout=5)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone red scarf monitor")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/config.yaml"),
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--mode",
        choices=("gui", "web"),
        default="gui",
        help="Run in GUI mode (OpenCV window) or web streaming mode",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host interface for web mode",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for web mode",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    settings = load_settings(args.config)
    try:
        if args.mode == "web":
            run_web_monitor(settings, host=args.host, port=args.port)
        else:
            run_monitor(settings)
    except KeyboardInterrupt:
        print("Interrupted by user")



