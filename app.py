"""Standalone red scarf monitoring script.

This script combines YOLOv8 person detection with a manual red-color analysis
around the neck region to identify students not wearing a red scarf. It handles
tracking, cooldowns, alerting (sound + voice), snapshot saving, SQLite logging,
CSV export, and can stream annotated frames to a lightweight Flask backend for
browser viewing.

Run with:

    python app.py                 # OpenCV window
    python app.py --mode web      # MJPEG web stream

Press 'q' to quit, 'e' to export violations to CSV on demand (GUI mode).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from camera_backend import init_videocapture

import yaml

from flask import Flask, Response, abort, jsonify, redirect, request, send_from_directory, session, url_for

from attendance import AttendanceRecord, AttendanceStore, RecognitionResult, Student, StudentRecognizer
from email_sender import EmailSender
from email_scheduler import EmailScheduler
from email_logger import EmailLogStore

_HAAR_FACE_CASCADE = None

try:
    import mediapipe as mp  # type: ignore

    MP_POSE = mp.solutions.pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    MP_FACE = mp.solutions.face_detection.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.6,
    )
except Exception:  # pragma: no cover - optional dependency
    MP_POSE = None
    MP_FACE = None

try:
    cascade_root = getattr(cv2.data, "haarcascades", "")
    if cascade_root:
        cascade_path = Path(cascade_root) / "haarcascade_frontalface_default.xml"
        if cascade_path.exists():
            candidate = cv2.CascadeClassifier(str(cascade_path))
            if not candidate.empty():
                _HAAR_FACE_CASCADE = candidate
except Exception:  # pragma: no cover - optional dependency
    _HAAR_FACE_CASCADE = None

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
_LATEST_RAW_FRAME: Optional[np.ndarray] = None
_FRAME_LOCK = threading.Lock()
_STATS_LOCK = threading.Lock()
_LATEST_STATS: Dict[str, Any] = {
    "timestamp": None,
    "total": 0,
    "with_scarf": 0,
    "missing": 0,
    "fps": 0.0,
    "monitoring": False,
    "recognized": 0,
    "capture_active": True,
    "message": "Initializing",
}
_GLOBAL_SETTINGS: Optional["Settings"] = None
_VIOLATION_STORE: Optional["ViolationStore"] = None
_ATTENDANCE_STORE: Optional[AttendanceStore] = None
_STUDENT_RECOGNIZER: Optional[StudentRecognizer] = None
_ATTENDANCE_MANAGER: Optional["AttendanceManager"] = None
_EMAIL_SENDER: Optional[EmailSender] = None
_EMAIL_SCHEDULER: Optional[EmailScheduler] = None
_EMAIL_LOG_STORE: Optional[EmailLogStore] = None
_CONFIG_PATH: Optional[Path] = None
_FACE_LOCK = threading.Lock()
_CAPTURE_LOCK = threading.Lock()
_CAPTURE_ACTIVE = True
_SOUND_LOCK = threading.Lock()
_SOUND_ENABLED = True


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
    red_min_pixels: int
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
    attendance_enabled: bool
    student_samples_dir: Path
    attendance_confidence_threshold: float
    attendance_cooldown_seconds: float
    recognition_min_samples: int
    face_distance_threshold: float
    admin_password: str
    email_enabled: bool
    email_sender: str
    email_password: str
    email_receiver: str
    email_schedule_hour: int
    email_schedule_minute: int


_CONFIG_FIELD_TYPES: Dict[str, type] = {
    "model_path": str,
    "camera_index": int,
    "img_size": int,
    "confidence_threshold": float,
    "red_ratio_threshold": float,
    "red_min_pixels": int,
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
    "attendance_enabled": bool,
    "attendance_confidence_threshold": float,
    "attendance_cooldown_seconds": float,
    "student_samples_dir": str,
    "recognition_min_samples": int,
    "face_distance_threshold": float,
    "email_enabled": bool,
    "email_sender": str,
    "email_password": str,
    "email_receiver": str,
    "email_schedule_hour": int,
    "email_schedule_minute": int,
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
    if isinstance(alarm_file, str) and not alarm_file.strip():
        alarm_path = None
    else:
        alarm_path = resolve_path(alarm_file, "assets/no_scarf.mp3")

    return Settings(
        model_path=resolve_path(data.get("model_path"), "model/best.pt"),
        camera_index=int(data.get("camera_index", 0)),
        imgsz=int(data.get("img_size", 640)),
        confidence=float(data.get("confidence_threshold", 0.6)),
        red_ratio_threshold=float(data.get("red_ratio_threshold", 0.08)),
        red_min_pixels=int(data.get("red_min_pixels", 1500)),
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
        attendance_enabled=bool(data.get("attendance_enabled", True)),
        student_samples_dir=resolve_path(data.get("student_samples_dir"), "images/students"),
        attendance_confidence_threshold=float(data.get("attendance_confidence_threshold", 0.45)),
        attendance_cooldown_seconds=float(data.get("attendance_cooldown_seconds", 45)),
        recognition_min_samples=int(data.get("recognition_min_samples", 1)),
        face_distance_threshold=float(data.get("face_distance_threshold", 70.0)),
        admin_password=str(data.get("admin_password", "changeme123")),
        email_enabled=bool(data.get("email_enabled", False)),
        email_sender=str(data.get("email_sender", "")),
        email_password=str(data.get("email_password", "")),
        email_receiver=str(data.get("email_receiver", "")),
        email_schedule_hour=int(data.get("email_schedule_hour", 18)),
        email_schedule_minute=int(data.get("email_schedule_minute", 0)),
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


def _require_admin_api() -> None:
    if not session.get("admin_authenticated"):
        abort(401, description="Admin authentication required")


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


class AttendanceManager:
    def __init__(self, store: AttendanceStore, cooldown_seconds: float, confidence_threshold: float) -> None:
        self._store = store
        self._cooldown = cooldown_seconds
        self._confidence_threshold = confidence_threshold
        self._lock = threading.Lock()
        self._last_logged: Dict[int, float] = {}

    def mark_attendance(self, recognition: RecognitionResult) -> Optional[AttendanceRecord]:
        if recognition.confidence < self._confidence_threshold:
            return None

        now = time.time()
        with self._lock:
            last = self._last_logged.get(recognition.student_id, 0.0)
            if now - last < self._cooldown:
                return None
            record = self._store.record_attendance(recognition.student_id, confidence=recognition.confidence)
            self._last_logged[recognition.student_id] = now
        return record

    @property
    def confidence_threshold(self) -> float:
        return self._confidence_threshold


# ---------------------------------------------------------------------------
# Alerts and logging
# ---------------------------------------------------------------------------


class AlertManager:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._voice_lock = threading.Lock()

    def sound(self) -> None:
        if not _get_sound_enabled() or playsound is None:
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
    bw = max(x2 - x1, 1)
    bh = max(y2 - y1, 1)

    cx = x1 + bw // 2
    # Keep the neck crop near the upper torso but not too tall.
    cy = y1 + int(bh * 0.28)

    neck_width = int(np.clip(bw * 0.45, 80, 220))
    neck_height = int(np.clip(bh * 0.30, 60, 160))

    nx1 = max(cx - neck_width // 2, 0)
    ny1 = max(cy - neck_height // 2, 0)
    nx2 = min(nx1 + neck_width, w)
    ny2 = min(ny1 + neck_height, h)

    # If the neck box was clamped by the frame bounds, adjust the origin to preserve size.
    nx1 = max(nx2 - neck_width, 0)
    ny1 = max(ny2 - neck_height, 0)

    return nx1, ny1, nx2 - nx1, ny2 - ny1


def neck_box_from_pose(
    frame: np.ndarray,
    person_bbox: Optional[Tuple[int, int, int, int]] = None,
) -> Optional[Tuple[int, int, int, int]]:
    if MP_POSE is None:
        return None

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = MP_POSE.process(rgb)
    if not results.pose_landmarks:
        return None

    landmarks = results.pose_landmarks.landmark
    if len(landmarks) <= 12:
        return None

    h, w = frame.shape[:2]

    def _to_px(index: int) -> Optional[Tuple[int, int, float]]:
        if index >= len(landmarks):
            return None
        landmark = landmarks[index]
        visibility = getattr(landmark, "visibility", 0.0)
        if visibility < 0.2:
            return None
        x = int(np.clip(landmark.x * w, 0, max(w - 1, 0)))
        y = int(np.clip(landmark.y * h, 0, max(h - 1, 0)))
        return x, y, visibility

    left = _to_px(11)
    right = _to_px(12)
    nose = _to_px(0)

    if left is None or right is None or nose is None:
        return None

    lx, ly, _ = left
    rx, ry, _ = right
    nx, ny, _ = nose

    shoulder_dist = float(np.hypot(rx - lx, ry - ly))
    if shoulder_dist < 20:
        return None

    cx = int((lx + rx) * 0.5)
    shoulder_y = int((ly + ry) * 0.5)
    neck_center_y = int(ny + 0.6 * (shoulder_y - ny))

    neck_width = int(np.clip(shoulder_dist * 1.05, 80, 240))
    neck_height = int(np.clip(shoulder_dist * 0.9, 60, 200))

    nx1 = max(cx - neck_width // 2, 0)
    ny1 = max(neck_center_y - neck_height // 2, 0)
    nx2 = min(cx + neck_width // 2, w)
    ny2 = min(neck_center_y + neck_height // 2, h)

    if person_bbox is not None:
        px1, py1, px2, py2 = person_bbox
        nx1 = max(nx1, px1)
        ny1 = max(ny1, py1)
        nx2 = min(nx2, px2)
        ny2 = min(ny2, py2)
        if nx2 <= nx1 or ny2 <= ny1:
            return None

    if nx2 <= nx1 or ny2 <= ny1:
        return None

    return nx1, ny1, nx2 - nx1, ny2 - ny1


def mask_neck_upper_band(crop: Optional[np.ndarray], ratio: float = 0.3) -> Optional[np.ndarray]:
    if crop is None or crop.size == 0:
        return crop
    h = crop.shape[0]
    guard = int(max(0, min(h * ratio, h - 1)))
    if guard <= 0:
        return crop
    guarded = crop.copy()
    guarded[:guard, :, :] = 0
    return guarded


def detect_faces(frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
    faces: List[Tuple[int, int, int, int, float]] = []

    if MP_FACE is not None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with _FACE_LOCK:
            results = MP_FACE.process(rgb)
        if results and results.detections:
            h, w = frame.shape[:2]
            for detection in results.detections:
                location = detection.location_data
                if not location or not location.HasField("relative_bounding_box"):
                    continue
                bbox = location.relative_bounding_box
                xmin = max(0.0, bbox.xmin)
                ymin = max(0.0, bbox.ymin)
                width = min(1.0 - xmin, bbox.width)
                height = min(1.0 - ymin, bbox.height)
                if width <= 0 or height <= 0:
                    continue
                x1 = int(xmin * w)
                y1 = int(ymin * h)
                x2 = int((xmin + width) * w)
                y2 = int((ymin + height) * h)
                confidence = float(detection.score[0]) if detection.score else 0.0
                faces.append((x1, y1, x2, y2, confidence))
            if faces:
                return faces

    if _HAAR_FACE_CASCADE is not None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        try:
            detections = _HAAR_FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        except cv2.error:
            detections = ()
        for (x, y, w, h) in detections:
            faces.append((int(x), int(y), int(x + w), int(y + h), 0.85))

    return faces


def crop_face(frame: np.ndarray, bbox: Tuple[int, int, int, int], padding: float = 0.15) -> Optional[np.ndarray]:
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    bw = x2 - x1
    bh = y2 - y1
    pad_x = int(bw * padding)
    pad_y = int(bh * padding)
    nx1 = max(x1 - pad_x, 0)
    ny1 = max(y1 - pad_y, 0)
    nx2 = min(x2 + pad_x, w)
    ny2 = min(y2 + pad_y, h)
    if nx1 >= nx2 or ny1 >= ny2:
        return None
    return frame[ny1:ny2, nx1:nx2]


def face_for_person(frame: np.ndarray, person_bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    faces = detect_faces(frame)
    if not faces:
        return None
    x1, y1, x2, y2 = person_bbox
    best_face: Optional[Tuple[int, int, int, int, float]] = None
    best_score = -1.0
    for fx1, fy1, fx2, fy2, score in faces:
        cx = (fx1 + fx2) // 2
        cy = (fy1 + fy2) // 2
        if not (x1 <= cx <= x2 and y1 <= cy <= y2):
            continue
        if score > best_score:
            best_face = (fx1, fy1, fx2, fy2, score)
            best_score = score
    if best_face is None:
        # fall back to highest confidence face overall
        best_face = max(faces, key=lambda item: item[4])
    cropped = crop_face(frame, best_face[:4])
    return cropped

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


def red_ratio(crop: np.ndarray) -> Tuple[float, int, Optional[Tuple[int, int, int, int]]]:
    if crop is None or crop.size == 0:
        return 0.0, 0, None

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
        return 0.0, 0, None

    red_pixels = cv2.countNonZero(mask)
    ratio = red_pixels / float(total_pixels)

    if red_pixels == 0:
        return ratio, red_pixels, None

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return ratio, red_pixels, None

    largest = max(contours, key=cv2.contourArea)
    rx, ry, rw, rh = cv2.boundingRect(largest)
    return ratio, red_pixels, (rx, ry, rw, rh)


def save_snapshot(settings: Settings, frame: np.ndarray, track_id: int) -> Path:
    settings.snapshot_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = settings.snapshot_dir / f"violation_{timestamp}_id{track_id}.jpg"
    cv2.imwrite(str(filename), frame)
    cleanup_old_violation_images(settings.snapshot_dir, max_images=1000)
    return filename


def cleanup_old_violation_images(snapshot_dir: Path, max_images: int = 1000) -> None:
    """Keep only the last max_images violation images, delete older ones.
    
    Args:
        snapshot_dir: Directory containing violation images
        max_images: Maximum number of images to keep (default: 1000)
    """
    if not snapshot_dir.exists():
        return
    
    # Get all violation image files
    image_files = list(snapshot_dir.glob("violation_*.jpg"))
    
    if len(image_files) <= max_images:
        return
    
    # Sort by modification time (newest first)
    image_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    
    # Delete older images beyond the limit
    images_to_delete = image_files[max_images:]
    deleted_count = 0
    for img_file in images_to_delete:
        try:
            img_file.unlink()
            deleted_count += 1
        except OSError as e:
            print(f"[WARNING] Failed to delete {img_file}: {e}")
    
    if deleted_count > 0:
        print(f"[INFO] Cleaned up {deleted_count} old violation images, keeping {max_images} newest")


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
    
    # Clean up old violation images on startup
    cleanup_old_violation_images(settings.snapshot_dir, max_images=1000)
    
    tracker = SimpleTracker(settings.tracker_distance, settings.tracker_expiry)
    cooldown = CooldownTracker(settings.cooldown_seconds)
    alerts = AlertManager(settings)
    store = ViolationStore(settings)
    attendance_store: Optional[AttendanceStore] = None
    recognizer: Optional[StudentRecognizer] = None
    attendance_manager: Optional[AttendanceManager] = None

    if settings.attendance_enabled:
        attendance_store = AttendanceStore(settings.db_path, settings.student_samples_dir)
        recognizer = StudentRecognizer(
            attendance_store,
            face_size=(200, 200),
            distance_threshold=settings.face_distance_threshold,
            min_samples=settings.recognition_min_samples,
        )
        recognizer.rebuild()
        attendance_manager = AttendanceManager(
            attendance_store,
            cooldown_seconds=settings.attendance_cooldown_seconds,
            confidence_threshold=settings.attendance_confidence_threshold,
        )

    global _GLOBAL_SETTINGS, _VIOLATION_STORE, _ATTENDANCE_STORE, _STUDENT_RECOGNIZER, _ATTENDANCE_MANAGER
    _GLOBAL_SETTINGS = settings
    _VIOLATION_STORE = store
    _ATTENDANCE_STORE = attendance_store
    _STUDENT_RECOGNIZER = recognizer
    _ATTENDANCE_MANAGER = attendance_manager
    _set_capture_active(settings.save_violation_images)
    _set_sound_enabled(settings.enable_sound_alert)

    # Always use OpenCV VideoCapture backend only
    print(f"[DEBUG] Attempting to open camera with index {settings.camera_index}...")
    cap = init_videocapture(index=settings.camera_index, try_alternate=False)
    if cap is None:
        import os
        cam_devices = [d for d in os.listdir('/dev') if d.startswith('video')]
        print(f"[ERROR] Unable to open camera (VideoCapture) at index {settings.camera_index}.")
        print(f"[DEBUG] /dev/video* devices found: {cam_devices}")
        raise RuntimeError(f"Unable to open camera (VideoCapture) at index {settings.camera_index}. Devices: {cam_devices}")
    print(f"[INFO] Using OpenCV VideoCapture backend (index {settings.camera_index})")

    print("Running...")
    if display:
        print("PRESS 'q' to quit, 'e' to export CSV, 's' to save manual snapshot")
    elif frame_callback:
        print("Streaming frames to web clients. Press Ctrl+C to stop.")

    last_violation_text = ""
    last_attendance_text = ""
    fps = 0.0
    last_time = time.time()
    # Track when attendance threshold was reached for each track_id (to maintain blue color for 1 second)
    attendance_threshold_timestamps: Dict[int, float] = {}

    _set_latest_stats(
        total=0,
        with_scarf=0,
        missing=0,
        recognized=0,
        fps=0.0,
        capture_active=_get_capture_active(),
        monitoring=False,
        message="Starting camera...",
    )

    while True:
        if stop_event and stop_event.is_set():
            break

        # Capture frame from OpenCV VideoCapture only
        frame = None
        try:
            success, frame = cap.read()
            if not success or frame is None:
                print(f"[ERROR] cap.read() failed (success={success}, frame={type(frame)})")
                frame = None
        except Exception as e:
            print(f"[WARN] Camera capture exception: {e}")
            frame = None

        if frame is None:
            print("[WARN] Camera read failed. Reconnecting VideoCapture...")
            try:
                cap.release()
            except Exception:
                pass
            time.sleep(1)
            cap = init_videocapture(index=settings.camera_index, try_alternate=False)
            if cap is None:
                print("[ERROR] Reopening VideoCapture failed; exiting")
                break
            continue

        raw_frame = frame.copy()
        _update_latest_raw_frame(raw_frame)

        if not within_time_window(settings):
            cv2.putText(frame, "Outside monitoring window", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            _set_latest_stats(
                total=0,
                with_scarf=0,
                missing=0,
                recognized=0,
                fps=0.0,
                capture_active=_get_capture_active(),
                monitoring=False,
                message="Outside monitoring window",
            )
            if frame_callback:
                frame_callback(frame)
            if display:
                cv2.imshow("Student Monitor", frame)
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
        recognized_count = 0

        capture_active = _get_capture_active()

        for idx, bbox in enumerate(person_boxes):
            x1, y1, x2, y2 = bbox
            track_id = assigned_ids[idx] if idx < len(assigned_ids) else -1
            confidence = confidences[idx]

            neck = neck_box_from_pose(frame, bbox)
            if neck is None:
                neck = neck_box_from_bbox(bbox, frame.shape)

            nx, ny, nw, nh = neck
            crop = frame[ny : ny + nh, nx : nx + nw]
            crop = mask_neck_upper_band(crop)
            ratio, red_pixels, red_bbox = red_ratio(crop)
            has_scarf = ratio >= settings.red_ratio_threshold or red_pixels >= settings.red_min_pixels

            person_color = (70, 135, 240)
            scarf_color = (0, 255, 0)  # Green for "Scarf: YES"
            no_scarf_color = (0, 0, 255)  # Red for "Scarf: NO"
            highlight_color = (72, 201, 91)
            attendance_threshold_color = (255, 0, 0)  # Blue for attendance threshold reached
            recognized_highlight = False
            attendance_threshold_reached = False
            name_label: Optional[str] = None
            label_box: Optional[Tuple[int, int, int, int]] = None
            
            # Check if attendance threshold was reached for this track_id within the last 1 second
            current_time = time.time()
            if track_id >= 0 and track_id in attendance_threshold_timestamps:
                time_since_threshold = current_time - attendance_threshold_timestamps[track_id]
                if time_since_threshold < 1.0:  # Maintain blue for at least 1 second
                    attendance_threshold_reached = True
                else:
                    # Remove old entries (older than 1 second)
                    del attendance_threshold_timestamps[track_id]

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

            if has_scarf:
                scarf_count += 1
            else:
                violations += 1
                violation_ready = track_id >= 0 and cooldown.ready(track_id)
                if violation_ready:
                    cooldown.mark(track_id)
                    snapshot_path: Optional[Path] = None
                    if settings.save_violation_images and capture_active:
                        snapshot_path = save_snapshot(settings, frame, track_id)
                    store.log(track_id, confidence, snapshot_path)
                    alerts.sound()
                    alerts.voice("Warning, student without red scarf")
                    last_violation_text = f"Violation ID {track_id} @ {dt.datetime.now().strftime('%H:%M:%S')}"

            if attendance_store and recognizer and recognizer.is_ready:
                face_crop = face_for_person(frame, bbox)
                if face_crop is not None:
                    recognition = recognizer.predict(face_crop)
                    if recognition is not None:
                        student = attendance_store.get_student(recognition.student_id)
                        if student is not None:
                            recognized_count += 1
                            recognized_highlight = True
                            name_label = f"{student.name} {recognition.confidence * 100:.0f}%"
                            (text_width, text_height), baseline = cv2.getTextSize(
                                name_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                            )
                            label_left = max(x1, 0)
                            label_bottom = max(y1 - 6, text_height + baseline + 6)
                            label_top = max(label_bottom - text_height - baseline - 6, 0)
                            label_right = min(label_left + text_width + 12, frame.shape[1] - 1)
                            label_box = (label_left, label_top, label_right, label_bottom)
                            if attendance_manager:
                                record = attendance_manager.mark_attendance(recognition)
                                if record:
                                    attendance_threshold_reached = True
                                    # Store timestamp when threshold is reached for this track_id
                                    if track_id >= 0:
                                        attendance_threshold_timestamps[track_id] = current_time
                                    event_label = record.event_type.replace("_", " ").title()
                                    last_attendance_text = (
                                        f"{student.name} {event_label} @ {dt.datetime.now().strftime('%H:%M:%S')}"
                                    )
                                else:
                                    threshold = attendance_manager.confidence_threshold
                                    if recognition.confidence < threshold:
                                        last_attendance_text = (
                                            f"{student.name} below attendance threshold "
                                            f"({recognition.confidence * 100:.0f}% < {threshold * 100:.0f}%)"
                                        )
                                    elif not last_attendance_text:
                                        last_attendance_text = (
                                            f"{student.name} seen recently (cooldown active)"
                                        )

            # Use blue if attendance threshold reached, otherwise use green highlight or status color
            if attendance_threshold_reached:
                status_color = attendance_threshold_color
                highlight_color = attendance_threshold_color
            elif recognized_highlight:
                status_color = highlight_color
            cv2.putText(frame, status, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)

            if recognized_highlight:
                current_highlight_color = attendance_threshold_color if attendance_threshold_reached else highlight_color
                cv2.rectangle(frame, (x1, y1), (x2, y2), current_highlight_color, 3)
                if label_box is not None and name_label is not None:
                    lx1, ly1, lx2, ly2 = label_box
                    cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), current_highlight_color, cv2.FILLED)
                    # Use white text for blue background, dark text for green background
                    text_color = (255, 255, 255) if attendance_threshold_reached else (12, 30, 12)
                    cv2.putText(
                        frame,
                        name_label,
                        (lx1 + 6, ly2 - baseline - 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        text_color,
                        2,
                    )

        now = time.time()
        elapsed = now - last_time
        if elapsed > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / elapsed) if fps > 0 else 1.0 / elapsed
        last_time = now

        hud = (
            f"Total: {len(person_boxes)}  With Scarf: {scarf_count}  Missing: {violations}  "
            f"Recognized: {recognized_count}  FPS: {fps:.1f}"
        )
        cv2.putText(frame, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2)
        if last_violation_text:
            cv2.putText(frame, last_violation_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 2)
        if last_attendance_text:
            cv2.putText(frame, last_attendance_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (152, 251, 152), 2)

        _set_latest_stats(
            total=len(person_boxes),
            with_scarf=scarf_count,
            missing=violations,
            recognized=recognized_count,
            fps=fps,
            capture_active=capture_active,
            monitoring=True,
            message=last_attendance_text or last_violation_text or "Monitoring...",
        )

        if frame_callback:
            frame_callback(frame)

        if display:
            cv2.imshow("Student Monitor", frame)

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

    # Cleanup camera backends
    try:
        if cap is not None:
            cap.release()
    except Exception:
        pass
    try:
        if picam2 is not None:
            picam2.stop()
    except Exception:
        pass
    try:
        if rpicam_cam is not None:
            # adapter exposes stop/release
            if hasattr(rpicam_cam, 'stop'):
                try:
                    rpicam_cam.stop()
                except Exception:
                    pass
    except Exception:
        pass
    if display:
        cv2.destroyAllWindows()


def _update_latest_frame(frame: np.ndarray) -> None:
    global _LATEST_FRAME
    with _FRAME_LOCK:
        _LATEST_FRAME = frame.copy()


def _update_latest_raw_frame(frame: np.ndarray) -> None:
    global _LATEST_RAW_FRAME
    with _FRAME_LOCK:
        _LATEST_RAW_FRAME = frame.copy()


def _set_latest_stats(
    *,
    total: int,
    with_scarf: int,
    missing: int,
    recognized: int,
    fps: float,
    capture_active: bool,
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
                "recognized": int(recognized),
                "fps": round(float(fps), 2),
                "capture_active": bool(capture_active),
                "monitoring": monitoring,
                "message": message,
            }
        )


def _set_capture_active(enabled: bool) -> bool:
    global _CAPTURE_ACTIVE
    with _CAPTURE_LOCK:
        _CAPTURE_ACTIVE = bool(enabled)
        return _CAPTURE_ACTIVE


def _get_capture_active() -> bool:
    with _CAPTURE_LOCK:
        return _CAPTURE_ACTIVE


def _set_sound_enabled(enabled: bool) -> bool:
    global _SOUND_ENABLED
    with _SOUND_LOCK:
        _SOUND_ENABLED = bool(enabled)
        return _SOUND_ENABLED


def _get_sound_enabled() -> bool:
    with _SOUND_LOCK:
        return _SOUND_ENABLED


def _generate_mjpeg_frames(*, raw: bool = False) -> Iterable[bytes]:
    jpeg_params = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
    while True:
        with _FRAME_LOCK:
            if raw:
                frame = (
                    None
                    if _LATEST_RAW_FRAME is None
                    else _LATEST_RAW_FRAME.copy()
                )
                if frame is None and _LATEST_FRAME is not None:
                    frame = _LATEST_FRAME.copy()
            else:
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
    from flask import g
    import traceback
    camera_error_holder = {'error': None}
    def monitor_wrapper():
        try:
            run_monitor(settings, frame_callback=_update_latest_frame, display=False, stop_event=stop_event)
        except Exception as e:
            camera_error_holder['error'] = f"Camera error: {e}\n" + traceback.format_exc()
            print(f"[ERROR] {camera_error_holder['error']}")

    monitor_thread = threading.Thread(
        target=monitor_wrapper,
        daemon=True,
    )
    monitor_thread.start()

    # Wait briefly, but do not block Flask startup
    time.sleep(0.5)

    # Initialize email sender and scheduler if enabled
    global _EMAIL_SENDER, _EMAIL_SCHEDULER, _EMAIL_LOG_STORE
    if settings.email_enabled and settings.email_sender and settings.email_password and settings.email_receiver:
        if _ATTENDANCE_STORE is not None:
            try:
                # Initialize email log store
                _EMAIL_LOG_STORE = EmailLogStore(settings.db_path)
                
                _EMAIL_SENDER = EmailSender(
                    sender_email=settings.email_sender,
                    sender_password=settings.email_password,
                    receiver_email=settings.email_receiver,
                    email_log_store=_EMAIL_LOG_STORE,
                )
                _EMAIL_SCHEDULER = EmailScheduler(
                    attendance_store=_ATTENDANCE_STORE,
                    email_sender=_EMAIL_SENDER,
                    schedule_hour=settings.email_schedule_hour,
                    schedule_minute=settings.email_schedule_minute,
                    email_log_store=_EMAIL_LOG_STORE,
                )
                _EMAIL_SCHEDULER.start()
                print(f"[INFO] Email scheduler initialized and started")
            except Exception as e:
                print(f"[WARN] Failed to initialize email scheduler: {e}")
                _EMAIL_SENDER = None
                _EMAIL_SCHEDULER = None
                _EMAIL_LOG_STORE = None
        else:
            print("[WARN] Email scheduler not started: attendance store not available")
            _EMAIL_SENDER = None
            _EMAIL_SCHEDULER = None
            _EMAIL_LOG_STORE = None
    else:
        _EMAIL_SENDER = None
        _EMAIL_SCHEDULER = None
        _EMAIL_LOG_STORE = None


    app = Flask(__name__, static_folder=str(Path(__file__).parent), static_url_path="")
    app.secret_key = settings.admin_password or "changeme123"
    app.config["SESSION_COOKIE_NAME"] = "admin_session"
    app.permanent_session_lifetime = dt.timedelta(hours=8)

    # Make camera error available in all requests
    @app.before_request
    def inject_camera_error():
        from flask import g
        g.camera_error = camera_error_holder['error']

    @app.route("/")
    def root() -> Response:
        from flask import g
        if g.get('camera_error'):
            return f"<h1>Camera Error</h1><pre>{g.camera_error}</pre>", 500
        return redirect("/webcam.html")

    @app.route("/webcam.html")
    def webcam_page() -> Response:
        from flask import g
        if g.get('camera_error'):
            return f"<h1>Camera Error</h1><pre>{g.camera_error}</pre>", 500
        # Try to serve webcam.html, fallback to inline HTML if missing
        import os
        webcam_path = os.path.join(app.static_folder, "webcam.html")
        if os.path.exists(webcam_path):
            return send_from_directory(app.static_folder, "webcam.html")
        return "<h1>Webcam Page</h1><p>No webcam.html found. If you see this, the server is running but the webcam page is missing.</p>", 200

    @app.route("/captures.html")
    def captures_page() -> Response:
        return send_from_directory(app.static_folder, "captures.html")

    @app.route("/settings.html")
    def settings_page() -> Response:
        if not session.get("admin_authenticated"):
            return redirect(url_for("admin_login", next=request.path))
        return send_from_directory(app.static_folder, "settings.html")

    @app.route("/admin.html")
    def admin_page() -> Response:
        if not session.get("admin_authenticated"):
            return redirect(url_for("admin_login", next=request.path))
        return send_from_directory(app.static_folder, "admin.html")

    @app.route("/email_history.html")
    def email_history_page() -> Response:
        if not session.get("admin_authenticated"):
            return redirect(url_for("admin_login", next=request.path))
        return send_from_directory(app.static_folder, "email_history.html")

    @app.route("/admin-login", methods=["GET", "POST"])
    def admin_login() -> Response:
        if session.get("admin_authenticated"):
            return redirect(url_for("admin_page"))

        if request.method == "POST":
            password = ""
            if request.is_json:
                payload = request.get_json(silent=True) or {}
                password = str(payload.get("password", ""))
            else:
                password = str(request.form.get("password", ""))

            if password == settings.admin_password:
                session["admin_authenticated"] = True
                next_url = request.args.get("next") or request.form.get("next")
                if next_url and next_url.startswith("/"):
                    return redirect(next_url)
                return redirect(url_for("admin_page"))
            next_param = request.args.get("next") or request.form.get("next")
            params = {"error": "1"}
            if next_param and next_param.startswith("/"):
                params["next"] = next_param
            return redirect(url_for("admin_login", **params))

        return send_from_directory(app.static_folder, "admin_login.html")

    @app.route("/admin-logout")
    def admin_logout() -> Response:
        session.pop("admin_authenticated", None)
        return redirect(url_for("admin_login"))

    @app.route("/video_feed")
    def video_feed() -> Response:
        raw_param = (request.args.get("raw") or "").strip().lower()
        raw = raw_param in {"1", "true", "yes", "on", "raw"}
        return Response(
            _generate_mjpeg_frames(raw=raw),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.route("/api/status")
    def api_status() -> Response:
        with _STATS_LOCK:
            payload = dict(_LATEST_STATS)
        payload.setdefault("timestamp", dt.datetime.now(dt.timezone.utc).isoformat())
        payload["capture_active"] = _get_capture_active()
        payload["sound_enabled"] = _get_sound_enabled()
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
        _require_admin_api()
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

    @app.route("/api/capture", methods=["POST"])
    def api_capture_toggle() -> Response:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            abort(400, description="Expected JSON object")
        if "enabled" not in payload:
            abort(400, description="Field 'enabled' is required")
        enabled = payload["enabled"]
        if not isinstance(enabled, bool):
            abort(400, description="'enabled' must be a boolean")

        settings = _GLOBAL_SETTINGS
        if settings is not None and not settings.save_violation_images and enabled:
            actual = _set_capture_active(False)
            return jsonify({"enabled": actual, "allowed": False, "reason": "Snapshot saving disabled in configuration."}), 409

        actual = _set_capture_active(enabled)
        return jsonify({"enabled": actual, "allowed": True})

    @app.route("/api/sound", methods=["GET", "POST"])
    def api_sound_toggle() -> Response:
        settings = _GLOBAL_SETTINGS
        if request.method == "GET":
            return jsonify(
                {
                    "enabled": _get_sound_enabled(),
                    "default": settings.enable_sound_alert if settings is not None else None,
                    "supported": playsound is not None,
                }
            )

        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or "enabled" not in payload:
            abort(400, description="Expected JSON object with 'enabled' field")

        enabled = payload["enabled"]
        if isinstance(enabled, bool):
            new_state = _set_sound_enabled(enabled)
        elif isinstance(enabled, (int, float)):
            new_state = _set_sound_enabled(bool(enabled))
        else:
            abort(400, description="'enabled' must be a boolean")

        if settings is not None:
            settings.enable_sound_alert = new_state

        return jsonify({"enabled": new_state})

    @app.route("/api/students", methods=["GET", "POST"])
    def api_students() -> Response:
        if _ATTENDANCE_STORE is None:
            if request.method == "GET":
                return jsonify({"enabled": False, "students": [], "attendance": {"check_ins": 0, "check_outs": 0}})
            abort(503, description="Attendance module is disabled")

        store = _ATTENDANCE_STORE

        if request.method == "GET":
            students = [
                {
                    "id": student.id,
                    "name": student.name,
                    "created_at": student.created_at.isoformat(),
                    "last_seen_at": student.last_seen_at.isoformat() if student.last_seen_at else None,
                    "sample_count": student.sample_count,
                }
                for student in store.list_students()
            ]
            summary = store.attendance_summary()
            return jsonify({"enabled": True, "students": students, "attendance": summary})

        payload = request.get_json(silent=True) or {}
        name = str(payload.get("name", "")).strip()
        if not name:
            abort(400, description="Student name is required")

        try:
            student = store.add_student(name)
        except ValueError as exc:
            abort(409, description=str(exc))
        if _STUDENT_RECOGNIZER is not None:
            _STUDENT_RECOGNIZER.rebuild()

        return (
            jsonify(
                {
                    "student": {
                        "id": student.id,
                        "name": student.name,
                        "created_at": student.created_at.isoformat(),
                        "last_seen_at": None,
                        "sample_count": 0,
                    }
                }
            ),
            201,
        )

    @app.route("/api/students/<int:student_id>", methods=["PATCH", "DELETE"])
    def api_student_detail(student_id: int) -> Response:
        if _ATTENDANCE_STORE is None:
            abort(503, description="Attendance module is disabled")

        store = _ATTENDANCE_STORE

        if request.method == "PATCH":
            payload = request.get_json(silent=True) or {}
            if "name" not in payload:
                abort(400, description="Field 'name' is required")
            try:
                student = store.update_student(student_id, payload["name"])
            except LookupError:
                abort(404, description="Student not found")
            except ValueError as exc:
                message = str(exc)
                status = 409 if "exists" in message.lower() else 400
                abort(status, description=message)

            if _STUDENT_RECOGNIZER is not None:
                _STUDENT_RECOGNIZER.rebuild()

            return jsonify(
                {
                    "student": {
                        "id": student.id,
                        "name": student.name,
                        "created_at": student.created_at.isoformat(),
                        "last_seen_at": student.last_seen_at.isoformat() if student.last_seen_at else None,
                        "sample_count": student.sample_count,
                    }
                }
            )

        # DELETE
        deleted = store.delete_student(student_id)
        if not deleted:
            abort(404, description="Student not found")

        if _STUDENT_RECOGNIZER is not None:
            _STUDENT_RECOGNIZER.rebuild()

        return jsonify({"deleted": True, "student_id": student_id})

    @app.route("/api/students/<int:student_id>/samples", methods=["GET"])
    def api_student_samples(student_id: int) -> Response:
        if _ATTENDANCE_STORE is None:
            abort(503, description="Attendance module is disabled")

        store = _ATTENDANCE_STORE
        student = store.get_student(student_id)
        if student is None:
            abort(404, description="Student not found")

        samples = [
            {
                "id": sample.id,
                "image_url": f"/student_samples/{sample.image_path.relative_to(store.samples_dir).as_posix()}",
                "created_at": sample.created_at.isoformat(),
            }
            for sample in store.list_samples(student_id)
        ]
        return jsonify({"student": {"id": student.id, "name": student.name}, "samples": samples})

    @app.route("/api/students/<int:student_id>/samples/<int:sample_id>", methods=["DELETE", "OPTIONS"])
    def api_student_sample_delete(student_id: int, sample_id: int) -> Response:
        if request.method == "OPTIONS":
            response = jsonify({"allowed": ["DELETE"]})
            response.headers["Allow"] = "DELETE, OPTIONS"
            return response

        if _ATTENDANCE_STORE is None:
            abort(503, description="Attendance module is disabled")

        store = _ATTENDANCE_STORE
        student = store.get_student(student_id)
        if student is None:
            abort(404, description="Student not found")

        sample = store.get_sample(sample_id)
        if sample is None or sample.student_id != student_id:
            abort(404, description="Sample not found")

        deleted = store.delete_sample(sample_id)
        if not deleted:
            abort(500, description="Unable to delete sample")

        if _STUDENT_RECOGNIZER is not None:
            try:
                _STUDENT_RECOGNIZER.rebuild()
            except Exception as exc:  # pragma: no cover - defensive
                abort(500, description=f"Sample removed but recognizer rebuild failed: {exc}")

        return jsonify({"deleted": True, "sample_id": sample_id})

    @app.route("/api/students/<int:student_id>/capture", methods=["POST"])
    def api_student_capture(student_id: int) -> Response:
        if _ATTENDANCE_STORE is None:
            abort(503, description="Attendance module is disabled")

        store = _ATTENDANCE_STORE
        student = store.get_student(student_id)
        if student is None:
            abort(404, description="Student not found")

        with _FRAME_LOCK:
            if _LATEST_RAW_FRAME is not None:
                frame = _LATEST_RAW_FRAME.copy()
            elif _LATEST_FRAME is not None:
                frame = _LATEST_FRAME.copy()
            else:
                frame = None

        if frame is None:
            abort(503, description="Camera frame unavailable")

        faces = detect_faces(frame)
        if not faces:
            abort(422, description="No face detected in current frame")

        largest = max(faces, key=lambda f: (f[2] - f[0]) * (f[3] - f[1]))
        face_image = crop_face(frame, largest[:4])
        if face_image is None or face_image.size == 0:
            abort(422, description="Unable to crop face from frame")

        try:
            sample = store.add_sample(student_id, face_image)
        except Exception as exc:
            abort(500, description=f"Unable to store sample: {exc}")

        if _STUDENT_RECOGNIZER is not None:
            try:
                _STUDENT_RECOGNIZER.rebuild()
            except Exception as exc:  # pragma: no cover - defensive
                abort(500, description=f"Sample saved but recognizer rebuild failed: {exc}")

        return (
            jsonify(
                {
                    "sample": {
                        "id": sample.id,
                        "image_url": f"/student_samples/{sample.image_path.relative_to(store.samples_dir).as_posix()}",
                        "created_at": sample.created_at.isoformat(),
                    }
                }
            ),
            201,
        )

    @app.route("/api/attendance/logs")
    def api_attendance_logs() -> Response:
        if _ATTENDANCE_STORE is None:
            return jsonify({"enabled": False, "records": [], "pagination": {}})

        store = _ATTENDANCE_STORE

        # Parse pagination parameters
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

        # Parse filter parameters
        start_date = request.args.get("start_date")
        end_date = request.args.get("end_date")
        student_name = request.args.get("student_name", "").strip() or None

        # Validate date parameters
        if start_date:
            try:
                dt.date.fromisoformat(start_date)
            except ValueError:
                abort(400, description="Invalid start_date parameter")
        if end_date:
            try:
                dt.date.fromisoformat(end_date)
            except ValueError:
                abort(400, description="Invalid end_date parameter")

        records, total_count = store.attendance_logs_filtered(
            limit=per_page,
            offset=offset,
            start_date=start_date,
            end_date=end_date,
            student_name=student_name,
        )

        student_map: Dict[int, Student] = {student.id: student for student in store.list_students()}
        items = []
        for record in records:
            student = student_map.get(record.student_id)
            items.append(
                {
                    "id": record.id,
                    "student_id": record.student_id,
                    "student_name": student.name if student else f"#{record.student_id}",
                    "event_type": record.event_type,
                    "timestamp": record.timestamp.isoformat(),
                    "confidence": record.confidence,
                }
            )

        total_pages = (total_count + per_page - 1) // per_page if total_count else 0
        pagination = {
            "page": page,
            "per_page": per_page,
            "total": total_count,
            "total_pages": total_pages,
            "has_prev": page > 1 and total_count > 0,
            "has_next": page < total_pages,
        }

        summary = store.attendance_summary()
        return jsonify({"enabled": True, "records": items, "attendance": summary, "pagination": pagination})

    @app.route("/api/email/send", methods=["POST"])
    def api_email_send() -> Response:
        _require_admin_api()
        if _EMAIL_SCHEDULER is None:
            return jsonify({"success": False, "message": "Email scheduler is not enabled or not initialized"}), 503
        
        try:
            success = _EMAIL_SCHEDULER.send_now()
            if success:
                return jsonify({"success": True, "message": "Timesheet email sent successfully"})
            else:
                return jsonify({"success": False, "message": "Failed to send email. Check server logs for details"}), 500
        except Exception as e:
            return jsonify({"success": False, "message": f"Error sending email: {str(e)}"}), 500

    @app.route("/api/email/logs")
    def api_email_logs() -> Response:
        _require_admin_api()
        if _EMAIL_LOG_STORE is None:
            return jsonify({"enabled": False, "logs": [], "statistics": {}})

        limit = request.args.get("limit", default=50, type=int)
        start_date = request.args.get("start_date")
        end_date = request.args.get("end_date")

        start = dt.date.fromisoformat(start_date) if start_date else None
        end = dt.date.fromisoformat(end_date) if end_date else None

        if start or end:
            logs = _EMAIL_LOG_STORE.get_logs_by_date_range(start_date=start, end_date=end, limit=limit)
        else:
            logs = _EMAIL_LOG_STORE.get_recent_logs(limit=limit)

        statistics = _EMAIL_LOG_STORE.get_statistics()

        items = []
        for log in logs:
            items.append(
                {
                    "id": log.id,
                    "sent_at": log.sent_at.isoformat(),
                    "receiver_email": log.receiver_email,
                    "report_date": log.report_date.isoformat(),
                    "status": log.status,
                    "error_message": log.error_message,
                    "student_count": log.student_count,
                }
            )

        return jsonify({"enabled": True, "logs": items, "statistics": statistics})

    @app.route("/api/attendance/report")
    def api_attendance_report() -> Response:
        if _ATTENDANCE_STORE is None:
            return jsonify(
                {
                    "enabled": False,
                    "records": [],
                    "filters": {"years": [], "months": [], "days": []},
                    "selected": {"year": None, "month": None, "day": None},
                }
            )

        def normalize_year(value: Optional[str]) -> Optional[str]:
            if value is None:
                return None
            value = value.strip()
            if not value:
                return None
            if not value.isdigit():
                abort(400, description="Invalid year")
            return f"{int(value):04d}"

        def normalize_month(value: Optional[str]) -> Optional[str]:
            if value is None:
                return None
            value = value.strip()
            if not value:
                return None
            if not value.isdigit():
                abort(400, description="Invalid month")
            month_value = int(value)
            if month_value < 1 or month_value > 12:
                abort(400, description="Invalid month")
            return f"{month_value:02d}"

        def normalize_day(value: Optional[str]) -> Optional[str]:
            if value is None:
                return None
            value = value.strip()
            if not value:
                return None
            if not value.isdigit():
                abort(400, description="Invalid day")
            day_value = int(value)
            if day_value < 1 or day_value > 31:
                abort(400, description="Invalid day")
            return f"{day_value:02d}"

        year_param = normalize_year(request.args.get("year"))
        month_param = normalize_month(request.args.get("month"))
        day_param = normalize_day(request.args.get("day"))
        format_param = (request.args.get("format") or "").lower()
        default_scope = (request.args.get("default") or "").strip().lower()

        if default_scope == "today" and not any([year_param, month_param, day_param]):
            today_local = dt.datetime.now().astimezone()
            year_param = today_local.strftime("%Y")
            month_param = today_local.strftime("%m")
            day_param = today_local.strftime("%d")

        store = _ATTENDANCE_STORE
        filters = store.attendance_report_filters(year=year_param, month=month_param)
        records = store.attendance_report(year=year_param, month=month_param, day=day_param)

        def _ensure_option(values: List[str], selected: Optional[str]) -> List[str]:
            if not selected:
                return values
            if selected not in values:
                values = values + [selected]
                values.sort(key=lambda item: int(item), reverse=True)
            return values

        filters["years"] = _ensure_option(filters.get("years", []), year_param)
        filters["months"] = _ensure_option(filters.get("months", []), month_param)
        filters["days"] = _ensure_option(filters.get("days", []), day_param)

        if format_param == "csv":
            buffer = io.StringIO()
            writer = csv.writer(buffer)
            writer.writerow(["#", "Student", "First Check In", "Last Check Out"])
            for index, record in enumerate(records, start=1):
                writer.writerow(
                    [
                        index,
                        record["student_name"],
                        record["first_check_in"] or "",
                        record["last_check_out"] or "",
                    ]
                )
            filename_parts = [part for part in (year_param, month_param, day_param) if part]
            label = "-".join(filename_parts) if filename_parts else "all"
            response = Response(buffer.getvalue(), mimetype="text/csv")
            response.headers["Content-Disposition"] = f'attachment; filename="attendance_report_{label}.csv"'
            return response

        return jsonify(
            {
                "enabled": True,
                "records": records,
                "filters": filters,
                "selected": {"year": year_param, "month": month_param, "day": day_param},
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

    @app.route("/student_samples/<path:filename>")
    def student_samples(filename: str) -> Response:
        if _ATTENDANCE_STORE is None:
            abort(404)
        sample_dir = _ATTENDANCE_STORE.samples_dir.resolve()
        target = (sample_dir / Path(filename)).resolve()
        if not str(target).startswith(str(sample_dir)) or not target.exists():
            abort(404)
        relative = target.relative_to(sample_dir)
        return send_from_directory(str(sample_dir), relative.as_posix())

    try:
        app.run(host=host, port=port, debug=False, threaded=True, use_reloader=False)
    finally:
        stop_event.set()
        if _EMAIL_SCHEDULER:
            _EMAIL_SCHEDULER.stop()
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



