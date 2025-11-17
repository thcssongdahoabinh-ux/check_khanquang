"""
Attendance management utilities for the red scarf monitoring system.

This module handles student enrollment, sample capture metadata, attendance
logging, and in-memory face recognition via OpenCV's LBPH classifier.
"""

from __future__ import annotations

import datetime as dt
import shutil
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


@dataclass
class Student:
    id: int
    name: str
    created_at: dt.datetime
    last_seen_at: Optional[dt.datetime]
    sample_count: int


@dataclass
class StudentSample:
    id: int
    student_id: int
    image_path: Path
    created_at: dt.datetime


@dataclass
class AttendanceRecord:
    id: int
    student_id: int
    event_type: str
    timestamp: dt.datetime
    confidence: Optional[float]


@dataclass
class RecognitionResult:
    student_id: int
    distance: float
    confidence: float


class AttendanceStore:
    """
    SQLite-backed persistence layer for students, samples, and attendance logs.
    """

    def __init__(self, db_path: Path, samples_dir: Path) -> None:
        self._db_path = db_path
        self._samples_dir = samples_dir
        self._samples_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._ensure_schema()

    # ------------------------------------------------------------------ schema

    def _ensure_schema(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS students (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    created_at TEXT NOT NULL,
                    last_seen_at TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS student_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id INTEGER NOT NULL REFERENCES students(id) ON DELETE CASCADE,
                    image_path TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS attendance_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id INTEGER NOT NULL REFERENCES students(id) ON DELETE CASCADE,
                    event_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    confidence REAL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_student_samples_student ON student_samples(student_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_attendance_student ON attendance_logs(student_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_attendance_timestamp ON attendance_logs(timestamp)")

    # ---------------------------------------------------------------- students

    def add_student(self, name: str) -> Student:
        sanitized = " ".join(part for part in name.strip().split() if part)
        if not sanitized:
            raise ValueError("Student name must not be empty")

        now = _utc_now().isoformat()
        try:
            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.execute(
                    "INSERT INTO students (name, created_at) VALUES (?, ?)",
                    (sanitized, now),
                )
                student_id = cursor.lastrowid
        except sqlite3.IntegrityError as exc:
            raise ValueError("Student name already exists") from exc

        return Student(id=int(student_id), name=sanitized, created_at=dt.datetime.fromisoformat(now), last_seen_at=None, sample_count=0)

    def update_student(self, student_id: int, name: str) -> Student:
        sanitized = " ".join(part for part in str(name).strip().split() if part)
        if not sanitized:
            raise ValueError("Student name must not be empty")

        try:
            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.execute(
                    "UPDATE students SET name = ? WHERE id = ?",
                    (sanitized, student_id),
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError("Student name already exists") from exc

        if cursor.rowcount == 0:
            raise LookupError("Student not found")

        updated = self.get_student(student_id)
        if updated is None:
            raise LookupError("Student not found")
        return updated

    def list_students(self) -> List[Student]:
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                """
                SELECT s.id, s.name, s.created_at, s.last_seen_at, COUNT(ss.id) AS sample_count
                FROM students AS s
                LEFT JOIN student_samples AS ss ON ss.student_id = s.id
                GROUP BY s.id
                ORDER BY s.name COLLATE NOCASE
                """
            ).fetchall()

        students: List[Student] = []
        for row in rows:
            created = dt.datetime.fromisoformat(row[2])
            last_seen = dt.datetime.fromisoformat(row[3]) if row[3] else None
            students.append(
                Student(
                    id=int(row[0]),
                    name=str(row[1]),
                    created_at=created,
                    last_seen_at=last_seen,
                    sample_count=int(row[4] or 0),
                )
            )
        return students

    def get_student(self, student_id: int) -> Optional[Student]:
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                """
                SELECT s.id, s.name, s.created_at, s.last_seen_at,
                       (SELECT COUNT(*) FROM student_samples WHERE student_id = s.id)
                FROM students AS s
                WHERE s.id = ?
                """,
                (student_id,),
            ).fetchone()

        if row is None:
            return None
        created = dt.datetime.fromisoformat(row[2])
        last_seen = dt.datetime.fromisoformat(row[3]) if row[3] else None
        return Student(id=int(row[0]), name=str(row[1]), created_at=created, last_seen_at=last_seen, sample_count=int(row[4] or 0))

    def delete_student(self, student_id: int) -> bool:
        sample_paths = [path for _, path in self.iter_sample_paths(student_id)]
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute("DELETE FROM students WHERE id = ?", (student_id,))

        if cursor.rowcount == 0:
            return False

        for path in sample_paths:
            try:
                if path.exists():
                    path.unlink()
            except OSError:
                pass

        student_dir = self._samples_dir / f"student_{student_id:04d}"
        try:
            shutil.rmtree(student_dir, ignore_errors=True)
        except OSError:
            pass
        return True

    # ---------------------------------------------------------------- samples

    def add_sample(self, student_id: int, image: np.ndarray) -> StudentSample:
        if image.size == 0:
            raise ValueError("Sample image must not be empty")

        student_dir = self._samples_dir / f"student_{student_id:04d}"
        student_dir.mkdir(parents=True, exist_ok=True)
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = student_dir / f"sample_{timestamp}.jpg"
        if not cv2.imwrite(str(filename), image):
            raise RuntimeError("Failed to save sample image")

        now = _utc_now().isoformat()
        rel_path = filename.relative_to(self._samples_dir)

        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO student_samples (student_id, image_path, created_at)
                VALUES (?, ?, ?)
                """,
                (student_id, rel_path.as_posix(), now),
            )
            sample_id = cursor.lastrowid

        return StudentSample(
            id=int(sample_id),
            student_id=student_id,
            image_path=filename,
            created_at=dt.datetime.fromisoformat(now),
        )

    def iter_sample_paths(self, student_id: Optional[int] = None) -> Iterable[Tuple[int, Path]]:
        query = "SELECT student_id, image_path FROM student_samples"
        params: Sequence[object] = ()
        if student_id is not None:
            query += " WHERE student_id = ?"
            params = (student_id,)
        query += " ORDER BY created_at"

        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(query, params).fetchall()

        for student, rel_path in rows:
            path = (self._samples_dir / str(rel_path)).resolve()
            if path.exists():
                yield int(student), path

    def list_samples(self, student_id: int) -> List[StudentSample]:
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                """
                SELECT id, image_path, created_at
                FROM student_samples
                WHERE student_id = ?
                ORDER BY created_at DESC
                """,
                (student_id,),
            ).fetchall()

        samples: List[StudentSample] = []
        for row in rows:
            absolute = (self._samples_dir / str(row[1])).resolve()
            samples.append(
                StudentSample(
                    id=int(row[0]),
                    student_id=student_id,
                    image_path=absolute,
                    created_at=dt.datetime.fromisoformat(row[2]),
                )
            )
        return samples

    def get_sample(self, sample_id: int) -> Optional[StudentSample]:
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                """
                SELECT student_id, image_path, created_at
                FROM student_samples
                WHERE id = ?
                """,
                (sample_id,),
            ).fetchone()

        if row is None:
            return None

        student_id, rel_path, created = row
        absolute = (self._samples_dir / str(rel_path)).resolve()
        return StudentSample(
            id=sample_id,
            student_id=int(student_id),
            image_path=absolute,
            created_at=dt.datetime.fromisoformat(str(created)),
        )

    def delete_sample(self, sample_id: int) -> bool:
        sample = self.get_sample(sample_id)
        if sample is None:
            return False

        with sqlite3.connect(self._db_path) as conn:
            conn.execute("DELETE FROM student_samples WHERE id = ?", (sample_id,))

        try:
            if sample.image_path.exists():
                sample.image_path.unlink()
        except OSError:
            pass
        return True

    # ------------------------------------------------------------- attendance

    def _determine_event_type(self, student_id: int, timestamp: dt.datetime) -> str:
        local_date = timestamp.astimezone().date().isoformat()
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                """
                SELECT event_type FROM attendance_logs
                WHERE student_id = ?
                  AND date(timestamp, 'localtime') = ?
                ORDER BY timestamp DESC LIMIT 1
                """,
                (student_id, local_date),
            ).fetchone()

        if row is None or row[0] == "check_out":
            return "check_in"
        return "check_out"

    def record_attendance(self, student_id: int, *, confidence: Optional[float], event_type: Optional[str] = None) -> AttendanceRecord:
        now = _utc_now()
        event = event_type or self._determine_event_type(student_id, now)
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO attendance_logs (student_id, event_type, timestamp, confidence)
                VALUES (?, ?, ?, ?)
                """,
                (student_id, event, now.isoformat(), confidence),
            )
            record_id = cursor.lastrowid
            conn.execute(
                "UPDATE students SET last_seen_at = ? WHERE id = ?",
                (now.isoformat(), student_id),
            )

        return AttendanceRecord(
            id=int(record_id),
            student_id=student_id,
            event_type=event,
            timestamp=now,
            confidence=confidence,
        )

    def recent_attendance(self, limit: int = 50) -> List[AttendanceRecord]:
        limit = max(1, min(int(limit), 500))
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                """
                SELECT id, student_id, event_type, timestamp, confidence
                FROM attendance_logs
                ORDER BY timestamp DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()

        records: List[AttendanceRecord] = []
        for row in rows:
            records.append(
                AttendanceRecord(
                    id=int(row[0]),
                    student_id=int(row[1]),
                    event_type=str(row[2]),
                    timestamp=dt.datetime.fromisoformat(row[3]),
                    confidence=float(row[4]) if row[4] is not None else None,
                )
            )
        return records

    def attendance_logs_filtered(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        student_name: Optional[str] = None,
    ) -> Tuple[List[AttendanceRecord], int]:
        """
        Query attendance logs with filtering and pagination.
        Returns (records, total_count).
        """
        limit = max(1, min(int(limit), 500))
        offset = max(0, int(offset))

        where_clauses: List[str] = []
        params: List[object] = []

        if start_date:
            where_clauses.append("date(timestamp, 'localtime') >= ?")
            params.append(start_date)
        if end_date:
            where_clauses.append("date(timestamp, 'localtime') <= ?")
            params.append(end_date)
        if student_name:
            where_clauses.append("student_id IN (SELECT id FROM students WHERE name LIKE ?)")
            params.append(f"%{student_name}%")

        where_sql = f" WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        with sqlite3.connect(self._db_path) as conn:
            # Get total count
            total_row = conn.execute(
                f"SELECT COUNT(*) FROM attendance_logs{where_sql}",
                params,
            ).fetchone()
            total_count = int(total_row[0]) if total_row else 0

            # Get paginated records
            rows = conn.execute(
                f"""
                SELECT id, student_id, event_type, timestamp, confidence
                FROM attendance_logs
                {where_sql}
                ORDER BY timestamp DESC LIMIT ? OFFSET ?
                """,
                params + [limit, offset],
            ).fetchall()

        records: List[AttendanceRecord] = []
        for row in rows:
            records.append(
                AttendanceRecord(
                    id=int(row[0]),
                    student_id=int(row[1]),
                    event_type=str(row[2]),
                    timestamp=dt.datetime.fromisoformat(row[3]),
                    confidence=float(row[4]) if row[4] is not None else None,
                )
            )
        return records, total_count

    # -------------------------------------------------------------- statistics

    def attendance_summary(self, *, date: Optional[dt.date] = None) -> Dict[str, int]:
        """Return total check-ins and check-outs for the specified local date."""
        target = date or dt.datetime.now().astimezone().date()
        prefix = target.isoformat()
        with sqlite3.connect(self._db_path) as conn:
            totals = conn.execute(
                """
                SELECT
                    SUM(CASE WHEN event_type = 'check_in' THEN 1 ELSE 0 END),
                    SUM(CASE WHEN event_type = 'check_out' THEN 1 ELSE 0 END)
                FROM attendance_logs
                WHERE date(timestamp, 'localtime') = ?
                """,
                (prefix,),
            ).fetchone()

        check_ins = int(totals[0] or 0) if totals else 0
        check_outs = int(totals[1] or 0) if totals else 0
        return {"check_ins": check_ins, "check_outs": check_outs}

    def attendance_report_filters(self, *, year: Optional[str] = None, month: Optional[str] = None) -> Dict[str, List[str]]:
        with sqlite3.connect(self._db_path) as conn:
            year_rows = conn.execute(
                """
                SELECT DISTINCT strftime('%Y', timestamp, 'localtime') AS year
                FROM attendance_logs
                WHERE timestamp IS NOT NULL
                ORDER BY year DESC
                """
            ).fetchall()
            years = [str(row[0]) for row in year_rows if row[0] is not None]

            month_query = "SELECT DISTINCT strftime('%m', timestamp, 'localtime') AS month FROM attendance_logs"
            month_params: List[object] = []
            if year:
                month_query += " WHERE strftime('%Y', timestamp, 'localtime') = ?"
                month_params.append(year)
            month_query += " ORDER BY month DESC"
            month_rows = conn.execute(month_query, month_params).fetchall()
            months = [str(row[0]) for row in month_rows if row[0] is not None]

            days: List[str] = []
            if year or month:
                day_query = "SELECT DISTINCT strftime('%d', timestamp, 'localtime') AS day FROM attendance_logs"
                conditions: List[str] = []
                params: List[object] = []
                if year:
                    conditions.append("strftime('%Y', timestamp, 'localtime') = ?")
                    params.append(year)
                if month:
                    conditions.append("strftime('%m', timestamp, 'localtime') = ?")
                    params.append(month)
                if conditions:
                    day_query += " WHERE " + " AND ".join(conditions)
                    day_query += " ORDER BY day DESC"
                    day_rows = conn.execute(day_query, params).fetchall()
                    days = [str(row[0]) for row in day_rows if row[0] is not None]

        return {"years": years, "months": months, "days": days}

    def attendance_report(
        self,
        *,
        year: Optional[str] = None,
        month: Optional[str] = None,
        day: Optional[str] = None,
    ) -> List[Dict[str, Optional[str]]]:
        join_conditions = ["a.student_id = s.id"]
        params: List[object] = []

        if year:
            join_conditions.append("strftime('%Y', a.timestamp, 'localtime') = ?")
            params.append(year)
        if month:
            join_conditions.append("strftime('%m', a.timestamp, 'localtime') = ?")
            params.append(month)
        if day:
            join_conditions.append("strftime('%d', a.timestamp, 'localtime') = ?")
            params.append(day)

        join_clause = " AND ".join(join_conditions)

        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"""
                SELECT
                    s.id AS student_id,
                    s.name AS student_name,
                    MIN(CASE WHEN a.event_type = 'check_in' THEN a.timestamp END) AS first_check_in,
                    MAX(CASE WHEN a.event_type = 'check_out' THEN a.timestamp END) AS last_check_out
                FROM students s
                LEFT JOIN attendance_logs a
                    ON {join_clause}
                GROUP BY s.id, s.name
                ORDER BY s.name COLLATE NOCASE
                """,
                params,
            ).fetchall()

        records: List[Dict[str, Optional[str]]] = []
        for row in rows:
            records.append(
                {
                    "student_id": int(row["student_id"]),
                    "student_name": str(row["student_name"]),
                    "first_check_in": str(row["first_check_in"]) if row["first_check_in"] is not None else None,
                    "last_check_out": str(row["last_check_out"]) if row["last_check_out"] is not None else None,
                }
            )

        return records

    # -------------------------------------------------------------- utilities

    @property
    def samples_dir(self) -> Path:
        return self._samples_dir


class StudentRecognizer:
    """
    LBPH-based face recognizer built from stored student samples.
    """

    def __init__(
        self,
        store: AttendanceStore,
        *,
        face_size: Tuple[int, int] = (200, 200),
        distance_threshold: float = 70.0,
        min_samples: int = 1,
    ) -> None:
        self._store = store
        self._face_size = face_size
        self._distance_threshold = distance_threshold
        self._min_samples = max(1, min_samples)
        self._lock = threading.Lock()
        self._recognizer = self._create_recognizer()
        self._is_trained = False

    @staticmethod
    def _create_recognizer() -> Optional[Any]:
        if not hasattr(cv2, "face") or not hasattr(cv2.face, "LBPHFaceRecognizer_create"):
            return None
        return cv2.face.LBPHFaceRecognizer_create()

    @property
    def available(self) -> bool:
        return self._recognizer is not None

    @property
    def is_ready(self) -> bool:
        return self.available and self._is_trained

    def rebuild(self) -> None:
        if not self.available:
            self._is_trained = False
            return

        images: List[np.ndarray] = []
        labels: List[int] = []
        for student_id, path in self._store.iter_sample_paths():
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            processed = self._preprocess(image)
            if processed is None:
                continue
            images.append(processed)
            labels.append(int(student_id))

        if len(images) < self._min_samples:
            self._is_trained = False
            return

        with self._lock:
            assert self._recognizer is not None
            self._recognizer.train(images, np.array(labels))
            self._is_trained = True

    def _preprocess(self, image: np.ndarray) -> Optional[np.ndarray]:
        if image.ndim == 3:
            channels = image.shape[2]
            if channels == 1:
                gray = image.squeeze(axis=2)
            else:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        if gray is None or gray.size == 0:
            return None
        if gray.ndim != 2:
            gray = np.squeeze(gray)
            if gray.ndim != 2:
                return None
        resized = cv2.resize(gray, self._face_size, interpolation=cv2.INTER_CUBIC)
        return resized

    def predict(self, face_image: np.ndarray) -> Optional[RecognitionResult]:
        if not self.is_ready:
            return None
        processed = self._preprocess(face_image)
        if processed is None:
            return None
        with self._lock:
            assert self._recognizer is not None
            label, distance = self._recognizer.predict(processed)

        confidence = self._distance_to_confidence(distance)
        if distance > self._distance_threshold:
            return None
        return RecognitionResult(student_id=int(label), distance=float(distance), confidence=confidence)

    def _distance_to_confidence(self, distance: float) -> float:
        # Convert LBPH distance (lower is better) into 0..1 confidence score.
        normalized = max(0.0, min(1.0, (self._distance_threshold - distance) / max(self._distance_threshold, 1e-6)))
        return float(normalized)

