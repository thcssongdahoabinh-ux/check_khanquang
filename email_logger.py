"""
Email logging utilities for tracking email send results.
"""

from __future__ import annotations

import datetime as dt
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class EmailLog:
    id: int
    sent_at: dt.datetime
    receiver_email: str
    report_date: dt.date
    status: str  # 'success' or 'failed'
    error_message: Optional[str]
    student_count: int


class EmailLogStore:
    """
    SQLite-backed persistence layer for email logs.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """Create email_logs table if it doesn't exist."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS email_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sent_at TEXT NOT NULL,
                    receiver_email TEXT NOT NULL,
                    report_date TEXT NOT NULL,
                    status TEXT NOT NULL,
                    error_message TEXT,
                    student_count INTEGER NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_email_logs_sent_at ON email_logs(sent_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_email_logs_report_date ON email_logs(report_date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_email_logs_status ON email_logs(status)")

    def log_email(
        self,
        receiver_email: str,
        report_date: dt.date,
        status: str,
        student_count: int,
        error_message: Optional[str] = None,
    ) -> EmailLog:
        """
        Log an email send attempt.

        Args:
            receiver_email: Email address the report was sent to
            report_date: Date of the attendance report
            status: 'success' or 'failed'
            student_count: Number of students in the report
            error_message: Error message if status is 'failed'

        Returns:
            EmailLog instance
        """
        sent_at = dt.datetime.now(dt.timezone.utc)
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO email_logs (sent_at, receiver_email, report_date, status, error_message, student_count)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    sent_at.isoformat(),
                    receiver_email,
                    report_date.isoformat(),
                    status,
                    error_message,
                    student_count,
                ),
            )
            log_id = cursor.lastrowid

        return EmailLog(
            id=int(log_id),
            sent_at=sent_at,
            receiver_email=receiver_email,
            report_date=report_date,
            status=status,
            error_message=error_message,
            student_count=student_count,
        )

    def get_recent_logs(self, limit: int = 50) -> List[EmailLog]:
        """
        Get recent email logs.

        Args:
            limit: Maximum number of logs to return

        Returns:
            List of EmailLog instances
        """
        limit = max(1, min(int(limit), 500))
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                """
                SELECT id, sent_at, receiver_email, report_date, status, error_message, student_count
                FROM email_logs
                ORDER BY sent_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        logs: List[EmailLog] = []
        for row in rows:
            logs.append(
                EmailLog(
                    id=int(row[0]),
                    sent_at=dt.datetime.fromisoformat(row[1]),
                    receiver_email=str(row[2]),
                    report_date=dt.date.fromisoformat(row[3]),
                    status=str(row[4]),
                    error_message=str(row[5]) if row[5] else None,
                    student_count=int(row[6]),
                )
            )
        return logs

    def get_logs_by_date_range(
        self,
        start_date: Optional[dt.date] = None,
        end_date: Optional[dt.date] = None,
        limit: int = 100,
    ) -> List[EmailLog]:
        """
        Get email logs filtered by date range.

        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            limit: Maximum number of logs to return

        Returns:
            List of EmailLog instances
        """
        limit = max(1, min(int(limit), 500))
        where_clauses: List[str] = []
        params: List[object] = []

        if start_date:
            where_clauses.append("report_date >= ?")
            params.append(start_date.isoformat())
        if end_date:
            where_clauses.append("report_date <= ?")
            params.append(end_date.isoformat())

        where_sql = f" WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                f"""
                SELECT id, sent_at, receiver_email, report_date, status, error_message, student_count
                FROM email_logs{where_sql}
                ORDER BY sent_at DESC
                LIMIT ?
                """,
                params + [limit],
            ).fetchall()

        logs: List[EmailLog] = []
        for row in rows:
            logs.append(
                EmailLog(
                    id=int(row[0]),
                    sent_at=dt.datetime.fromisoformat(row[1]),
                    receiver_email=str(row[2]),
                    report_date=dt.date.fromisoformat(row[3]),
                    status=str(row[4]),
                    error_message=str(row[5]) if row[5] else None,
                    student_count=int(row[6]),
                )
            )
        return logs

    def get_statistics(self) -> dict:
        """
        Get email sending statistics.

        Returns:
            Dictionary with statistics
        """
        with sqlite3.connect(self._db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM email_logs").fetchone()[0]
            success = conn.execute("SELECT COUNT(*) FROM email_logs WHERE status = 'success'").fetchone()[0]
            failed = conn.execute("SELECT COUNT(*) FROM email_logs WHERE status = 'failed'").fetchone()[0]

            latest = conn.execute(
                "SELECT sent_at, status FROM email_logs ORDER BY sent_at DESC LIMIT 1"
            ).fetchone()

        return {
            "total": int(total or 0),
            "success": int(success or 0),
            "failed": int(failed or 0),
            "latest_sent_at": latest[0] if latest else None,
            "latest_status": latest[1] if latest else None,
        }

