"""
Scheduled email job for sending daily timesheet reports.

This module handles scheduling and executing the daily email job at 6PM.
"""

from __future__ import annotations

import datetime as dt
import threading
import time
from typing import Optional

from attendance import AttendanceStore
from email_sender import EmailSender
from email_logger import EmailLogStore


class EmailScheduler:
    """
    Scheduler for daily timesheet email reports.
    """

    def __init__(
        self,
        attendance_store: Optional[AttendanceStore],
        email_sender: Optional[EmailSender],
        schedule_hour: int = 18,  # 6PM
        schedule_minute: int = 0,
        email_log_store: Optional[EmailLogStore] = None,
    ) -> None:
        """
        Initialize email scheduler.

        Args:
            attendance_store: AttendanceStore instance
            email_sender: EmailSender instance
            schedule_hour: Hour of day to send email (0-23)
            schedule_minute: Minute of hour to send email (0-59)
            email_log_store: Optional EmailLogStore instance for logging
        """
        self.attendance_store = attendance_store
        self.email_sender = email_sender
        self.schedule_hour = schedule_hour
        self.schedule_minute = schedule_minute
        self.email_log_store = email_log_store
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Start the scheduler thread."""
        if self._running:
            return

        if not self.attendance_store or not self.email_sender:
            print("[WARN] Email scheduler not started: missing attendance_store or email_sender")
            return

        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        print(f"[INFO] Email scheduler started. Will send reports daily at {self.schedule_hour:02d}:{self.schedule_minute:02d}")

    def stop(self) -> None:
        """Stop the scheduler thread."""
        if not self._running:
            return

        self._running = False
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        print("[INFO] Email scheduler stopped")

    def _run_loop(self) -> None:
        """Main scheduler loop."""
        while self._running and not self._stop_event.is_set():
            try:
                now = dt.datetime.now()
                target_time = now.replace(
                    hour=self.schedule_hour,
                    minute=self.schedule_minute,
                    second=0,
                    microsecond=0,
                )

                # If target time has passed today, schedule for tomorrow
                if target_time <= now:
                    target_time += dt.timedelta(days=1)

                # Calculate seconds until target time
                wait_seconds = (target_time - now).total_seconds()

                print(f"[INFO] Next email scheduled for {target_time.strftime('%Y-%m-%d %H:%M:%S')} (in {wait_seconds/3600:.1f} hours)")

                # Wait until target time or stop event
                if self._stop_event.wait(timeout=min(wait_seconds, 3600)):
                    # Stop event was set
                    break

                # Check if it's time to send (with 1 minute tolerance)
                now = dt.datetime.now()
                if (
                    now.hour == self.schedule_hour
                    and now.minute == self.schedule_minute
                    and not self._stop_event.is_set()
                ):
                    self._send_daily_report()

                # Sleep for a minute to avoid busy waiting
                time.sleep(60)

            except Exception as e:
                print(f"[ERROR] Error in email scheduler loop: {e}")
                # Wait a bit before retrying
                time.sleep(60)

    def _send_daily_report(self) -> None:
        """Send the daily timesheet report."""
        if not self.attendance_store or not self.email_sender:
            return

        try:
            today = dt.datetime.now().astimezone().date()
            print(f"[INFO] Sending daily timesheet report for {today}")
            success = self.email_sender.send_timesheet_report(
                self.attendance_store,
                date=today,
            )
            if success:
                print(f"[INFO] Daily timesheet report sent successfully for {today}")
            else:
                print(f"[WARN] Failed to send daily timesheet report for {today}")
        except Exception as e:
            print(f"[ERROR] Error sending daily report: {e}")

    def send_now(self) -> bool:
        """
        Manually trigger sending the report for today.

        Returns:
            True if sent successfully, False otherwise
        """
        if not self.attendance_store or not self.email_sender:
            return False

        try:
            today = dt.datetime.now().astimezone().date()
            return self.email_sender.send_timesheet_report(
                self.attendance_store,
                date=today,
            )
        except Exception as e:
            print(f"[ERROR] Error sending manual report: {e}")
            return False

    @property
    def is_running(self) -> bool:
        """Check if scheduler is running."""
        return self._running

