"""
Email sending utilities for the attendance monitoring system.

This module handles sending daily timesheet reports via Gmail SMTP.
"""

from __future__ import annotations

import datetime as dt
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import List, Optional

from attendance import AttendanceStore
from email_logger import EmailLogStore


class EmailSender:
    """
    Gmail SMTP-based email sender for timesheet reports.
    """

    GMAIL_SMTP_SERVER = "smtp.gmail.com"
    GMAIL_SMTP_PORT = 587

    def __init__(
        self,
        sender_email: str,
        sender_password: str,
        receiver_email: str,
        email_log_store: Optional[EmailLogStore] = None,
    ) -> None:
        """
        Initialize email sender.

        Args:
            sender_email: Gmail address to send from
            sender_password: Gmail app-specific password
            receiver_email: Email address to send reports to
            email_log_store: Optional EmailLogStore instance for logging
        """
        self.sender_email = sender_email
        self.sender_password = sender_password
        self.receiver_email = receiver_email
        self.email_log_store = email_log_store

    def send_timesheet_report(
        self,
        attendance_store: AttendanceStore,
        date: Optional[dt.date] = None,
    ) -> bool:
        """
        Generate and send daily timesheet report for all students.

        Args:
            attendance_store: AttendanceStore instance to query data from
            date: Date to generate report for (defaults to today)

        Returns:
            True if email sent successfully, False otherwise
        """
        if date is None:
            date = dt.datetime.now().astimezone().date()

        # Get attendance report for the specified date
        year = date.strftime("%Y")
        month = date.strftime("%m")
        day = date.strftime("%d")

        records = attendance_store.attendance_report(year=year, month=month, day=day)

        # Generate email content
        subject = f"Daily Timesheet Report - {date.strftime('%Y-%m-%d')}"
        html_body = self._generate_html_report(records, date)
        text_body = self._generate_text_report(records, date)

        try:
            success = self._send_email(subject, html_body, text_body)
            # Log the result
            if self.email_log_store:
                self.email_log_store.log_email(
                    receiver_email=self.receiver_email,
                    report_date=date,
                    status="success" if success else "failed",
                    student_count=len(records),
                    error_message=None if success else "Email sending failed",
                )
            return success
        except Exception as e:
            error_msg = str(e)
            print(f"[ERROR] Failed to send email: {e}")
            # Log the failure
            if self.email_log_store:
                try:
                    self.email_log_store.log_email(
                        receiver_email=self.receiver_email,
                        report_date=date,
                        status="failed",
                        student_count=len(records),
                        error_message=error_msg,
                    )
                except Exception as log_error:
                    print(f"[ERROR] Failed to log email error: {log_error}")
            return False

    def _generate_html_report(
        self,
        records: List[dict],
        date: dt.date,
    ) -> str:
        """Generate HTML email body with timesheet table."""
        date_str = date.strftime("%B %d, %Y")
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                h1 {{
                    color: #2c3e50;
                    border-bottom: 3px solid #3498db;
                    padding-bottom: 10px;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                th {{
                    background-color: #3498db;
                    color: white;
                    padding: 12px;
                    text-align: left;
                    font-weight: bold;
                }}
                td {{
                    padding: 10px;
                    border-bottom: 1px solid #ddd;
                }}
                tr:nth-child(even) {{
                    background-color: #f9f9f9;
                }}
                tr:hover {{
                    background-color: #f5f5f5;
                }}
                .no-data {{
                    text-align: center;
                    color: #999;
                    font-style: italic;
                    padding: 20px;
                }}
                .summary {{
                    background-color: #ecf0f1;
                    padding: 15px;
                    border-radius: 5px;
                    margin: 20px 0;
                }}
            </style>
        </head>
        <body>
            <h1>Daily Timesheet Report</h1>
            <p><strong>Date:</strong> {date_str}</p>
            
            <div class="summary">
                <p><strong>Total Students:</strong> {len(records)}</p>
            </div>
            
            <table>
                <thead>
                    <tr>
                        <th>No.</th>
                        <th>Student Name</th>
                        <th>First Check In</th>
                        <th>Last Check Out</th>
                    </tr>
                </thead>
                <tbody>
        """

        if not records:
            html += """
                    <tr>
                        <td colspan="4" class="no-data">No attendance records for this date</td>
                    </tr>
            """
        else:
            for idx, record in enumerate(records, 1):
                student_name = record.get("student_name", "Unknown")
                first_check_in = self._format_datetime(record.get("first_check_in"))
                last_check_out = self._format_datetime(record.get("last_check_out"))
                
                html += f"""
                    <tr>
                        <td>{idx}</td>
                        <td>{student_name}</td>
                        <td>{first_check_in}</td>
                        <td>{last_check_out}</td>
                    </tr>
                """

        html += """
                </tbody>
            </table>
            
            <p style="color: #7f8c8d; font-size: 0.9em; margin-top: 30px;">
                This is an automated email from the Student Attendance Monitoring System.
            </p>
        </body>
        </html>
        """
        return html

    def _generate_text_report(
        self,
        records: List[dict],
        date: dt.date,
    ) -> str:
        """Generate plain text email body."""
        date_str = date.strftime("%B %d, %Y")
        
        text = f"Daily Timesheet Report\n"
        text += f"Date: {date_str}\n"
        text += f"\nTotal Students: {len(records)}\n"
        text += "\n" + "=" * 80 + "\n\n"

        if not records:
            text += "No attendance records for this date.\n"
        else:
            text += f"{'No.':<5} {'Student Name':<30} {'First Check In':<25} {'Last Check Out':<25}\n"
            text += "-" * 80 + "\n"
            
            for idx, record in enumerate(records, 1):
                student_name = record.get("student_name", "Unknown")
                first_check_in = self._format_datetime(record.get("first_check_in"))
                last_check_out = self._format_datetime(record.get("last_check_out"))
                
                text += f"{idx:<5} {student_name:<30} {first_check_in:<25} {last_check_out:<25}\n"

        text += "\n" + "=" * 80 + "\n"
        text += "\nThis is an automated email from the Student Attendance Monitoring System.\n"
        
        return text

    def _format_datetime(self, timestamp_str: Optional[str]) -> str:
        """Format ISO timestamp string to readable format."""
        if not timestamp_str:
            return "--"
        try:
            dt_obj = dt.datetime.fromisoformat(timestamp_str)
            return dt_obj.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return timestamp_str or "--"

    def _send_email(
        self,
        subject: str,
        html_body: str,
        text_body: str,
    ) -> bool:
        """
        Send email via Gmail SMTP.

        Args:
            subject: Email subject
            html_body: HTML email body
            text_body: Plain text email body

        Returns:
            True if sent successfully, False otherwise
        """
        if not self.sender_email or not self.sender_password or not self.receiver_email:
            print("[WARN] Email configuration incomplete. Cannot send email.")
            return False

        try:
            # Create message
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = self.sender_email
            msg["To"] = self.receiver_email

            # Attach both plain text and HTML versions
            part1 = MIMEText(text_body, "plain")
            part2 = MIMEText(html_body, "html")

            msg.attach(part1)
            msg.attach(part2)

            # Send email
            with smtplib.SMTP(self.GMAIL_SMTP_SERVER, self.GMAIL_SMTP_PORT) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)

            print(f"[INFO] Timesheet email sent successfully to {self.receiver_email}")
            return True

        except smtplib.SMTPAuthenticationError as e:
            error_msg = f"SMTP authentication failed: {e}"
            print(f"[ERROR] {error_msg}")
            raise Exception(error_msg) from e
        except smtplib.SMTPException as e:
            error_msg = f"SMTP error: {e}"
            print(f"[ERROR] {error_msg}")
            raise Exception(error_msg) from e
        except Exception as e:
            error_msg = f"Failed to send email: {e}"
            print(f"[ERROR] {error_msg}")
            raise

