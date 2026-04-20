from __future__ import annotations

import os
import smtplib
from email.message import EmailMessage
from pathlib import Path

from pain_monitoring.config import PainMonitoringConfig


def _resolve_sender(config: PainMonitoringConfig) -> str:
    return (
        config.notification_email_from.strip()
        or os.getenv("PAIN_MONITOR_EMAIL_FROM", "").strip()
    )


def _resolve_password(config: PainMonitoringConfig) -> str:
    return (
        config.notification_email_password.strip()
        or os.getenv("PAIN_MONITOR_EMAIL_PASSWORD", "").strip()
    )


def _resolve_recipient(config: PainMonitoringConfig) -> str:
    return (
        config.notification_email_to.strip()
        or os.getenv("PAIN_MONITOR_EMAIL_TO", "").strip()
    )


def _parse_recipients(raw_value: str) -> list[str]:
    return [item.strip() for item in raw_value.replace(";", ",").split(",") if item.strip()]


def resolve_recipient_list(config: PainMonitoringConfig) -> list[str]:
    return _parse_recipients(_resolve_recipient(config))


def email_notifications_ready(config: PainMonitoringConfig) -> bool:
    if not config.email_notifications_enabled:
        return False
    return bool(_resolve_sender(config) and _resolve_password(config) and resolve_recipient_list(config))


def _build_message(subject: str, body: str, sender: str, recipients: list[str], attachments: list[Path] | None = None) -> EmailMessage:
    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = sender
    message["To"] = ", ".join(recipients)
    message.set_content(body)

    for attachment in attachments or []:
        if not attachment.exists() or not attachment.is_file():
            continue
        data = attachment.read_bytes()
        message.add_attachment(data, maintype="application", subtype="octet-stream", filename=attachment.name)
    return message


def send_email_notification(
    config: PainMonitoringConfig,
    subject: str,
    body: str,
    attachments: list[Path] | None = None,
) -> tuple[bool, str]:
    if not email_notifications_ready(config):
        return False, "Email notifications are not fully configured."

    sender = _resolve_sender(config)
    password = _resolve_password(config)
    recipients = resolve_recipient_list(config)
    message = _build_message(subject, body, sender, recipients, attachments)

    try:
        if config.smtp_use_tls:
            with smtplib.SMTP(config.smtp_host, config.smtp_port, timeout=20) as server:
                server.starttls()
                server.login(sender, password)
                server.send_message(message)
        else:
            with smtplib.SMTP_SSL(config.smtp_host, config.smtp_port, timeout=20) as server:
                server.login(sender, password)
                server.send_message(message)
    except Exception as exc:
        return False, str(exc)
    return True, ", ".join(recipients)


def build_alert_subject(patient_id: int, alert_type: str) -> str:
    return f"Patient {patient_id} {alert_type} alert"


def build_alert_body(
    patient_id: int,
    alert_type: str,
    pain_score: float,
    wheeze_probability: float,
    calibration_text: str,
) -> str:
    return (
        f"Patient ID: {patient_id}\n"
        f"Alert type: {alert_type}\n"
        f"Pain score: {pain_score:.2f}/10\n"
        f"Wheeze probability: {wheeze_probability:.2f}\n"
        f"Monitor status: {calibration_text}\n"
    )


def build_session_report_subject(patient_id: int) -> str:
    return f"Patient {patient_id} monitoring report"


def build_session_report_body(patient_id: int, summary: dict) -> str:
    return (
        f"Patient ID: {patient_id}\n"
        f"Rows logged: {summary.get('rows', 0)}\n"
        f"Episodes detected: {summary.get('episodes_detected', 0)}\n"
        f"Total pain duration: {summary.get('total_pain_duration_s', 0.0):.2f}s\n"
        f"Mean pain score: {summary.get('mean_pain_score', 0.0):.2f}\n"
        f"Max pain score: {summary.get('max_pain_score', 0.0):.2f}\n"
        f"Mean wheeze probability: {summary.get('mean_wheeze_probability', 0.0):.2f}\n"
        f"Max wheeze probability: {summary.get('max_wheeze_probability', 0.0):.2f}\n"
        f"Session CSV: {summary.get('session_csv', '')}\n"
    )
