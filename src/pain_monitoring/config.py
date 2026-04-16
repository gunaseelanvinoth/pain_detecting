from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class PainMonitoringConfig:
    camera_index: int = 0
    frame_width: int = 960
    frame_height: int = 540
    min_face_area: int = 6000
    pain_start_threshold: float = 3.0
    pain_end_threshold: float = 2.0
    start_hold_seconds: float = 0.6
    end_hold_seconds: float = 1.0
    smoothing_alpha: float = 0.45
    prediction_smoothing_alpha: float = 0.35
    calibration_seconds: float = 6.0
    calibration_min_samples: int = 25
    neutral_anchor_score: float = 1.0
    save_live_data: bool = True
    log_every_n_frames: int = 4
    patient_id: int = 1
    enable_audio_monitoring: bool = True
    audio_window_seconds: float = 1.5
    audio_sample_rate: int = 16000
    audio_channels: int = 1
    dataset_augmentation_factor: int = 20
    training_ridge_alpha: float = 0.8
    overlay_scale: float = 0.85
    overlay_anchor: str = "top_right"
    pain_display_alpha_still: float = 0.24
    pain_display_alpha_change: float = 0.68
    wheeze_display_alpha: float = 0.34
    expression_change_threshold: float = 0.08
    pain_expression_boost: float = 3.2
    micro_expression_trigger_threshold: float = 0.18
    wheeze_support_boost: float = 0.22
    wheeze_alert_threshold: float = 0.30
    notification_cooldown_seconds: float = 45.0
    email_notifications_enabled: bool = False
    notification_email_to: str = ""
    notification_email_from: str = ""
    notification_email_password: str = ""
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_use_tls: bool = True
    email_send_instant_alerts: bool = True
    email_send_session_report: bool = True

    @staticmethod
    def from_json(path: Path | None) -> "PainMonitoringConfig":
        if path is None:
            return PainMonitoringConfig()
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        defaults = asdict(PainMonitoringConfig())
        defaults.update(payload)
        config = PainMonitoringConfig(**defaults)
        config.validate()
        return config

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    def validate(self) -> None:
        if self.frame_width <= 0 or self.frame_height <= 0:
            raise ValueError("Frame dimensions must be positive.")
        if self.min_face_area <= 0:
            raise ValueError("min_face_area must be positive.")
        if not (0.0 <= self.smoothing_alpha <= 1.0):
            raise ValueError("smoothing_alpha must be between 0 and 1.")
        if not (0.0 <= self.prediction_smoothing_alpha <= 1.0):
            raise ValueError("prediction_smoothing_alpha must be between 0 and 1.")
        if not (0.0 <= self.pain_end_threshold <= self.pain_start_threshold <= 10.0):
            raise ValueError("Thresholds must satisfy 0 <= end <= start <= 10.")
        if self.start_hold_seconds < 0.0 or self.end_hold_seconds < 0.0:
            raise ValueError("Hold durations must be non-negative.")
        if self.calibration_seconds < 0.0:
            raise ValueError("calibration_seconds must be non-negative.")
        if self.calibration_min_samples < 1:
            raise ValueError("calibration_min_samples must be >= 1.")
        if not (0.0 <= self.neutral_anchor_score <= 10.0):
            raise ValueError("neutral_anchor_score must be between 0 and 10.")
        if self.log_every_n_frames <= 0:
            raise ValueError("log_every_n_frames must be >= 1.")
        if self.audio_window_seconds <= 0.0:
            raise ValueError("audio_window_seconds must be positive.")
        if self.audio_sample_rate < 4000:
            raise ValueError("audio_sample_rate must be >= 4000.")
        if self.audio_channels < 1:
            raise ValueError("audio_channels must be >= 1.")
        if self.dataset_augmentation_factor < 0:
            raise ValueError("dataset_augmentation_factor must be >= 0.")
        if self.training_ridge_alpha < 0.0:
            raise ValueError("training_ridge_alpha must be >= 0.")
        if self.overlay_scale <= 0.4:
            raise ValueError("overlay_scale must be > 0.4.")
        if self.overlay_anchor not in {"top_left", "top_right"}:
            raise ValueError("overlay_anchor must be 'top_left' or 'top_right'.")
        if not (0.0 <= self.pain_display_alpha_still <= 1.0):
            raise ValueError("pain_display_alpha_still must be between 0 and 1.")
        if not (0.0 <= self.pain_display_alpha_change <= 1.0):
            raise ValueError("pain_display_alpha_change must be between 0 and 1.")
        if not (0.0 <= self.wheeze_display_alpha <= 1.0):
            raise ValueError("wheeze_display_alpha must be between 0 and 1.")
        if not (0.0 <= self.expression_change_threshold <= 1.0):
            raise ValueError("expression_change_threshold must be between 0 and 1.")
        if self.pain_expression_boost < 0.0:
            raise ValueError("pain_expression_boost must be >= 0.")
        if not (0.0 <= self.micro_expression_trigger_threshold <= 1.0):
            raise ValueError("micro_expression_trigger_threshold must be between 0 and 1.")
        if self.wheeze_support_boost < 0.0:
            raise ValueError("wheeze_support_boost must be >= 0.")
        if not (0.0 <= self.wheeze_alert_threshold <= 1.0):
            raise ValueError("wheeze_alert_threshold must be between 0 and 1.")
        if self.notification_cooldown_seconds < 0.0:
            raise ValueError("notification_cooldown_seconds must be >= 0.")
        if not (1 <= self.smtp_port <= 65535):
            raise ValueError("smtp_port must be between 1 and 65535.")
