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
    pain_start_threshold: float = 4.5
    pain_end_threshold: float = 3.0
    start_hold_seconds: float = 1.4
    end_hold_seconds: float = 1.8
    smoothing_alpha: float = 0.45
    prediction_smoothing_alpha: float = 0.35
    calibration_seconds: float = 6.0
    calibration_min_samples: int = 25
    neutral_anchor_score: float = 1.0
    save_live_data: bool = True
    log_every_n_frames: int = 4
    patient_id: int = 1

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
