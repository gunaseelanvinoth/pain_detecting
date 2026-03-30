from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from pain_monitoring.types import DurationStatus, FramePainFeatures


class PainLiveLogger:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_file = self.output_dir / f"pain_session_{stamp}.csv"
        self.episode_file = self.output_dir / f"pain_episode_events_{stamp}.csv"
        self._write_header()
        self._write_episode_header()

    def _write_header(self) -> None:
        with self.output_file.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "timestamp",
                    "frame_idx",
                    "patient_id",
                    "pain_score_0_10",
                    "pain_level",
                    "pain_active",
                    "active_episode_id",
                    "current_episode_duration_s",
                    "total_pain_duration_s",
                    "eye_closure",
                    "brow_tension",
                    "mouth_tension",
                    "smile_absence",
                    "motion_score",
                    "face_detected",
                ]
            )

    def _write_episode_header(self) -> None:
        with self.episode_file.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "timestamp",
                    "event",
                    "episode_id",
                    "start_time",
                    "end_time",
                    "duration_seconds",
                    "max_score",
                    "avg_score",
                ]
            )

    def log(
        self,
        frame_idx: int,
        patient_id: int,
        score_0_10: float,
        level: str,
        features: FramePainFeatures,
        duration: DurationStatus,
    ) -> None:
        now_iso = datetime.now().isoformat(timespec="seconds")

        with self.output_file.open("a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    now_iso,
                    frame_idx,
                    patient_id,
                    f"{score_0_10:.3f}",
                    level,
                    int(duration.pain_active),
                    duration.active_episode_id if duration.active_episode_id is not None else "",
                    f"{duration.current_episode_duration_s:.3f}",
                    f"{duration.total_pain_duration_s:.3f}",
                    f"{features.eye_closure:.4f}",
                    f"{features.brow_tension:.4f}",
                    f"{features.mouth_tension:.4f}",
                    f"{features.smile_absence:.4f}",
                    f"{features.motion_score:.4f}",
                    int(features.face_detected),
                ]
            )

        if duration.started_now:
            self._log_episode_event(now_iso, "START", duration.active_episode_id, None)
        if duration.ended_now and duration.finished_episode is not None:
            self._log_episode_event(now_iso, "END", duration.finished_episode.episode_id, duration.finished_episode)

    def _log_episode_event(self, timestamp: str, event: str, episode_id: int | None, summary) -> None:
        with self.episode_file.open("a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    timestamp,
                    event,
                    episode_id if episode_id is not None else "",
                    "" if summary is None else f"{summary.start_time:.4f}",
                    "" if summary is None else f"{summary.end_time:.4f}",
                    "" if summary is None else f"{summary.duration_seconds:.4f}",
                    "" if summary is None else f"{summary.max_score:.4f}",
                    "" if summary is None else f"{summary.avg_score:.4f}",
                ]
            )
