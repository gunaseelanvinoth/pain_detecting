from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FramePainFeatures:
    face_detected: bool
    face_box: Optional[tuple[int, int, int, int]]
    eye_closure: float
    brow_tension: float
    mouth_tension: float
    smile_absence: float
    motion_score: float
    eye_symmetry: float = 0.0
    brow_energy: float = 0.0
    mouth_opening: float = 0.0
    lower_face_motion: float = 0.0
    face_edge_density: float = 0.0
    nasal_tension: float = 0.0
    respiratory_motion: float = 0.0
    wheeze_tonality: float = 0.0
    wheeze_band_energy: float = 0.0
    wheeze_entropy: float = 0.0
    wheeze_probability: float = 0.0


@dataclass
class PainPrediction:
    score_0_10: float
    level: str
    is_pain: bool


@dataclass
class EpisodeState:
    episode_id: int
    start_time: float
    last_time: float
    max_score: float
    score_sum: float
    frame_count: int


@dataclass
class EpisodeSummary:
    episode_id: int
    start_time: float
    end_time: float
    duration_seconds: float
    max_score: float
    avg_score: float


@dataclass
class DurationStatus:
    pain_active: bool
    active_episode_id: Optional[int]
    current_episode_duration_s: float
    total_pain_duration_s: float
    started_now: bool = False
    ended_now: bool = False
    finished_episode: Optional[EpisodeSummary] = None


@dataclass
class RuntimeState:
    previous_gray: Optional[object] = None
    previous_face_box: Optional[tuple[int, int, int, int]] = None
    previous_timestamp: Optional[float] = None
    smoothed_score: float = 0.0
    smoothed_pain_score: float = 0.0
    smoothed_wheeze_probability: float = 0.0
    previous_raw_pain_score: float = 0.0
    calibration_scores: list[float] = field(default_factory=list)
    baseline_score: Optional[float] = None
    time_above_start: float = 0.0
    time_below_end: float = 0.0
    active_episode: Optional[EpisodeState] = None
    closed_duration_s: float = 0.0
    closed_episodes: list[EpisodeSummary] = field(default_factory=list)
    episode_counter: int = 0
