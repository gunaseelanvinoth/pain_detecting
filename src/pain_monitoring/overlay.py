from __future__ import annotations

import cv2

from pain_monitoring.types import DurationStatus, FramePainFeatures


def level_from_score(score_0_10: float) -> str:
    if score_0_10 >= 7.0:
        return "Severe"
    if score_0_10 >= 4.5:
        return "Moderate"
    if score_0_10 >= 2.5:
        return "Mild"
    return "None"


def draw_overlay(
    frame,
    features: FramePainFeatures,
    score_0_10: float,
    level: str,
    duration: DurationStatus,
    calibration_text: str = "",
) -> None:
    color = (0, 220, 0)
    if level == "Mild":
        color = (0, 200, 255)
    elif level in {"Moderate", "Severe"}:
        color = (0, 80, 255)

    if features.face_detected and features.face_box is not None:
        x, y, w, h = features.face_box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            frame,
            f"Pain {score_0_10:0.1f}/10 ({level})",
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )

    cv2.rectangle(frame, (10, 10), (650, 245), (20, 20, 20), -1)
    cv2.putText(frame, "Patient Pain Detection & Duration Monitor", (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.63, (255, 255, 255), 2)
    cv2.putText(frame, f"Pain score: {score_0_10:0.2f} / 10", (20, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.60, color, 2)
    cv2.putText(frame, f"Pain level: {level}", (20, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.60, color, 2)
    cv2.putText(
        frame,
        f"Face analysis: {'TRACKED' if features.face_detected else 'NOT DETECTED'}",
        (20, 122),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (220, 220, 220),
        2,
    )
    cv2.putText(frame, f"Pain active: {'YES' if duration.pain_active else 'NO'}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (220, 220, 220), 2)
    cv2.putText(frame, f"Current episode: {duration.current_episode_duration_s:0.1f} s", (20, 178), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (220, 220, 220), 2)
    cv2.putText(frame, f"Total pain duration: {duration.total_pain_duration_s:0.1f} s", (20, 206), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (180, 255, 180), 2)
    if calibration_text:
        cv2.putText(frame, calibration_text, (20, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
