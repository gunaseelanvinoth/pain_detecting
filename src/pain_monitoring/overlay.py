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


def wheeze_level_from_probability(probability: float) -> str:
    if probability >= 0.75:
        return "High"
    if probability >= 0.45:
        return "Medium"
    if probability >= 0.2:
        return "Low"
    return "None"


def pain_detection_label(level: str) -> str:
    return "Pain Detected" if level != "None" else "No Pain"


def wheeze_detection_label(wheeze_level: str) -> str:
    return "Wheezing Detected" if wheeze_level != "None" else "No Wheezing"


def _panel_origin(frame, panel_width: int, anchor: str) -> tuple[int, int]:
    margin = 12
    x = margin if anchor == "top_left" else max(margin, frame.shape[1] - panel_width - margin)
    return x, margin


def draw_overlay(
    frame,
    features: FramePainFeatures,
    score_0_10: float,
    level: str,
    duration: DurationStatus,
    wheeze_level: str = "None",
    calibration_text: str = "",
    overlay_scale: float = 0.85,
    overlay_anchor: str = "top_right",
) -> None:
    color = (0, 220, 0)
    if level == "Mild":
        color = (0, 200, 255)
    elif level in {"Moderate", "Severe"}:
        color = (0, 80, 255)

    pain_status = pain_detection_label(level)
    wheeze_status = wheeze_detection_label(wheeze_level)

    if features.face_detected and features.face_box is not None:
        x, y, w, h = features.face_box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            frame,
            f"{pain_status} | {score_0_10:0.1f}/10",
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )

    font_scale = max(0.45, 0.62 * overlay_scale)
    small_scale = max(0.38, 0.54 * overlay_scale)
    line_gap = int(34 * overlay_scale)
    panel_width = int(300 * overlay_scale + 70)
    panel_height = int(205 * overlay_scale + 40)
    panel_x, panel_y = _panel_origin(frame, panel_width, overlay_anchor)

    panel = frame.copy()
    cv2.rectangle(panel, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (18, 18, 18), -1)
    cv2.addWeighted(panel, 0.56, frame, 0.44, 0, frame)

    text_x = panel_x + 14
    y = panel_y + 24
    cv2.putText(frame, "Pain + Wheeze Monitor", (text_x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
    y += line_gap
    cv2.putText(frame, f"Pain status: {pain_status}", (text_x, y), cv2.FONT_HERSHEY_SIMPLEX, small_scale, color, 2)
    y += line_gap
    cv2.putText(frame, f"Pain score: {score_0_10:0.2f}/10 ({level})", (text_x, y), cv2.FONT_HERSHEY_SIMPLEX, small_scale, color, 2)
    y += line_gap
    cv2.putText(
        frame,
        f"Wheeze: {wheeze_status} ({features.wheeze_probability:0.2f} | {wheeze_level})",
        (text_x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        small_scale,
        (255, 230, 120),
        2,
    )
    y += line_gap
    cv2.putText(frame, f"Face: {'TRACKED' if features.face_detected else 'NOT DETECTED'}", (text_x, y), cv2.FONT_HERSHEY_SIMPLEX, small_scale, (220, 220, 220), 2)
    y += line_gap
    cv2.putText(frame, f"Pain active: {'YES' if duration.pain_active else 'NO'}", (text_x, y), cv2.FONT_HERSHEY_SIMPLEX, small_scale, (220, 220, 220), 2)
    y += line_gap
    cv2.putText(frame, f"Episode: {duration.current_episode_duration_s:0.1f}s", (text_x, y), cv2.FONT_HERSHEY_SIMPLEX, small_scale, (220, 220, 220), 2)
    y += line_gap
    cv2.putText(frame, f"Total duration: {duration.total_pain_duration_s:0.1f}s", (text_x, y), cv2.FONT_HERSHEY_SIMPLEX, small_scale, (180, 255, 180), 2)
    if calibration_text:
        cv2.putText(frame, calibration_text, (text_x, panel_y + panel_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.37 + 0.08 * overlay_scale, (180, 180, 180), 1)
