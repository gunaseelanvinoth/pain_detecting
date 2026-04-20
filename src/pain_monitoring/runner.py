from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from pain_monitoring.audio_features import FileAudioFeatureProvider, MicrophoneAudioFeatureProvider
from pain_monitoring.config import PainMonitoringConfig
from pain_monitoring.dataset import prepare_training_dataset
from pain_monitoring.episode_tracker import update_duration_state
from pain_monitoring.io_utils import write_json
from pain_monitoring.logger import PainLiveLogger
from pain_monitoring.model import FEATURE_COLUMNS, TARGET_COLUMN, WHEEZE_TARGET_COLUMN, PainLinearModel, train_linear_model_from_frame
from pain_monitoring.notifications import (
    build_alert_body,
    build_alert_subject,
    build_session_report_body,
    build_session_report_subject,
    email_notifications_ready,
    send_email_notification,
)
from pain_monitoring.reporting import summarize_session_csv
from pain_monitoring.types import RuntimeState


def _smooth_value(current_smoothed: float, new_value: float, alpha: float) -> float:
    if current_smoothed <= 0.0:
        return float(new_value)
    return float(alpha * new_value + (1.0 - alpha) * current_smoothed)


def _expression_change_strength(raw_score: float, previous_raw_score: float, features) -> float:
    expressive_motion = (
        0.28 * features.lower_face_motion
        + 0.22 * features.motion_score
        + 0.20 * features.mouth_micro_motion
        + 0.16 * max(features.brow_motion, features.eyebrow_contraction)
        + 0.14 * features.mouth_opening
    )
    score_delta = abs(raw_score - previous_raw_score)
    return float(np.clip(expressive_motion + 0.12 * score_delta, 0.0, 1.0))


def _estimate_expression_pain_boost(features, change_strength: float, config: PainMonitoringConfig) -> float:
    facial_signal = (
        0.24 * features.eye_closure
        + 0.18 * features.brow_tension
        + 0.18 * features.brow_energy
        + 0.14 * features.brow_motion
        + 0.20 * features.eyebrow_contraction
        + 0.12 * features.eyebrow_angle_score
        + 0.18 * features.eyebrow_pain_confidence
        + 0.08 * features.brow_position
        + 0.17 * features.mouth_tension
        + 0.16 * features.mouth_opening
        + 0.16 * features.mouth_micro_motion
        + 0.15 * features.lower_face_motion
        + 0.13 * features.face_edge_density
        + 0.10 * features.eye_symmetry
        + 0.10 * features.nasal_tension
        + 0.10 * features.nose_contrast
    )
    trigger = max(change_strength, facial_signal)
    if trigger < config.micro_expression_trigger_threshold:
        return 0.0
    return float(np.clip((trigger - config.micro_expression_trigger_threshold) * config.pain_expression_boost, 0.0, 4.0))


def _estimate_brow_edge_boost(features, config: PainMonitoringConfig) -> float:
    if (
        features.brow_tension < config.calm_brow_threshold
        and features.brow_energy < config.calm_brow_threshold
        and features.brow_motion < config.calm_brow_threshold
        and features.face_edge_density < config.calm_edge_threshold
        and features.nose_contrast < config.calm_edge_threshold
    ):
        return 0.0
    brow_edge_signal = (
        0.26 * features.brow_tension
        + 0.22 * features.brow_energy
        + 0.16 * features.brow_motion
        + 0.20 * features.eyebrow_contraction
        + 0.12 * features.eyebrow_angle_score
        + 0.16 * features.eyebrow_pain_confidence
        + 0.16 * features.face_edge_density
        + 0.12 * features.eye_symmetry
        + 0.08 * features.nasal_tension
        + 0.08 * features.nose_contrast
    )
    return float(np.clip(brow_edge_signal * config.brow_edge_pain_boost, 0.0, 1.5))


def _facial_pain_evidence(features) -> float:
    return float(
        np.clip(
            0.20 * features.brow_tension
            + 0.18 * features.brow_energy
            + 0.22 * features.eyebrow_contraction
            + 0.10 * features.eyebrow_angle_score
            + 0.18 * features.eyebrow_pain_confidence
            + 0.16 * features.brow_position
            + 0.14 * features.mouth_tension
            + 0.12 * features.mouth_opening
            + 0.10 * features.nose_contrast
            + 0.06 * features.nasal_tension
            + 0.04 * features.eye_closure,
            0.0,
            1.0,
        )
    )


def _relative_facial_pain_evidence(features, baseline_facial_evidence: float | None) -> float:
    evidence = _facial_pain_evidence(features)
    baseline = baseline_facial_evidence if baseline_facial_evidence is not None else 0.0
    return float(np.clip(evidence - baseline, 0.0, 1.0))


def _pain_region_count(features, config: PainMonitoringConfig, baseline_facial_evidence: float | None = None) -> int:
    relative_evidence = _relative_facial_pain_evidence(features, baseline_facial_evidence)
    brow_active = (
        features.eyebrow_pain_confidence >= config.eyebrow_pain_confidence_threshold
        or
        features.brow_motion >= config.neutral_motion_threshold
        or features.brow_tension >= config.calm_brow_threshold + 0.08
        or features.brow_energy >= config.calm_brow_threshold + 0.08
    )
    mouth_active = (
        features.mouth_micro_motion >= config.neutral_motion_threshold
        or features.mouth_opening >= config.neutral_expression_signal_threshold
        or features.mouth_tension >= config.neutral_expression_signal_threshold + 0.06
    )
    nose_active = (
        features.nose_contrast >= config.calm_edge_threshold + 0.08
        or features.nasal_tension >= config.neutral_expression_signal_threshold
    )
    eye_active = features.eye_closure >= 0.55
    relative_active = relative_evidence >= config.neutral_relative_evidence_margin
    return sum([brow_active, mouth_active, nose_active, eye_active, relative_active])


def _wheeze_evidence(features, probability: float) -> float:
    return float(
        np.clip(
            0.36 * probability
            + 0.26 * features.wheeze_band_energy
            + 0.18 * features.wheeze_tonality
            + 0.12 * features.respiratory_motion
            + 0.08 * features.wheeze_entropy,
            0.0,
            1.0,
        )
    )


def _update_evidence_seconds(current_seconds: float, evidence: float, threshold: float, dt: float) -> float:
    if evidence >= threshold:
        return float(min(60.0, current_seconds + max(0.0, dt)))
    return float(max(0.0, current_seconds - max(0.0, dt) * 1.5))


def _estimate_sustained_pain_boost(
    features,
    evidence_seconds: float,
    config: PainMonitoringConfig,
    baseline_facial_evidence: float | None = None,
) -> float:
    evidence = _relative_facial_pain_evidence(features, baseline_facial_evidence)
    if evidence < config.sustained_pain_signal_threshold:
        return 0.0
    hold_ratio = np.clip(evidence_seconds / config.sustained_pain_seconds_to_full_boost, 0.0, 1.0)
    intensity_ratio = np.clip((evidence - config.sustained_pain_signal_threshold) / 0.45, 0.0, 1.0)
    return float(np.clip(config.sustained_pain_boost * hold_ratio * (0.45 + 0.55 * intensity_ratio), 0.0, config.sustained_pain_boost))


def _is_neutral_expression(
    features,
    change_strength: float,
    config: PainMonitoringConfig,
    baseline_facial_evidence: float | None = None,
) -> bool:
    if not features.face_detected:
        return True
    evidence = _facial_pain_evidence(features)
    relative_evidence = _relative_facial_pain_evidence(features, baseline_facial_evidence)
    return (
        (evidence < config.neutral_expression_signal_threshold or relative_evidence < config.neutral_relative_evidence_margin)
        and change_strength < config.neutral_motion_threshold
        and features.brow_motion < config.neutral_motion_threshold
        and features.eyebrow_pain_confidence < config.eyebrow_pain_confidence_threshold
        and features.mouth_micro_motion < config.neutral_motion_threshold
        and features.lower_face_motion < config.neutral_motion_threshold
        and features.nose_contrast < config.calm_edge_threshold
        and features.eye_closure < 0.45
    )


def _is_confirmed_eyebrow_pain(
    features,
    config: PainMonitoringConfig,
    baseline_eyebrow_distance_ratio: float | None = None,
) -> bool:
    if features.eyebrow_pain_confidence < config.eyebrow_pain_confidence_threshold:
        return False
    if features.eyebrow_distance_ratio <= 0.0:
        return False
    if baseline_eyebrow_distance_ratio is not None and baseline_eyebrow_distance_ratio > 0.0:
        required_drop = config.eyebrow_contraction_drop_threshold * 0.55
        distance_confirmed = features.eyebrow_distance_ratio <= baseline_eyebrow_distance_ratio - required_drop
    else:
        distance_confirmed = features.eyebrow_distance_ratio <= config.eyebrow_distance_pain_threshold * 0.75
    angle_confirmed = features.eyebrow_angle_score >= 0.45
    return distance_confirmed or (features.eyebrow_contraction >= 0.75 and angle_confirmed)


def _apply_neutral_face_guard(
    score: float,
    features,
    change_strength: float,
    config: PainMonitoringConfig,
    baseline_facial_evidence: float | None = None,
    baseline_eyebrow_distance_ratio: float | None = None,
) -> float:
    guarded_score = float(score)
    if _is_confirmed_eyebrow_pain(features, config, baseline_eyebrow_distance_ratio):
        eyebrow_score = config.pain_start_threshold + max(0.2, config.eyebrow_pain_score_boost * features.eyebrow_pain_confidence)
        return min(10.0, max(guarded_score, eyebrow_score))
    if _pain_region_count(features, config, baseline_facial_evidence) < config.minimum_pain_regions_required:
        guarded_score = min(guarded_score, config.single_region_score_cap)
    if _is_neutral_expression(features, change_strength, config, baseline_facial_evidence):
        guarded_score = min(guarded_score, config.neutral_face_score_cap)
    return guarded_score


def _estimate_sustained_wheeze_boost(features, probability: float, evidence_seconds: float, config: PainMonitoringConfig) -> float:
    evidence = _wheeze_evidence(features, probability)
    if evidence < config.sustained_wheeze_signal_threshold:
        return 0.0
    hold_ratio = np.clip(evidence_seconds / config.sustained_wheeze_seconds_to_full_boost, 0.0, 1.0)
    intensity_ratio = np.clip((evidence - config.sustained_wheeze_signal_threshold) / 0.50, 0.0, 1.0)
    return float(np.clip(config.sustained_wheeze_boost * hold_ratio * (0.40 + 0.60 * intensity_ratio), 0.0, config.sustained_wheeze_boost))


def _support_wheeze_probability(features, probability: float, config: PainMonitoringConfig) -> float:
    support_signal = (
        0.45 * probability
        + 0.25 * features.respiratory_motion
        + 0.20 * features.wheeze_band_energy
        + 0.10 * features.wheeze_tonality
    )
    boosted = probability + config.wheeze_support_boost * support_signal
    return float(np.clip(max(probability, boosted), 0.0, 1.0))


def _predict_array(model: PainLinearModel, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frame_rows = []
    for row in x:
        feature_map = {name: float(value) for name, value in zip(FEATURE_COLUMNS, row)}
        from pain_monitoring.types import FramePainFeatures

        features = FramePainFeatures(
            face_detected=True,
            face_box=None,
            eye_closure=feature_map.get("eye_closure", 0.0),
            brow_tension=feature_map.get("brow_tension", 0.0),
            mouth_tension=feature_map.get("mouth_tension", 0.0),
            smile_absence=feature_map.get("smile_absence", 0.0),
            motion_score=feature_map.get("motion_score", 0.0),
            eye_symmetry=feature_map.get("eye_symmetry", 0.0),
            brow_energy=feature_map.get("brow_energy", 0.0),
            brow_position=feature_map.get("brow_position", 0.0),
            brow_motion=feature_map.get("brow_motion", 0.0),
            eyebrow_distance_ratio=feature_map.get("eyebrow_distance_ratio", 0.0),
            eyebrow_contraction=feature_map.get("eyebrow_contraction", 0.0),
            eyebrow_pain_confidence=feature_map.get("eyebrow_pain_confidence", 0.0),
            mouth_opening=feature_map.get("mouth_opening", 0.0),
            mouth_micro_motion=feature_map.get("mouth_micro_motion", 0.0),
            lower_face_motion=feature_map.get("lower_face_motion", 0.0),
            face_edge_density=feature_map.get("face_edge_density", 0.0),
            nasal_tension=feature_map.get("nasal_tension", 0.0),
            nose_contrast=feature_map.get("nose_contrast", 0.0),
            respiratory_motion=feature_map.get("respiratory_motion", 0.0),
            wheeze_tonality=feature_map.get("wheeze_tonality", 0.0),
            wheeze_band_energy=feature_map.get("wheeze_band_energy", 0.0),
            wheeze_entropy=feature_map.get("wheeze_entropy", 0.0),
            wheeze_probability=feature_map.get("wheeze_probability", 0.0),
        )
        frame_rows.append((model.predict_score(features), model.predict_wheeze(features)))
    pain = np.array([item[0] for item in frame_rows], dtype=float)
    wheeze = np.array([item[1] for item in frame_rows], dtype=float)
    return pain, wheeze


def _make_audio_provider(audio_path: Path | None, config: PainMonitoringConfig):
    if not config.enable_audio_monitoring:
        return None
    if audio_path is not None:
        return FileAudioFeatureProvider(audio_path=audio_path, window_seconds=config.audio_window_seconds)
    try:
        return MicrophoneAudioFeatureProvider(
            sample_rate=config.audio_sample_rate,
            window_seconds=config.audio_window_seconds,
            channels=config.audio_channels,
        )
    except Exception:
        return None


def run_live_monitor(
    model_path: Path | None = None,
    video_path: Path | None = None,
    audio_path: Path | None = None,
    config: PainMonitoringConfig | None = None,
) -> dict:
    import cv2

    from pain_monitoring.features import EyebrowLandmarkDetector, apply_audio_snapshot, extract_frame_features
    from pain_monitoring.overlay import draw_overlay, level_from_score, wheeze_level_from_probability

    cfg = config or PainMonitoringConfig()
    cfg.validate()
    model = PainLinearModel.load(model_path) if model_path is not None and model_path.exists() else PainLinearModel()

    source = str(video_path) if video_path is not None else cfg.camera_index
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError("Could not open camera/video source. Check file path or camera permissions.")

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.frame_width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.frame_height)

    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    smile_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
    eyebrow_detector = EyebrowLandmarkDetector(enabled=cfg.enable_eyebrow_landmarks)

    runtime = RuntimeState()
    logger = PainLiveLogger(Path.cwd() / "sample_data") if cfg.save_live_data else None
    audio_provider = _make_audio_provider(audio_path=audio_path, config=cfg)

    frame_idx = 0
    processed = 0
    start_wall = time.time()
    calibration_done = cfg.calibration_seconds <= 0.0
    calibration_text = "Calibration skipped"
    latest_wheeze_probability = 0.0
    last_pain_alert_time = -cfg.notification_cooldown_seconds
    last_wheeze_alert_time = -cfg.notification_cooldown_seconds

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            now = time.time()
            elapsed = now - start_wall
            dt = 0.0 if runtime.previous_frame_timestamp is None else max(0.0, now - runtime.previous_frame_timestamp)
            runtime.previous_frame_timestamp = now
            features, gray = extract_frame_features(
                frame=frame,
                previous_gray=runtime.previous_gray,
                previous_face_box=runtime.previous_face_box,
                face_detector=face_detector,
                eye_detector=eye_detector,
                smile_detector=smile_detector,
                config=cfg,
                eyebrow_detector=eyebrow_detector,
                baseline_eyebrow_distance_ratio=runtime.baseline_eyebrow_distance_ratio,
            )
            if audio_provider is not None:
                snapshot = audio_provider.get_snapshot(elapsed)
                features = apply_audio_snapshot(features, snapshot)

            runtime.previous_gray = gray
            runtime.previous_face_box = features.face_box if features.face_detected else runtime.previous_face_box

            raw_score = model.predict_score(features)
            latest_wheeze_probability = model.predict_wheeze(features)
            latest_wheeze_probability = _support_wheeze_probability(features, latest_wheeze_probability, cfg)
            runtime.wheeze_evidence_seconds = _update_evidence_seconds(
                runtime.wheeze_evidence_seconds,
                _wheeze_evidence(features, latest_wheeze_probability),
                cfg.sustained_wheeze_signal_threshold,
                dt,
            )
            latest_wheeze_probability = np.clip(
                latest_wheeze_probability
                + _estimate_sustained_wheeze_boost(features, latest_wheeze_probability, runtime.wheeze_evidence_seconds, cfg),
                0.0,
                1.0,
            )
            features.wheeze_probability = latest_wheeze_probability

            elapsed_calibration = now - start_wall
            if not calibration_done:
                if features.face_detected:
                    runtime.calibration_scores.append(raw_score)
                    runtime.calibration_facial_evidence.append(_facial_pain_evidence(features))
                    runtime.calibration_wheeze_evidence.append(_wheeze_evidence(features, latest_wheeze_probability))
                    if features.eyebrow_distance_ratio > 0.0:
                        runtime.calibration_eyebrow_distance_ratios.append(features.eyebrow_distance_ratio)
                if elapsed_calibration >= cfg.calibration_seconds and len(runtime.calibration_scores) >= cfg.calibration_min_samples:
                    runtime.baseline_score = float(np.median(runtime.calibration_scores))
                    runtime.baseline_facial_evidence = float(np.median(runtime.calibration_facial_evidence))
                    runtime.baseline_wheeze_evidence = float(np.median(runtime.calibration_wheeze_evidence))
                    if runtime.calibration_eyebrow_distance_ratios:
                        runtime.baseline_eyebrow_distance_ratio = float(np.median(runtime.calibration_eyebrow_distance_ratios))
                    calibration_done = True

            adjusted_score = raw_score
            if runtime.baseline_score is not None:
                adjusted_score = np.clip(raw_score - runtime.baseline_score + cfg.neutral_anchor_score, 0.0, 10.0)

            change_strength = _expression_change_strength(adjusted_score, runtime.previous_raw_pain_score, features)
            runtime.facial_pain_evidence_seconds = _update_evidence_seconds(
                runtime.facial_pain_evidence_seconds,
                _relative_facial_pain_evidence(features, runtime.baseline_facial_evidence),
                cfg.sustained_pain_signal_threshold,
                dt,
            )
            adjusted_score = np.clip(
                adjusted_score
                + _estimate_expression_pain_boost(features, change_strength, cfg)
                + _estimate_brow_edge_boost(features, cfg),
                0.0,
                10.0,
            )
            adjusted_score = np.clip(
                adjusted_score + _estimate_sustained_pain_boost(features, runtime.facial_pain_evidence_seconds, cfg, runtime.baseline_facial_evidence),
                0.0,
                10.0,
            )
            adjusted_score = _apply_neutral_face_guard(
                adjusted_score,
                features,
                change_strength,
                cfg,
                runtime.baseline_facial_evidence,
                runtime.baseline_eyebrow_distance_ratio,
            )
            adaptive_alpha = cfg.pain_display_alpha_still
            if change_strength >= cfg.expression_change_threshold:
                adaptive_alpha = cfg.pain_display_alpha_change

            runtime.smoothed_pain_score = _smooth_value(runtime.smoothed_pain_score, adjusted_score, adaptive_alpha)
            runtime.smoothed_wheeze_probability = _smooth_value(
                runtime.smoothed_wheeze_probability,
                latest_wheeze_probability,
                cfg.wheeze_display_alpha,
            )
            latest_wheeze_probability = runtime.smoothed_wheeze_probability
            features.wheeze_probability = latest_wheeze_probability
            runtime.previous_raw_pain_score = float(adjusted_score)

            score = runtime.smoothed_pain_score
            if not calibration_done:
                score = min(score, cfg.neutral_face_score_cap)
            level = level_from_score(score)
            wheeze_level = wheeze_level_from_probability(latest_wheeze_probability)
            duration_score = score if calibration_done else 0.0
            duration = update_duration_state(runtime, duration_score, now, cfg)

            if not calibration_done:
                calibration_text = (
                    f"Calibrating neutral face: {elapsed_calibration:0.1f}/{cfg.calibration_seconds:0.1f}s, "
                    f"samples={len(runtime.calibration_scores)}"
                )
            else:
                baseline = runtime.baseline_score if runtime.baseline_score is not None else 0.0
                calibration_text = (
                    f"Calibration ready | baseline={baseline:0.2f} | raw={raw_score:0.2f} | wheeze={latest_wheeze_probability:0.2f}"
                )

            draw_overlay(
                frame,
                features,
                score,
                level,
                duration,
                wheeze_level=wheeze_level,
                calibration_text=calibration_text,
                overlay_scale=cfg.overlay_scale,
                overlay_anchor=cfg.overlay_anchor,
                pain_display_enabled=calibration_done,
                eyebrow_confidence_threshold=cfg.eyebrow_pain_confidence_threshold,
            )

            if logger is not None and frame_idx % cfg.log_every_n_frames == 0:
                timestamp_iso = datetime.now().astimezone().isoformat(timespec="seconds")
                logger.log(
                    timestamp_iso=timestamp_iso,
                    frame_idx=frame_idx,
                    elapsed_seconds=elapsed,
                    patient_id=cfg.patient_id,
                    score_0_10=score,
                    level=level,
                    features=features,
                    duration=duration,
                )

            if email_notifications_ready(cfg) and cfg.email_send_instant_alerts:
                if duration.started_now and now - last_pain_alert_time >= cfg.notification_cooldown_seconds:
                    send_email_notification(
                        cfg,
                        subject=build_alert_subject(cfg.patient_id, "pain"),
                        body=build_alert_body(cfg.patient_id, "pain", score, latest_wheeze_probability, calibration_text),
                    )
                    last_pain_alert_time = now
                if latest_wheeze_probability >= cfg.wheeze_alert_threshold and now - last_wheeze_alert_time >= cfg.notification_cooldown_seconds:
                    send_email_notification(
                        cfg,
                        subject=build_alert_subject(cfg.patient_id, "wheeze"),
                        body=build_alert_body(cfg.patient_id, "wheeze", score, latest_wheeze_probability, calibration_text),
                    )
                    last_wheeze_alert_time = now

            cv2.imshow("Patient Pain Monitor", frame)
            frame_idx += 1
            processed += 1

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        if audio_provider is not None and hasattr(audio_provider, "close"):
            audio_provider.close()
        eyebrow_detector.close()
        cv2.destroyAllWindows()

    elapsed = max(0.001, time.time() - start_wall)
    total_duration = runtime.closed_duration_s
    if runtime.active_episode is not None:
        total_duration += max(0.0, time.time() - runtime.active_episode.start_time)

    report_summary = None
    if logger is not None and cfg.email_send_session_report and email_notifications_ready(cfg):
        report_summary = summarize_session_csv(logger.output_file)
        send_email_notification(
            cfg,
            subject=build_session_report_subject(cfg.patient_id),
            body=build_session_report_body(cfg.patient_id, report_summary),
            attachments=[logger.output_file, logger.episode_file],
        )

    result = {
        "frames_processed": processed,
        "elapsed_seconds": elapsed,
        "fps": processed / elapsed,
        "total_pain_duration_seconds": total_duration,
        "latest_wheeze_probability": latest_wheeze_probability,
        "log_file": str(logger.output_file) if logger is not None else None,
        "episode_event_file": str(logger.episode_file) if logger is not None else None,
        "calibration_done": calibration_done,
        "baseline_score": runtime.baseline_score,
    }
    if report_summary is not None:
        result["session_report"] = report_summary
    return result


def extract_features_from_video(
    video_path: Path,
    output_csv_path: Path,
    config: PainMonitoringConfig | None = None,
    sample_every_n_frames: int = 3,
    fixed_label: float | None = None,
    fixed_wheeze_label: float | None = None,
    audio_path: Path | None = None,
    show_preview: bool = True,
) -> dict:
    import cv2

    from pain_monitoring.features import EyebrowLandmarkDetector, apply_audio_snapshot, extract_frame_features

    cfg = config or PainMonitoringConfig()
    cfg.validate()
    if sample_every_n_frames <= 0:
        raise ValueError("sample_every_n_frames must be >= 1")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    smile_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
    eyebrow_detector = EyebrowLandmarkDetector(enabled=cfg.enable_eyebrow_landmarks)
    audio_provider = FileAudioFeatureProvider(audio_path=audio_path, window_seconds=cfg.audio_window_seconds) if audio_path else None

    fps = capture.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 25.0

    prev_gray = None
    prev_face_box = None
    frame_idx = 0
    rows: list[dict] = []

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            elapsed_seconds = frame_idx / fps
            features, gray = extract_frame_features(
                frame=frame,
                previous_gray=prev_gray,
                previous_face_box=prev_face_box,
                face_detector=face_detector,
                eye_detector=eye_detector,
                smile_detector=smile_detector,
                config=cfg,
                eyebrow_detector=eyebrow_detector,
            )
            if audio_provider is not None:
                features = apply_audio_snapshot(features, audio_provider.get_snapshot(elapsed_seconds))

            prev_gray = gray
            if features.face_detected:
                prev_face_box = features.face_box

            if frame_idx % sample_every_n_frames == 0 and features.face_detected:
                row = {"video_file": str(video_path.name), "frame_idx": frame_idx, **{column: float(getattr(features, column, 0.0)) for column in FEATURE_COLUMNS}}
                row["wheeze_probability"] = float(features.wheeze_probability)
                if fixed_label is not None:
                    row[TARGET_COLUMN] = float(fixed_label)
                if fixed_wheeze_label is not None:
                    row[WHEEZE_TARGET_COLUMN] = float(np.clip(fixed_wheeze_label, 0.0, 1.0))
                rows.append(row)

            if show_preview:
                preview = frame.copy()
                if features.face_detected and features.face_box is not None:
                    x, y, w, h = features.face_box
                    cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 200, 0), 2)
                cv2.putText(preview, f"Frame: {frame_idx} | Rows: {len(rows)}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                cv2.putText(preview, f"Wheeze: {features.wheeze_probability:0.2f}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
                cv2.imshow("Feature Extraction Preview", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1
    finally:
        capture.release()
        eyebrow_detector.close()
        if show_preview:
            cv2.destroyWindow("Feature Extraction Preview")

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv_path, index=False)

    return {
        "video": str(video_path),
        "audio": str(audio_path) if audio_path else None,
        "output_csv": str(output_csv_path),
        "rows": len(rows),
        "label_attached": fixed_label is not None,
        "wheeze_label_attached": fixed_wheeze_label is not None,
    }


def train_from_labeled_csv(
    csv_path: Path,
    model_out: Path,
    metrics_out: Path | None = None,
    ridge_alpha: float = 0.8,
) -> dict:
    frame = pd.read_csv(csv_path)
    model, metrics = train_linear_model_from_frame(frame, ridge_alpha=ridge_alpha)
    model.save(model_out)

    payload = {"source_csv": str(csv_path), "model_path": str(model_out), **metrics}
    if metrics_out is not None:
        write_json(metrics_out, payload)
    return payload


def evaluate_from_csv(model_path: Path, csv_path: Path) -> dict:
    model = PainLinearModel.load(model_path)
    frame = pd.read_csv(csv_path)

    for column in FEATURE_COLUMNS:
        if column not in frame.columns:
            frame[column] = 0.0
    required = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns for evaluation: {missing}")

    clean = frame[required + ([WHEEZE_TARGET_COLUMN] if WHEEZE_TARGET_COLUMN in frame.columns else [])].dropna(subset=[TARGET_COLUMN])
    if clean.empty:
        raise ValueError("No valid rows to evaluate.")

    x = clean[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_true = clean[TARGET_COLUMN].to_numpy(dtype=float)
    pain_pred, wheeze_pred = _predict_array(model, x)

    payload = {
        "model_path": str(model_path),
        "csv_path": str(csv_path),
        "rows_evaluated": int(clean.shape[0]),
        "mae": float(np.mean(np.abs(pain_pred - y_true))),
        "rmse": float(np.sqrt(np.mean((pain_pred - y_true) ** 2))),
        "within_1_point_percent": float(np.mean(np.abs(pain_pred - y_true) <= 1.0) * 100.0),
        "within_2_points_percent": float(np.mean(np.abs(pain_pred - y_true) <= 2.0) * 100.0),
    }
    if WHEEZE_TARGET_COLUMN in clean.columns and clean[WHEEZE_TARGET_COLUMN].notna().any():
        y_wheeze = clean[WHEEZE_TARGET_COLUMN].fillna(0.0).to_numpy(dtype=float)
        payload["wheeze_mae"] = float(np.mean(np.abs(wheeze_pred - y_wheeze)))
    return payload


def summarize_session(session_csv: Path, summary_out: Path | None = None) -> dict:
    return summarize_session_csv(session_csv_path=session_csv, summary_out=summary_out)


def build_training_dataset(
    csv_paths: list[Path],
    output_csv: Path,
    config: PainMonitoringConfig | None = None,
    augment_factor: int | None = None,
) -> dict:
    cfg = config or PainMonitoringConfig()
    return prepare_training_dataset(
        csv_paths=csv_paths,
        output_csv_path=output_csv,
        augment_factor=cfg.dataset_augmentation_factor if augment_factor is None else augment_factor,
    )
