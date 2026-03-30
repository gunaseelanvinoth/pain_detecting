from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from pain_monitoring.config import PainMonitoringConfig
from pain_monitoring.episode_tracker import update_duration_state
from pain_monitoring.io_utils import write_json
from pain_monitoring.logger import PainLiveLogger
from pain_monitoring.model import FEATURE_COLUMNS, TARGET_COLUMN, PainLinearModel, train_linear_model_from_frame
from pain_monitoring.reporting import summarize_session_csv
from pain_monitoring.types import RuntimeState


def _predict_array(model: PainLinearModel, x: np.ndarray) -> np.ndarray:
    coeffs = np.array(
        [
            model.eye_closure,
            model.brow_tension,
            model.mouth_tension,
            model.smile_absence,
            model.motion_score,
            model.bias,
        ],
        dtype=float,
    )
    x_augmented = np.concatenate([x, np.ones((x.shape[0], 1), dtype=float)], axis=1)
    return np.clip(x_augmented @ coeffs, 0.0, 10.0)


def run_live_monitor(
    model_path: Path | None = None,
    video_path: Path | None = None,
    config: PainMonitoringConfig | None = None,
) -> dict:
    import cv2

    from pain_monitoring.features import extract_frame_features
    from pain_monitoring.overlay import draw_overlay, level_from_score

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

    runtime = RuntimeState()
    logger = PainLiveLogger(Path.cwd() / "sample_data") if cfg.save_live_data else None

    source_text = str(video_path) if video_path is not None else f"camera_index={cfg.camera_index}"
    print(f"[INFO] Starting live pain monitor on source: {source_text}")
    print("[INFO] Preview window opened. Press 'q' to stop.")
    if cfg.calibration_seconds > 0:
        print(f"[INFO] Calibration running for ~{cfg.calibration_seconds:.1f}s. Keep a neutral face.")

    frame_idx = 0
    processed = 0
    start_wall = time.time()
    calibration_done = cfg.calibration_seconds <= 0.0
    calibration_text = "Calibration skipped"

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            now = time.time()
            features, gray = extract_frame_features(
                frame=frame,
                previous_gray=runtime.previous_gray,
                previous_face_box=runtime.previous_face_box,
                face_detector=face_detector,
                eye_detector=eye_detector,
                smile_detector=smile_detector,
                config=cfg,
            )
            runtime.previous_gray = gray
            runtime.previous_face_box = features.face_box if features.face_detected else runtime.previous_face_box

            raw_score = model.predict_score(features)

            elapsed_calibration = now - start_wall
            if not calibration_done:
                if features.face_detected:
                    runtime.calibration_scores.append(raw_score)
                if (
                    elapsed_calibration >= cfg.calibration_seconds
                    and len(runtime.calibration_scores) >= cfg.calibration_min_samples
                ):
                    runtime.baseline_score = float(np.median(runtime.calibration_scores))
                    calibration_done = True

            adjusted_score = raw_score
            if runtime.baseline_score is not None:
                adjusted_score = np.clip(
                    raw_score - runtime.baseline_score + cfg.neutral_anchor_score,
                    0.0,
                    10.0,
                )

            if runtime.smoothed_pain_score <= 0.0:
                runtime.smoothed_pain_score = float(adjusted_score)
            else:
                runtime.smoothed_pain_score = float(
                    cfg.prediction_smoothing_alpha * adjusted_score
                    + (1.0 - cfg.prediction_smoothing_alpha) * runtime.smoothed_pain_score
                )

            score = runtime.smoothed_pain_score
            level = level_from_score(score)
            duration = update_duration_state(runtime, score, now, cfg)

            if not calibration_done:
                calibration_text = (
                    f"Calibrating neutral face: {elapsed_calibration:0.1f}/{cfg.calibration_seconds:0.1f}s, "
                    f"samples={len(runtime.calibration_scores)}"
                )
            else:
                baseline = runtime.baseline_score if runtime.baseline_score is not None else 0.0
                calibration_text = f"Calibration ready | baseline={baseline:0.2f} | raw={raw_score:0.2f}"

            draw_overlay(frame, features, score, level, duration, calibration_text=calibration_text)

            if logger is not None and frame_idx % cfg.log_every_n_frames == 0:
                timestamp_iso = datetime.now().astimezone().isoformat(timespec="seconds")
                logger.log(
                    timestamp_iso=timestamp_iso,
                    frame_idx=frame_idx,
                    elapsed_seconds=(now - start_wall),
                    patient_id=cfg.patient_id,
                    score_0_10=score,
                    level=level,
                    features=features,
                    duration=duration,
                )

            cv2.imshow("Patient Pain Monitor", frame)
            frame_idx += 1
            processed += 1

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()

    elapsed = max(0.001, time.time() - start_wall)
    total_duration = runtime.closed_duration_s
    if runtime.active_episode is not None:
        total_duration += max(0.0, time.time() - runtime.active_episode.start_time)

    summary = {
        "frames_processed": processed,
        "elapsed_seconds": elapsed,
        "fps": processed / elapsed,
        "total_pain_duration_seconds": total_duration,
        "closed_episodes": [
            {
                "episode_id": item.episode_id,
                "start_time": item.start_time,
                "end_time": item.end_time,
                "duration_seconds": item.duration_seconds,
                "max_score": item.max_score,
                "avg_score": item.avg_score,
            }
            for item in runtime.closed_episodes
        ],
        "log_file": str(logger.output_file) if logger is not None else None,
        "episode_event_file": str(logger.episode_file) if logger is not None else None,
        "calibration_done": calibration_done,
        "baseline_score": runtime.baseline_score,
    }
    return summary


def extract_features_from_video(
    video_path: Path,
    output_csv_path: Path,
    config: PainMonitoringConfig | None = None,
    sample_every_n_frames: int = 3,
    fixed_label: float | None = None,
    show_preview: bool = True,
) -> dict:
    import cv2

    from pain_monitoring.features import extract_frame_features

    cfg = config or PainMonitoringConfig()
    cfg.validate()
    if sample_every_n_frames <= 0:
        raise ValueError("sample_every_n_frames must be >= 1")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    print(f"[INFO] Extracting from video: {video_path}")
    if show_preview:
        print("[INFO] Preview window opened. Press 'q' to stop early.")

    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    smile_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

    prev_gray = None
    prev_face_box = None
    frame_idx = 0
    rows: list[dict] = []

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            features, gray = extract_frame_features(
                frame=frame,
                previous_gray=prev_gray,
                previous_face_box=prev_face_box,
                face_detector=face_detector,
                eye_detector=eye_detector,
                smile_detector=smile_detector,
                config=cfg,
            )
            prev_gray = gray
            if features.face_detected:
                prev_face_box = features.face_box

            if frame_idx % sample_every_n_frames == 0 and features.face_detected:
                row = {
                    "video_file": str(video_path.name),
                    "frame_idx": frame_idx,
                    "eye_closure": features.eye_closure,
                    "brow_tension": features.brow_tension,
                    "mouth_tension": features.mouth_tension,
                    "smile_absence": features.smile_absence,
                    "motion_score": features.motion_score,
                }
                if fixed_label is not None:
                    row[TARGET_COLUMN] = float(fixed_label)
                rows.append(row)

            if show_preview:
                preview = frame.copy()
                if features.face_detected and features.face_box is not None:
                    x, y, w, h = features.face_box
                    cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 200, 0), 2)
                    cv2.putText(
                        preview,
                        "Face detected",
                        (x, max(20, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 200, 0),
                        2,
                    )
                else:
                    cv2.putText(
                        preview,
                        "Face not detected",
                        (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 80, 255),
                        2,
                    )
                cv2.putText(
                    preview,
                    f"Frame: {frame_idx} | Rows: {len(rows)}",
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow("Feature Extraction Preview", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1
    finally:
        capture.release()
        if show_preview:
            cv2.destroyWindow("Feature Extraction Preview")

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv_path, index=False)

    return {
        "video": str(video_path),
        "output_csv": str(output_csv_path),
        "rows": len(rows),
        "label_attached": fixed_label is not None,
    }


def train_from_labeled_csv(csv_path: Path, model_out: Path, metrics_out: Path | None = None) -> dict:
    frame = pd.read_csv(csv_path)
    model, metrics = train_linear_model_from_frame(frame)
    model.save(model_out)

    payload = {
        "source_csv": str(csv_path),
        "model_path": str(model_out),
        **metrics,
    }

    if metrics_out is not None:
        write_json(metrics_out, payload)

    return payload


def evaluate_from_csv(model_path: Path, csv_path: Path) -> dict:
    model = PainLinearModel.load(model_path)
    frame = pd.read_csv(csv_path)

    required = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns for evaluation: {missing}")

    clean = frame[required].dropna()
    if clean.empty:
        raise ValueError("No valid rows to evaluate.")

    x = clean[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_true = clean[TARGET_COLUMN].to_numpy(dtype=float)

    y_pred = _predict_array(model, x)
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

    return {
        "model_path": str(model_path),
        "csv_path": str(csv_path),
        "rows_evaluated": int(clean.shape[0]),
        "mae": mae,
        "rmse": rmse,
        "within_1_point_percent": float(np.mean(np.abs(y_pred - y_true) <= 1.0) * 100.0),
        "within_2_points_percent": float(np.mean(np.abs(y_pred - y_true) <= 2.0) * 100.0),
    }


def summarize_session(session_csv: Path, summary_out: Path | None = None) -> dict:
    return summarize_session_csv(session_csv_path=session_csv, summary_out=summary_out)
