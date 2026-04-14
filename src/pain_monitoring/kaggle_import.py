from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from pain_monitoring.audio_features import FileAudioFeatureProvider
from pain_monitoring.config import PainMonitoringConfig
from pain_monitoring.features import extract_frame_features
from pain_monitoring.model import FEATURE_COLUMNS, TARGET_COLUMN, WHEEZE_TARGET_COLUMN


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PAIN_KEYWORDS = {
    "no_pain": 0.0,
    "nopain": 0.0,
    "neutral": 0.5,
    "control": 0.5,
    "mild": 3.5,
    "moderate": 6.0,
    "severe": 8.5,
    "pain": 7.0,
    "distress": 6.5,
}


def _empty_feature_row() -> dict[str, float]:
    return {column: 0.0 for column in FEATURE_COLUMNS}


def _resolve_pain_label(path: Path, default_label: float | None = None) -> float | None:
    text = " ".join(part.lower() for part in path.parts)
    for key, value in PAIN_KEYWORDS.items():
        if key in text:
            return value
    return default_label


def import_kaggle_respiratory_dataset(
    dataset_dir: Path,
    output_csv_path: Path,
    config: PainMonitoringConfig | None = None,
) -> dict:
    cfg = config or PainMonitoringConfig()
    wav_files = sorted(dataset_dir.rglob("*.wav"))
    rows: list[dict] = []

    for wav_path in wav_files:
        annotation_path = wav_path.with_suffix(".txt")
        if not annotation_path.exists():
            continue

        provider = FileAudioFeatureProvider(wav_path, window_seconds=cfg.audio_window_seconds)
        try:
            annotations = pd.read_csv(
                annotation_path,
                sep=r"\s+",
                header=None,
                names=["start", "end", "crackles", "wheezes"],
                engine="python",
            )
        except Exception:
            continue

        for cycle_idx, item in annotations.iterrows():
            midpoint = float(item["start"] + item["end"]) / 2.0
            snapshot = provider.get_snapshot(midpoint)
            row = _empty_feature_row()
            row.update(
                {
                    "respiratory_motion": snapshot.respiratory_motion,
                    "wheeze_tonality": snapshot.wheeze_tonality,
                    "wheeze_band_energy": snapshot.wheeze_band_energy,
                    "wheeze_entropy": snapshot.wheeze_entropy,
                    WHEEZE_TARGET_COLUMN: float(item["wheezes"]),
                    "source_audio_file": str(wav_path),
                    "cycle_index": int(cycle_idx),
                    "cycle_start_s": float(item["start"]),
                    "cycle_end_s": float(item["end"]),
                    "cycle_has_crackles": int(item["crackles"]),
                }
            )
            rows.append(row)

    frame = pd.DataFrame(rows)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv_path, index=False)
    return {
        "dataset_dir": str(dataset_dir),
        "output_csv": str(output_csv_path),
        "rows": int(frame.shape[0]),
        "wav_files_found": len(wav_files),
        "wheeze_labels_present": int(frame[WHEEZE_TARGET_COLUMN].notna().sum()) if not frame.empty else 0,
    }


def import_kaggle_face_dataset(
    dataset_dir: Path,
    output_csv_path: Path,
    config: PainMonitoringConfig | None = None,
    default_label: float | None = None,
) -> dict:
    cfg = config or PainMonitoringConfig()
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    smile_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

    image_files = sorted(path for path in dataset_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
    rows: list[dict] = []
    detected_faces = 0

    for image_path in image_files:
        label = _resolve_pain_label(image_path, default_label=default_label)
        if label is None:
            continue

        frame = cv2.imread(str(image_path))
        if frame is None:
            continue

        features, _ = extract_frame_features(
            frame=frame,
            previous_gray=None,
            previous_face_box=None,
            face_detector=face_detector,
            eye_detector=eye_detector,
            smile_detector=smile_detector,
            config=cfg,
        )
        if not features.face_detected:
            continue

        detected_faces += 1
        row = {column: float(getattr(features, column, 0.0)) for column in FEATURE_COLUMNS}
        row.update(
            {
                TARGET_COLUMN: float(label),
                "image_file": str(image_path),
                "derived_label_source": "folder_name",
            }
        )
        rows.append(row)

    frame = pd.DataFrame(rows)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv_path, index=False)
    return {
        "dataset_dir": str(dataset_dir),
        "output_csv": str(output_csv_path),
        "images_found": len(image_files),
        "faces_detected": detected_faces,
        "rows": int(frame.shape[0]),
        "pain_labels_present": int(frame[TARGET_COLUMN].notna().sum()) if not frame.empty else 0,
    }
