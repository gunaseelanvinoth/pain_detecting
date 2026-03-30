from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pain_monitoring.types import FramePainFeatures


def _clamp(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


@dataclass
class PainLinearModel:
    eye_closure: float = 3.0
    brow_tension: float = 2.4
    mouth_tension: float = 1.8
    smile_absence: float = 1.3
    motion_score: float = 1.5
    bias: float = 0.0

    def predict_score(self, features: FramePainFeatures) -> float:
        raw = (
            self.eye_closure * features.eye_closure
            + self.brow_tension * features.brow_tension
            + self.mouth_tension * features.mouth_tension
            + self.smile_absence * features.smile_absence
            + self.motion_score * features.motion_score
            + self.bias
        )
        return _clamp(raw, 0.0, 10.0)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "eye_closure": self.eye_closure,
            "brow_tension": self.brow_tension,
            "mouth_tension": self.mouth_tension,
            "smile_absence": self.smile_absence,
            "motion_score": self.motion_score,
            "bias": self.bias,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def load(path: Path) -> "PainLinearModel":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return PainLinearModel(
            eye_closure=float(payload.get("eye_closure", 3.0)),
            brow_tension=float(payload.get("brow_tension", 2.4)),
            mouth_tension=float(payload.get("mouth_tension", 1.8)),
            smile_absence=float(payload.get("smile_absence", 1.3)),
            motion_score=float(payload.get("motion_score", 1.5)),
            bias=float(payload.get("bias", 0.0)),
        )


FEATURE_COLUMNS = [
    "eye_closure",
    "brow_tension",
    "mouth_tension",
    "smile_absence",
    "motion_score",
]
TARGET_COLUMN = "pain_label_0_10"


def train_linear_model_from_frame(df) -> tuple[PainLinearModel, dict]:
    required = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for training: {missing}")

    clean = df[required].dropna()
    if clean.empty:
        raise ValueError("No valid rows found for training after dropping NaN values.")

    x = clean[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = clean[TARGET_COLUMN].to_numpy(dtype=float)

    x_augmented = np.concatenate([x, np.ones((x.shape[0], 1), dtype=float)], axis=1)
    coeffs, _, _, _ = np.linalg.lstsq(x_augmented, y, rcond=None)

    model = PainLinearModel(
        eye_closure=float(coeffs[0]),
        brow_tension=float(coeffs[1]),
        mouth_tension=float(coeffs[2]),
        smile_absence=float(coeffs[3]),
        motion_score=float(coeffs[4]),
        bias=float(coeffs[5]),
    )

    predictions = np.clip(x_augmented @ coeffs, 0.0, 10.0)
    mae = float(np.mean(np.abs(predictions - y)))
    rmse = float(np.sqrt(np.mean((predictions - y) ** 2)))

    metrics = {
        "rows_used": int(clean.shape[0]),
        "mae": mae,
        "rmse": rmse,
        "coefficients": {
            "eye_closure": model.eye_closure,
            "brow_tension": model.brow_tension,
            "mouth_tension": model.mouth_tension,
            "smile_absence": model.smile_absence,
            "motion_score": model.motion_score,
            "bias": model.bias,
        },
    }
    return model, metrics
