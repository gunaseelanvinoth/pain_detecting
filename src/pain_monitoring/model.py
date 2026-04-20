from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from pain_monitoring.types import FramePainFeatures


def _clamp(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def _brow_edge_pain_signal(features: FramePainFeatures) -> float:
    return (
        1.8 * features.brow_tension
        + 1.5 * features.brow_energy
        + 1.2 * features.brow_motion
        + 2.0 * features.eyebrow_contraction
        + 1.2 * features.eyebrow_angle_score
        + 1.6 * features.eyebrow_pain_confidence
        + 0.6 * features.brow_position
        + 1.1 * features.face_edge_density
        + 0.8 * features.eye_symmetry
        + 0.5 * features.nasal_tension
        + 0.5 * features.nose_contrast
    )


FACE_FEATURE_COLUMNS = [
    "eye_closure",
    "brow_tension",
    "mouth_tension",
    "smile_absence",
    "motion_score",
    "eye_symmetry",
    "brow_energy",
    "brow_position",
    "brow_motion",
    "eyebrow_distance_ratio",
    "eyebrow_contraction",
    "eyebrow_pain_confidence",
    "mouth_opening",
    "mouth_micro_motion",
    "lower_face_motion",
    "face_edge_density",
    "nasal_tension",
    "nose_contrast",
]

RESPIRATORY_FEATURE_COLUMNS = [
    "respiratory_motion",
    "wheeze_tonality",
    "wheeze_band_energy",
    "wheeze_entropy",
]

FEATURE_COLUMNS = FACE_FEATURE_COLUMNS + RESPIRATORY_FEATURE_COLUMNS
TARGET_COLUMN = "pain_label_0_10"
WHEEZE_TARGET_COLUMN = "wheeze_label_0_1"


def _sigmoid(values):
    array = np.asarray(values, dtype=float)
    return 1.0 / (1.0 + np.exp(-array))


def _ensure_feature_frame(df):
    frame = df.copy()
    for column in FEATURE_COLUMNS:
        if column not in frame.columns:
            frame[column] = 0.0
    return frame


def _build_design_matrix(x: np.ndarray) -> np.ndarray:
    squares = x**2
    interactions = np.column_stack(
        [
            x[:, 0] * x[:, 1],
            x[:, 1] * x[:, 2],
            x[:, 2] * x[:, 4],
            x[:, 0] * x[:, 12],
            x[:, 3] * x[:, 12],
            x[:, 4] * x[:, 14],
            x[:, 5] * x[:, 15],
            x[:, 2] * x[:, 16],
            x[:, 18] * x[:, 19],
            x[:, 19] * x[:, 20],
            x[:, 18] * x[:, 20],
            x[:, 17] * x[:, 18],
        ]
    )
    return np.concatenate([x, squares, interactions], axis=1)


def _standardize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = x.mean(axis=0)
    scales = x.std(axis=0)
    scales[scales < 1e-6] = 1.0
    return (x - means) / scales, means, scales


def _fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    x_std, means, scales = _standardize(x)
    design = np.concatenate([x_std, np.ones((x_std.shape[0], 1), dtype=float)], axis=1)
    identity = np.eye(design.shape[1], dtype=float)
    identity[-1, -1] = 0.0
    coeffs = np.linalg.solve(design.T @ design + alpha * identity, design.T @ y)
    return np.concatenate([means, scales, coeffs[:-1], np.array([coeffs[-1]], dtype=float)])


def _predict_linear(blob: np.ndarray, x: np.ndarray) -> np.ndarray:
    feature_count = x.shape[1]
    means = blob[:feature_count]
    scales = blob[feature_count : feature_count * 2]
    coeffs = blob[feature_count * 2 : feature_count * 3]
    bias = blob[-1]
    x_std = (x - means) / scales
    return x_std @ coeffs + bias


@dataclass
class PainLinearModel:
    version: str = "multimodal-ridge-v2"
    feature_columns: list[str] = field(default_factory=lambda: FEATURE_COLUMNS.copy())
    design_feature_count: int = 0
    pain_blob: list[float] = field(default_factory=list)
    wheeze_blob: list[float] = field(default_factory=list)
    pain_training_rows: int = 0
    wheeze_training_rows: int = 0

    def _feature_vector(self, features: FramePainFeatures) -> np.ndarray:
        values = np.array([float(getattr(features, column, 0.0)) for column in self.feature_columns], dtype=float)
        return _build_design_matrix(values[None, :])[0]

    def predict_score(self, features: FramePainFeatures) -> float:
        if self.pain_blob:
            raw = float(_predict_linear(np.array(self.pain_blob, dtype=float), self._feature_vector(features)[None, :])[0])
        else:
            raw = (
                3.3 * features.eye_closure
                + 2.0 * features.brow_tension
                + 1.6 * features.brow_energy
                + 0.7 * features.brow_position
                + 1.2 * features.brow_motion
                + 2.4 * features.eyebrow_contraction
                + 1.3 * features.eyebrow_angle_score
                + 1.8 * features.eyebrow_pain_confidence
                + 2.4 * features.mouth_tension
                + 1.4 * features.motion_score
                + 1.0 * features.mouth_opening
                + 1.4 * features.mouth_micro_motion
                + 0.8 * features.lower_face_motion
                + 0.8 * features.face_edge_density
                + 0.4 * features.eye_symmetry
                + 0.6 * features.nasal_tension
                + 0.9 * features.nose_contrast
                + 0.6 * features.respiratory_motion
                + 1.2 * features.wheeze_probability
            )
            raw += 0.10 * _brow_edge_pain_signal(features)
        return _clamp(raw, 0.0, 10.0)

    def predict_wheeze(self, features: FramePainFeatures) -> float:
        if self.wheeze_blob:
            raw = float(_predict_linear(np.array(self.wheeze_blob, dtype=float), self._feature_vector(features)[None, :])[0])
            return _clamp(float(_sigmoid([raw])[0]), 0.0, 1.0)
        heuristic = (
            0.40 * features.wheeze_tonality
            + 0.30 * features.wheeze_band_energy
            + 0.20 * features.wheeze_entropy
            + 0.10 * features.respiratory_motion
        )
        return _clamp(heuristic, 0.0, 1.0)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.version,
            "feature_columns": self.feature_columns,
            "design_feature_count": self.design_feature_count,
            "pain_blob": self.pain_blob,
            "wheeze_blob": self.wheeze_blob,
            "pain_training_rows": self.pain_training_rows,
            "wheeze_training_rows": self.wheeze_training_rows,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def load(path: Path) -> "PainLinearModel":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "pain_blob" not in payload:
            return PainLinearModel()
        feature_columns = list(payload.get("feature_columns", FEATURE_COLUMNS))
        if feature_columns != FEATURE_COLUMNS:
            return PainLinearModel()
        return PainLinearModel(
            version=str(payload.get("version", "multimodal-ridge-v2")),
            feature_columns=feature_columns,
            design_feature_count=int(payload.get("design_feature_count", 0)),
            pain_blob=list(payload.get("pain_blob", [])),
            wheeze_blob=list(payload.get("wheeze_blob", [])),
            pain_training_rows=int(payload.get("pain_training_rows", 0)),
            wheeze_training_rows=int(payload.get("wheeze_training_rows", 0)),
        )


def train_linear_model_from_frame(df, ridge_alpha: float = 0.8) -> tuple[PainLinearModel, dict]:
    frame = _ensure_feature_frame(df)
    if TARGET_COLUMN not in frame.columns and WHEEZE_TARGET_COLUMN not in frame.columns:
        raise ValueError(f"Missing required columns for training: ['{TARGET_COLUMN}', '{WHEEZE_TARGET_COLUMN}']")

    pain_rows = frame.dropna(subset=[TARGET_COLUMN]) if TARGET_COLUMN in frame.columns else frame.iloc[0:0]
    wheeze_rows = frame.dropna(subset=[WHEEZE_TARGET_COLUMN]) if WHEEZE_TARGET_COLUMN in frame.columns else frame.iloc[0:0]

    if pain_rows.empty and wheeze_rows.empty:
        raise ValueError("No valid rows found for training after dropping NaN values.")

    if not pain_rows.empty:
        x = pain_rows[FEATURE_COLUMNS].to_numpy(dtype=float)
        x_design = _build_design_matrix(x)
        y_pain = pain_rows[TARGET_COLUMN].to_numpy(dtype=float)
        pain_blob = _fit_ridge(x_design, y_pain, ridge_alpha)
        pain_predictions = np.clip(_predict_linear(pain_blob, x_design), 0.0, 10.0)
        pain_mae = float(np.mean(np.abs(pain_predictions - y_pain)))
        pain_rmse = float(np.sqrt(np.mean((pain_predictions - y_pain) ** 2)))
        design_feature_count = int(x_design.shape[1])
    else:
        pain_blob = np.array([], dtype=float)
        pain_mae = None
        pain_rmse = None
        design_feature_count = 0

    if not wheeze_rows.empty:
        x_wheeze = _build_design_matrix(wheeze_rows[FEATURE_COLUMNS].to_numpy(dtype=float))
        y_wheeze = wheeze_rows[WHEEZE_TARGET_COLUMN].to_numpy(dtype=float)
        wheeze_blob = _fit_ridge(x_wheeze, y_wheeze, ridge_alpha)
        wheeze_predictions = np.clip(_sigmoid(_predict_linear(wheeze_blob, x_wheeze)), 0.0, 1.0)
        wheeze_mae = float(np.mean(np.abs(wheeze_predictions - y_wheeze)))
        if design_feature_count == 0:
            design_feature_count = int(x_wheeze.shape[1])
    else:
        wheeze_blob = np.array([], dtype=float)
        wheeze_mae = None

    model = PainLinearModel(
        feature_columns=FEATURE_COLUMNS.copy(),
        design_feature_count=design_feature_count,
        pain_blob=pain_blob.tolist(),
        wheeze_blob=wheeze_blob.tolist(),
        pain_training_rows=int(pain_rows.shape[0]),
        wheeze_training_rows=int(wheeze_rows.shape[0]),
    )

    metrics = {
        "rows_used": int(max(pain_rows.shape[0], wheeze_rows.shape[0])),
        "mae": pain_mae,
        "rmse": pain_rmse,
        "wheeze_mae": wheeze_mae,
        "pain_training_rows": int(pain_rows.shape[0]),
        "wheeze_training_rows": int(wheeze_rows.shape[0]),
        "feature_columns": FEATURE_COLUMNS,
        "design_feature_count": design_feature_count,
    }
    return model, metrics
