from __future__ import annotations

from pain_monitoring.config import PainMonitoringConfig
from pain_monitoring.types import FramePainFeatures


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def eye_contrast_score(features: FramePainFeatures) -> float:
    return _clip01(0.65 * features.eye_closure + 0.35 * max(features.eye_symmetry, features.brow_energy))


def mouth_contrast_score(features: FramePainFeatures) -> float:
    return _clip01(0.55 * features.mouth_tension + 0.30 * features.mouth_opening + 0.15 * features.lower_face_motion)


def nose_contrast_score(features: FramePainFeatures) -> float:
    return _clip01(0.7 * features.nasal_tension + 0.3 * features.face_edge_density)


def facial_expression_strength(features: FramePainFeatures) -> float:
    eye_score = eye_contrast_score(features)
    mouth_score = mouth_contrast_score(features)
    nose_score = nose_contrast_score(features)
    return _clip01(
        0.35 * eye_score
        + 0.35 * mouth_score
        + 0.20 * nose_score
        + 0.10 * max(features.motion_score, features.lower_face_motion)
    )


def pain_detected_from_face(features: FramePainFeatures, score_0_10: float, config: PainMonitoringConfig) -> bool:
    if not features.face_detected:
        return False

    region_threshold = config.pain_feature_region_threshold
    active_regions = sum(
        [
            eye_contrast_score(features) >= region_threshold,
            mouth_contrast_score(features) >= region_threshold,
            nose_contrast_score(features) >= region_threshold,
        ]
    )
    expression_strength = facial_expression_strength(features)
    return (
        score_0_10 >= config.pain_detection_score_threshold
        and active_regions >= config.pain_min_active_regions
        and expression_strength >= config.pain_expression_threshold
    )


def pain_status_text(is_detected: bool) -> str:
    return "Pain is detected" if is_detected else "There is no pain detected"
