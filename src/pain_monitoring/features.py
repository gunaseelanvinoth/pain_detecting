from __future__ import annotations

import cv2
import numpy as np

from pain_monitoring.audio_features import AudioFeatureSnapshot
from pain_monitoring.config import PainMonitoringConfig
from pain_monitoring.types import FramePainFeatures


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _largest_box(boxes) -> tuple[int, int, int, int] | None:
    if len(boxes) == 0:
        return None
    return max(boxes, key=lambda rect: rect[2] * rect[3])


def _roi_std(region: np.ndarray) -> float:
    return float(region.std()) if region.size > 0 else 0.0


def _texture_score(region: np.ndarray, scale: float) -> float:
    if region.size == 0:
        return 0.0
    lap = cv2.Laplacian(region, cv2.CV_64F)
    return _clip01(float(lap.std()) / scale)


def _edge_density(region: np.ndarray, scale: float) -> float:
    if region.size == 0:
        return 0.0
    edges = cv2.Canny(region, 30, 90)
    return _clip01(float(edges.mean()) / scale)


def _dark_gap_score(region: np.ndarray) -> float:
    if region.size == 0:
        return 0.0
    return _clip01((140.0 - float(region.mean())) / 90.0)


def apply_audio_snapshot(features: FramePainFeatures, snapshot: AudioFeatureSnapshot | None) -> FramePainFeatures:
    if snapshot is None:
        return features
    features.respiratory_motion = snapshot.respiratory_motion
    features.wheeze_tonality = snapshot.wheeze_tonality
    features.wheeze_band_energy = snapshot.wheeze_band_energy
    features.wheeze_entropy = snapshot.wheeze_entropy
    features.wheeze_probability = snapshot.wheeze_probability
    return features


def extract_frame_features(
    frame: np.ndarray,
    previous_gray: np.ndarray | None,
    previous_face_box: tuple[int, int, int, int] | None,
    face_detector: cv2.CascadeClassifier,
    eye_detector: cv2.CascadeClassifier,
    smile_detector: cv2.CascadeClassifier,
    config: PainMonitoringConfig,
) -> tuple[FramePainFeatures, np.ndarray]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)
    faces = face_detector.detectMultiScale(gray_eq, scaleFactor=1.1, minNeighbors=4, minSize=(70, 70))
    face_box = _largest_box(faces)

    if face_box is None and previous_face_box is not None:
        px, py, pw, ph = previous_face_box
        margin_x = max(20, int(0.35 * pw))
        margin_y = max(20, int(0.35 * ph))
        rx1 = max(0, px - margin_x)
        ry1 = max(0, py - margin_y)
        rx2 = min(gray_eq.shape[1], px + pw + margin_x)
        ry2 = min(gray_eq.shape[0], py + ph + margin_y)
        local = gray_eq[ry1:ry2, rx1:rx2]
        if local.size > 0:
            local_faces = face_detector.detectMultiScale(local, scaleFactor=1.08, minNeighbors=3, minSize=(50, 50))
            local_face = _largest_box(local_faces)
            if local_face is not None:
                lx, ly, lw, lh = local_face
                face_box = (rx1 + lx, ry1 + ly, lw, lh)

    if face_box is None:
        return FramePainFeatures(False, None, 0.0, 0.0, 0.0, 0.0, 0.0), gray

    x, y, w, h = face_box
    if w * h < config.min_face_area:
        return FramePainFeatures(False, (x, y, w, h), 0.0, 0.0, 0.0, 0.0, 0.0), gray

    face_roi = gray_eq[y : y + h, x : x + w]
    upper = face_roi[: max(1, h // 2), :]
    lower = face_roi[h // 2 :, :]
    forehead = face_roi[: max(1, h // 3), :]
    nose_band = face_roi[h // 3 : (2 * h) // 3, w // 3 : (2 * w) // 3]
    mouth_band = lower[max(0, lower.shape[0] // 4) : min(lower.shape[0], (3 * lower.shape[0]) // 4), :]
    left_eye_region = upper[:, : max(1, upper.shape[1] // 2)]
    right_eye_region = upper[:, max(1, upper.shape[1] // 2) :]

    eyes = eye_detector.detectMultiScale(upper, scaleFactor=1.1, minNeighbors=4, minSize=(18, 18))
    smiles = smile_detector.detectMultiScale(lower, scaleFactor=1.4, minNeighbors=18, minSize=(25, 25))

    eye_closure = _clip01(1.0 - min(len(eyes), 2) / 2.0)
    brow_tension = _texture_score(forehead, 32.0)
    mouth_tension = _texture_score(lower, 38.0)
    smile_absence = 0.0 if len(smiles) > 0 else 1.0

    left_std = _roi_std(left_eye_region)
    right_std = _roi_std(right_eye_region)
    eye_symmetry = _clip01(abs(left_std - right_std) / max(12.0, left_std + right_std, 1.0))
    brow_energy = _clip01(float(cv2.Sobel(forehead, cv2.CV_64F, 1, 0, ksize=3).std()) / 30.0) if forehead.size > 0 else 0.0
    mouth_opening = _dark_gap_score(mouth_band)
    face_edge_density = _edge_density(face_roi, 55.0)
    nasal_tension = _texture_score(nose_band, 30.0)

    if previous_gray is None:
        motion_score = 0.0
        lower_face_motion = 0.0
    else:
        previous_face = previous_gray[y : y + h, x : x + w]
        if previous_face.shape != face_roi.shape:
            motion_score = 0.0
            lower_face_motion = 0.0
        else:
            diff = cv2.absdiff(face_roi, previous_face)
            motion_score = _clip01(float(diff.mean()) / 22.0)
            lower_diff = diff[h // 2 :, :] if diff.shape[0] > 1 else diff
            lower_face_motion = _clip01(float(lower_diff.mean()) / 18.0)

    features = FramePainFeatures(
        face_detected=True,
        face_box=(x, y, w, h),
        eye_closure=eye_closure,
        brow_tension=brow_tension,
        mouth_tension=mouth_tension,
        smile_absence=smile_absence,
        motion_score=motion_score,
        eye_symmetry=eye_symmetry,
        brow_energy=brow_energy,
        mouth_opening=mouth_opening,
        lower_face_motion=lower_face_motion,
        face_edge_density=face_edge_density,
        nasal_tension=nasal_tension,
    )
    return features, gray
