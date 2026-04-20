from __future__ import annotations

import cv2
import numpy as np

from pain_monitoring.audio_features import AudioFeatureSnapshot
from pain_monitoring.config import PainMonitoringConfig
from pain_monitoring.types import FramePainFeatures


RIGHT_INNER_BROW_LANDMARKS = (107, 66, 105)
LEFT_INNER_BROW_LANDMARKS = (336, 296, 334)
RIGHT_OUTER_BROW_LANDMARKS = (70, 63, 46)
LEFT_OUTER_BROW_LANDMARKS = (300, 293, 276)


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


def _local_contrast(region: np.ndarray, scale: float) -> float:
    if region.size == 0:
        return 0.0
    return _clip01(float(region.std()) / scale)


def _vertical_edge_position(region: np.ndarray) -> float:
    if region.size == 0:
        return 0.0
    gradients = np.abs(cv2.Sobel(region, cv2.CV_64F, 0, 1, ksize=3))
    weights = gradients.mean(axis=1)
    total = float(weights.sum())
    if total <= 1e-6:
        return 0.0
    positions = np.linspace(0.0, 1.0, num=weights.shape[0], dtype=float)
    return _clip01(float(np.dot(weights, positions) / total))


def _dark_gap_score(region: np.ndarray) -> float:
    if region.size == 0:
        return 0.0
    return _clip01((140.0 - float(region.mean())) / 90.0)


def _region_motion(current: np.ndarray, previous: np.ndarray | None, scale: float) -> float:
    if previous is None or current.size == 0 or previous.shape != current.shape:
        return 0.0
    return _clip01(float(cv2.absdiff(current, previous).mean()) / scale)


def _landmark_mean(landmarks, indexes: tuple[int, ...]) -> tuple[float, float]:
    x = float(np.mean([landmarks[index].x for index in indexes]))
    y = float(np.mean([landmarks[index].y for index in indexes]))
    return x, y


def _face_box_from_landmarks(landmarks, frame_shape) -> tuple[int, int, int, int]:
    height, width = frame_shape[:2]
    xs = np.array([point.x for point in landmarks], dtype=float)
    ys = np.array([point.y for point in landmarks], dtype=float)
    x1 = int(np.clip(xs.min() * width, 0, width - 1))
    y1 = int(np.clip(ys.min() * height, 0, height - 1))
    x2 = int(np.clip(xs.max() * width, x1 + 1, width))
    y2 = int(np.clip(ys.max() * height, y1 + 1, height))
    return x1, y1, max(1, x2 - x1), max(1, y2 - y1)


def eyebrow_contraction_from_landmarks(
    landmarks,
    frame_shape,
    config: PainMonitoringConfig,
    baseline_ratio: float | None = None,
) -> tuple[float, float, float, float]:
    """Return eyebrow gap ratio, angle score, contraction strength, and pain confidence."""
    right_inner = _landmark_mean(landmarks, RIGHT_INNER_BROW_LANDMARKS)
    left_inner = _landmark_mean(landmarks, LEFT_INNER_BROW_LANDMARKS)
    right_outer = _landmark_mean(landmarks, RIGHT_OUTER_BROW_LANDMARKS)
    left_outer = _landmark_mean(landmarks, LEFT_OUTER_BROW_LANDMARKS)
    face_box = _face_box_from_landmarks(landmarks, frame_shape)
    face_width_ratio = max(float(face_box[2]) / max(float(frame_shape[1]), 1.0), 1e-6)
    eyebrow_gap_ratio = abs(left_inner[0] - right_inner[0]) / face_width_ratio

    if baseline_ratio is not None and baseline_ratio > 0.0:
        contraction = (baseline_ratio - eyebrow_gap_ratio) / config.eyebrow_contraction_drop_threshold
    else:
        contraction = (config.eyebrow_distance_pain_threshold - eyebrow_gap_ratio) / max(config.eyebrow_distance_pain_threshold, 1e-6)

    contraction = _clip01(contraction)
    # Contracted/pain brows often pull inward and the inner brow points drop.
    # Normalizing by face width makes the angle signal stable across distances.
    inner_lowering = max(0.0, (right_inner[1] - right_outer[1] + left_inner[1] - left_outer[1]) * 0.5)
    angle_score = _clip01(inner_lowering / max(config.eyebrow_angle_pain_threshold, 1e-6))
    combined_signal = _clip01(0.78 * contraction + 0.22 * angle_score)
    confidence = _clip01(0.15 + 0.85 * combined_signal) if combined_signal > 0.0 else 0.0
    return float(eyebrow_gap_ratio), float(angle_score), float(contraction), float(confidence)


class EyebrowLandmarkDetector:
    """Optional MediaPipe FaceMesh eyebrow detector with OpenCV fallback compatibility."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.face_mesh = None
        if not enabled:
            return
        try:
            import mediapipe as mp

            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        except Exception:
            self.face_mesh = None

    def detect_landmarks(self, frame: np.ndarray):
        if self.face_mesh is None:
            return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb)
        if not result.multi_face_landmarks:
            return None
        return result.multi_face_landmarks[0].landmark

    def apply(
        self,
        frame: np.ndarray,
        features: FramePainFeatures,
        config: PainMonitoringConfig,
        baseline_ratio: float | None = None,
    ) -> FramePainFeatures:
        landmarks = self.detect_landmarks(frame)
        if landmarks is None:
            return features
        distance_ratio, angle_score, contraction, confidence = eyebrow_contraction_from_landmarks(landmarks, frame.shape, config, baseline_ratio)
        features.face_detected = True
        features.face_box = features.face_box or _face_box_from_landmarks(landmarks, frame.shape)
        features.eyebrow_distance_ratio = distance_ratio
        features.eyebrow_angle_score = angle_score
        features.eyebrow_contraction = contraction
        features.eyebrow_pain_confidence = confidence
        return features

    def close(self) -> None:
        if self.face_mesh is not None:
            self.face_mesh.close()


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
    eyebrow_detector: EyebrowLandmarkDetector | None = None,
    baseline_eyebrow_distance_ratio: float | None = None,
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
        features = FramePainFeatures(False, None, 0.0, 0.0, 0.0, 0.0, 0.0)
        if eyebrow_detector is not None:
            features = eyebrow_detector.apply(frame, features, config, baseline_eyebrow_distance_ratio)
        return features, gray_eq

    x, y, w, h = face_box
    if w * h < config.min_face_area:
        features = FramePainFeatures(False, (x, y, w, h), 0.0, 0.0, 0.0, 0.0, 0.0)
        if eyebrow_detector is not None:
            features = eyebrow_detector.apply(frame, features, config, baseline_eyebrow_distance_ratio)
        return features, gray_eq

    face_roi = gray_eq[y : y + h, x : x + w]
    upper = face_roi[: max(1, h // 2), :]
    lower = face_roi[h // 2 :, :]
    forehead = face_roi[: max(1, h // 3), :]
    nose_band = face_roi[h // 3 : (2 * h) // 3, w // 3 : (2 * w) // 3]
    mouth_band = lower[max(0, lower.shape[0] // 4) : min(lower.shape[0], (3 * lower.shape[0]) // 4), :]
    brow_band = face_roi[max(0, int(0.16 * h)) : max(1, int(0.43 * h)), max(0, int(0.12 * w)) : min(w, int(0.88 * w))]
    mouth_focus = face_roi[max(0, int(0.58 * h)) : min(h, int(0.84 * h)), max(0, int(0.18 * w)) : min(w, int(0.82 * w))]
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
    brow_position = _vertical_edge_position(brow_band)
    mouth_opening = _dark_gap_score(mouth_band)
    face_edge_density = _edge_density(face_roi, 55.0)
    nasal_tension = _texture_score(nose_band, 30.0)
    nose_contrast = _clip01(0.65 * _local_contrast(nose_band, 44.0) + 0.35 * _edge_density(nose_band, 45.0))

    if previous_gray is None:
        motion_score = 0.0
        lower_face_motion = 0.0
        brow_motion = 0.0
        mouth_micro_motion = 0.0
    else:
        previous_face = previous_gray[y : y + h, x : x + w]
        if previous_face.shape != face_roi.shape:
            motion_score = 0.0
            lower_face_motion = 0.0
            brow_motion = 0.0
            mouth_micro_motion = 0.0
        else:
            diff = cv2.absdiff(face_roi, previous_face)
            motion_score = _clip01(float(diff.mean()) / 22.0)
            lower_diff = diff[h // 2 :, :] if diff.shape[0] > 1 else diff
            lower_face_motion = _clip01(float(lower_diff.mean()) / 18.0)
            previous_brow = previous_face[
                max(0, int(0.16 * h)) : max(1, int(0.43 * h)),
                max(0, int(0.12 * w)) : min(w, int(0.88 * w)),
            ]
            previous_mouth = previous_face[
                max(0, int(0.58 * h)) : min(h, int(0.84 * h)),
                max(0, int(0.18 * w)) : min(w, int(0.82 * w)),
            ]
            brow_motion = _region_motion(brow_band, previous_brow, 16.0)
            mouth_micro_motion = _region_motion(mouth_focus, previous_mouth, 13.0)

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
        brow_position=brow_position,
        brow_motion=brow_motion,
        mouth_opening=mouth_opening,
        mouth_micro_motion=mouth_micro_motion,
        lower_face_motion=lower_face_motion,
        face_edge_density=face_edge_density,
        nasal_tension=nasal_tension,
        nose_contrast=nose_contrast,
    )
    if eyebrow_detector is not None:
        features = eyebrow_detector.apply(frame, features, config, baseline_eyebrow_distance_ratio)
    return features, gray_eq
