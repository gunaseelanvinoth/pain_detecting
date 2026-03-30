from __future__ import annotations

import cv2
import numpy as np

from pain_monitoring.config import PainMonitoringConfig
from pain_monitoring.types import FramePainFeatures


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _largest_box(boxes) -> tuple[int, int, int, int] | None:
    if len(boxes) == 0:
        return None
    return max(boxes, key=lambda rect: rect[2] * rect[3])


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
        features = FramePainFeatures(
            face_detected=False,
            face_box=None,
            eye_closure=0.0,
            brow_tension=0.0,
            mouth_tension=0.0,
            smile_absence=0.0,
            motion_score=0.0,
        )
        return features, gray

    x, y, w, h = face_box
    if w * h < config.min_face_area:
        features = FramePainFeatures(
            face_detected=False,
            face_box=(x, y, w, h),
            eye_closure=0.0,
            brow_tension=0.0,
            mouth_tension=0.0,
            smile_absence=0.0,
            motion_score=0.0,
        )
        return features, gray

    face_roi = gray_eq[y : y + h, x : x + w]
    upper = face_roi[: max(1, h // 2), :]
    lower = face_roi[h // 2 :, :]
    forehead = face_roi[: max(1, h // 3), :]

    eyes = eye_detector.detectMultiScale(upper, scaleFactor=1.1, minNeighbors=4, minSize=(18, 18))
    smiles = smile_detector.detectMultiScale(lower, scaleFactor=1.4, minNeighbors=20, minSize=(25, 25))

    eye_closure = _clip01(1.0 - min(len(eyes), 2) / 2.0)

    brow_texture = cv2.Laplacian(forehead, cv2.CV_64F).std() if forehead.size > 0 else 0.0
    brow_tension = _clip01(float(brow_texture) / 35.0)

    mouth_texture = cv2.Laplacian(lower, cv2.CV_64F).std() if lower.size > 0 else 0.0
    mouth_tension = _clip01(float(mouth_texture) / 42.0)

    smile_absence = 0.0 if len(smiles) > 0 else 1.0

    if previous_gray is None:
        motion_score = 0.0
    else:
        previous_face = previous_gray[y : y + h, x : x + w]
        if previous_face.shape != face_roi.shape:
            motion_score = 0.0
        else:
            diff = cv2.absdiff(face_roi, previous_face)
            motion_score = _clip01(float(diff.mean()) / 28.0)

    features = FramePainFeatures(
        face_detected=True,
        face_box=(x, y, w, h),
        eye_closure=eye_closure,
        brow_tension=brow_tension,
        mouth_tension=mouth_tension,
        smile_absence=smile_absence,
        motion_score=motion_score,
    )
    return features, gray
