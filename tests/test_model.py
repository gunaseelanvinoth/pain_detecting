import sys
import unittest
import wave
from pathlib import Path

import numpy as np
import pandas as pd

SRC_ROOT = Path(__file__).resolve().parent.parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pain_monitoring.dataset import prepare_training_dataset
from pain_monitoring.kaggle_import import import_kaggle_respiratory_dataset
from pain_monitoring.model import PainLinearModel, _brow_edge_pain_signal, train_linear_model_from_frame
from pain_monitoring.notifications import build_alert_body, build_session_report_body, email_notifications_ready, resolve_recipient_list
from pain_monitoring.overlay import pain_detection_label, wheeze_detection_label
from pain_monitoring.runner import (
    _estimate_brow_edge_boost,
    _estimate_expression_pain_boost,
    _estimate_sustained_pain_boost,
    _estimate_sustained_wheeze_boost,
    _apply_neutral_face_guard,
    _is_confirmed_eyebrow_pain,
    _is_neutral_expression,
    _pain_region_count,
    _relative_facial_pain_evidence,
    _support_wheeze_probability,
    _update_evidence_seconds,
    _wheeze_evidence,
)
from pain_monitoring.config import PainMonitoringConfig
from pain_monitoring.features import eyebrow_contraction_from_landmarks
from pain_monitoring.types import FramePainFeatures


class FakeLandmark:
    def __init__(self, x: float, y: float = 0.5) -> None:
        self.x = x
        self.y = y


def _fake_face_landmarks(right_brow_x: float, left_brow_x: float):
    landmarks = [FakeLandmark(0.5, 0.5) for _ in range(468)]
    for index in (107, 66, 105):
        landmarks[index] = FakeLandmark(right_brow_x, 0.35)
    for index in (336, 296, 334):
        landmarks[index] = FakeLandmark(left_brow_x, 0.35)
    for index in (70, 63, 46):
        landmarks[index] = FakeLandmark(right_brow_x - 0.05, 0.35)
    for index in (300, 293, 276):
        landmarks[index] = FakeLandmark(left_brow_x + 0.05, 0.35)
    landmarks[10] = FakeLandmark(0.2, 0.1)
    landmarks[152] = FakeLandmark(0.8, 0.9)
    return landmarks


class ModelTests(unittest.TestCase):
    def test_predict_clamps_to_0_10(self):
        model = PainLinearModel()
        features = FramePainFeatures(
            True,
            (0, 0, 100, 100),
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            mouth_opening=1.0,
            lower_face_motion=1.0,
            wheeze_probability=1.0,
        )
        score = model.predict_score(features)
        self.assertLessEqual(score, 10.0)
        self.assertGreaterEqual(score, 0.0)

    def test_train_returns_metrics(self):
        frame = pd.DataFrame(
            {
                "eye_closure": [0.1, 0.2, 0.3, 0.4],
                "brow_tension": [0.1, 0.2, 0.3, 0.4],
                "mouth_tension": [0.1, 0.2, 0.3, 0.4],
                "smile_absence": [0.3, 0.4, 0.5, 0.6],
                "motion_score": [0.1, 0.2, 0.3, 0.4],
                "eye_symmetry": [0.05, 0.08, 0.11, 0.12],
                "brow_energy": [0.1, 0.14, 0.18, 0.25],
                "brow_position": [0.18, 0.24, 0.34, 0.42],
                "brow_motion": [0.02, 0.06, 0.13, 0.24],
                "mouth_opening": [0.1, 0.2, 0.4, 0.5],
                "mouth_micro_motion": [0.02, 0.05, 0.16, 0.28],
                "lower_face_motion": [0.05, 0.1, 0.22, 0.3],
                "face_edge_density": [0.2, 0.22, 0.25, 0.31],
                "nasal_tension": [0.12, 0.18, 0.24, 0.29],
                "nose_contrast": [0.08, 0.16, 0.30, 0.36],
                "respiratory_motion": [0.1, 0.15, 0.22, 0.3],
                "wheeze_tonality": [0.05, 0.1, 0.3, 0.45],
                "wheeze_band_energy": [0.08, 0.14, 0.35, 0.5],
                "wheeze_entropy": [0.02, 0.08, 0.25, 0.4],
                "pain_label_0_10": [1.0, 2.5, 5.4, 7.8],
                "wheeze_label_0_1": [0.0, 0.1, 0.5, 0.8],
            }
        )
        model, metrics = train_linear_model_from_frame(frame)
        self.assertIsInstance(model, PainLinearModel)
        self.assertIn("mae", metrics)
        self.assertEqual(metrics["rows_used"], 4)
        self.assertGreater(metrics["design_feature_count"], len(frame.columns))

    def test_prepare_dataset_augments_rows(self):
        temp_dir = Path(__file__).resolve().parent / "_temp"
        temp_dir.mkdir(exist_ok=True)
        input_csv = temp_dir / "input.csv"
        output_csv = temp_dir / "prepared.csv"
        pd.DataFrame(
            {
                "eye_closure": [0.1, 0.6],
                "brow_tension": [0.2, 0.5],
                "mouth_tension": [0.2, 0.6],
                "smile_absence": [0.1, 0.9],
                "motion_score": [0.1, 0.5],
                "pain_label_0_10": [1.0, 8.0],
            }
        ).to_csv(input_csv, index=False)
        payload = prepare_training_dataset([input_csv], output_csv, augment_factor=3)
        prepared = pd.read_csv(output_csv)
        self.assertGreater(payload["prepared_rows"], payload["base_rows"])
        self.assertEqual(payload["prepared_rows"], len(prepared))

    def test_train_can_use_separate_pain_and_wheeze_rows(self):
        frame = pd.DataFrame(
            {
                "eye_closure": [0.1, 0.2, 0.0, 0.0],
                "brow_tension": [0.2, 0.4, 0.0, 0.0],
                "mouth_tension": [0.2, 0.5, 0.0, 0.0],
                "smile_absence": [0.1, 0.9, 0.0, 0.0],
                "motion_score": [0.1, 0.5, 0.0, 0.0],
                "wheeze_tonality": [0.0, 0.0, 0.2, 0.8],
                "wheeze_band_energy": [0.0, 0.0, 0.3, 0.9],
                "wheeze_entropy": [0.0, 0.0, 0.2, 0.7],
                "respiratory_motion": [0.0, 0.0, 0.3, 0.8],
                "pain_label_0_10": [1.0, 7.0, np.nan, np.nan],
                "wheeze_label_0_1": [np.nan, np.nan, 0.0, 1.0],
            }
        )
        model, metrics = train_linear_model_from_frame(frame)
        self.assertIsInstance(model, PainLinearModel)
        self.assertEqual(metrics["pain_training_rows"], 2)
        self.assertEqual(metrics["wheeze_training_rows"], 2)

    def test_import_kaggle_respiratory_dataset_creates_rows(self):
        temp_dir = Path(__file__).resolve().parent / "_temp_resp"
        temp_dir.mkdir(exist_ok=True)
        wav_path = temp_dir / "sample.wav"
        txt_path = temp_dir / "sample.txt"
        output_csv = temp_dir / "resp.csv"

        sample_rate = 16000
        duration_s = 1.0
        t = np.linspace(0.0, duration_s, int(sample_rate * duration_s), endpoint=False)
        signal = (0.35 * np.sin(2 * np.pi * 600 * t)).astype(np.float32)
        pcm = (signal * 32767).astype(np.int16)
        with wave.open(str(wav_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm.tobytes())

        txt_path.write_text("0.0 1.0 0 1\n", encoding="utf-8")
        payload = import_kaggle_respiratory_dataset(temp_dir, output_csv)
        frame = pd.read_csv(output_csv)
        self.assertEqual(payload["rows"], 1)
        self.assertEqual(int(frame["wheeze_label_0_1"].iloc[0]), 1)

    def test_overlay_detection_labels_are_clear(self):
        self.assertEqual(pain_detection_label("None"), "No Pain")
        self.assertEqual(pain_detection_label("Moderate"), "Pain Detected")
        self.assertEqual(wheeze_detection_label("None"), "No Wheezing")
        self.assertEqual(wheeze_detection_label("Low"), "Wheezing Detected")

    def test_expression_boost_helps_small_real_changes(self):
        config = PainMonitoringConfig()
        features = FramePainFeatures(
            True,
            (0, 0, 100, 100),
            0.35,
            0.32,
            0.28,
            1.0,
            0.20,
            eye_symmetry=0.18,
            brow_energy=0.42,
            brow_position=0.36,
            brow_motion=0.26,
            mouth_opening=0.34,
            mouth_micro_motion=0.29,
            lower_face_motion=0.31,
            face_edge_density=0.27,
            nasal_tension=0.22,
            nose_contrast=0.24,
        )
        boost = _estimate_expression_pain_boost(features, 0.24, config)
        self.assertGreater(boost, 0.0)

    def test_brow_edge_signal_and_boost_raise_pain_sensitivity(self):
        config = PainMonitoringConfig()
        features = FramePainFeatures(
            True,
            (0, 0, 100, 100),
            0.15,
            0.48,
            0.12,
            0.8,
            0.10,
            eye_symmetry=0.22,
            brow_energy=0.55,
            brow_position=0.40,
            brow_motion=0.32,
            face_edge_density=0.44,
            nasal_tension=0.20,
            nose_contrast=0.31,
        )
        signal = _brow_edge_pain_signal(features)
        boost = _estimate_brow_edge_boost(features, config)
        self.assertGreater(signal, 0.0)
        self.assertGreater(boost, 0.0)

    def test_eyebrow_landmarks_classify_contracted_brows_as_pain(self):
        config = PainMonitoringConfig()
        landmarks = _fake_face_landmarks(0.49, 0.51)
        distance, angle_score, contraction, confidence = eyebrow_contraction_from_landmarks(landmarks, (480, 640, 3), config)
        self.assertLess(distance, config.eyebrow_distance_pain_threshold)
        self.assertEqual(angle_score, 0.0)
        self.assertGreaterEqual(contraction, 0.4)
        self.assertGreaterEqual(confidence, 0.5)

    def test_eyebrow_landmarks_classify_normal_brows_as_no_pain(self):
        config = PainMonitoringConfig()
        landmarks = _fake_face_landmarks(0.40, 0.60)
        distance, angle_score, contraction, confidence = eyebrow_contraction_from_landmarks(landmarks, (480, 640, 3), config)
        self.assertGreater(distance, config.eyebrow_distance_pain_threshold)
        self.assertEqual(angle_score, 0.0)
        self.assertEqual(contraction, 0.0)
        self.assertEqual(confidence, 0.0)

    def test_eyebrow_angle_raises_confidence_for_inner_brow_lowering(self):
        config = PainMonitoringConfig()
        landmarks = _fake_face_landmarks(0.49, 0.51)
        for index in (107, 66, 105, 336, 296, 334):
            landmarks[index].y = 0.42
        distance, angle_score, contraction, confidence = eyebrow_contraction_from_landmarks(landmarks, (480, 640, 3), config)
        self.assertLess(distance, config.eyebrow_distance_pain_threshold)
        self.assertGreater(angle_score, 0.0)
        self.assertGreater(confidence, contraction)

    def test_eyebrow_confidence_needs_calibrated_distance_drop(self):
        config = PainMonitoringConfig()
        features = FramePainFeatures(
            True,
            (0, 0, 100, 100),
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            eyebrow_distance_ratio=0.20,
            eyebrow_contraction=0.75,
            eyebrow_pain_confidence=0.85,
        )
        self.assertFalse(_is_confirmed_eyebrow_pain(features, config, baseline_eyebrow_distance_ratio=0.24))

    def test_confirmed_eyebrow_drop_can_trigger_pain(self):
        config = PainMonitoringConfig()
        features = FramePainFeatures(
            True,
            (0, 0, 100, 100),
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            eyebrow_distance_ratio=0.14,
            eyebrow_contraction=0.85,
            eyebrow_pain_confidence=0.90,
        )
        self.assertTrue(_is_confirmed_eyebrow_pain(features, config, baseline_eyebrow_distance_ratio=0.24))

    def test_calm_brows_do_not_add_brow_edge_boost(self):
        config = PainMonitoringConfig()
        features = FramePainFeatures(
            True,
            (0, 0, 100, 100),
            0.10,
            0.10,
            0.10,
            0.6,
            0.05,
            eye_symmetry=0.08,
            brow_energy=0.12,
            brow_position=0.18,
            brow_motion=0.06,
            face_edge_density=0.10,
            nasal_tension=0.08,
            nose_contrast=0.09,
        )
        boost = _estimate_brow_edge_boost(features, config)
        self.assertEqual(boost, 0.0)

    def test_mouth_and_nose_micro_features_raise_heuristic_score(self):
        base = FramePainFeatures(True, (0, 0, 100, 100), 0.15, 0.18, 0.20, 0.8, 0.08)
        attentive = FramePainFeatures(
            True,
            (0, 0, 100, 100),
            0.15,
            0.18,
            0.20,
            0.8,
            0.08,
            brow_motion=0.18,
            mouth_micro_motion=0.30,
            nose_contrast=0.34,
        )
        model = PainLinearModel()
        self.assertGreater(model.predict_score(attentive), model.predict_score(base))

    def test_wheeze_probability_gets_support_from_audio_features(self):
        config = PainMonitoringConfig()
        features = FramePainFeatures(
            True,
            (0, 0, 100, 100),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            respiratory_motion=0.55,
            wheeze_tonality=0.60,
            wheeze_band_energy=0.58,
            wheeze_probability=0.28,
        )
        boosted = _support_wheeze_probability(features, 0.28, config)
        self.assertGreater(boosted, 0.28)

    def test_sustained_pain_expression_adds_confidence_over_time(self):
        config = PainMonitoringConfig()
        features = FramePainFeatures(
            True,
            (0, 0, 100, 100),
            0.24,
            0.46,
            0.38,
            1.0,
            0.08,
            brow_energy=0.50,
            brow_position=0.44,
            mouth_opening=0.32,
            nasal_tension=0.24,
            nose_contrast=0.34,
        )
        short_boost = _estimate_sustained_pain_boost(features, 0.5, config)
        held_boost = _estimate_sustained_pain_boost(features, 8.0, config)
        self.assertGreater(held_boost, short_boost)
        self.assertGreater(held_boost, 0.0)

    def test_neutral_face_guard_keeps_normal_face_below_pain_detection(self):
        config = PainMonitoringConfig()
        features = FramePainFeatures(
            True,
            (0, 0, 100, 100),
            0.12,
            0.12,
            0.10,
            1.0,
            0.03,
            brow_energy=0.10,
            brow_position=0.12,
            brow_motion=0.02,
            mouth_opening=0.08,
            mouth_micro_motion=0.02,
            lower_face_motion=0.03,
            nasal_tension=0.08,
            nose_contrast=0.08,
        )
        guarded = _apply_neutral_face_guard(3.6, features, 0.04, config)
        self.assertTrue(_is_neutral_expression(features, 0.04, config))
        self.assertLess(guarded, config.pain_start_threshold)

    def test_calibrated_neutral_face_caps_trained_model_false_positive(self):
        config = PainMonitoringConfig()
        features = FramePainFeatures(
            True,
            (0, 0, 100, 100),
            0.18,
            0.36,
            0.22,
            1.0,
            0.04,
            brow_energy=0.34,
            brow_position=0.30,
            brow_motion=0.03,
            mouth_opening=0.12,
            mouth_micro_motion=0.03,
            lower_face_motion=0.04,
            nasal_tension=0.16,
            nose_contrast=0.16,
        )
        baseline = 0.34
        guarded = _apply_neutral_face_guard(5.0, features, 0.05, config, baseline)
        self.assertLess(_relative_facial_pain_evidence(features, baseline), config.neutral_relative_evidence_margin)
        self.assertLess(guarded, config.pain_start_threshold)

    def test_eyebrow_only_activity_is_not_enough_for_pain_detected(self):
        config = PainMonitoringConfig()
        features = FramePainFeatures(
            True,
            (0, 0, 100, 100),
            0.16,
            0.55,
            0.12,
            1.0,
            0.08,
            brow_energy=0.52,
            brow_position=0.44,
            brow_motion=0.18,
            mouth_opening=0.08,
            mouth_micro_motion=0.03,
            lower_face_motion=0.04,
            nasal_tension=0.10,
            nose_contrast=0.10,
        )
        guarded = _apply_neutral_face_guard(4.0, features, 0.09, config, baseline_facial_evidence=0.30)
        self.assertLess(_pain_region_count(features, config, 0.30), config.minimum_pain_regions_required)
        self.assertLess(guarded, config.pain_start_threshold)

    def test_neutral_guard_does_not_hide_real_pain_expression(self):
        config = PainMonitoringConfig()
        features = FramePainFeatures(
            True,
            (0, 0, 100, 100),
            0.30,
            0.46,
            0.40,
            1.0,
            0.16,
            brow_energy=0.50,
            brow_position=0.45,
            brow_motion=0.20,
            mouth_opening=0.36,
            mouth_micro_motion=0.22,
            lower_face_motion=0.18,
            nasal_tension=0.26,
            nose_contrast=0.34,
        )
        guarded = _apply_neutral_face_guard(4.2, features, 0.18, config, baseline_facial_evidence=0.12)
        self.assertFalse(_is_neutral_expression(features, 0.18, config, baseline_facial_evidence=0.12))
        self.assertEqual(guarded, 4.2)

    def test_evidence_seconds_increase_and_decay(self):
        gained = _update_evidence_seconds(0.0, 0.6, 0.3, 1.0)
        decayed = _update_evidence_seconds(gained, 0.1, 0.3, 0.5)
        self.assertGreater(gained, 0.0)
        self.assertLess(decayed, gained)

    def test_sustained_wheeze_boost_increases_held_wheeze_probability(self):
        config = PainMonitoringConfig()
        features = FramePainFeatures(
            True,
            (0, 0, 100, 100),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            respiratory_motion=0.50,
            wheeze_tonality=0.62,
            wheeze_band_energy=0.66,
            wheeze_entropy=0.45,
        )
        probability = 0.36
        evidence = _wheeze_evidence(features, probability)
        boost = _estimate_sustained_wheeze_boost(features, probability, 5.0, config)
        self.assertGreater(evidence, config.sustained_wheeze_signal_threshold)
        self.assertGreater(boost, 0.0)

    def test_email_notifications_require_sender_receiver_and_password(self):
        config = PainMonitoringConfig(
            email_notifications_enabled=True,
            notification_email_from="sender@gmail.com",
            notification_email_to="receiver@gmail.com, second@gmail.com",
            notification_email_password="app-password",
        )
        self.assertTrue(email_notifications_ready(config))
        self.assertEqual(resolve_recipient_list(config), ["receiver@gmail.com", "second@gmail.com"])

    def test_notification_body_contains_patient_and_scores(self):
        alert_body = build_alert_body(7, "pain", 5.2, 0.44, "Calibration ready")
        report_body = build_session_report_body(
            7,
            {"rows": 12, "episodes_detected": 2, "total_pain_duration_s": 8.4, "max_pain_score": 6.5},
        )
        self.assertIn("Patient ID: 7", alert_body)
        self.assertIn("Pain score: 5.20/10", alert_body)
        self.assertIn("Episodes detected: 2", report_body)


if __name__ == "__main__":
    unittest.main()
