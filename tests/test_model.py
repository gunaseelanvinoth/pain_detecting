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
from pain_monitoring.config import PainMonitoringConfig
from pain_monitoring.decision import pain_detected_from_face, pain_status_text
from pain_monitoring.kaggle_import import import_kaggle_respiratory_dataset
from pain_monitoring.model import PainLinearModel, train_linear_model_from_frame
from pain_monitoring.overlay import pain_detection_label, wheeze_detection_label
from pain_monitoring.types import FramePainFeatures


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
                "mouth_opening": [0.1, 0.2, 0.4, 0.5],
                "lower_face_motion": [0.05, 0.1, 0.22, 0.3],
                "face_edge_density": [0.2, 0.22, 0.25, 0.31],
                "nasal_tension": [0.12, 0.18, 0.24, 0.29],
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
        self.assertEqual(pain_detection_label("None"), "There is no pain detected")
        self.assertEqual(pain_detection_label("Moderate"), "Pain is detected")
        self.assertEqual(wheeze_detection_label("None"), "No Wheezing")
        self.assertEqual(wheeze_detection_label("Low"), "Wheezing Detected")

    def test_face_contrast_rule_detects_pain_when_eye_mouth_nose_change(self):
        features = FramePainFeatures(
            True,
            (0, 0, 100, 100),
            0.75,
            0.42,
            0.70,
            1.0,
            0.28,
            eye_symmetry=0.35,
            brow_energy=0.40,
            mouth_opening=0.62,
            lower_face_motion=0.38,
            face_edge_density=0.36,
            nasal_tension=0.58,
        )
        self.assertTrue(pain_detected_from_face(features, 5.1, PainMonitoringConfig()))
        self.assertEqual(pain_status_text(True), "Pain is detected")

    def test_face_contrast_rule_returns_no_pain_for_neutral_face(self):
        features = FramePainFeatures(
            True,
            (0, 0, 100, 100),
            0.05,
            0.08,
            0.10,
            0.0,
            0.04,
            eye_symmetry=0.03,
            brow_energy=0.05,
            mouth_opening=0.08,
            lower_face_motion=0.04,
            face_edge_density=0.10,
            nasal_tension=0.08,
        )
        self.assertFalse(pain_detected_from_face(features, 3.8, PainMonitoringConfig()))
        self.assertEqual(pain_status_text(False), "There is no pain detected")


if __name__ == "__main__":
    unittest.main()
