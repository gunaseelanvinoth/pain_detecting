import sys
import unittest
from pathlib import Path

import pandas as pd

SRC_ROOT = Path(__file__).resolve().parent.parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pain_monitoring.model import PainLinearModel, train_linear_model_from_frame
from pain_monitoring.types import FramePainFeatures


class ModelTests(unittest.TestCase):
    def test_predict_clamps_to_0_10(self):
        model = PainLinearModel(10, 10, 10, 10, 10, 10)
        features = FramePainFeatures(True, (0, 0, 100, 100), 1.0, 1.0, 1.0, 1.0, 1.0)
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
                "pain_label_0_10": [1.0, 2.0, 3.0, 4.0],
            }
        )
        model, metrics = train_linear_model_from_frame(frame)
        self.assertIsInstance(model, PainLinearModel)
        self.assertIn("mae", metrics)
        self.assertEqual(metrics["rows_used"], 4)


if __name__ == "__main__":
    unittest.main()
