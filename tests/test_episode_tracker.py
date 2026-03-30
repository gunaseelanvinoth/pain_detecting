import sys
import unittest
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parent.parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pain_monitoring.config import PainMonitoringConfig
from pain_monitoring.episode_tracker import update_duration_state
from pain_monitoring.types import RuntimeState


class EpisodeTrackerTests(unittest.TestCase):
    def test_episode_starts_and_ends_after_hold_windows(self):
        config = PainMonitoringConfig(
            pain_start_threshold=4.0,
            pain_end_threshold=2.0,
            start_hold_seconds=1.0,
            end_hold_seconds=1.0,
            smoothing_alpha=1.0,
        )
        runtime = RuntimeState()

        samples = [
            (0.0, 4.2),
            (0.6, 4.3),
            (1.2, 4.5),
            (1.8, 4.6),
            (2.4, 1.7),
            (3.0, 1.6),
        ]

        states = []
        for ts, score in samples:
            states.append(update_duration_state(runtime, score, ts, config))

        self.assertTrue(any(item.started_now for item in states))
        self.assertTrue(any(item.ended_now for item in states))
        self.assertGreaterEqual(runtime.closed_duration_s, 0.5)
        self.assertEqual(len(runtime.closed_episodes), 1)


if __name__ == "__main__":
    unittest.main()
