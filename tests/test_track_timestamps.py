
import unittest

import numpy as np

from flowmetric.domain import Track
from flowmetric.events import load_tracks


class TrackTimestampTests(unittest.TestCase):
    def test_explicit_timestamps_define_track_duration(self):
        track = Track(
            "1",
            "customer",
            np.array([0, 1, 2]),
            np.array([[0, 0, 10, 20]] * 3),
            60.0,
            np.array([0.0, 0.05, 0.15]),
        )

        self.assertAlmostEqual(track.t0, 0.0)
        self.assertAlmostEqual(track.t1, 0.15)
        self.assertAlmostEqual(track.life, 0.15)

    def test_legacy_track_uses_fps(self):
        track = Track(
            "1",
            "customer",
            np.array([0, 10]),
            np.array([[0, 0, 10, 20]] * 2),
            10.0,
        )

        self.assertAlmostEqual(track.life, 1.0)

    def test_load_tracks_reads_timestamps(self):
        data = {
            "fps": 60.0,
            "tracks": {
                "1": {
                    "frames": [0, 1, 2],
                    "times": [0.0, 0.05, 0.15],
                    "boxes": [[0, 0, 10, 20]] * 3,
                }
            },
        }

        track = load_tracks(data)[0]
        self.assertAlmostEqual(track.life, 0.15)


if __name__ == "__main__":
    unittest.main()

