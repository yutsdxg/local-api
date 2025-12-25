from __future__ import annotations

import unittest

import numpy as np

from app.services.similar_tones import MultiSegmentSelector


class TestMultiSegmentSelector(unittest.TestCase):
    def test_extract_returns_full_audio_if_too_short(self) -> None:
        selector = MultiSegmentSelector(
            segment_seconds=1.0,
            sample_rate=10,
            rms_window_seconds=0.2,
            target_db_offset=14.0,
        )
        audio = np.array([0.0, 0.5, -0.25], dtype=np.float32)

        result = selector.extract_segments(audio)

        self.assertEqual(1, len(result))
        np.testing.assert_array_equal(audio, result[0])

    def test_extract_uses_first_peak_occurrence(self) -> None:
        selector = MultiSegmentSelector(
            segment_seconds=0.2,
            sample_rate=10,
            rms_window_seconds=0.2,
            target_db_offset=14.0,
        )
        audio = np.array([0.0, 1.0, 1.0, 0.0, 0.0], dtype=np.float32)

        result = selector.extract_segments(audio)

        self.assertEqual(2, len(result))
        np.testing.assert_array_equal(np.array([0.0, 1.0], dtype=np.float32), result[0])

    def test_extract_clamps_to_end(self) -> None:
        selector = MultiSegmentSelector(
            segment_seconds=0.4,
            sample_rate=10,
            rms_window_seconds=0.2,
            target_db_offset=14.0,
        )
        audio = np.array([0.0, 0.2, 0.1, 0.3, 1.0], dtype=np.float32)

        result = selector.extract_segments(audio)

        self.assertEqual(2, len(result))
        np.testing.assert_array_equal(
            np.array([0.2, 0.1, 0.3, 1.0], dtype=np.float32),
            result[0],
        )

    def test_extract_falls_back_to_peak_only_for_silence(self) -> None:
        selector = MultiSegmentSelector(
            segment_seconds=0.2,
            sample_rate=10,
            rms_window_seconds=0.2,
            target_db_offset=14.0,
        )
        audio = np.zeros(5, dtype=np.float32)

        result = selector.extract_segments(audio)

        self.assertEqual(1, len(result))
        np.testing.assert_array_equal(np.array([0.0, 0.0], dtype=np.float32), result[0])
