from __future__ import annotations

import unittest

import numpy as np

from app.services.similar_tones import _combine_weighted_scores, _update_max_scores


class TestUpdateMaxScores(unittest.TestCase):
    def test_update_max_scores_prefers_higher(self) -> None:
        scores_by_path: dict[str, float] = {}
        file_paths = ["a.wav", "b.wav", "a.wav"]
        similarities = np.array([0.1, 0.3, 0.8], dtype=np.float32)

        _update_max_scores(file_paths, similarities, scores_by_path)

        self.assertEqual(set(scores_by_path.keys()), {"a.wav", "b.wav"})
        self.assertAlmostEqual(0.8, scores_by_path["a.wav"], places=6)
        self.assertAlmostEqual(0.3, scores_by_path["b.wav"], places=6)


class TestCombineWeightedScores(unittest.TestCase):
    def test_combine_weighted_scores_applies_weights(self) -> None:
        scores_by_segment = [
            {"a.wav": 0.9, "b.wav": 0.1},
            {"a.wav": 0.2, "b.wav": 0.8},
        ]
        weights = [0.7, 0.3]

        combined = _combine_weighted_scores(scores_by_segment, weights)

        self.assertAlmostEqual(0.69, combined["a.wav"], places=6)
        self.assertAlmostEqual(0.31, combined["b.wav"], places=6)
