from __future__ import annotations

import unittest

import numpy as np

from app.services.similar_tones import (
    _combine_weighted_scores,
    _update_max_scores,
    _update_max_scores_with_source,
)


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
            {"a.wav": {"score": 0.9, "index_path": "a.pkl"}, "b.wav": {"score": 0.1, "index_path": "a.pkl"}},
            {"a.wav": {"score": 0.2, "index_path": "b.pkl"}, "b.wav": {"score": 0.8, "index_path": "b.pkl"}},
        ]
        weights = [0.7, 0.3]

        combined = _combine_weighted_scores(scores_by_segment, weights)

        self.assertAlmostEqual(0.69, combined["a.wav"]["score"], places=6)
        self.assertAlmostEqual(0.31, combined["b.wav"]["score"], places=6)
        self.assertEqual("a.pkl", combined["a.wav"]["index_path"])
        self.assertEqual("b.pkl", combined["b.wav"]["index_path"])


class TestUpdateMaxScoresWithSource(unittest.TestCase):
    def test_update_max_scores_with_source_prefers_higher(self) -> None:
        scores_by_path: dict[str, dict[str, float]] = {}
        file_paths = ["a.wav", "b.wav", "a.wav"]
        source_index_paths = ["a.pkl", "a.pkl", "b.pkl"]
        similarities = np.array([0.1, 0.3, 0.8], dtype=np.float32)

        _update_max_scores_with_source(
            file_paths, source_index_paths, similarities, scores_by_path
        )

        self.assertEqual(set(scores_by_path.keys()), {"a.wav", "b.wav"})
        self.assertAlmostEqual(0.8, scores_by_path["a.wav"]["score"], places=6)
        self.assertEqual("b.pkl", scores_by_path["a.wav"]["index_path"])
