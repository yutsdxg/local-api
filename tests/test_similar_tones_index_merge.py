from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from app.services.similar_tones import VectorStore, _load_combined_index


class TestLoadCombinedIndex(unittest.TestCase):
    def test_load_combined_index_concatenates_vectors(self) -> None:
        with TemporaryDirectory() as tmpdir:
            store = VectorStore()
            index_a = Path(tmpdir) / "a.pkl"
            index_b = Path(tmpdir) / "b.pkl"

            store.save_index(
                np.array([[1.0, 0.0]], dtype=np.float32), ["a.wav"], index_a
            )
            store.save_index(
                np.array([[0.0, 1.0]], dtype=np.float32), ["b.wav"], index_b
            )

            vectors, file_paths, index_paths = _load_combined_index(
                [index_a, index_b], store
            )

            np.testing.assert_array_equal(
                np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
                vectors,
            )
            self.assertEqual(["a.wav", "b.wav"], file_paths)
            self.assertEqual([str(index_a), str(index_b)], index_paths)
