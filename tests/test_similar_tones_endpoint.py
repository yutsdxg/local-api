from __future__ import annotations

import asyncio
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from fastapi import HTTPException

from app import main
from app.services import similar_tones as similar_tones_service


class TestSimilarTonesEndpoint(unittest.TestCase):
    def test_create_index_returns_summary(self) -> None:
        with TemporaryDirectory() as tmpdir:
            preset_dir = Path(tmpdir) / "presets"
            output_path = Path(tmpdir) / "index.pkl"
            expected = {"index_path": str(output_path), "indexed_count": 10, "skipped_count": 2}

            with mock.patch.object(
                main.similar_tones_service, "create_index", return_value=expected
            ) as mocked_index:
                result = asyncio.run(
                    main.create_similar_tones_index(
                        preset_dir=str(preset_dir), output_path=str(output_path)
                    )
                )

            mocked_index.assert_called_once_with(
                main.settings, preset_dir, output_path
            )
            self.assertEqual(expected, result)

    def test_create_index_returns_400_on_path_error(self) -> None:
        with TemporaryDirectory() as tmpdir:
            preset_dir = Path(tmpdir) / "presets"
            output_path = Path(tmpdir) / "index.pkl"

            with mock.patch.object(
                main.similar_tones_service,
                "create_index",
                side_effect=similar_tones_service.SimilarTonesPathError("bad path"),
            ):
                with self.assertRaises(HTTPException) as excinfo:
                    asyncio.run(
                        main.create_similar_tones_index(
                            preset_dir=str(preset_dir), output_path=str(output_path)
                        )
                    )

            self.assertEqual(400, excinfo.exception.status_code)
            self.assertEqual("bad path", excinfo.exception.detail)

    def test_search_returns_text_and_results(self) -> None:
        with TemporaryDirectory() as tmpdir:
            target_path = Path(tmpdir) / "target.wav"
            index_path = Path(tmpdir) / "index.pkl"
            expected = {
                "text": "Rank  Similarity  File Name  Path\n",
                "results": [{"file_path": "/x.wav", "file_name": "x.wav", "similarity_score": 1.0, "rank": 1}],
            }

            with mock.patch.object(
                main.similar_tones_service, "search_similar", return_value=expected
            ) as mocked_search:
                result = asyncio.run(
                    main.search_similar_tones(
                        target_path=str(target_path), index_path=str(index_path), top_k=3
                    )
                )

            mocked_search.assert_called_once_with(
                main.settings, target_path, index_path, 3
            )
            self.assertEqual(expected, result)

    def test_search_returns_500_on_execution_error(self) -> None:
        with TemporaryDirectory() as tmpdir:
            target_path = Path(tmpdir) / "target.wav"
            index_path = Path(tmpdir) / "index.pkl"

            with mock.patch.object(
                main.similar_tones_service,
                "search_similar",
                side_effect=similar_tones_service.SimilarTonesExecutionError("failed"),
            ):
                with self.assertRaises(HTTPException) as excinfo:
                    asyncio.run(
                        main.search_similar_tones(
                            target_path=str(target_path), index_path=str(index_path), top_k=3
                        )
                    )

            self.assertEqual(500, excinfo.exception.status_code)
            self.assertEqual("failed", excinfo.exception.detail)
