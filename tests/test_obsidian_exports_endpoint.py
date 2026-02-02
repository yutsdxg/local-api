from __future__ import annotations

import asyncio
import unittest
from unittest import mock

from fastapi import HTTPException

from app import main
from app.services import obsidian_exports as obsidian_exports_service


class TestObsidianExportsEndpoint(unittest.TestCase):
    def test_merge_exports_returns_summary(self) -> None:
        expected = {
            "vault_root": "/vault",
            "output_dir": "/exports",
            "generated": [],
            "skipped": {"excluded_tag": 0, "no_target_tag": 0, "parse_error": 0},
        }

        with mock.patch.object(
            main.obsidian_exports_service, "merge_exports", return_value=expected
        ) as mocked_merge:
            result = asyncio.run(main.merge_obsidian_exports())

        mocked_merge.assert_called_once_with(main.settings)
        self.assertEqual(expected, result)

    def test_merge_exports_returns_400_on_path_error(self) -> None:
        with mock.patch.object(
            main.obsidian_exports_service,
            "merge_exports",
            side_effect=obsidian_exports_service.ObsidianExportPathError("bad path"),
        ):
            with self.assertRaises(HTTPException) as excinfo:
                asyncio.run(main.merge_obsidian_exports())

        self.assertEqual(400, excinfo.exception.status_code)
        self.assertEqual("bad path", excinfo.exception.detail)

    def test_merge_exports_returns_400_on_front_matter_error(self) -> None:
        with mock.patch.object(
            main.obsidian_exports_service,
            "merge_exports",
            side_effect=obsidian_exports_service.ObsidianFrontMatterError("bad front matter"),
        ):
            with self.assertRaises(HTTPException) as excinfo:
                asyncio.run(main.merge_obsidian_exports())

        self.assertEqual(400, excinfo.exception.status_code)
        self.assertEqual("bad front matter", excinfo.exception.detail)
