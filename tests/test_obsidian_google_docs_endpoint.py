from __future__ import annotations

import asyncio
import unittest
from unittest import mock

from fastapi import HTTPException

from app import main
from app.services import obsidian_google_docs as obsidian_google_docs_service


class TestObsidianGoogleDocsEndpoint(unittest.TestCase):
    def test_google_docs_export_returns_summary(self) -> None:
        expected = obsidian_google_docs_service.GoogleDocsExportResult(
            source_path="/exports/others.md",
            document_id="doc-id",
            document_url="https://docs.google.com/document/d/doc-id/edit",
            title="others",
            folder_id=None,
            skipped=False,
        )

        with mock.patch.object(
            main.obsidian_google_docs_service,
            "export_markdown_to_google_docs",
            return_value=expected,
        ) as mocked_export:
            result = asyncio.run(
                main.export_obsidian_exports_to_google_docs("others.md", None, None)
            )

        mocked_export.assert_called_once_with(
            main.settings,
            mock.ANY,
            title=None,
            folder_id=None,
        )
        self.assertEqual(
            {
                "source_path": "/exports/others.md",
                "document_id": "doc-id",
                "document_url": "https://docs.google.com/document/d/doc-id/edit",
                "title": "others",
                "folder_id": None,
                "skipped": False,
            },
            result,
        )

    def test_google_docs_export_returns_400_on_path_error(self) -> None:
        with mock.patch.object(
            main.obsidian_google_docs_service,
            "export_markdown_to_google_docs",
            side_effect=obsidian_google_docs_service.GoogleDocsPathError("bad path"),
        ):
            with self.assertRaises(HTTPException) as excinfo:
                asyncio.run(
                    main.export_obsidian_exports_to_google_docs("others.md", None, None)
                )

        self.assertEqual(400, excinfo.exception.status_code)
        self.assertEqual("bad path", excinfo.exception.detail)

    def test_google_docs_export_returns_500_on_dependency_error(self) -> None:
        with mock.patch.object(
            main.obsidian_google_docs_service,
            "export_markdown_to_google_docs",
            side_effect=obsidian_google_docs_service.GoogleDocsDependencyError("missing"),
        ):
            with self.assertRaises(HTTPException) as excinfo:
                asyncio.run(
                    main.export_obsidian_exports_to_google_docs("others.md", None, None)
                )

        self.assertEqual(500, excinfo.exception.status_code)
        self.assertEqual("missing", excinfo.exception.detail)
