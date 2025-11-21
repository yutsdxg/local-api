from __future__ import annotations

import asyncio
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from fastapi import HTTPException

from app import main
from app.services import rsync as rsync_service


class TestRsyncEndpoint(unittest.TestCase):
    def test_rsync_endpoint_returns_synced_list(self) -> None:
        with TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            src = base / "src"
            dst = base / "dst"

            expected = [{"name": "alpha", "changes": [">f.st some/file"]}]
            with mock.patch.object(main.rsync_service, "sync_directories", return_value=expected) as mocked_sync:
                result = asyncio.run(main.sync_directories(src=str(src), dst=str(dst)))

            mocked_sync.assert_called_once_with(Path(src), Path(dst), main.settings.rsync_bin)
            self.assertEqual({"synced": expected}, result)

    def test_rsync_endpoint_returns_400_on_path_error(self) -> None:
        with TemporaryDirectory() as tmpdir:
            params = {"src": str(Path(tmpdir) / "src"), "dst": str(Path(tmpdir) / "dst")}

            with mock.patch.object(
                main.rsync_service, "sync_directories", side_effect=rsync_service.RsyncPathError("invalid path")
            ):
                with self.assertRaises(HTTPException) as excinfo:
                    asyncio.run(main.sync_directories(**params))

            self.assertEqual(400, excinfo.exception.status_code)
            self.assertEqual("invalid path", excinfo.exception.detail)

    def test_rsync_endpoint_returns_500_on_rsync_failure(self) -> None:
        with TemporaryDirectory() as tmpdir:
            params = {"src": str(Path(tmpdir) / "src"), "dst": str(Path(tmpdir) / "dst")}

            with mock.patch.object(
                main.rsync_service, "sync_directories", side_effect=rsync_service.RsyncExecutionError("rsync broke")
            ):
                with self.assertRaises(HTTPException) as excinfo:
                    asyncio.run(main.sync_directories(**params))

            self.assertEqual(500, excinfo.exception.status_code)
            self.assertEqual("rsync broke", excinfo.exception.detail)
