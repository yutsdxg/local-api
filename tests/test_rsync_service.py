from __future__ import annotations

import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from app.services import rsync


class TestSyncDirectories(unittest.TestCase):
    def test_sync_directories_runs_rsync_per_child(self) -> None:
        with TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            src = base / "src"
            dst = base / "dst"
            src.mkdir()
            dst.mkdir()

            (src / "bravo").mkdir()
            (src / "alpha").mkdir()

            called_commands: list[list[str]] = []

            def fake_run(cmd: list[str], check: bool) -> None:
                called_commands.append(cmd)

            with mock.patch.object(rsync.subprocess, "run", side_effect=fake_run):
                synced = rsync.sync_directories(src, dst, rsync_bin="rsync")

            self.assertEqual(["alpha", "bravo"], synced)
            self.assertEqual(
                [
                    ["rsync", "-av", "--delete", f"{src / 'alpha'}/", f"{dst / 'alpha'}/"],
                    ["rsync", "-av", "--delete", f"{src / 'bravo'}/", f"{dst / 'bravo'}/"],
                ],
                called_commands,
            )

    def test_sync_directories_requires_existing_paths(self) -> None:
        with TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            dst = base / "dst"
            dst.mkdir()

            with self.assertRaises(rsync.RsyncPathError):
                rsync.sync_directories(base / "missing", dst)

            src = base / "src"
            src.mkdir()

            with self.assertRaises(rsync.RsyncPathError):
                rsync.sync_directories(src, base / "missing-dst")

    def test_sync_directories_raises_on_rsync_failure(self) -> None:
        with TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            src = base / "src"
            dst = base / "dst"
            src.mkdir()
            dst.mkdir()

            (src / "only").mkdir()

            def fake_run(cmd: list[str], check: bool) -> None:
                raise subprocess.CalledProcessError(cmd=cmd, returncode=1)

            with mock.patch.object(rsync.subprocess, "run", side_effect=fake_run):
                with self.assertRaises(rsync.RsyncExecutionError):
                    rsync.sync_directories(src, dst)
