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
            outputs: list[str] = []

            stdout = "\n".join(
                [
                    "sending incremental file list",
                    ">f++++++ some/newfile.txt",
                    ">f.st... modified/file1.txt",
                    "*deleting   old/removed_file.txt",
                    "",
                    "sent 123 bytes  received 456 bytes  789.00 bytes/sec",
                    "total size is 1,234  speedup is 2.00",
                ]
            )

            def fake_run(cmd: list[str], check: bool, capture_output: bool, text: bool):
                called_commands.append(cmd)
                return subprocess.CompletedProcess(cmd, 0, stdout=stdout)

            with mock.patch.object(rsync.subprocess, "run", side_effect=fake_run):
                synced = rsync.sync_directories(src, dst, rsync_bin="rsync")

            self.assertEqual(
                [
                    {
                        "name": "alpha",
                        "changes": [
                            ">f++++++ some/newfile.txt",
                            ">f.st... modified/file1.txt",
                            "*deleting   old/removed_file.txt",
                        ],
                    },
                    {
                        "name": "bravo",
                        "changes": [
                            ">f++++++ some/newfile.txt",
                            ">f.st... modified/file1.txt",
                            "*deleting   old/removed_file.txt",
                        ],
                    },
                ],
                synced,
            )
            self.assertEqual(
                [
                    ["rsync", "-aiv", "--delete", f"{src / 'alpha'}/", f"{dst / 'alpha'}/"],
                    ["rsync", "-aiv", "--delete", f"{src / 'bravo'}/", f"{dst / 'bravo'}/"],
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

            def fake_run(cmd: list[str], check: bool, capture_output: bool, text: bool):
                raise subprocess.CalledProcessError(cmd=cmd, returncode=1)

            with mock.patch.object(rsync.subprocess, "run", side_effect=fake_run):
                with self.assertRaises(rsync.RsyncExecutionError):
                    rsync.sync_directories(src, dst)
