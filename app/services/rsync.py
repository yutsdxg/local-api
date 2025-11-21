from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable

from typing_extensions import TypedDict


class SyncResult(TypedDict):
    name: str
    changes: list[str]


class RsyncPathError(ValueError):
    """Raised when source or destination paths are invalid."""


class RsyncExecutionError(RuntimeError):
    """Raised when rsync fails for a directory."""


def sync_directories(
    src_root: Path | str, dst_root: Path | str, rsync_bin: str = "rsync"
) -> list[SyncResult]:
    """
    Sync the first-level subdirectories from src_root into dst_root using rsync.
    """

    src_path = Path(src_root)
    dst_path = Path(dst_root)

    _ensure_directory(src_path, "Source directory not found or not a directory")
    _ensure_directory(dst_path, "Destination directory not found or not a directory")

    synced: list[SyncResult] = []
    for directory in _child_directories(src_path):
        name = directory.name
        dest_dir = dst_path / name
        cmd = [rsync_bin, "-aiv", "--delete", f"{directory}/", f"{dest_dir}/"]

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            raise RsyncExecutionError(f"rsync failed for directory: {name}") from exc

        changes = _parse_changes(result.stdout)
        synced.append({"name": name, "changes": changes})

    return synced


def _child_directories(path: Path) -> Iterable[Path]:
    return sorted((item for item in path.iterdir() if item.is_dir()), key=lambda p: p.name)


def _ensure_directory(path: Path, message: str) -> None:
    if not path.is_dir():
        raise RsyncPathError(f"{message}: {path}")


def _parse_changes(output: str) -> list[str]:
    changes: list[str] = []
    for raw_line in output.splitlines():
        line = raw_line.rstrip("\n")
        if not line:
            continue
        if line in ("sending incremental file list",):
            continue
        if line.startswith("sent ") or line.startswith("total size "):
            continue
        changes.append(line)
    return changes
