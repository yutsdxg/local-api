from __future__ import annotations

import subprocess
import uuid
from pathlib import Path

from app.config import Settings


class VideoIdValidationError(ValueError):
    """Raised when the provided video ID is invalid."""


class YtDlpError(RuntimeError):
    """Raised when yt-dlp fails or output is missing."""


ALLOWED_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_")


def download_audio(video_id: str, settings: Settings) -> tuple[Path, str]:
    output_dir = settings.ytdlp_output_dir
    if not output_dir.exists():
        raise YtDlpError(f"Output dir missing: {output_dir}")

    sanitized_id = _sanitize_video_id(video_id)
    youtube_url = f"https://www.youtube.com/watch?v={sanitized_id}"

    session_id = uuid.uuid4().hex
    output_prefix = output_dir / f"input_{session_id}"
    output_template = f"{output_prefix}.%(ext)s"

    cmd = [
        settings.ytdlp_bin,
        "-x",
        "--audio-format",
        "wav",
        "-o",
        output_template,
        youtube_url,
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise YtDlpError("yt-dlp command failed.") from exc

    output_file = output_prefix.with_suffix(".wav")
    if not output_file.exists():
        raise YtDlpError("yt-dlp output not found.")

    return output_file, youtube_url


def _sanitize_video_id(video_id: str) -> str:
    if not video_id:
        raise VideoIdValidationError("video_id is required.")

    sanitized = video_id.strip()
    if not sanitized:
        raise VideoIdValidationError("video_id is required.")

    if not all(char in ALLOWED_CHARS for char in sanitized):
        raise VideoIdValidationError("video_id contains invalid characters.")

    return sanitized
