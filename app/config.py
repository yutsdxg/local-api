from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


class BaseSettings:
    """Minimal settings helper that reads values from environment variables."""

    env_prefix: str = "LOCAL_API_"

    @classmethod
    def _env(cls, key: str, default: str) -> str:
        env_key = f"{cls.env_prefix}{key}"
        return os.getenv(env_key, default)


@dataclass(slots=True)
class Settings(BaseSettings):
    whisper_bin: str
    whisper_model_path: str
    ffmpeg_bin: str
    whisper_tmp_dir: Path
    ytdlp_bin: str
    ytdlp_output_dir: Path
    rsync_bin: str

    @classmethod
    def load(cls) -> "Settings":
        return cls(
            whisper_bin=cls._env(
                "WHISPER_BIN", "/opt/homebrew/Cellar/whisper-cpp/1.8.2/bin/whisper-cli"
            ),
            whisper_model_path=cls._env(
                "WHISPER_MODEL_PATH", "model/ggml-medium.bin"
            ),
            ffmpeg_bin=cls._env("FFMPEG_BIN", "ffmpeg"),
            whisper_tmp_dir=Path(cls._env("WHISPER_TMP_DIR", "data/tmp/whisper")),
            ytdlp_bin=cls._env("YTDLP_BIN", "yt-dlp"),
            ytdlp_output_dir=Path(cls._env("YTDLP_OUTPUT_DIR", "data/tmp/yt-dlp")),
            rsync_bin=cls._env("RSYNC_BIN", "rsync"),
        )


def get_settings() -> Settings:
    """Factory that can be reused with FastAPI dependencies if needed."""
    return Settings.load()
