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
    chord_melody_input_dir: Path
    chord_melody_log_dir: Path
    chord_melody_time_unit: float
    chord_melody_poly_threshold: float
    chord_melody_poly_note_count: int
    chord_melody_stability_threshold: float
    similar_tones_cache_dir: Path
    similar_tones_device: str
    similar_tones_segment_seconds: float
    similar_tones_rms_window_seconds: float
    similar_tones_target_db_offset: float
    similar_tones_peak_weight: float
    similar_tones_target_weight: float

    @classmethod
    def load(cls) -> "Settings":
        return cls(
            whisper_bin=cls._env(
                "WHISPER_BIN", "/Users/yuts/Data/Dev/whisper.cpp/build/bin/whisper-cli"
            ),
            whisper_model_path=cls._env(
                "WHISPER_MODEL_PATH", "/Users/yuts/Data/Dev/whisper.cpp/models/ggml-medium.bin"
            ),
            ffmpeg_bin=cls._env("FFMPEG_BIN", "ffmpeg"),
            whisper_tmp_dir=Path(cls._env("WHISPER_TMP_DIR", "data/tmp/whisper")),
            ytdlp_bin=cls._env("YTDLP_BIN", "yt-dlp"),
            ytdlp_output_dir=Path(cls._env("YTDLP_OUTPUT_DIR", "data/tmp/yt-dlp")),
            rsync_bin=cls._env("RSYNC_BIN", "rsync"),
            chord_melody_input_dir=Path(cls._env("CHORD_MELODY_INPUT_DIR", "data/chord-melody/input")),
            chord_melody_log_dir=Path(cls._env("CHORD_MELODY_LOG_DIR", "data/chord-melody/logs")),
            chord_melody_time_unit=float(cls._env("CHORD_MELODY_TIME_UNIT", "0.1")),
            chord_melody_poly_threshold=float(cls._env("CHORD_MELODY_POLY_THRESHOLD", "0.4")),
            chord_melody_poly_note_count=int(cls._env("CHORD_MELODY_POLY_NOTE_COUNT", "3")),
            chord_melody_stability_threshold=float(cls._env("CHORD_MELODY_STABILITY_THRESHOLD", "0.7")),
            similar_tones_cache_dir=Path(cls._env("SIMILAR_TONES_CACHE_DIR", "data/similar-tones/cache")),
            similar_tones_device=cls._env("SIMILAR_TONES_DEVICE", "cpu"),
            similar_tones_segment_seconds=float(cls._env("SIMILAR_TONES_SEGMENT_SECONDS", "0.2")),
            similar_tones_rms_window_seconds=float(
                cls._env("SIMILAR_TONES_RMS_WINDOW_SECONDS", "0.02")
            ),
            similar_tones_target_db_offset=float(
                cls._env("SIMILAR_TONES_TARGET_DB_OFFSET", "14.0")
            ),
            similar_tones_peak_weight=float(cls._env("SIMILAR_TONES_PEAK_WEIGHT", "0.7")),
            similar_tones_target_weight=float(cls._env("SIMILAR_TONES_TARGET_WEIGHT", "0.3")),
        )


def get_settings() -> Settings:
    """Factory that can be reused with FastAPI dependencies if needed."""
    return Settings.load()
