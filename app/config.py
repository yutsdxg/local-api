from __future__ import annotations

import os
import shlex
from dataclasses import dataclass
from pathlib import Path


class BaseSettings:
    """Minimal settings helper that reads values from environment variables."""

    env_prefix: str = "LOCAL_API_"

    @classmethod
    def _env(cls, key: str, default: str) -> str:
        env_key = f"{cls.env_prefix}{key}"
        return os.getenv(env_key, default)

    @classmethod
    def _split_csv(cls, key: str, default: str) -> tuple[str, ...]:
        raw_value = cls._env(key, default)
        items = [item.strip() for item in raw_value.split(",") if item.strip()]
        return tuple(items)

    @classmethod
    def _split_args(cls, key: str, default: str) -> tuple[str, ...]:
        raw_value = cls._env(key, default)
        if not raw_value.strip():
            return ()
        return tuple(shlex.split(raw_value))

    @classmethod
    def _bool(cls, key: str, default: bool) -> bool:
        raw_value = cls._env(key, str(default)).strip().lower()
        if raw_value in {"1", "true", "yes", "on"}:
            return True
        if raw_value in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"{cls.env_prefix}{key} must be a boolean.")


@dataclass(slots=True)
class Settings(BaseSettings):
    whisper_bin: str
    whisper_model_path: str
    whisper_args: tuple[str, ...]
    ffmpeg_bin: str
    whisper_tmp_dir: Path
    ytdlp_bin: str
    ytdlp_output_dir: Path
    rsync_bin: str
    obsidian_vault_root: Path
    obsidian_export_dir: Path
    obsidian_target_dirs: tuple[str, ...]
    obsidian_exclude_tags: tuple[str, ...]
    obsidian_journal_tag: str
    obsidian_topic_prefix: str
    obsidian_others_group_name: str
    google_docs_credentials_path: Path | None
    google_docs_folder_id: str | None
    google_oauth_client_id: str | None
    google_oauth_client_secret: str | None
    google_oauth_refresh_token: str | None
    google_oauth_token_uri: str
    whisper_preprocessing: str = "vad"
    whisper_normalize: bool = False
    whisper_vad_model_path: Path = Path("data/models/ggml-silero-v6.2.0.bin")
    whisper_vad_threshold: float = 0.5
    whisper_vad_min_speech_duration_ms: int = 100
    whisper_vad_min_silence_duration_ms: int = 500
    whisper_vad_speech_pad_ms: int = 200
    whisper_deepfilter_bin: str = (
        "data/models/deepfilternet/deep-filter-0.5.6-aarch64-apple-darwin"
    )
    whisper_deepfilter_model_path: Path = Path(
        "data/models/deepfilternet/DeepFilterNet3_onnx.tar.gz"
    )
    whisper_deepfilter_attenuation_limit_db: float = 12.0

    @classmethod
    def load(cls) -> "Settings":
        google_docs_credentials = cls._env("GOOGLE_DOCS_CREDENTIALS_PATH", "").strip()
        google_docs_folder_id = cls._env("GOOGLE_DOCS_FOLDER_ID", "").strip() or None
        google_oauth_client_id = cls._env("GOOGLE_OAUTH_CLIENT_ID", "").strip() or None
        google_oauth_client_secret = cls._env("GOOGLE_OAUTH_CLIENT_SECRET", "").strip() or None
        google_oauth_refresh_token = cls._env("GOOGLE_OAUTH_REFRESH_TOKEN", "").strip() or None
        google_oauth_token_uri = cls._env(
            "GOOGLE_OAUTH_TOKEN_URI", "https://oauth2.googleapis.com/token"
        ).strip()

        return cls(
            whisper_bin=cls._env(
                "WHISPER_BIN", "/Users/yuts/Data/Dev/whisper.cpp/build/bin/whisper-cli"
            ),
            whisper_model_path=cls._env(
                "WHISPER_MODEL_PATH", "/Users/yuts/Data/Dev/whisper.cpp/models/ggml-medium.bin"
            ),
            whisper_args=cls._split_args("WHISPER_ARGS", "-ng -nt -np"),
            ffmpeg_bin=cls._env("FFMPEG_BIN", "ffmpeg"),
            whisper_tmp_dir=Path(cls._env("WHISPER_TMP_DIR", "data/tmp/whisper")),
            ytdlp_bin=cls._env("YTDLP_BIN", "yt-dlp"),
            ytdlp_output_dir=Path(cls._env("YTDLP_OUTPUT_DIR", "data/tmp/yt-dlp")),
            rsync_bin=cls._env("RSYNC_BIN", "rsync"),
            obsidian_vault_root=Path(
                cls._env("OBSIDIAN_VAULT_ROOT", "/Users/yuts/Library/Mobile Documents/iCloud~md~obsidian/Documents/yuts")
            ),
            obsidian_export_dir=Path(
                cls._env("OBSIDIAN_EXPORT_DIR", "/Users/yuts/Library/Mobile Documents/iCloud~md~obsidian/Documents/yuts/integration")
            ),
            obsidian_target_dirs=cls._split_csv("OBSIDIAN_TARGET_DIRS", "inbox,journal"),
            obsidian_exclude_tags=cls._split_csv("OBSIDIAN_EXCLUDE_TAGS", "type/snippet,type/account"),
            obsidian_journal_tag=cls._env("OBSIDIAN_JOURNAL_TAG", "type/journal"),
            obsidian_topic_prefix=cls._env("OBSIDIAN_TOPIC_PREFIX", "topic/"),
            obsidian_others_group_name=cls._env("OBSIDIAN_OTHERS_GROUP_NAME", "others"),
            google_docs_credentials_path=Path(google_docs_credentials)
            if google_docs_credentials
            else None,
            google_docs_folder_id=google_docs_folder_id,
            google_oauth_client_id=google_oauth_client_id,
            google_oauth_client_secret=google_oauth_client_secret,
            google_oauth_refresh_token=google_oauth_refresh_token,
            google_oauth_token_uri=google_oauth_token_uri,
            whisper_preprocessing=cls._env("WHISPER_PREPROCESSING", "vad").strip().lower(),
            whisper_normalize=cls._bool("WHISPER_NORMALIZE", False),
            whisper_vad_model_path=Path(cls._env(
                "WHISPER_VAD_MODEL_PATH", "data/models/ggml-silero-v6.2.0.bin"
            )),
            whisper_vad_threshold=float(cls._env("WHISPER_VAD_THRESHOLD", "0.5")),
            whisper_vad_min_speech_duration_ms=int(cls._env(
                "WHISPER_VAD_MIN_SPEECH_DURATION_MS", "100"
            )),
            whisper_vad_min_silence_duration_ms=int(cls._env(
                "WHISPER_VAD_MIN_SILENCE_DURATION_MS", "500"
            )),
            whisper_vad_speech_pad_ms=int(cls._env("WHISPER_VAD_SPEECH_PAD_MS", "200")),
            whisper_deepfilter_bin=cls._env(
                "WHISPER_DEEPFILTER_BIN",
                "data/models/deepfilternet/deep-filter-0.5.6-aarch64-apple-darwin",
            ),
            whisper_deepfilter_model_path=Path(cls._env(
                "WHISPER_DEEPFILTER_MODEL_PATH",
                "data/models/deepfilternet/DeepFilterNet3_onnx.tar.gz",
            )),
            whisper_deepfilter_attenuation_limit_db=float(cls._env(
                "WHISPER_DEEPFILTER_ATTENUATION_LIMIT_DB", "12"
            )),
        )


def get_settings() -> Settings:
    """Factory that can be reused with FastAPI dependencies if needed."""
    return Settings.load()
