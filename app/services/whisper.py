from __future__ import annotations

import subprocess
import tempfile
import uuid
from pathlib import Path

from fastapi import UploadFile

from app.config import Settings


class WhisperError(RuntimeError):
    """Raised when Whisper processing fails."""


async def transcribe_upload(
    file: UploadFile,
    settings: Settings,
    language: str = "ja",
) -> str:
    """
    Convert an uploaded audio file into text using whisper-cli.
    """

    tmp_dir = settings.whisper_tmp_dir
    if not tmp_dir.exists():
        raise WhisperError(f"Temporary directory not found: {tmp_dir}")

    if not file.filename:
        file.filename = f"{uuid.uuid4()}.tmp"

    session_id = uuid.uuid4().hex
    output_prefix = tmp_dir / f"whisper_output_{session_id}"
    output_path = output_prefix.with_suffix(".txt")

    if output_path.exists():
        output_path.unlink()

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / file.filename
        input_path.write_bytes(await file.read())

        whisper_input_path = tmp_dir / f"whisper_input_{session_id}.wav"
        audio_filters = (
            "highpass=f=120, "
            "lowpass=f=8000, "
            "dynaudnorm=f=200:g=7, "
            "silenceremove=start_periods=1:start_duration=0.8:"
            "start_threshold=-50dB:stop_periods=-1:stop_duration=0.8:"
            "stop_threshold=-50dB"
        )
        convert_cmd = [
            settings.ffmpeg_bin,
            "-i",
            str(input_path),
            "-af",
            audio_filters,
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            "-y",
            str(whisper_input_path),
        ]
        _run_command(convert_cmd, "ffmpeg conversion failed.")

        whisper_cmd = [
            settings.whisper_bin,
            "-m",
            settings.whisper_model_path,
            "-f",
            str(whisper_input_path),
            "-otxt",
            "-of",
            str(output_prefix),
            "-l",
            language,
            *settings.whisper_args,
        ]
        _run_command(whisper_cmd, "Whisper failed to process the audio file.")

        if not output_path.exists():
            raise WhisperError("Whisper output not found.")

        return output_path.read_text(encoding="utf-8")


def _run_command(cmd: list[str], error_message: str) -> None:
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise WhisperError(error_message) from exc
