from __future__ import annotations

import asyncio
import math
import shutil
import subprocess
import tempfile
import threading
import uuid
import wave
from pathlib import Path

from fastapi import UploadFile

from app.config import Settings


class WhisperError(RuntimeError):
    """Raised when Whisper processing fails."""


LEGACY_AUDIO_FILTERS = (
    "highpass=f=120, "
    "lowpass=f=8000, "
    "dynaudnorm=f=200:g=7, "
    "silenceremove=start_periods=1:start_duration=0.8:"
    "start_threshold=-50dB:stop_periods=-1:stop_duration=0.8:"
    "stop_threshold=-50dB"
)
NORMALIZE_AUDIO_FILTER = "dynaudnorm=f=200:g=7"
_TRANSCRIPTION_LOCK = threading.Lock()


async def transcribe_upload(
    file: UploadFile,
    settings: Settings,
    language: str = "ja",
) -> str:
    """Read the upload, then run the blocking pipeline outside the event loop."""
    audio = await file.read()
    return await asyncio.to_thread(_transcribe_serialized, audio, settings, language)


def _transcribe_serialized(audio: bytes, settings: Settings, language: str) -> str:
    # This lock serializes the full pipeline within one API process, including
    # requests submitted from different event loops. Separate server processes
    # have separate locks. Once this worker starts, cancelling the await (for
    # example on client disconnect) does not stop its thread or subprocesses.
    # The worker therefore owns the lock and temporary files until completion.
    with _TRANSCRIPTION_LOCK:
        return _transcribe_audio(audio, settings, language)


def _transcribe_audio(audio: bytes, settings: Settings, language: str) -> str:
    """Synchronous preparation and recognition boundary; caller owns the lock."""

    tmp_dir = settings.whisper_tmp_dir
    if not tmp_dir.exists():
        raise WhisperError(f"Temporary directory not found: {tmp_dir}")

    _validate_preprocessing(settings)

    session_id = uuid.uuid4().hex
    output_prefix = tmp_dir / f"whisper_output_{session_id}"
    output_path = output_prefix.with_suffix(".txt")

    if output_path.exists():
        output_path.unlink()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Client-supplied names may contain absolute paths or parent components.
        input_path = Path(tmpdir) / "upload.audio"
        input_path.write_bytes(audio)

        whisper_input_path = tmp_dir / f"whisper_input_{session_id}.wav"
        _prepare_audio(input_path, whisper_input_path, Path(tmpdir), settings)
        if _wav_frame_count(whisper_input_path) == 0:
            output_path.write_text("", encoding="utf-8")
            return ""

        vad_args = []
        if settings.whisper_preprocessing != "legacy":
            vad_args = [
                "--vad",
                "--vad-model", str(settings.whisper_vad_model_path),
                "--vad-threshold", str(settings.whisper_vad_threshold),
                "--vad-min-speech-duration-ms", str(settings.whisper_vad_min_speech_duration_ms),
                "--vad-min-silence-duration-ms", str(settings.whisper_vad_min_silence_duration_ms),
                "--vad-speech-pad-ms", str(settings.whisper_vad_speech_pad_ms),
            ]

        whisper_cmd = [
            settings.whisper_bin,
            "-m",
            settings.whisper_model_path,
            "-f",
            str(whisper_input_path),
            "-otxt",
            "-of",
            str(output_prefix),
            *vad_args,
            "-l",
            language,
            *settings.whisper_args,
        ]
        _run_command(whisper_cmd, "Whisper failed to process the audio file.")

        if not output_path.exists():
            raise WhisperError("Whisper output not found.")

        return output_path.read_text(encoding="utf-8")


def _validate_preprocessing(settings: Settings) -> None:
    profile = settings.whisper_preprocessing
    if profile not in {"legacy", "vad", "deepfilter"}:
        raise WhisperError(f"Unknown Whisper preprocessing profile: {profile}")
    if profile == "legacy":
        return
    if not settings.whisper_vad_model_path.is_file():
        raise WhisperError(f"VAD model not found: {settings.whisper_vad_model_path}")
    if not 0 <= settings.whisper_vad_threshold <= 1:
        raise WhisperError("VAD threshold must be between 0 and 1.")
    if any(value < 0 for value in (
        settings.whisper_vad_min_speech_duration_ms,
        settings.whisper_vad_min_silence_duration_ms,
        settings.whisper_vad_speech_pad_ms,
    )):
        raise WhisperError("VAD durations must not be negative.")
    if profile == "deepfilter":
        if shutil.which(settings.whisper_deepfilter_bin) is None:
            raise WhisperError(f"DeepFilterNet executable not found: {settings.whisper_deepfilter_bin}")
        if not settings.whisper_deepfilter_model_path.is_file():
            raise WhisperError(f"DeepFilterNet model not found: {settings.whisper_deepfilter_model_path}")
        attenuation = settings.whisper_deepfilter_attenuation_limit_db
        if not math.isfinite(attenuation) or attenuation < 0.01:
            raise WhisperError("DeepFilterNet attenuation limit must be finite and at least 0.01 dB.")


def _prepare_audio(input_path: Path, output_path: Path, work_dir: Path, settings: Settings) -> None:
    profile = settings.whisper_preprocessing
    if profile == "legacy":
        _convert_audio(input_path, output_path, settings, filters=LEGACY_AUDIO_FILTERS)
        return

    filters = NORMALIZE_AUDIO_FILTER if settings.whisper_normalize else None
    if profile == "deepfilter":
        decoded_path = work_dir / "decoded-48k.wav"
        _convert_audio(input_path, decoded_path, settings, sample_rate=48000)
        frame_count = _wav_frame_count(decoded_path)
        if frame_count == 0:
            _convert_audio(decoded_path, output_path, settings)
            return

        padded_path = work_dir / "deepfilter-input.wav"
        _pad_deepfilter_input(decoded_path, padded_path)
        enhanced_dir = work_dir / "enhanced"
        enhanced_dir.mkdir()
        _run_command([
            settings.whisper_deepfilter_bin,
            "-m", str(settings.whisper_deepfilter_model_path),
            "-a", str(settings.whisper_deepfilter_attenuation_limit_db),
            "-D",
            "-o", str(enhanced_dir),
            str(padded_path),
        ], "DeepFilterNet failed to process the audio file.")
        input_path = enhanced_dir / padded_path.name
        if not input_path.is_file():
            raise WhisperError("DeepFilterNet output not found.")
        if _wav_frame_count(input_path) < frame_count:
            raise WhisperError("DeepFilterNet output is shorter than the original audio.")
        # -D compensates the model delay; remove only the padding added below.
        trim_filter = f"atrim=end_sample={frame_count}"
        filters = f"{trim_filter},{filters}" if filters else trim_filter

    _convert_audio(input_path, output_path, settings, filters=filters)


def _pad_deepfilter_input(source: Path, destination: Path) -> None:
    # DFN3 uses a 480-sample hop and 1440-sample delay at 48 kHz. Its CLI
    # truncates partial hops and removes the delay with -D. Reflect 40 ms of
    # the tail to flush the model: zero padding triggers its silence shortcut
    # and can discard the original final speech samples instead.
    with wave.open(str(source), "rb") as reader, wave.open(str(destination), "wb") as writer:
        writer.setparams(reader.getparams())
        frame_count = reader.getnframes()
        frame_width = reader.getsampwidth() * reader.getnchannels()
        tail_frames = min(frame_count, 1920)
        reader.setpos(frame_count - tail_frames)
        tail = reader.readframes(tail_frames)
        reader.rewind()
        while chunk := reader.readframes(65536):
            writer.writeframesraw(chunk)
        if tail:
            reflected = b"".join(
                tail[index:index + frame_width]
                for index in reversed(range(0, len(tail), frame_width))
            )
            padding = (reflected * ((1920 + tail_frames - 1) // tail_frames))[:1920 * frame_width]
            writer.writeframesraw(padding)


def _wav_frame_count(path: Path) -> int:
    try:
        with wave.open(str(path), "rb") as audio:
            return audio.getnframes()
    except (OSError, EOFError, wave.Error) as exc:
        raise WhisperError(f"Unable to read converted WAV: {path}") from exc


def _convert_audio(
    source: Path,
    destination: Path,
    settings: Settings,
    *,
    sample_rate: int = 16000,
    filters: str | None = None,
) -> None:
    cmd = [settings.ffmpeg_bin, "-i", str(source)]
    if filters:
        cmd.extend(["-af", filters])
    cmd.extend(["-ar", str(sample_rate), "-ac", "1", "-c:a", "pcm_s16le", "-y", str(destination)])
    _run_command(cmd, "ffmpeg conversion failed.")


def _run_command(cmd: list[str], error_message: str) -> None:
    try:
        subprocess.run(cmd, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise WhisperError(error_message) from exc
