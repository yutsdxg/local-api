"""Keep one local ASR model loaded for sequential JSON Lines evaluation.

Run with the dedicated ASR Python environment, not the API environment::

    python scripts/asr_worker.py --backend qwen --model /absolute/model-directory

After ``ready``, send one JSON object per line with ``id``, ``audio_path``,
``language`` (default ``ja``), optional ``prompt`` and optional ``options``.
The audio must already be 16 kHz, mono, PCM16 WAV. This worker performs no VAD
or denoising; the caller owns segmentation and maps relative timestamps back
to the original recording. Qwen segments describe input chunks, not alignment.
Send {"action": "shutdown"} or close stdin to exit. Only protocol JSON goes
to stdout. Install dependencies and download all model assets beforehand.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.metadata
import json
import math
import os
import sys
import time
import wave
from pathlib import Path
from typing import Any, TextIO


BACKENDS = ("mlx-whisper", "qwen", "parakeet")
DEFAULT_OPTIONS: dict[str, dict[str, Any]] = {
    "mlx-whisper": {"temperature": 0.0, "condition_on_previous_text": False},
    "qwen": {"temperature": 0.0, "max_tokens": 8192,
             "chunk_duration": 1200.0, "min_chunk_duration": 1.0},
    "parakeet": {"chunk_duration": None},
}


def local_path(value: Any, *, directory: bool = False) -> Path:
    if not isinstance(value, str) or not value or not Path(value).is_absolute():
        raise ValueError("An absolute local path is required.")
    path = Path(value).resolve(strict=True)
    if not (path.is_dir() if directory else path.is_file()):
        raise ValueError(f"Expected a local {'directory' if directory else 'file'}: {path}")
    return path


def read_pcm(path: Path) -> tuple[bytes, float]:
    with wave.open(str(path), "rb") as reader:
        if (reader.getframerate(), reader.getnchannels(), reader.getsampwidth(),
                reader.getcomptype()) != (16000, 1, 2, "NONE"):
            raise ValueError("Audio must be a prepared 16 kHz mono PCM16 WAV.")
        frames = reader.getnframes()
        if frames == 0:
            raise ValueError("Audio contains no samples.")
        pcm = reader.readframes(frames)
    if len(pcm) != frames * 2:
        raise ValueError("WAV audio is truncated.")
    return pcm, frames / 16000


def request_options(backend: str, supplied: Any) -> dict[str, Any]:
    if not isinstance(supplied, dict):
        raise ValueError("options must be an object.")
    defaults = DEFAULT_OPTIONS[backend]
    unknown = set(supplied) - defaults.keys()
    if unknown:
        raise ValueError(f"Unsupported {backend} options: {', '.join(sorted(unknown))}")
    options = defaults | supplied
    for key, value in options.items():
        if key == "condition_on_previous_text":
            if not isinstance(value, bool):
                raise ValueError(f"{key} must be a boolean.")
        elif key == "chunk_duration" and backend == "parakeet" and value is None:
            continue
        elif key == "max_tokens":
            if type(value) is not int or value <= 0:
                raise ValueError("max_tokens must be a positive integer.")
        elif type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"{key} must be a finite number.")
        elif value < 0 or (key != "temperature" and value == 0):
            raise ValueError(f"Invalid {key}: {value}")
    if backend == "qwen" and options["min_chunk_duration"] > options["chunk_duration"]:
        raise ValueError("min_chunk_duration must not exceed chunk_duration.")
    return options


def normalize_segments(segments: Any, duration: float) -> list[dict[str, Any]]:
    normalized = []
    for segment in segments:
        def get(name: str) -> Any:
            return segment[name] if isinstance(segment, dict) else getattr(segment, name)

        start, end = float(get("start")), float(get("end"))
        text = get("text")
        if not math.isfinite(start) or not math.isfinite(end) or end < start:
            raise ValueError("Backend returned invalid segment timestamps.")
        if not isinstance(text, str):
            raise ValueError("Backend returned non-text segment content.")
        # Qwen pads sub-second inputs; never report that padding as source audio.
        normalized.append({"start": min(duration, max(0.0, start)),
                           "end": min(duration, max(0.0, end)), "text": text})
    return normalized


class LocalBackend:
    def __init__(self, name: str, model_path: Path):
        self.name = name
        self.model_path = model_path
        self.np = importlib.import_module("numpy")
        self.mx = importlib.import_module("mlx.core")
        if name == "mlx-whisper":
            self.whisper = importlib.import_module("mlx_whisper.transcribe")
            self.model = self.whisper.ModelHolder.get_model(str(model_path), self.mx.float16)
            self.mx.eval(self.model.parameters())
        else:
            load = importlib.import_module("mlx_audio.stt.utils").load
            # A Path bypasses Hugging Face repository resolution entirely.
            self.model = load(model_path, lazy=False, strict=True,
                              model_type="qwen3_asr" if name == "qwen" else "parakeet")

    def transcribe(self, pcm: bytes, duration: float, language: str | None,
                   prompt: str | None, options: dict[str, Any]) -> dict[str, Any]:
        audio = self.np.frombuffer(pcm, dtype="<i2").astype(self.np.float32) / self.np.float32(32768)
        self.mx.reset_peak_memory()
        warnings = []
        generation_tokens = None
        if self.name == "mlx-whisper":
            result = self.whisper.transcribe(
                audio, path_or_hf_repo=str(self.model_path), language=language,
                initial_prompt=prompt, verbose=None, fp16=True, **options,
            )
            text, segments = result["text"], result["segments"]
            timestamp_kind = "whisper-segment"
        elif self.name == "qwen":
            if language not in {None, "ja", "Japanese"}:
                raise ValueError("Qwen evaluation supports language='ja' or 'auto'.")
            result = self.model.generate(
                audio, language="Japanese" if language else None, system_prompt=prompt,
                batch_size=1, stream=False, verbose=False, **options,
            )
            text, segments = result.text, result.segments
            timestamp_kind = "input-chunk"
            generation_tokens = int(result.generation_tokens)
            if generation_tokens >= options["max_tokens"]:
                warnings.append("Generation token budget reached; inspect the output for truncation.")
        else:
            if language not in {None, "ja", "Japanese"}:
                raise ValueError("This Parakeet evaluation model supports Japanese only.")
            if prompt:
                raise ValueError("Parakeet does not support a transcription prompt.")
            result = self.model.generate(
                self.mx.array(audio), dtype=self.mx.float32,
                stream=False, verbose=False, **options,
            )
            text, segments = result.text, result.sentences
            timestamp_kind = "parakeet-sentence"
        if not isinstance(text, str):
            raise ValueError("Backend returned non-text transcription content.")
        return {"text": text, "segments": normalize_segments(segments, duration),
                "timestamp_kind": timestamp_kind, "generation_tokens": generation_tokens,
                "peak_memory_bytes": int(self.mx.get_peak_memory()),
                "memory_scope": "MLX active allocation peak for this request; excludes host RSS and allocator cache",
                "warnings": warnings}


def load_backend(name: str, model_path: Path) -> LocalBackend:
    return LocalBackend(name, model_path)


def write_event(output: TextIO, event: dict[str, Any]) -> None:
    output.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
    output.flush()


def error_event(event: str, exc: Exception, request_id: Any = None) -> dict[str, Any]:
    return {"event": event, "id": request_id,
            "error": {"type": type(exc).__name__, "message": str(exc)}}


def package_versions() -> dict[str, str]:
    versions = {}
    for package in ("mlx", "mlx-whisper", "mlx-audio", "numpy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return versions


def serve(backend_name: str, model: str, source: TextIO, output: TextIO) -> int:
    # Set these before importing ML packages. Missing local tokenizer/processor
    # assets must fail rather than being fetched implicitly during model load.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    try:
        model_path = local_path(model, directory=True)
        if backend_name not in BACKENDS:
            raise ValueError(f"Unknown backend: {backend_name}")
        started = time.perf_counter()
        with contextlib.redirect_stdout(sys.stderr):
            backend = load_backend(backend_name, model_path)
        load_seconds = time.perf_counter() - started
        write_event(output, {"event": "ready", "backend": backend_name,
                             "model": str(model_path), "load_seconds": load_seconds,
                             "packages": package_versions(),
                             "default_options": DEFAULT_OPTIONS[backend_name],
                             "vad": "none; caller supplies prepared segments"})
    except Exception as exc:
        write_event(output, error_event("startup_error", exc))
        return 1

    for line in source:
        request_id = None
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("Each request must be a JSON object.")
            request_id = request.get("id")
            if request_id is not None and type(request_id) not in (str, int):
                request_id = None
                raise ValueError("id must be a string, integer, or null.")
            if request.get("action") == "shutdown":
                write_event(output, {"event": "stopped", "id": request_id})
                return 0
            if request.get("action", "transcribe") != "transcribe":
                raise ValueError("Unknown action.")
            unknown = set(request) - {"action", "id", "audio_path", "language", "prompt", "options"}
            if unknown:
                raise ValueError(f"Unknown request fields: {', '.join(sorted(unknown))}")
            path = local_path(request.get("audio_path"))
            language = request.get("language", "ja")
            if language is not None and (not isinstance(language, str) or not language):
                raise ValueError("language must be a nonempty string or null.")
            if language == "auto":
                language = None
            prompt = request.get("prompt")
            if prompt is not None and not isinstance(prompt, str):
                raise ValueError("prompt must be a string or null.")
            options = request_options(backend_name, request.get("options", {}))
            started = time.perf_counter()
            pcm, duration = read_pcm(path)
            with contextlib.redirect_stdout(sys.stderr):
                result = backend.transcribe(pcm, duration, language, prompt, options)
            write_event(output, {"event": "result", "id": request_id, **result,
                                 "elapsed_seconds": time.perf_counter() - started,
                                 "audio_seconds": duration, "options": options})
        except Exception as exc:
            write_event(output, error_event("error", exc, request_id))
    return 0


def protocol_stdout() -> TextIO:
    """Reserve the original stdout for JSON, including against native logs."""
    sys.stdout.flush()
    output = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=BACKENDS)
    parser.add_argument("--model", required=True, help="Absolute local model directory")
    args = parser.parse_args()
    with protocol_stdout() as output:
        return serve(args.backend, args.model, sys.stdin, output)


if __name__ == "__main__":
    raise SystemExit(main())
