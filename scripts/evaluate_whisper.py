"""Compare local Whisper preprocessing on one identical excerpt per profile.

Run from the repository root. Audio and transcripts stay in the chosen output
directory; use a directory under ignored data/ for private recordings.
The existing transcript is a comparison artifact, never a ground-truth label.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
import wave
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import UploadFile

from app.config import Settings
from app.services.whisper import transcribe_upload


PROFILES = ("legacy", "vad", "vad_normalized", "deepfilter", "deepfilter_normalized")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def wav_info(path: Path) -> dict[str, int | float]:
    with wave.open(str(path), "rb") as audio:
        return {
            "sample_rate": audio.getframerate(),
            "channels": audio.getnchannels(),
            "sample_width_bytes": audio.getsampwidth(),
            "frames": audio.getnframes(),
            "seconds": audio.getnframes() / audio.getframerate(),
        }


def metrics(text: str) -> dict[str, int]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return {
        "characters": len(text),
        "nonempty_lines": len(lines),
        "adjacent_duplicate_lines": sum(a == b for a, b in zip(lines, lines[1:])),
    }


def run_profile(source: Path, output: Path, profile: str, language: str) -> dict:
    preprocessing = profile.removesuffix("_normalized")
    settings = replace(
        Settings.load(),
        whisper_tmp_dir=output,
        whisper_preprocessing=preprocessing,
        whisper_normalize=profile.endswith("_normalized"),
    )
    with source.open("rb") as stream:
        upload = UploadFile(filename=source.name, file=stream)
        text = asyncio.run(transcribe_upload(upload, settings, language=language))
    (output / "transcript.txt").write_text(text, encoding="utf-8")
    inputs = sorted(output.glob("whisper_input_*.wav"))
    return {
        "profile": profile,
        "effective_settings": {
            "whisper_preprocessing": settings.whisper_preprocessing,
            "whisper_normalize": settings.whisper_normalize,
        },
        "transcript": str(output / "transcript.txt"),
        "transcript_sha256": sha256(output / "transcript.txt"),
        "metrics": metrics(text),
        "prepared_audio": wav_info(inputs[0]) if inputs else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--profiles", nargs="+", choices=PROFILES, default=list(PROFILES[:4]))
    parser.add_argument("--language", default="ja")
    parser.add_argument("--existing-transcript", type=Path)
    parser.add_argument("--worker", choices=PROFILES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not math.isfinite(args.start) or args.start < 0:
        parser.error("--start must be finite and nonnegative")
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0):
        parser.error("--duration must be finite and positive")
    if len(set(args.profiles)) != len(args.profiles):
        parser.error("duplicate profiles are not allowed")

    source = args.audio.resolve(strict=True)
    output = args.output_dir.resolve()
    if args.worker:
        result = run_profile(source, output, args.worker, args.language)
        (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        return

    # Refuse reuse: this prevents a prior excerpt or setting from being mistaken
    # for the result of the current invocation.
    output.mkdir(parents=True, exist_ok=False)
    settings = Settings.load()
    excerpt = output / "source.wav"
    command = [settings.ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-ss", str(args.start), "-i", str(source)]
    if args.duration is not None:
        command += ["-t", str(args.duration)]
    # Preserve native channels/rate and up to 32-bit integer precision (the
    # supplied ALAC recordings are 24-bit) until the production pipeline.
    command += ["-map", "0:a:0", "-c:a", "pcm_s32le", str(excerpt)]
    subprocess.run(command, check=True)
    if wav_info(excerpt)["frames"] == 0:
        raise ValueError("The selected excerpt contains no audio samples.")

    model_paths = {
        "whisper": Path(settings.whisper_model_path),
        "vad": Path(settings.whisper_vad_model_path),
        "deepfilter": Path(settings.whisper_deepfilter_model_path),
    }
    binaries = {}
    for name, configured in {
        "ffmpeg": settings.ffmpeg_bin,
        "whisper": settings.whisper_bin,
        "deepfilter": settings.whisper_deepfilter_bin,
    }.items():
        resolved = shutil.which(configured)
        path = Path(resolved).resolve() if resolved else None
        binaries[name] = {"configured": configured, "resolved": str(path) if path else None, "sha256": sha256(path) if path else None}
    root = Path(__file__).resolve().parents[1]
    report = {
        "source": str(source),
        "source_sha256": sha256(source),
        "start_seconds": args.start,
        "duration_requested_seconds": args.duration,
        "excerpt": wav_info(excerpt),
        "excerpt_sha256": sha256(excerpt),
        "language": args.language,
        "whisper_bin": settings.whisper_bin,
        "whisper_args": list(settings.whisper_args),
        "binaries": binaries,
        "implementation_sha256": {name: sha256(root / name) for name in ("app/config.py", "app/services/whisper.py", "scripts/evaluate_whisper.py")},
        "models": {name: {"path": str(path), "sha256": sha256(path) if path.is_file() else None} for name, path in model_paths.items()},
        "settings": {
            key: str(value) if isinstance(value, Path) else value
            for key in settings.__dataclass_fields__
            if key.startswith("whisper_") and key not in {"whisper_tmp_dir", "whisper_bin", "whisper_args", "whisper_model_path", "whisper_preprocessing", "whisper_normalize"}
            for value in [getattr(settings, key)]
        },
        "existing_transcript": None,
        "results": [],
        "limitations": "No ground truth supplied. Text differences and duplicate counts are not accuracy scores. Cropping changes Whisper context and filter boundaries.",
    }
    if args.existing_transcript:
        reference = args.existing_transcript.resolve(strict=True)
        report["existing_transcript"] = {"path": str(reference), "sha256": sha256(reference), "scope": "historical full recording; not ground truth"}
    report_path = output / "report.json"
    for profile in args.profiles:
        profile_dir = output / profile
        profile_dir.mkdir()
        started = time.monotonic()
        with (profile_dir / "process.log").open("w") as log:
            completed = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), str(excerpt), "--output-dir", str(profile_dir), "--language", args.language, "--worker", profile],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        elapsed = round(time.monotonic() - started, 3)
        if completed.returncode:
            result = {"profile": profile, "error": f"exit {completed.returncode}; see process.log"}
        else:
            result = json.loads((profile_dir / "result.json").read_text())
        result["elapsed_seconds"] = elapsed
        report["results"].append(result)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"{profile}: {elapsed:.1f}s {'FAILED' if completed.returncode else 'OK'}", flush=True)
    if any("error" in result for result in report["results"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
