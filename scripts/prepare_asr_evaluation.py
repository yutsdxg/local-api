#!/usr/bin/env python3
"""Prepare one shared, VAD-filtered PCM input for ASR comparisons.

This evaluation-only helper binds the whisper.cpp 1.8.2 VAD ABI, runs Silero
on the CPU, and preserves original PCM16 samples. It does not transcribe.
Output directory reuse is deliberately forbidden.

The concatenation follows whisper.cpp's whisper_vad(): extend each non-final
segment by 100 ms and insert 100 ms of zeros between segments. Two deliberate
differences are recorded in every manifest: correct exclusive-end allocation
avoids 1.8.2's n_samples-1 sizing / n_samples copy mismatch; source mappings
describe the actual copied samples, including overlap, instead of interpolating
the extended segment onto the unextended VAD timestamps.
"""

from __future__ import annotations

import argparse
import array
import ctypes as c
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import wave


SAMPLE_RATE = 16000
SUPPORTED_WHISPER_VERSION = "1.8.2"
OVERLAP_SAMPLES = 1600
GAP_SAMPLES = 1600


class ContextParams(c.Structure):
    _fields_ = [("n_threads", c.c_int), ("use_gpu", c.c_bool), ("gpu_device", c.c_int)]


class VadParams(c.Structure):
    _fields_ = [
        ("threshold", c.c_float),
        ("min_speech_duration_ms", c.c_int),
        ("min_silence_duration_ms", c.c_int),
        ("max_speech_duration_s", c.c_float),
        ("speech_pad_ms", c.c_int),
        ("samples_overlap", c.c_float),
    ]


class WhisperVad:
    """Version-gated C API binding; no ASR model or GPU is used."""

    def __init__(self, library_path: Path, model_path: Path):
        self.library = c.CDLL(str(library_path))
        version = self._bind("whisper_version", c.c_char_p)()
        self.version = version.decode("ascii") if version else "unknown"
        if self.version != SUPPORTED_WHISPER_VERSION:
            raise ValueError(
                f"Unsupported whisper.cpp ABI: {self.version}; expected "
                f"{SUPPORTED_WHISPER_VERSION}. Review ctypes structures before upgrading."
            )
        default_context = self._bind("whisper_vad_default_context_params", ContextParams)
        default_params = self._bind("whisper_vad_default_params", VadParams)
        self.context_params = default_context()
        self.context_params.use_gpu = False
        self.params = default_params()
        self.params.threshold = 0.5
        self.params.min_speech_duration_ms = 100
        self.params.min_silence_duration_ms = 500
        self.params.speech_pad_ms = 200
        self.params.samples_overlap = 0.1
        self.model_path = model_path
        self._init = self._bind(
            "whisper_vad_init_from_file_with_params", c.c_void_p, c.c_char_p, ContextParams
        )
        self._segments = self._bind(
            "whisper_vad_segments_from_samples", c.c_void_p,
            c.c_void_p, VadParams, c.POINTER(c.c_float), c.c_int,
        )
        self._count = self._bind("whisper_vad_segments_n_segments", c.c_int, c.c_void_p)
        self._t0 = self._bind(
            "whisper_vad_segments_get_segment_t0", c.c_float, c.c_void_p, c.c_int
        )
        self._t1 = self._bind(
            "whisper_vad_segments_get_segment_t1", c.c_float, c.c_void_p, c.c_int
        )
        self._free_segments = self._bind("whisper_vad_free_segments", None, c.c_void_p)
        self._free = self._bind("whisper_vad_free", None, c.c_void_p)

    def _bind(self, name, result, *args):
        function = getattr(self.library, name)
        function.restype = result
        function.argtypes = list(args)
        return function

    def settings(self) -> dict:
        return {
            "context": {name: getattr(self.context_params, name)
                        for name, _ in ContextParams._fields_},
            "vad": {name: getattr(self.params, name) for name, _ in VadParams._fields_},
            "concatenation": {
                "end_overlap_samples": OVERLAP_SAMPLES,
                "end_overlap_seconds": 0.1,
                "zero_gap_samples": GAP_SAMPLES,
                "zero_gap_seconds": 0.1,
            },
        }

    def detect(self, samples: array.array) -> list[tuple[float, float]]:
        if len(samples) > 2**31 - 1:
            raise ValueError("Input exceeds the C API's signed 32-bit sample count")
        if not samples:
            return []
        # Same PCM16 -> float conversion as whisper-cli, without requantization.
        audio = array.array("f", (value / 32768.0 for value in samples))
        buffer = (c.c_float * len(audio)).from_buffer(audio)
        context = self._init(os.fsencode(self.model_path), self.context_params)
        if not context:
            raise RuntimeError("Failed to initialize Silero VAD context")
        try:
            regions = self._segments(context, self.params, buffer, len(audio))
            if not regions:
                raise RuntimeError("Silero VAD failed to return speech segments")
            try:
                count = self._count(regions)
                if count < 0 or count > len(samples):
                    raise RuntimeError("Invalid VAD segment count")
                return [(self._t0(regions, i), self._t1(regions, i)) for i in range(count)]
            finally:
                self._free_segments(regions)
        finally:
            self._free(context)


def read_pcm16(path: Path) -> array.array:
    with wave.open(str(path), "rb") as source:
        if (source.getframerate(), source.getnchannels(), source.getsampwidth(),
                source.getcomptype()) != (SAMPLE_RATE, 1, 2, "NONE"):
            raise ValueError("Source must be uncompressed 16000 Hz, mono, PCM16 WAV")
        frames = source.getnframes()
        if frames > 2**31 - 1:
            raise ValueError("Input exceeds the C API's signed 32-bit sample count")
        raw = source.readframes(frames)
        if len(raw) != frames * 2:
            raise ValueError("Truncated WAV: PCM data does not match the frame count")
    samples = array.array("h")
    samples.frombytes(raw)
    if sys.byteorder != "little":
        samples.byteswap()
    return samples


def _sample_range(start: int, end: int) -> dict:
    return {
        "start_sample": start,
        "end_sample": end,
        "start_seconds": start / SAMPLE_RATE,
        "end_seconds": end / SAMPLE_RATE,
    }


def extract_segments(
    samples: array.array, segments_cs: list[tuple[float, float]],
) -> tuple[array.array, list[dict], list[dict]]:
    """Return copied PCM, source mappings, and unmapped synthetic silence gaps.

    All sample ranges are half-open [start, end). VAD returns centiseconds
    rounded from its 32 ms analysis windows; endpoints may exceed the WAV EOF.
    Such endpoints are clamped. Invalid, non-finite or unordered data is rejected.
    """
    if samples.typecode != "h":
        raise ValueError("Expected signed PCM16 samples in array('h')")
    selected = []
    previous_start = -1
    for index, (start_cs, end_cs) in enumerate(segments_cs):
        if (not math.isfinite(start_cs) or not math.isfinite(end_cs)
                or start_cs < 0 or end_cs < start_cs or start_cs < previous_start
                or not float(start_cs).is_integer() or not float(end_cs).is_integer()):
            raise ValueError(f"Invalid VAD interval at index {index}: {(start_cs, end_cs)}")
        previous_start = start_cs
        # The C API exposes integer centiseconds as float. Match cs_to_samples.
        start = min(int(start_cs) * 160, len(samples))
        end = min(int(end_cs) * 160, len(samples))
        extended_end = min(
            int(end_cs) * 160 + (OVERLAP_SAMPLES if index < len(segments_cs) - 1 else 0),
            len(samples),
        )
        if end <= start:
            continue
        selected.append((index, start_cs, end_cs, start, end, extended_end))

    output = array.array("h")
    mapping = []
    gaps = []
    for position, (index, start_cs, end_cs, start, end, extended_end) in enumerate(selected):
        output_start = len(output)
        output.extend(samples[start:extended_end])
        mapping.append({
            "segment_index": index,
            "vad_start_centiseconds": start_cs,
            "vad_end_centiseconds": end_cs,
            "source": _sample_range(start, extended_end),
            "source_before_overlap": _sample_range(start, end),
            "output": _sample_range(output_start, len(output)),
            "added_end_overlap_samples": extended_end - end,
        })
        if position < len(selected) - 1:
            gap_start = len(output)
            output.extend(array.array("h", [0]) * GAP_SAMPLES)
            gaps.append({"output": _sample_range(gap_start, len(output)), "source": None})
    return output, mapping, gaps


def _sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def prepare_evaluation(
    source: Path, output_dir: Path, whisper_lib: Path, vad_model: Path,
) -> dict:
    source, output_dir, whisper_lib, vad_model = (
        Path(path).absolute() for path in (source, output_dir, whisper_lib, vad_model)
    )
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Output directory already exists; reuse is refused: {output_dir}")
    for path in (source, whisper_lib, vad_model):
        if not path.is_file():
            raise FileNotFoundError(f"Required input is not a file: {path}")
    samples = read_pcm16(source)
    vad = WhisperVad(whisper_lib, vad_model)
    intervals = vad.detect(samples)
    output, segments, gaps = extract_segments(samples, intervals)
    artifacts = {
        "source": {"path": str(source), "sha256": _sha256(source),
                   "samples": len(samples), "duration_seconds": len(samples) / SAMPLE_RATE},
        "whisper_library": {"path": str(whisper_lib), "sha256": _sha256(whisper_lib),
                            "version": vad.version},
        "vad_model": {"path": str(vad_model), "sha256": _sha256(vad_model)},
    }
    # mkdir(exist_ok=False) also rejects a competing creator after the initial check.
    output_dir.mkdir(parents=True, exist_ok=False)
    prepared = output_dir / "prepared.wav"
    raw_output = array.array("h", output)
    if sys.byteorder != "little":
        raw_output.byteswap()
    with wave.open(str(prepared), "wb") as target:
        target.setparams((1, 2, SAMPLE_RATE, 0, "NONE", "not compressed"))
        target.writeframes(raw_output.tobytes())
    manifest = {
        "schema_version": 1,
        "purpose": "shared_vad_input_for_asr_evaluation_only",
        "status": "empty_input" if not samples else ("ready" if segments else "no_speech"),
        "sample_rate": SAMPLE_RATE,
        "sample_format": "PCM16_LE",
        "channels": 1,
        **artifacts,
        "output": {"path": str(prepared), "sha256": _sha256(prepared),
                   "samples": len(output), "duration_seconds": len(output) / SAMPLE_RATE},
        "settings": vad.settings(),
        "segments": segments,
        "zero_gaps": gaps,
        "raw_vad_segments_centiseconds": intervals,
        "mapping_convention": "half-open sample ranges; source/output offset is 1:1; gaps have no source",
        "implementation_notes": [
            "VAD ABI is restricted to whisper.cpp 1.8.2; model/context defaults come from the library.",
            "CPU VAD only. ASR evaluation must disable additional VAD on prepared.wav.",
            "Non-final VAD segments receive up to 100 ms of source audio at their end; retained segments are separated by 100 ms of zeros.",
            "1.8.2 whisper_vad sizes its buffer with end <= n_samples-1 but copies with end <= n_samples; this helper uses consistent exclusive endpoints and does not reproduce the potential one-sample overrun per segment reaching EOF.",
            "Unlike the internal interpolated mapping, these mappings retain the true source positions of copied overlap samples.",
            "Intervals are clamped to WAV bounds; zero-length intervals are omitted. No surviving speech produces a valid zero-frame WAV; skip ASR for no_speech/empty_input.",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="16000 Hz mono PCM16 WAV")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory, must not exist")
    parser.add_argument("--whisper-lib", type=Path, required=True, help="whisper.cpp 1.8.2 shared library")
    parser.add_argument("--vad-model", type=Path, required=True, help="Silero VAD ggml model")
    args = parser.parse_args(argv)
    try:
        prepare_evaluation(args.source, args.output_dir, args.whisper_lib, args.vad_model)
    except (OSError, EOFError, ValueError, RuntimeError, wave.Error) as error:
        parser.exit(2, f"error: {error}\n")
    print((args.output_dir / "manifest.json").absolute())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
