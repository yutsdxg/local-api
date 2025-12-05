from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from app.config import Settings

logger = logging.getLogger(__name__)


class ChordMelodyPathError(ValueError):
    """Raised when configured paths are invalid."""


class ChordMelodyExecutionError(RuntimeError):
    """Raised when analysis or rename fails."""


@dataclass
class TimeSlice:
    start_time: float
    end_time: float
    notes: List[int]
    is_poly: bool

    def get_lowest_note(self) -> int:
        if not self.notes:
            return -1
        return min(self.notes)

    def has_same_lowest_note(self, other: "TimeSlice") -> bool:
        if not (self.is_poly and other.is_poly):
            return False
        return self.get_lowest_note() == other.get_lowest_note()


class MidiAnalyzer:
    def __init__(
        self,
        time_unit: float,
        poly_threshold: float,
        poly_note_count: int,
        chord_stability_threshold: float,
    ) -> None:
        self.time_unit = time_unit
        self.poly_threshold = poly_threshold
        self.poly_note_count = poly_note_count
        self.chord_stability_threshold = chord_stability_threshold
        self._logger = logger.getChild("midi_analyzer")

    def analyze_file(self, file_path: Path) -> str:
        if not file_path.exists():
            raise ChordMelodyPathError(f"File not found: {file_path}")
        if not file_path.is_file():
            raise ChordMelodyPathError(f"Expected file, got directory: {file_path}")

        try:
            model_output = _predict_audio(str(file_path))
            return self._analyze_midi_data(model_output)
        except MemoryError as exc:
            raise ChordMelodyExecutionError(f"Out of memory while analyzing: {file_path}") from exc
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("Analyze failed for %s", file_path)
            raise ChordMelodyExecutionError(
                f"Failed to analyze audio file: {file_path} (cause: {exc})"
            ) from exc

    def _analyze_midi_data(self, model_output: Tuple) -> str:
        midi_notes, onsets, frames = model_output  # noqa: F841
        time_slices: list[TimeSlice] = []

        max_time = 0.0
        for instrument in onsets.instruments:
            if instrument.notes:
                max_time = max(max_time, max(note.end for note in instrument.notes))

        current_time = 0.0
        while current_time < max_time:
            start_time = current_time
            end_time = min(current_time + self.time_unit, max_time)

            current_notes = set()
            for instrument in onsets.instruments:
                for note in instrument.notes:
                    if note.start <= end_time and note.end >= start_time:
                        current_notes.add(note.pitch)

            is_poly = len(current_notes) >= self.poly_note_count
            time_slice = TimeSlice(
                start_time=start_time,
                end_time=end_time,
                notes=list(current_notes),
                is_poly=is_poly,
            )
            time_slices.append(time_slice)
            current_time += self.time_unit

        if not time_slices:
            return "MELODY"

        poly_count = sum(1 for slice_ in time_slices if slice_.is_poly)
        poly_ratio = poly_count / len(time_slices)

        if poly_ratio < self.poly_threshold:
            return "MELODY"

        poly_slices = [slice_ for slice_ in time_slices if slice_.is_poly]
        if len(poly_slices) <= 1:
            return "CHORD"

        base_slice = poly_slices[0]
        stable_count = sum(1 for slice_ in poly_slices[1:] if base_slice.has_same_lowest_note(slice_))
        stability_ratio = stable_count / (len(poly_slices) - 1)

        return "CHORD1" if stability_ratio >= self.chord_stability_threshold else "CHORD"


def analyze_chord_melody(settings: Settings) -> list[dict[str, str]]:
    input_dir = settings.chord_melody_input_dir
    log_dir = settings.chord_melody_log_dir
    _ensure_directory(input_dir, "Input directory not found or not a directory")
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "analysis.log"
    _configure_logger(log_file)

    analyzer = MidiAnalyzer(
        time_unit=settings.chord_melody_time_unit,
        poly_threshold=settings.chord_melody_poly_threshold,
        poly_note_count=settings.chord_melody_poly_note_count,
        chord_stability_threshold=settings.chord_melody_stability_threshold,
    )

    results: list[dict[str, str]] = []
    target_files = _find_target_files(input_dir)
    logger.info("Target directory: %s", input_dir)
    logger.info("Target file count: %d", len(target_files))
    for relative_path in target_files:
        file_path = input_dir / relative_path
        result = analyzer.analyze_file(file_path)
        logger.info("%s\t%s", relative_path.as_posix(), result)

        try:
            rename_analyzed_file(file_path, result)
        except Exception as exc:  # noqa: BLE001
            raise ChordMelodyExecutionError(f"Failed to rename file: {file_path}") from exc

        results.append({"path": relative_path.as_posix(), "result": result})

    return results


def _find_target_files(input_dir: Path) -> list[Path]:
    files = [
        path.relative_to(input_dir)
        for path in sorted(
            input_dir.rglob("*"),
            key=lambda p: (p.parent.relative_to(input_dir).as_posix(), p.name),
        )
        if path.is_file() and path.suffix.lower() == ".wav" and _is_target_file(path)
    ]
    return files


def _is_target_file(path: Path) -> bool:
    stem = path.stem
    return stem.endswith("_MLD") or stem.endswith("_CHP") or stem.endswith("_CH1")


def rename_analyzed_file(file_path: Path, result: str) -> Path:
    if result not in ("CHORD", "MELODY", "CHORD1"):
        raise ChordMelodyExecutionError(f"Unknown analysis result: {result}")

    if not file_path.exists():
        raise ChordMelodyPathError(f"File not found: {file_path}")
    if not file_path.is_file():
        raise ChordMelodyPathError(f"Expected file, got directory: {file_path}")

    stem = file_path.stem
    if not _is_target_file(file_path):
        raise ChordMelodyExecutionError(f"Unsupported filename format: {file_path}")

    base_name = stem[:-4]
    suffix_map = {"CHORD": "CHP", "MELODY": "MLD", "CHORD1": "CH1"}
    new_stem = f"{base_name}_{suffix_map[result]}"
    new_path = file_path.with_stem(new_stem)

    if file_path.stem == new_stem:
        return file_path
    if new_path.exists():
        raise ChordMelodyExecutionError(f"Target filename already exists: {new_path}")

    file_path.rename(new_path)
    return new_path


def _ensure_directory(path: Path, message: str) -> None:
    if not path.is_dir():
        raise ChordMelodyPathError(f"{message}: {path}")


def _configure_logger(log_file: Path) -> None:
    handler = logging.FileHandler(log_file)
    handler.setFormatter(logging.Formatter("%(message)s"))

    module_logger = logger
    module_logger.setLevel(logging.INFO)
    module_logger.propagate = False

    if module_logger.handlers:
        module_logger.handlers.clear()
    module_logger.addHandler(handler)


def _predict_audio(file_path: str) -> Tuple:
    from basic_pitch.inference import predict

    return predict(file_path)
